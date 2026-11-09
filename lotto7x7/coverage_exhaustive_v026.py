"""Exhaustive four-ticket membership-mask structural certificate (v0.26 P1).

Each labeled four-ticket configuration maps to 15 integer membership counts,
with four constraints of six numbers. Enumerate every feasible nonnegative
solution; exclude duplicate tickets; quotient by S4 ticket permutations.
Exact official-prize union is calculated once per canonical orbit.

Full mode reports a proved global optimum *only* if all valid orbits have
been scored. --max-classes explicitly marks an exploratory partial run.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import permutations
import json
from pathlib import Path

from .deficit_compensation_v025 import REFERENCE
from .official_prize_v026 import official_prize_coverage

SCHEMA = "lotto7x7.coverage_exhaustive.v0.26.p1"
MASKS = tuple(range(1, 16))
BIT_SETS = tuple(tuple(i for i in range(4) if m & (1 << i))
                 for m in range(16))
PERM_MASKS = tuple(
    tuple(sum((1 << i) for i, original in enumerate(perm)
              if mask & (1 << original)) for mask in range(16))
    for perm in permutations(range(4))
)


def feasible_signatures():
    """All 4x6 labeled structures; each one yielded exactly once."""
    counts = [0] * 16

    def visit(mask, remains):
        if mask == 16:
            if remains == (0, 0, 0, 0):
                yield tuple(counts[1:])
            return
        affected = BIT_SETS[mask]
        limit = min(remains[i] for i in affected)
        for qty in range(limit + 1):
            counts[mask] = qty
            next_remains = tuple(
                remains[i] - qty if i in affected else remains[i]
                for i in range(4))
            yield from visit(mask + 1, next_remains)
        counts[mask] = 0

    yield from visit(1, (6, 6, 6, 6))


def has_unique_tickets(signature):
    """No pair of rows has identical number sets."""
    vector = (0,) + tuple(signature)
    return all(
        any(vector[mask] and bool(mask & (1 << i)) != bool(mask & (1 << j))
            for mask in MASKS)
        for i in range(4) for j in range(i + 1, 4)
    )


def canonical_signature(signature):
    """Lexicographic orbit minimum under all permutations of four tickets."""
    if len(signature) != 15:
        raise ValueError("expected 15 membership classes")
    vector = (0,) + tuple(signature)
    return min(
        tuple(vector[mapping.index(mask)] for mask in MASKS)
        for mapping in PERM_MASKS
    )


def materialize_tickets(signature):
    """Assign actual labels consecutively to membership classes.

    All structures contain at most 24 distinct labels. Relabeling does not
    affect exact uniform 6+bonus outcome probabilities.
    """
    if len(signature) != 15 or any(
            not isinstance(v, int) or isinstance(v, bool) or v < 0
            for v in signature):
        raise ValueError("bad membership signature")
    if sum(signature) > 49:
        raise ValueError("more than 49 distinct labels")
    rows = [[], [], [], []]
    label = 1
    for mask, qty in enumerate(signature, 1):
        for _ in range(qty):
            for i in BIT_SETS[mask]:
                rows[i].append(label)
            label += 1
    result = tuple(tuple(r) for r in rows)
    if any(len(row) != 6 for row in result) or len(set(result)) != 4:
        raise ValueError("not a valid four-ticket configuration")
    return result


def exhaustive_audit(*, max_classes=None):
    """Return a full mathematical certificate, or clearly marked partial audit."""
    if max_classes is not None and (
            not isinstance(max_classes, int) or isinstance(max_classes, bool)
            or max_classes < 1):
        raise ValueError("max_classes must be a positive integer")
    seen = set()
    raw_count = valid_count = 0
    valid_orbit_count = 0
    evaluated = 0
    best_score = -1
    best_sig = None
    best_tickets = None
    histogram = {}
    sha = hashlib.sha256()
    complete = True
    for signature in feasible_signatures():
        raw_count += 1
        if not has_unique_tickets(signature):
            continue
        valid_count += 1
        canonical = canonical_signature(signature)
        if canonical in seen:
            continue
        seen.add(canonical)
        valid_orbit_count += 1
        if max_classes is not None and evaluated >= max_classes:
            complete = False
            break
        tickets = materialize_tickets(canonical)
        out = official_prize_coverage(tickets)
        score = out["any_prize_outcomes"]
        histogram[score] = histogram.get(score, 0) + 1
        evaluated += 1
        sha.update(("%s:%s\n" % (
            ",".join(map(str, canonical)), score)).encode("ascii"))
        if score > best_score:
            best_score, best_sig, best_tickets = score, canonical, tickets

    baseline = official_prize_coverage(REFERENCE)
    if complete:
        if raw_count != 52450 or valid_count != 50985 or evaluated != 2957:
            raise AssertionError(
                f"unexpected complete structural census "
                f"raw={raw_count} valid={valid_count} classes={evaluated}")
    proof = complete and best_score == baseline["any_prize_outcomes"]
    return {
        "schema": SCHEMA,
        "method": "exhaustive integer membership-mask census modulo S4",
        "completed": complete,
        "global_disjoint_optimality_proven": proof,
        "certification_scope": (
            "four distinct six-number tickets; uniform 6 of 49 mains plus one "
            "of 43 distinct bonus; exact eight official prize tiers"
        ),
        "raw_labeled_signatures": raw_count if complete else None,
        "valid_labeled_signatures": valid_count if complete else None,
        "invalid_duplicate_ticket_signatures":
            raw_count - valid_count if complete else None,
        "unique_valid_unlabeled_structures": evaluated if complete else None,
        "evaluated_classes": evaluated,
        "score_sha256": sha.hexdigest(),
        "reference_any_prize_outcomes": baseline["any_prize_outcomes"],
        "reference_any_prize_probability": baseline["any_prize_probability"],
        "best_any_prize_outcomes": best_score,
        "best_any_prize_probability": best_score / baseline["total_outcomes"],
        "best_signature": list(best_sig),
        "best_tickets": [list(t) for t in best_tickets],
        "maximum_improvement_outcomes": best_score - baseline["any_prize_outcomes"],
        "unique_scores": len(histogram),
        "number_of_maximizing_structures": (
            histogram.get(best_score) if complete else None),
        "warning": (
            "COMPLETE exhaustive structural certificate" if complete else
            "PARTIAL run: no global claim; max_classes halted enumeration"
        ),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-classes", type=int, default=None)
    p.add_argument("--out", type=Path, default=Path("results/v026_coverage_exhaustive.json"))
    a = p.parse_args(argv)
    r = exhaustive_audit(max_classes=a.max_classes)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
    print(f"[ENUM] complete={r['completed']} classes={r['evaluated_classes']}")
    print(f"[MAX] reference={r['reference_any_prize_probability']:.12f} "
          f"best={r['best_any_prize_probability']:.12f} "
          f"delta_outcomes={r['maximum_improvement_outcomes']}")
    print(f"[PROOF] four-ticket disjoint optimum={r['global_disjoint_optimality_proven']}")
    print(f"[SHA256] {r['score_sha256']}")
    print(f"[REPORT] {a.out}")


if __name__ == "__main__":
    main()
