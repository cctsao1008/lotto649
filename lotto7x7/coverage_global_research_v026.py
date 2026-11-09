"""v0.26 P1 research: global-optimum *evidence*, not a claimed proof.

Systematically compare all two-ticket intersection sizes (exact: up to label
relabeling, there are only six valid overlaps 0..5). For four tickets,
search additional legal intersection structures using deterministic randomized
constructions and include the pre-existing 15 profile templates. Every score
is computed by the exact official 6+special probability engine.

The four-ticket search is finite, seeded, and NOT exhaustive over all
non-isomorphic four-set systems. All hardcoded reference tickets stay sealed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random

from .deficit_compensation_v025 import REFERENCE
from .coverage_strategy_v026 import PROFILES, portfolio_profile, overlap_summary
from .official_prize_v026 import official_prize_coverage

SCHEMA = "lotto7x7.coverage_global_research.v0.26.p1"


def canonical_signature(tickets):
    """Canonical up to ticket permutation and relabeling of 1..49.

    The full counts of the 15 nonzero membership masks define the structure.
    Considering all 4! ticket permutations removes ticket-order sensitivity.
    """
    from itertools import permutations
    rows = tuple(map(set, tickets))
    signatures = []
    for perm in permutations(range(len(rows))):
        masks = [0] * (1 << len(rows))
        for label in set.union(*rows):
            mask = sum(1 << i for i, j in enumerate(perm) if label in rows[j])
            masks[mask] += 1
        signatures.append(tuple(masks[1:]))
    return min(signatures)


def two_ticket_overlap_study():
    """Exhaustive structural audit for *two* different six-number tickets."""
    first = tuple(range(1, 7))
    results = []
    for shared in range(6):
        second = tuple(range(1, shared + 1)) + tuple(range(7, 13 - shared))
        prize = official_prize_coverage((first, second))
        results.append({
            "shared_numbers": shared,
            "any_prize_probability": prize["any_prize_probability"],
            "both_awarded_probability": prize["awarded_ticket_count_probabilities"][2],
            "expected_awarded_tickets": prize["expected_awarded_tickets"],
            "any_prize_outcomes": prize["any_prize_outcomes"],
        })
    winner = max(results, key=lambda r: r["any_prize_outcomes"])
    return {
        "cases": results,
        "best_overlap": winner["shared_numbers"],
        "best_any_prize_probability": winner["any_prize_probability"],
        "strict_disjoint_optimum": (
            winner["shared_numbers"] == 0 and
            all(results[i]["any_prize_probability"] >
                results[i + 1]["any_prize_probability"]
                for i in range(5))
        ),
        "scope": "global among two distinct tickets, by exhaustive overlap classification",
    }


def random_overlap_portfolio(rng, *, mutations=8):
    """Generate one legal 4x6 portfolio by varied label reassignments.

    Starting from disjoint, reusing another ticket's number increases
    overlap; swap and revert moves generate mixed multi-way intersections.
    Every result has four distinct legal rows.
    """
    pool = list(range(1, 50))
    rng.shuffle(pool)
    rows = [set(pool[6 * i:6 * (i + 1)]) for i in range(4)]
    for _ in range(mutations):
        target, source = rng.sample(range(4), 2)
        old = rng.choice(tuple(sorted(rows[target])))
        if rng.random() < 0.80:
            new = rng.choice(tuple(sorted(rows[source])))
        else:
            new = rng.choice(pool)
        if new != old and new not in rows[target]:
            rows[target].remove(old)
            rows[target].add(new)
    result = tuple(tuple(sorted(row)) for row in rows)
    if len(set(result)) != 4:
        raise AssertionError("duplicate tickets after randomized moves")
    return result


def research(*, seed=649, samples=200):
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 0:
        raise ValueError("samples must be nonnegative integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    two = two_ticket_overlap_study()
    rng = Random(seed)
    templates = [(name, portfolio_profile(name, seed)) for name in PROFILES]
    templates.append(("purchased_reference", REFERENCE))
    attempts = templates + [
        (f"random_{i}", random_overlap_portfolio(rng, mutations=rng.randint(1, 32)))
        for i in range(samples)
    ]
    distinct = {}
    for label, tickets in attempts:
        signature = canonical_signature(tickets)
        if signature in distinct:
            continue
        result = official_prize_coverage(tickets)
        distinct[signature] = {
            "label": label,
            "tickets": [list(t) for t in tickets],
            "membership_signature": list(signature),
            "overlap": overlap_summary(tickets),
            "any_prize_outcomes": result["any_prize_outcomes"],
            "any_prize_probability": result["any_prize_probability"],
            "at_least_two_awards_probability": sum(
                result["awarded_ticket_count_probabilities"][2:]),
            "expected_awarded_tickets": result["expected_awarded_tickets"],
        }
    rows = sorted(distinct.values(),
                  key=lambda r: (-r["any_prize_outcomes"], r["membership_signature"]))
    reference = official_prize_coverage(REFERENCE)
    best = rows[0]
    invariant_expectation = all(
        abs(row["expected_awarded_tickets"] -
            reference["expected_awarded_tickets"]) < 1e-12 for row in rows
    )
    if not invariant_expectation:
        raise AssertionError("four-ticket expected awards violated")
    return {
        "schema": SCHEMA,
        "two_ticket_exact_structural_study": two,
        "four_ticket": {
            "search_type": "non-exhaustive seeded structural search",
            "samples_attempted": samples,
            "profile_templates": len(PROFILES),
            "distinct_structures_evaluated": len(rows),
            "relabeling_symmetry_reduction": "minimum counts over all 4! ticket permutations",
            "reference_any_prize_probability": reference["any_prize_probability"],
            "best_tested": best,
            "improvement_over_reference": (
                best["any_prize_probability"] - reference["any_prize_probability"]),
            "expected_awarded_tickets_invariant": invariant_expectation,
            "globally_optimal_proven": False,
            "top_10": rows[:10],
        },
        "limitations": [
            "Two tickets: overlap class 0..5 is an exhaustive structural partition, "
            "so the numerical strict ordering proves the two-ticket case for 6/49.",
            "Four tickets: sampled structure set is not exhaustive; no global "
            "optimality theorem is claimed or implied.",
            "No forecast, prize money, expected profit, or retrospective fit.",
        ],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=649)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--out", type=Path, default=Path("results/v026_coverage_global_research.json"))
    args = p.parse_args(argv)
    result = research(seed=args.seed, samples=args.samples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    pair = result["two_ticket_exact_structural_study"]
    four = result["four_ticket"]
    print(f"[TWO TICKETS] strict disjoint optimum={pair['strict_disjoint_optimum']}")
    for case in pair["cases"]:
        print(f"[PAIR] shared={case['shared_numbers']} any_prize={case['any_prize_probability']:.12f}")
    print(f"[FOUR TICKETS] structures={four['distinct_structures_evaluated']} "
          f"best={four['best_tested']['label']} any_prize={four['best_tested']['any_prize_probability']:.12f}")
    print(f"[REFERENCE] {four['reference_any_prize_probability']:.12f}; "
          f"gain={four['improvement_over_reference']:+.12f}")
    print("[CAUTION] Four-ticket global optimum NOT proven")
    print(f"[REPORT] {args.out}")


if __name__ == "__main__":
    main()
