"""v0.25 P2: L1/L2/L3 shape-class deficit audit and portfolio selection.

Uses the existing *complete, exactly enumerated* shape universe; category
frequencies are not estimated from historical data or random simulations.
All outputs are descriptive; no next-draw advantage is inferred.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

import numpy as np

from .coverage_v06 import exact_coverage
from .deficit_compensation_v025 import (
    REFERENCE, discrepancy, deficit_select, load_csv_history, validate_ticket,
)
from .shape_taxonomy_v015 import classify_numbers
from .shape_theory_v016 import (
    LEVELS, RULE_VERSION, UNIVERSE, category_name, classification_keys,
    exhaustive_class_counts,
)


def load_exact_baseline(path: Path):
    raw = path.read_bytes()
    baseline = json.loads(raw)
    if (baseline.get("complete") is not True
            or baseline.get("total") != UNIVERSE
            or baseline.get("schema") != RULE_VERSION):
        raise ValueError("need complete v0.16 exact 6/49 shape universe")
    counts = baseline.get("counts", {})
    if set(counts) != set(LEVELS):
        raise ValueError("wrong taxonomy levels")
    for level in LEVELS:
        values = counts[level]
        if (not isinstance(values, dict) or not values
                or any(int(k) < 0 or int(v) <= 0 for k, v in values.items())
                or sum(int(v) for v in values.values()) != UNIVERSE):
            raise ValueError(f"incomplete theoretical counts: {level}")
    return baseline, sha256(raw).hexdigest()


def _keys(tickets):
    if not tickets:
        return {name: np.empty(0, dtype=np.uint32) for name in LEVELS}
    arr = np.asarray([validate_ticket(t) for t in tickets], dtype=np.uint8)
    return classification_keys(classify_numbers(arr))


def _history_tickets(records):
    return [r[2] for r in records]


def level_summary(records, portfolio, baseline, *, alpha=4.0):
    """Regularized class discrepancy with fixed target and exact p.

    D = sum_c ((O_c + T_c)/(N+k) - p_c)^2 / (p_c + alpha/N).

    Regularization is fixed using historical N, including before/after,
    so a score decrease means smaller *declared* discrepancy. This is
    NOT a chi-square test, and alpha is a dimensionless p-floor multiplier.
    """
    if alpha < 0 or not isfinite(alpha):
        raise ValueError("alpha must be finite and >=0")
    n = len(records)
    if n == 0:
        raise ValueError("empty history")
    tickets = tuple(validate_ticket(t) for t in portfolio)
    if len(set(tickets)) != len(tickets):
        raise ValueError("duplicate portfolio tickets")
    historical = _keys(_history_tickets(records))
    added = _keys(tickets)
    summaries = {}
    for level in LEVELS:
        weights = {int(k): int(v) for k, v in baseline["counts"][level].items()}
        hist = Counter(map(int, historical[level]))
        extra = Counter(map(int, added[level]))
        if set(hist) - set(weights) or set(extra) - set(weights):
            raise ValueError(f"unknown {level} category")
        rows = []
        before = 0.0
        after = 0.0
        for code, mass in sorted(weights.items()):
            p = mass / UNIVERSE
            observed = hist.get(code, 0)
            chosen = extra.get(code, 0)
            denom = p + alpha / n
            if denom <= 0:
                raise ValueError("invalid regularization")
            b = (observed / n - p) ** 2 / denom
            a = ((observed + chosen) / (n + len(tickets)) - p) ** 2 / denom
            before += b
            after += a
            rows.append({
                "code": code,
                "class": category_name(level, code),
                "historical_observed": observed,
                "historical_expected": n * p,
                "historical_deficit": n * p - observed,
                "theory_probability": p,
                "selected_count": chosen,
                "before_contribution": b,
                "after_contribution": a,
            })
        summaries[level] = {
            "classes": len(rows), "before": before, "after": after,
            "improvement": before - after, "rows": rows,
        }
    return summaries


def shape_score(levels, weights):
    if not weights or set(weights) - set(LEVELS):
        raise ValueError("unknown or missing shape family weights")
    if any(not isfinite(float(v)) or v < 0 for v in weights.values()):
        raise ValueError("shape weights must be finite and >=0")
    if sum(weights.values()) <= 0:
        raise ValueError("at least one positive shape weight")
    return sum(float(weights.get(l, 0)) * levels[l]["after"] for l in LEVELS)


def select_shape_portfolio(records, baseline, *, pool_size=300,
                           seed=649, weights=None, alpha=4.0):
    """Finite-pool greedy + improving swaps on joint four-ticket shape score."""
    if weights is None:
        weights = {"component": 1.0, "partition": 1.0, "edge": 1.0}
    # Reuse the uniform ticket-pool construction, not the historically
    # conditioned number-deficit selector (avoids pre-filter bias).
    from random import Random
    if pool_size < 4 or pool_size > 5000 or seed < 0:
        raise ValueError("pool size must be 4..5000 and seed >=0")
    if alpha < 0 or not isfinite(alpha):
        raise ValueError("alpha must be finite and >=0")
    rng = Random(seed)
    pool = []
    seen = set()
    while len(pool) < pool_size:
        ticket = tuple(sorted(rng.sample(range(1, 50), 6)))
        if ticket not in seen:
            seen.add(ticket)
            pool.append(ticket)
    # Cache all six-number shape classifications; scoring uses 6 keys only.
    history_codes = _keys(_history_tickets(records))
    candidate_codes = _keys(pool)
    n = len(records)
    fixed = {}
    for level in LEVELS:
        counts = {int(k): int(v) for k, v in baseline["counts"][level].items()}
        history = Counter(map(int, history_codes[level]))
        fixed[level] = (
            {k: v / UNIVERSE for k, v in counts.items()},
            history,
            list(map(int, candidate_codes[level])),
        )

    def objective(indices):
        k = len(indices)
        score = 0.0
        for level in LEVELS:
            w = float(weights.get(level, 0))
            if w == 0:
                continue
            probabilities, hist, candidate_labels = fixed[level]
            additions = Counter(candidate_labels[i] for i in indices)
            for code, p in probabilities.items():
                diff = ((hist.get(code, 0) + additions.get(code, 0))
                        / (n + k) - p)
                score += w * diff * diff / (p + alpha / n)
        return score

    selected = []
    for _ in range(4):
        option = min((i for i in range(len(pool)) if i not in selected),
                     key=lambda i: (objective(selected + [i]), pool[i]))
        selected.append(option)
    for _ in range(3):
        changed = False
        for position in range(4):
            old = objective(selected)
            replacement = min(
                (i for i in range(len(pool))
                 if i not in selected or i == selected[position]),
                key=lambda i: (
                    objective(selected[:position] + [i] + selected[position + 1:]),
                    pool[i],
                ),
            )
            candidate = selected[:position] + [replacement] + selected[position + 1:]
            if objective(candidate) < old - 1e-15:
                selected = candidate
                changed = True
        if not changed:
            break
    tickets = tuple(sorted(pool[i] for i in selected))
    levels = level_summary(records, tickets, baseline, alpha=alpha)
    return {
        "tickets": [list(t) for t in tickets],
        "objective": shape_score(levels, weights),
        "weights": weights, "alpha": alpha, "seed": seed,
        "pool_size": pool_size, "method": "finite-pool greedy plus local swaps",
        "globally_optimal": False,
    }


def report(csv_dir: Path, universe: Path, *, cutoff=None,
           pool_size=300, seed=649, alpha=4.0):
    records, history_hash = load_csv_history(csv_dir, cutoff)
    baseline, universe_hash = load_exact_baseline(universe)
    selected = select_shape_portfolio(
        records, baseline, pool_size=pool_size, seed=seed, alpha=alpha)
    reference_summary = level_summary(records, REFERENCE, baseline, alpha=alpha)
    selected_tickets = selected["tickets"]
    selected_summary = level_summary(records, selected_tickets, baseline, alpha=alpha)
    weights = selected["weights"]
    empty_summary = level_summary(records, (), baseline, alpha=alpha)
    return {
        "schema": "lotto7x7.shape_deficit_compensation.v0.25.p2",
        "research_only": True,
        "historical_draws": len(records),
        "last_draw_id": records[-1][1], "last_date": records[-1][0],
        "csv_sha256": history_hash, "universe_sha256": universe_hash,
        "taxonomy_rule": RULE_VERSION,
        "alpha": alpha, "weights": weights,
        "baseline_score": shape_score(empty_summary, weights),
        "selected": {
            **selected,
            "levels": selected_summary,
            "exact_main_ball_coverage": exact_coverage(selected_tickets),
        },
        "reference": {
            "tickets": [list(t) for t in REFERENCE],
            "objective": shape_score(reference_summary, weights),
            "levels": reference_summary,
            "exact_main_ball_coverage": exact_coverage(REFERENCE),
        },
        "scope": "L1/L2/L3 shape only; number-shape covariance and validation pending",
        "note": "Historical deficits are not a conditional future-draw probability.",
    }


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, default=Path("data/csv"))
    parser.add_argument("--universe", type=Path,
                        default=Path("results/v015_v016_p1_shape_catalog_universe.json"))
    parser.add_argument("--cutoff", default=None)
    parser.add_argument("--pool-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=649)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--out", type=Path, default=Path("results/v025_shape_deficit.json"))
    args = parser.parse_args(argv)
    result = report(args.csv_dir, args.universe, cutoff=args.cutoff,
                    pool_size=args.pool_size, seed=args.seed, alpha=args.alpha)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[v0.25 P2] {result['historical_draws']} verified historical draws")
    print(f"[SHAPE] selected={result['selected']['tickets']}")
    print(f"[SCORE] baseline={result['baseline_score']:.8f}; "
          f"reference={result['reference']['objective']:.8f}; "
          f"selected={result['selected']['objective']:.8f}")
    print(f"[REPORT] {args.out}")


if __name__ == "__main__":
    main()
