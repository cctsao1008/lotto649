"""Compare exact main-ball coverage and covariance-aware repair for four tickets."""
from __future__ import annotations

import argparse
from functools import lru_cache
from hashlib import sha256
from itertools import combinations, islice
import json
from pathlib import Path

import numpy as np

from .coverage_audit_v06 import _masks, _popcount64
from .coverage_exhaustive_v026 import (
    canonical_signature as canonical_membership, feasible_signatures,
    has_unique_tickets, materialize_tickets,
)
from .coverage_global_research_v026 import canonical_signature
from .coverage_v06 import UNIVERSE, exact_coverage, joint_coverage_counts
from .deficit_compensation_v025 import load_csv_history
from .matched_covariance_v025 import (
    CUTOFF, EXACT, FIXTURE, HISTORY_DRAWS, LAST_DRAW, POOL_SIZE, RIDGE, SELECTOR_SEED,
    json_sha256, population_covariance,
)
from .multivariate_audit_v025 import (
    candidate_pool, covariance_model, distance, encode_features, registry,
)
from .official_prize_v026 import official_prize_coverage
from .shape_deficit_v025 import load_exact_baseline
from .shape_predictive_v017 import _baseline_tables
from .unified_selection_v021 import coverage_select, frozen_shape_scores

FREEZE = Path("docs/prospective_freeze_20261108.json")
PASSES = 3
THRESHOLDS = ("3", "4", "5")
FLOOR_FRACTIONS = ((1, 1), (999, 1000), (99, 100))
DISJOINT_REPRESENTATIVE = tuple(tuple(range(start, start + 6)) for start in (1, 7, 13, 19))


@lru_cache(maxsize=4096)
def coverage_by_structure(signature):
    counts = exact_coverage(materialize_tickets(signature))["at_least_counts"]
    return tuple(counts[t] for t in THRESHOLDS)


def main_ball_counts(tickets):
    return coverage_by_structure(canonical_signature(tickets))


def coverage_census():
    structures = set()
    for signature in feasible_signatures():
        if has_unique_tickets(signature):
            structures.add(canonical_membership(signature))
    if len(structures) != 2957:
        raise AssertionError("four-ticket structural census differs from the Burnside certificate")
    counts = np.asarray([coverage_by_structure(s) for s in sorted(structures)])
    best = counts.max(axis=0)
    return {
        "structures": len(structures),
        "maximum_outcomes": dict(zip(THRESHOLDS, best.tolist())),
        "structures_attaining_each_maximum": dict(zip(
            THRESHOLDS, (counts == best).sum(axis=0).tolist(),
        )),
        "structures_attaining_all_maxima": int((counts == best).all(axis=1).sum()),
    }


def full_universe_audit(portfolios):
    """Independent streaming intersection census of max hits and multiplicity."""
    masks = {name: _masks(tickets) for name, tickets in portfolios.items()}
    histograms = {name: np.zeros((5, 7), dtype=np.int64) for name in masks}
    draws = combinations(range(1, 50), 6)
    checked = 0
    while block := list(islice(draws, 100_000)):
        values = np.asarray(block, dtype=np.uint64)
        draw_masks = np.bitwise_or.reduce(np.uint64(1) << (values - 1), axis=1)
        checked += len(block)
        for name, ticket_masks in masks.items():
            hits = _popcount64(draw_masks[:, None] & ticket_masks[None, :])
            histograms[name][0] += np.bincount(hits.max(axis=1), minlength=7)
            for row, threshold in enumerate((3, 4, 5, 6), 1):
                histograms[name][row] += np.bincount(
                    (hits >= threshold).sum(axis=1), minlength=7,
                )
    if checked != UNIVERSE:
        raise AssertionError("incomplete independent main-ball enumeration")
    return {
        name: {
            "max_hits": rows[0].tolist(),
            "qualifying_ticket_counts": {
                str(t): rows[i, :len(portfolios[name]) + 1].tolist()
                for i, t in enumerate((3, 4, 5, 6), 1)
            },
        }
        for name, rows in histograms.items()
    }


def objective_key(objectives, policy, floors):
    coverage, repair, distinct_numbers, distinct_shapes = objectives
    diversity = (-distinct_numbers, -distinct_shapes)
    descending_coverage = tuple(-value for value in coverage)
    if policy == "coverage_only":
        return descending_coverage, diversity, repair
    if policy == "compensation_only":
        return repair, diversity, descending_coverage
    deficits = tuple(max(0, minimum - count) for minimum, count in zip(floors, coverage))
    return deficits, repair, diversity, descending_coverage


def swap_search(pool_size, score, policy, floors, *, passes=PASSES):
    """Deterministic best one-swap passes with the same cap for every policy."""
    visited = set()

    def key(indices):
        visited.add(indices)
        return objective_key(score(indices), policy, floors), indices

    chosen = tuple(range(4))
    current = key(chosen)
    for _ in range(passes):
        starting = chosen
        for slot in range(4):
            fixed = chosen[:slot] + chosen[slot + 1:]
            trials = (tuple(sorted((*fixed, i))) for i in range(pool_size) if i not in chosen)
            best = min((key(trial) for trial in trials), default=current)
            if best < current:
                current, chosen = best, best[1]
        if chosen == starting:
            break
    return chosen, len(visited)


def pareto_frontier(scored):
    """Return representatives of exactly nondominated vectors over the scored set."""
    groups = {}
    for indices, (coverage, repair, distinct_numbers, distinct_shapes) in sorted(scored.items()):
        losses = (repair, *(-x for x in coverage), -distinct_numbers, -distinct_shapes)
        groups.setdefault(losses, []).append(indices)
    front = []
    for losses in sorted(groups):
        if any(all(x <= y for x, y in zip(point, losses)) for point in front):
            continue
        front.append(losses)
    return [(groups[point][0], len(groups[point])) for point in front]


def portfolio_metrics(tickets, encoded, history_sum, n, model):
    max_hits = joint_coverage_counts(tickets)
    prize = official_prize_coverage(tickets)
    after = distance((history_sum + encoded.sum(axis=0)) / (n + len(tickets)), model)
    overlap = [len(set(a).intersection(b)) for a, b in combinations(tickets, 2)]
    return {
        "tickets": tickets,
        "repair_after": after,
        "repair_gain": distance(history_sum / n, model) - after,
        "distinct_numbers": len({v for ticket in tickets for v in ticket}),
        "pair_intersection_counts_0_to_6": np.bincount(overlap, minlength=7).tolist(),
        "shape_classes": [
            {level: int(name.rsplit("_", 1)[1])
             for level in ("component", "partition", "edge")
             for name, present in zip(model["names"][52:], row[52:])
             if present and name.startswith(level + "_")}
            for row in encoded
        ],
        "exact_main_ball": max_hits,
        "main_ball_at_least_counts": {str(t): sum(max_hits["max_hits"][t:]) for t in (3, 4, 5, 6)},
        "main_ball_at_least_once_in_ten_iid_draws": {
            str(t): 1 - (1 - sum(max_hits["max_hits"][t:]) / UNIVERSE) ** 10
            for t in (3, 4, 5, 6)
        },
        "official_any_prize_probability": prize["any_prize_probability"],
        "official_any_prize_outcomes": prize["any_prize_outcomes"],
        "official_any_ticket_by_tier": prize["any_ticket_by_tier"],
        "official_awarded_ticket_count_outcomes": prize["awarded_ticket_count_outcomes"],
        "official_any_prize_probability_at_least_once_in_ten_iid_draws": (
            1 - (1 - prize["any_prize_probability"]) ** 10
        ),
    }


def optimize(records, baseline, model, freeze, *, pool_size=POOL_SIZE, seed=SELECTOR_SEED):
    if not 6 <= pool_size <= 5000 or seed < 0:
        raise ValueError("pool size must be 6..5000 and seed nonnegative")
    pool = candidate_pool(pool_size, seed)
    history = encode_features([r[2] for r in records], model["catalog"])
    history_sum = history.sum(axis=0)
    n = len(records)
    encoded = encode_features(pool, model["catalog"])
    disjoint_coverage = main_ball_counts(DISJOINT_REPRESENTATIVE)
    floors = tuple((count * a + b - 1) // b
                   for count, (a, b) in zip(disjoint_coverage, FLOOR_FRACTIONS))
    budget = 1 + PASSES * 4 * (pool_size - 4)
    # Determine scores with one basis transform followed by four vector sums.
    transform = (model["basis"] / np.sqrt(model["eigenvalues"])[None, :]) / model["sd"][:, None]
    residual = (history_sum - n * model["theory"]) @ transform
    increments = (encoded - model["theory"]) @ transform
    shape_ids = encoded[:, -len(model["catalog"]["edge"]):].argmax(axis=1)
    scores = {}

    def score(indices):
        if indices not in scores:
            tickets = tuple(pool[i] for i in indices)
            combined = residual + increments[list(indices)].sum(axis=0)
            scores[indices] = (
                main_ball_counts(tickets), float(combined @ combined / ((n + 4) ** 2)),
                len({v for ticket in tickets for v in ticket}),
                len({shape_ids[i] for i in indices}),
            )
        return scores[indices]

    pool_array = np.asarray(pool, dtype=np.uint8)
    historical_ids = np.asarray([r[1] for r in records], dtype="<i8")
    historical_tickets = np.asarray([r[2] for r in records], dtype="u1")
    freeze_history = sha256(historical_ids.tobytes() + historical_tickets.tobytes()).hexdigest()
    if freeze_history != freeze["source"]["historical_record_sha256"]:
        raise ValueError("frozen shape scores must match the same historical records")
    ratios, _ = frozen_shape_scores(pool_array, freeze, _baseline_tables(baseline))
    shape_rank = sorted(range(pool_size), key=lambda i: (-ratios[i], i))
    coverage_indices, coverage_info = coverage_select(
        pool_array, 4, objective="3plus", max_evaluations=budget,
    )
    selections = {
        "uniform": (tuple(range(4)), 1),
        "coverage_v021": (tuple(sorted(coverage_indices)), coverage_info["oracle_evaluations"]),
        "shape_score_v021": (tuple(sorted(shape_rank[:4])), 1),
    }
    for policy in ("compensation_only", "coverage_only", "hybrid"):
        selections[policy] = swap_search(pool_size, score, policy, floors)

    outcomes = {}
    for name, (indices, evaluated) in selections.items():
        tickets = tuple(pool[i] for i in indices)
        coverage, repair, _, _ = score(indices)
        row = portfolio_metrics(tickets, encoded[list(indices)], history_sum, n, model)
        if not np.isclose(repair, row["repair_after"], atol=1e-12, rtol=0):
            raise AssertionError("repair score differs from direct recomputation")
        outcomes[name] = {
            **row, "pool_indices": indices, "evaluations": evaluated,
            "meets_hybrid_floor": all(c >= f for c, f in zip(coverage, floors)),
        }
    feasible = [
        indices for indices, (coverage, *_rest) in scores.items()
        if all(count >= floor for count, floor in zip(coverage, floors))
    ]
    best_feasible = min(
        feasible, key=lambda i: (objective_key(scores[i], "hybrid", floors), i),
        default=None,
    )
    frontier = []
    for indices, equivalent in pareto_frontier(scores):
        coverage, repair, numbers, shapes = scores[indices]
        frontier.append({
            "pool_indices": indices, "tickets": [pool[i] for i in indices],
            "coverage_outcomes_3_4_5": coverage, "repair_after": repair,
            "distinct_numbers": numbers, "distinct_l3_shapes": shapes,
            "equivalent_scored_portfolios": equivalent,
        })
    return {
        "config": {
            "pool_size": pool_size, "pool_seed": seed, "pool_sha256": json_sha256(pool),
            "portfolio_evaluation_cap_per_strategy": budget, "swap_passes": PASSES,
            "hybrid_coverage_floors": dict(zip(THRESHOLDS, floors)),
        },
        "strategies": outcomes,
        "pareto": {
            "best_repair_at_hybrid_floors_among_evaluated": (
                {
                    "pool_indices": best_feasible,
                    "tickets": [pool[i] for i in best_feasible],
                    "coverage_outcomes_3_4_5": scores[best_feasible][0],
                    "repair_after": scores[best_feasible][1],
                } if best_feasible is not None else None
            ),
            "scope": "complete portfolios evaluated by new swaps and final v0.21 and uniform selections",
            "portfolios_evaluated": len(scores), "distinct_frontier_vectors": len(frontier),
            "frontier": frontier,
        },
    }


def report(csv_dir, fixture, moments_path, freeze_path, *, pool_size=POOL_SIZE, seed=SELECTOR_SEED):
    records, csv_hash = load_csv_history(csv_dir, CUTOFF)
    if len(records) != HISTORY_DRAWS or records[-1][:2] != (CUTOFF, LAST_DRAW):
        raise ValueError("history must match the original 2,497-draw cutoff")
    baseline, fixture_hash = load_exact_baseline(fixture)
    moments_bytes = moments_path.read_bytes()
    names, theory, catalog = registry(baseline)
    covariance = population_covariance(json.loads(moments_bytes), names, theory)
    model = covariance_model(names, theory, catalog, covariance, np.diag(covariance), RIDGE, {
        "source": "exact population covariance", "universe": UNIVERSE,
    })
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    result = optimize(records, baseline, model, freeze, pool_size=pool_size, seed=seed)
    checked = {
        "synthetic_disjoint": DISJOINT_REPRESENTATIVE,
        "uniform": result["strategies"]["uniform"]["tickets"],
    }
    brute = full_universe_audit(checked)
    for name, tickets in checked.items():
        if brute[name] != joint_coverage_counts(tickets):
            raise AssertionError(f"independent universe differs from exact joint DP for {name}")
    return {
        "schema": "lotto649.v025.portfolio_pareto.v1",
        "research_decision": "DESCRIPTIVE_COMPLETE",
        "source": {
            "history_draws": len(records), "cutoff": CUTOFF, "last_draw": LAST_DRAW,
            "input_csv_sha256": csv_hash, "used_history_sha256": json_sha256(records),
            "shape_fixture_sha256": fixture_hash,
            "exact_moments_sha256": sha256(moments_bytes).hexdigest(),
            "shape_freeze_checksum": freeze["self_checksum_sha256"],
        },
        "model": model["meta"],
        **result,
        "exact_theoretical_coverage_census": coverage_census(),
        "independent_main_ball_audit": {"draws_per_portfolio": UNIVERSE,
                                        "full_dp_parity_pass": list(checked)},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, default=Path("data/csv"))
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--exact-moments", type=Path, default=EXACT)
    parser.add_argument("--shape-freeze", type=Path, default=FREEZE)
    parser.add_argument("--pool-size", type=int, default=POOL_SIZE)
    parser.add_argument("--seed", type=int, default=SELECTOR_SEED)
    parser.add_argument("--out", type=Path, default=Path("results/v025_portfolio_pareto.json"))
    args = parser.parse_args(argv)
    result = report(args.csv_dir, args.fixture, args.exact_moments, args.shape_freeze,
                    pool_size=args.pool_size, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, separators=(",", ":"))
        stream.write("\n")
    print("[PARETO]", result["pareto"]["distinct_frontier_vectors"], "frontier vectors")
    print("[REPORT]", args.out)


if __name__ == "__main__":
    main()
