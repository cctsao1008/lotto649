"""Compare covariance models on chronological, coverage-matched holdouts."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from research.lotto7x7.portfolio.deficit_compensation_v025 import REFERENCE, load_csv_history
from research.lotto7x7.portfolio.matched_covariance_v025 import (
    ARMS, CONSTRAINT, CUTOFF, EXACT, FIXTURE, HISTORY_DRAWS, LAST_DRAW,
    POOL_SIZE, SELECTOR_SEED, benchmark_models, json_sha256,
)
from research.lotto7x7.portfolio.multivariate_audit_v025 import candidate_pool, select_multivariate_portfolio
from research.lotto7x7.portfolio.shape_deficit_v025 import load_exact_baseline
from research.lotto7x7.portfolio.strategy_ab_iid_calibration_v026 import empirical_calibration, iid_full_history
from research.lotto7x7.portfolio.strategy_ab_v026 import (
    assert_official_coverage_parity, checked_bonus_by_draw, disjoint_portfolio, draw_awards,
)

TRAIN_MIN = 1800
HORIZON = 25
FOLDS = 8
FIRST_NULL_SEED = 2650
NULL_REPLICATES = 99
BASELINE_SEED = 649
WIN_ARMS = ("history_blind", *ARMS)


def walkforward(records, bonuses, models, *, train_min=TRAIN_MIN, horizon=HORIZON,
                fold_count=FOLDS):
    if train_min < 1 or horizon < 1 or fold_count < 1:
        raise ValueError("positive training size, horizon and fold count required")
    if len(records) < train_min + horizon * fold_count:
        raise ValueError("insufficient history for full holdout blocks")
    if any(a[0] >= b[0] or a[1] >= b[1] for a, b in zip(records, records[1:])):
        raise ValueError("chronological records with increasing draw IDs required")
    if set(models) != set(ARMS):
        raise ValueError("all three covariance models are required")

    folds = []
    reference = assert_official_coverage_parity(REFERENCE, REFERENCE)
    for fold in range(fold_count):
        start = train_min + fold * horizon
        train = records[:start]
        holdout = records[start:start + horizon]
        portfolios = {"history_blind": disjoint_portfolio(BASELINE_SEED + fold)}
        for arm in ARMS:
            result = select_multivariate_portfolio(
                train, models[arm], pool_size=POOL_SIZE, seed=SELECTOR_SEED,
                coverage_constraint=CONSTRAINT,
            )
            portfolios[arm] = tuple(tuple(ticket) for ticket in result["tickets"])
        for tickets in portfolios.values():
            assert_official_coverage_parity(REFERENCE, tickets)

        wins = dict.fromkeys(WIN_ARMS, 0)
        awarded = dict.fromkeys(WIN_ARMS, 0)
        paired = dict.fromkeys(("exact_only", "matched_mc_only", "both", "neither"), 0)
        categories = {
            (True, False): "exact_only", (False, True): "matched_mc_only",
            (True, True): "both", (False, False): "neither",
        }
        for day, draw_id, main in holdout:
            bonus = bonuses[day, draw_id]
            scored = {arm: draw_awards(tickets, main, bonus)
                      for arm, tickets in portfolios.items()}
            for arm, result in scored.items():
                wins[arm] += int(result["any_award"])
                awarded[arm] += result["awarded_tickets"]
            exact_wins = scored["exact"]["any_award"]
            mc_wins = scored["matched_mc"]["any_award"]
            paired[categories[exact_wins, mc_wins]] += 1

        folds.append({
            "train_draws": len(train), "train_end": train[-1][:2],
            "holdout_first": holdout[0][:2], "holdout_last": holdout[-1][:2],
            "holdout_draws": len(holdout), "portfolios": portfolios,
            "winning_draws": wins, "awarded_tickets": awarded, "paired": paired,
        })

    winning_draws = {arm: sum(fold["winning_draws"][arm] for fold in folds)
                     for arm in WIN_ARMS}
    paired = {key: sum(fold["paired"][key] for fold in folds) for key in folds[0]["paired"]}
    delta = winning_draws["exact"] - winning_draws["matched_mc"]
    if sum(paired.values()) != horizon * fold_count or delta != (
        paired["exact_only"] - paired["matched_mc_only"]
    ):
        raise AssertionError("paired holdout counts do not close")

    return {
        "folds": folds,
        "aggregate": {
            "tested_draws": horizon * fold_count, "winning_draws": winning_draws,
            "awarded_tickets": {arm: sum(fold["awarded_tickets"][arm] for fold in folds)
                                for arm in WIN_ARMS},
            "paired": paired, "exact_minus_matched_mc": delta,
            "same_exact_and_matched_mc_portfolio_folds": sum(
                fold["portfolios"]["exact"] == fold["portfolios"]["matched_mc"]
                for fold in folds
            ),
            "fair_any_award_probability_all_arms": reference["any_official_prize_probability"],
        },
    }


def report(csv_dir, fixture, moments_path):
    records, csv_hash = load_csv_history(csv_dir, CUTOFF)
    if len(records) != HISTORY_DRAWS or records[-1][:2] != (CUTOFF, LAST_DRAW):
        raise ValueError("history must match the original 2,497-draw cutoff")
    bonuses = checked_bonus_by_draw(csv_dir, records)
    baseline, fixture_hash = load_exact_baseline(fixture)
    moments_bytes = moments_path.read_bytes()
    models, _ = benchmark_models(baseline, json.loads(moments_bytes))

    observed = walkforward(records, bonuses, models)
    # The last 497 draws after the original eight holdouts do not enter a fold.
    used = records[:TRAIN_MIN + HORIZON * FOLDS]
    null = []
    for seed in range(FIRST_NULL_SEED, FIRST_NULL_SEED + NULL_REPLICATES):
        synthetic, synthetic_bonuses = iid_full_history(used, seed=seed)
        replay = walkforward(synthetic, synthetic_bonuses, models)
        null.append({"seed": seed, **replay["aggregate"]})

    observed_delta = observed["aggregate"]["exact_minus_matched_mc"]
    null_deltas = [item["exact_minus_matched_mc"] for item in null]
    calibration = empirical_calibration(observed_delta, null_deltas)
    upper = calibration["p_empirical_upper_exploratory"]
    return {
        "schema": "lotto649.v025.matched_covariance_walkforward.v1",
        "scope": "exploratory retrospective walk-forward under exact prize coverage equality",
        "source": {
            "draws": len(records), "last_date": CUTOFF, "last_draw_id": LAST_DRAW,
            "input_csv_sha256": csv_hash,
            "used_six_main_and_bonus_sha256": json_sha256([
                (*row, bonuses[row[:2]]) for row in used
            ]),
            "shape_fixture_sha256": fixture_hash,
            "exact_population_moments_sha256": sha256(moments_bytes).hexdigest(),
        },
        "config": {
            "train_min": TRAIN_MIN, "horizon": HORIZON, "folds": FOLDS,
            "pool_size": POOL_SIZE, "selector_seed": SELECTOR_SEED,
            "candidate_pool_sha256": json_sha256(candidate_pool(POOL_SIZE, SELECTOR_SEED)),
            "baseline_seed": BASELINE_SEED, "coverage_constraint": CONSTRAINT,
            "first_null_seed": FIRST_NULL_SEED, "last_null_seed": FIRST_NULL_SEED + NULL_REPLICATES - 1,
            "numpy": np.__version__,
        },
        "models": {arm: model["meta"] for arm, model in models.items()},
        "observed": observed,
        "null_calibration": {
            "replicates": len(null), "mean_delta": calibration["null_mean"],
            "delta_quantiles_2_5_and_97_5_pct": np.quantile(null_deltas, [0.025, 0.975]).tolist(),
            "upper_tail_count": sum(delta >= observed_delta for delta in null_deltas),
            "exploratory_upper_tail_p": upper,
            "p_resolution": calibration["p_resolution"],
            "null_by_seed": null,
        },
        "research_decision": (
            "EXPLORATORY_SIGNAL_REQUIRES_PROSPECTIVE_VALIDATION"
            if observed_delta > 0 and upper <= 0.05 else "NO_EVIDENCE"
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, default=Path("data/csv"))
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--exact-moments", type=Path, default=EXACT)
    parser.add_argument("--out", type=Path,
                        default=Path("results/v025_matched_covariance_walkforward.json"))
    args = parser.parse_args(argv)
    result = report(args.csv_dir, args.fixture, args.exact_moments)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(f"[WALK-FORWARD] {result['observed']['aggregate']['winning_draws']}")
    print(f"[IID] p={result['null_calibration']['exploratory_upper_tail_p']:.3f}")
    print(f"[REPORT] {args.out}")


if __name__ == "__main__":
    main()
