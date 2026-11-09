"""v0.26 P3: IID pipeline-matched calibration of A/B out-of-sample prize wins.

The null must replay the *entire* historical training and selection pipeline,
not shuffle observed winning indicators and not compare in-sample distance.
Each null replicate generates an independent fair 6+special draw sequence
for the same chronological index, then selects HDDC B *anew* in each fold
from only its synthetic training prefix. A remains history-blind.

Observed statistics are exploratory because the research design was chosen
after inspecting historical lottery outcomes. The test does not provide a
prospective prediction claim.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from math import isfinite
from pathlib import Path

import numpy as np

from .deficit_compensation_v025 import load_csv_history
from .multivariate_audit_v025 import fair_covariance
from .shape_deficit_v025 import load_exact_baseline
from .strategy_ab_v026 import (
    checked_bonus_by_draw, chronological_walkforward,
)

SCHEMA = "lotto7x7.iid_ab_walkforward_calibration.v0.26.p3"


def iid_full_history(template, *, seed):
    """Generate independent uniform 6+special draws with timeline-only metadata.

    Actual main/bonus outcomes are never used to produce simulated outcomes.
    For each draw, sampling seven labels without replacement gives the first
    six a uniform six-element main subset and the seventh a uniform special
    among the remaining 43. The same (date, draw_id) template is used solely
    for fold and no-lookahead assertions.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not template:
        raise ValueError("empty timeline")
    rng = np.random.default_rng(seed)
    sequence = []
    bonuses = {}
    for day, draw_id, _ in template:
        values = rng.choice(49, size=7, replace=False) + 1
        mains = tuple(sorted(int(x) for x in values[:6]))
        special = int(values[6])
        if special in mains:
            raise AssertionError("IID generator special ball repeats a main ball")
        key = (day, draw_id)
        if key in bonuses:
            raise ValueError("duplicate date/draw ID in timeline")
        sequence.append((day, draw_id, mains))
        bonuses[key] = special
    return sequence, bonuses


def empirical_calibration(observed_delta, null_deltas):
    """Upper- and two-sided empirical p, including the observed pseudocount.

    The null is centered at zero because A and B have the same unconditional
    fair draw union probability and B sees no held-out outcome.
    """
    vals = [int(v) for v in null_deltas]
    if not vals or isinstance(observed_delta, bool) or int(observed_delta) != observed_delta:
        raise ValueError("nonempty integer null deltas and observed delta required")
    observed_delta = int(observed_delta)
    n = len(vals)
    upper = (1 + sum(v >= observed_delta for v in vals)) / (n + 1)
    lower = (1 + sum(v <= observed_delta for v in vals)) / (n + 1)
    two_sided = (1 + sum(abs(v) >= abs(observed_delta) for v in vals)) / (n + 1)
    return {
        "replicates": n,
        "observed_b_minus_a": observed_delta,
        "null_b_minus_a": vals,
        "null_mean": sum(vals) / n,
        "null_min": min(vals),
        "null_max": max(vals),
        "p_empirical_upper_exploratory": upper,
        "p_empirical_lower_exploratory": lower,
        "p_empirical_two_sided_exploratory": two_sided,
        "p_resolution": 1 / (n + 1),
        "null_definition": "independent IID 6+special histories, complete retrained A/B folds",
        "inference": "exploratory, not prospectively preregistered",
    }


def calibrate(records, observed_bonuses, model, *,
              null_replicates=99, null_seed=2649,
              train_min=1800, horizon=25, stride=25, max_folds=8,
              selector_seed=649, pool_size=300, a_seed=649,
              on_progress=None):
    """Match all selection/fold settings, except independently IID outcomes."""
    if (not isinstance(null_replicates, int) or isinstance(null_replicates, bool)
            or not 1 <= null_replicates <= 1000):
        raise ValueError("null_replicates must be 1..1000")
    if not isinstance(null_seed, int) or isinstance(null_seed, bool) or null_seed < 0:
        raise ValueError("null_seed must be a nonnegative integer")
    observed = chronological_walkforward(
        records, observed_bonuses, model,
        train_min=train_min, horizon=horizon, stride=stride,
        max_folds=max_folds, selector_seed=selector_seed,
        pool_size=pool_size, a_seed=a_seed)
    observed_agg = observed["aggregate"]
    observed_delta = observed_agg["b_wins"] - observed_agg["a_wins"]
    null_deltas = []
    null_summaries = []
    expected_folds = [(f["train_end_id"], f["test_begin_id"],
                       f["test_end_id"]) for f in observed["folds"]]
    for ri in range(null_replicates):
        seed = null_seed + ri + 1
        synthetic, bonuses = iid_full_history(records, seed=seed)
        replay = chronological_walkforward(
            synthetic, bonuses, model,
            train_min=train_min, horizon=horizon, stride=stride,
            max_folds=max_folds, selector_seed=selector_seed,
            pool_size=pool_size, a_seed=a_seed)
        folds = [(f["train_end_id"], f["test_begin_id"], f["test_end_id"])
                 for f in replay["folds"]]
        if folds != expected_folds:
            raise AssertionError("IID null fold indices differ from observed folds")
        agg = replay["aggregate"]
        if agg["draws"] != observed_agg["draws"]:
            raise AssertionError("IID null tested draw count differs")
        delta = agg["b_wins"] - agg["a_wins"]
        null_deltas.append(delta)
        null_summaries.append({
            "replicate": ri, "history_seed": seed,
            "a_wins": agg["a_wins"], "b_wins": agg["b_wins"],
            "b_minus_a": delta,
            "a_only": agg["a_only"], "b_only": agg["b_only"],
            "both": agg["both"], "neither": agg["neither"],
            "draws": agg["draws"],
        })
        if on_progress is not None:
            on_progress(ri + 1, null_replicates, delta)
    calibrated = empirical_calibration(observed_delta, null_deltas)
    return {
        "schema": SCHEMA,
        "strategy_design": {
            "A": "seeded fair-history-blind disjoint four-ticket baseline",
            "B": "v0.25 multivariate HDDC re-optimized in each historical training fold",
            "theoretical_official_prize_coverage_locked": True,
            "iid_covariance_fit": "independent IID model; held constant in all replicates",
            "fold_selection": "strictly chronological, expanding training, nonoverlapping holdout",
            "train_min": train_min, "horizon": horizon, "stride": stride,
            "max_folds": max_folds, "selector_seed": selector_seed,
            "pool_size": pool_size, "a_seed": a_seed,
            "null_replicates": null_replicates, "null_seed": null_seed,
            "null_history_seeds": [null_seed + i + 1 for i in range(null_replicates)],
        },
        "observed": {
            "folds": len(observed["folds"]),
            "tested_draws": observed_agg["draws"],
            "a_wins": observed_agg["a_wins"],
            "b_wins": observed_agg["b_wins"],
            "b_minus_a": observed_delta,
            "a_only": observed_agg["a_only"],
            "b_only": observed_agg["b_only"],
            "both": observed_agg["both"],
            "neither": observed_agg["neither"],
            "iid_theoretical_any_prize_probability_both":
                observed["iid_theoretical_any_prize_probability_both"],
            "paired_conditional_exact_p_exploratory":
                observed_agg["paired_two_sided_exact_p_exploratory"],
            "fold_boundaries": [
                {"train_end_id": x["train_end_id"],
                 "test_begin_id": x["test_begin_id"],
                 "test_end_id": x["test_end_id"]} for x in observed["folds"]
            ],
        },
        "null_calibration": calibrated,
        "null_replay_summaries": null_summaries,
        "limitations": [
            "Research design and model hyperparameters were inspected retrospectively; empirical p-values are exploratory.",
            "Finite IID replicates give a coarse Monte Carlo p-resolution.",
            "Theoretical IID joint winning probabilities of A and B are identical for all folds.",
            "The pre-existing purchased A-D four-ticket draw seal is not a prospectively enrolled A/B trial.",
        ],
    }


def report(csv_dir, universe, *, cutoff="2026-11-06",
           iid_samples=8192, covariance_seed=649, ridge=0.25,
           pool_size=300, selector_seed=649, a_seed=649,
           train_min=1800, horizon=25, stride=25, max_folds=8,
           null_replicates=99, null_seed=2649,
           on_progress=None):
    if cutoff is None:
        raise ValueError("explicit historical cutoff required")
    records, csv_hash = load_csv_history(Path(csv_dir), cutoff)
    bonuses = checked_bonus_by_draw(Path(csv_dir), records)
    baseline, universe_hash = load_exact_baseline(Path(universe))
    model = fair_covariance(baseline, samples=iid_samples,
                            seed=covariance_seed, ridge=ridge)
    result = calibrate(
        records, bonuses, model,
        null_replicates=null_replicates, null_seed=null_seed,
        train_min=train_min, horizon=horizon, stride=stride,
        max_folds=max_folds, selector_seed=selector_seed,
        pool_size=pool_size, a_seed=a_seed, on_progress=on_progress)
    result.update({
        "historical_draws": len(records), "cutoff": cutoff,
        "history_last_date": records[-1][0],
        "history_last_draw_id": records[-1][1],
        "csv_sha256": csv_hash, "exact_universe_sha256": universe_hash,
        "covariance": model["meta"],
    })
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv-dir", type=Path, default=Path("data/csv"))
    p.add_argument("--universe", type=Path,
                   default=Path("results/v015_v016_p1_shape_catalog_universe.json"))
    p.add_argument("--cutoff", default="2026-11-06")
    p.add_argument("--iid-samples", type=int, default=8192)
    p.add_argument("--covariance-seed", type=int, default=649)
    p.add_argument("--ridge", type=float, default=0.25)
    p.add_argument("--pool-size", type=int, default=300)
    p.add_argument("--selector-seed", type=int, default=649)
    p.add_argument("--a-seed", type=int, default=649)
    p.add_argument("--train-min", type=int, default=1800)
    p.add_argument("--horizon", type=int, default=25)
    p.add_argument("--stride", type=int, default=25)
    p.add_argument("--max-folds", type=int, default=8)
    p.add_argument("--null-replicates", type=int, default=99)
    p.add_argument("--null-seed", type=int, default=2649)
    p.add_argument("--out", type=Path,
                   default=Path("results/v026_strategy_ab_iid_calibration.json"))
    a = p.parse_args(argv)
    def show_progress(finished, total, delta):
        print(f"[NULL] {finished}/{total} B_minus_A={delta:+d}", flush=True)
    result = report(
        a.csv_dir, a.universe, cutoff=a.cutoff,
        iid_samples=a.iid_samples, covariance_seed=a.covariance_seed,
        ridge=a.ridge, pool_size=a.pool_size, selector_seed=a.selector_seed,
        a_seed=a.a_seed, train_min=a.train_min, horizon=a.horizon,
        stride=a.stride, max_folds=a.max_folds, null_replicates=a.null_replicates,
        null_seed=a.null_seed, on_progress=show_progress)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    obs = result["observed"]
    null = result["null_calibration"]
    print(f"[P3] observed draws={obs['tested_draws']} A={obs['a_wins']} "
          f"B={obs['b_wins']} delta={obs['b_minus_a']:+d}")
    print(f"[CALIBRATION] IID null_mean={null['null_mean']:+.4f} "
          f"p_upper={null['p_empirical_upper_exploratory']:.4f} "
          f"p_two_sided={null['p_empirical_two_sided_exploratory']:.4f} "
          f"resolution={null['p_resolution']:.4f}")
    print("[CAUTION] exploratory retrospective analysis only; no prospective forecasting claim")
    print(f"[REPORT] {a.out}")


if __name__ == "__main__":
    main()
