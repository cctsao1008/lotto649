"""v0.25 P4 exploratory robustness and *pipeline-matched* IID null controls.

A lower in-sample discrepancy after optimization is expected even for fair
random histories. The null therefore reruns THE SAME optimizer and candidate
pool rules on independently generated IID pseudo-histories. This does not
estimate future lottery prediction skill or produce confirmatory p-values.
"""
from __future__ import annotations

import json
from math import isfinite
from pathlib import Path

import numpy as np

from .deficit_compensation_v025 import REFERENCE, load_csv_history
from .multivariate_audit_v025 import (
    fair_covariance, iid_tickets, portfolio_audit, select_multivariate_portfolio,
)
from .shape_deficit_v025 import load_exact_baseline


def _integer_sequence(text):
    """Ordered unique nonnegative integers from a comma-separated string."""
    try:
        result = tuple(int(s.strip()) for s in text.split(","))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("expected comma-separated nonnegative integers") from exc
    if not result or len(set(result)) != len(result) or any(v < 0 for v in result):
        raise ValueError("expected distinct nonnegative integers")
    return result


def _float_sequence(text):
    try:
        result = tuple(float(s.strip()) for s in text.split(","))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("expected comma-separated positive ridge floats") from exc
    if (not result or len(set(result)) != len(result)
            or any(not isfinite(v) or v <= 0 for v in result)):
        raise ValueError("expected distinct positive finite ridge floats")
    return result


def _historical_tuples(tickets):
    """Dates/IDs have no role in features; synthetic IDs must not imply actual draws."""
    return [(None, i + 1, tuple(int(x) for x in row))
            for i, row in enumerate(tickets)]


def compare_portfolio(records, model, *, pool_size, selector_seed,
                      coverage_constraint):
    reference = portfolio_audit(records, model, REFERENCE)
    selected = select_multivariate_portfolio(
        records, model, pool_size=pool_size, seed=selector_seed,
        coverage_constraint=coverage_constraint)
    if abs(selected["objective"] - portfolio_audit(
            records, model, selected["tickets"])["after"]) > 1e-9:
        raise AssertionError("selector score differs from direct recomputation")
    return {
        "reference_distance": reference["after"],
        "optimized_distance": selected["objective"],
        "reference_minus_optimized": reference["after"] - selected["objective"],
        "selected_tickets": selected["tickets"],
        "coverage_floor_pass": selected["meets_reference_3plus_floor"],
        "coverage_3plus": selected["exact_main_ball_coverage"]["at_least"]["3"],
    }


def robustness_audit(records, baseline, *, iid_sizes=(8192,),
                     covariance_seeds=(649,), ridges=(0.25,),
                     selector_seed=649, pool_size=300,
                     null_replicates=99, null_seed=1649,
                     coverage_constraint="reference-3plus"):
    """Calibration replicates MUST re-optimize on independently sampled IID history.

    Candidate pool uses same deterministic selector_seed as the real
    scenario; covariance model for each scenario is fitted on an independent
    IID cohort. Null draws use separate seed stream per scenario/replicate.
    Empirical p counts equally extreme *optimization gain* under fair IID.
    This is exploratory since model/metric/grid/seed were previously inspected.
    """
    if not records:
        raise ValueError("history must not be empty")
    if not isinstance(null_replicates, int) or not 1 <= null_replicates <= 10000:
        raise ValueError("null_replicates must be 1..10000")
    if (not isinstance(null_seed, int) or null_seed < 0
            or not isinstance(selector_seed, int) or selector_seed < 0):
        raise ValueError("seeds must be nonnegative")
    if not isinstance(pool_size, int) or not 4 <= pool_size <= 5000:
        raise ValueError("pool_size must be 4..5000")
    if coverage_constraint not in ("none", "reference-3plus"):
        raise ValueError("unknown coverage constraint")
    iid_sizes = tuple(iid_sizes)
    covariance_seeds = tuple(covariance_seeds)
    ridges = tuple(ridges)
    if (not iid_sizes or not covariance_seeds or not ridges
            or len(set(iid_sizes)) != len(iid_sizes)
            or len(set(covariance_seeds)) != len(covariance_seeds)
            or len(set(ridges)) != len(ridges)
            or any(not isinstance(s, int) or not 256 <= s <= 200000 for s in iid_sizes)
            or any(not isinstance(s, int) or s < 0 for s in covariance_seeds)
            or any(not isfinite(float(r)) or r <= 0 for r in ridges)):
        raise ValueError("invalid covariance robustness grid")

    # Precompute synthetic independent random-history labels ONCE per null
    # replicate and reuse across the ridge/seed grid for paired robustness
    # comparisons. Seed stream is explicit and independent from covariance.
    n = len(records)
    iid_histories = [
        _historical_tuples(iid_tickets(n, null_seed + 1 + r))
        for r in range(null_replicates)
    ]
    scenarios = []
    for size in iid_sizes:
        for cov_seed in covariance_seeds:
            for ridge in ridges:
                model = fair_covariance(baseline, samples=size, seed=cov_seed, ridge=ridge)
                observed = compare_portfolio(
                    records, model, pool_size=pool_size,
                    selector_seed=selector_seed,
                    coverage_constraint=coverage_constraint)
                gains = []
                for synthetic in iid_histories:
                    item = compare_portfolio(
                        synthetic, model, pool_size=pool_size,
                        selector_seed=selector_seed,
                        coverage_constraint=coverage_constraint)
                    gains.append(float(item["reference_minus_optimized"]))
                gains = np.asarray(gains, dtype=float)
                p_exploratory = (1 + int(np.count_nonzero(
                    gains >= observed["reference_minus_optimized"] - 1e-12))
                    ) / (null_replicates + 1)
                scenarios.append({
                    "iid_covariance_samples": size,
                    "covariance_seed": cov_seed,
                    "ridge": ridge,
                    "condition_number": model["meta"]["condition_number"],
                    "observed": observed,
                    "null_pipeline_matched": {
                        "replicates": null_replicates,
                        "mean_gain": float(gains.mean()),
                        "median_gain": float(np.median(gains)),
                        "min_gain": float(gains.min()),
                        "max_gain": float(gains.max()),
                        "p_empirical_exploratory_upper": p_exploratory,
                        "p_resolution": 1 / (null_replicates + 1),
                        "gains": gains.tolist(),
                    },
                })
    return {
        "schema": "lotto7x7.multivariate_robustness.v0.25.p4",
        "history_draws": n,
        "config": {
            "iid_sizes": list(iid_sizes),
            "covariance_seeds": list(covariance_seeds),
            "ridges": list(ridges),
            "selector_seed": selector_seed,
            "pool_size": pool_size,
            "null_replicates": null_replicates,
            "null_seed": null_seed,
            "coverage_constraint": coverage_constraint,
            "matched_null": "same selector and reference, independent uniform 6/49 histories",
        },
        "scenarios": scenarios,
        "interpretation": [
            "This is an in-sample objective optimization-gain null, NOT a prediction-skill test.",
            "The candidate pool is shared across real/null histories; selected portfolios re-optimized.",
            "Finite null replicates limit p-value resolution; grid exploration prevents confirmatory inference.",
            "Cross-feature covariance is estimated by IID Monte Carlo, not exact.",
            "The purchased four-ticket A-D set and previous forecast seals are unchanged.",
        ],
    }


def report(csv_dir: Path, universe: Path, *, cutoff=None, iid_sizes=(8192,),
           covariance_seeds=(649,), ridges=(0.25,), selector_seed=649,
           pool_size=300, null_replicates=99, null_seed=1649,
           coverage_constraint="reference-3plus"):
    records, history_hash = load_csv_history(csv_dir, cutoff)
    exact, universe_hash = load_exact_baseline(universe)
    output = robustness_audit(
        records, exact, iid_sizes=iid_sizes, covariance_seeds=covariance_seeds,
        ridges=ridges, selector_seed=selector_seed, pool_size=pool_size,
        null_replicates=null_replicates, null_seed=null_seed,
        coverage_constraint=coverage_constraint)
    output.update({
        "last_date": records[-1][0], "last_draw_id": records[-1][1],
        "csv_sha256": history_hash,
        "exact_universe_sha256": universe_hash,
    })
    return output


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, default=Path("data/csv"))
    parser.add_argument("--universe", type=Path,
                        default=Path("results/v015_v016_p1_shape_catalog_universe.json"))
    parser.add_argument("--cutoff", default=None)
    parser.add_argument("--iid-sizes", default="8192")
    parser.add_argument("--covariance-seeds", default="649")
    parser.add_argument("--ridges", default="0.25")
    parser.add_argument("--selector-seed", type=int, default=649)
    parser.add_argument("--pool-size", type=int, default=300)
    parser.add_argument("--null-replicates", type=int, default=99)
    parser.add_argument("--null-seed", type=int, default=1649)
    parser.add_argument("--coverage-constraint", choices=("none", "reference-3plus"),
                        default="reference-3plus")
    parser.add_argument("--out", type=Path, default=Path("results/v025_multivariate_robustness.json"))
    args = parser.parse_args(argv)
    result = report(
        args.csv_dir, args.universe, cutoff=args.cutoff,
        iid_sizes=_integer_sequence(args.iid_sizes),
        covariance_seeds=_integer_sequence(args.covariance_seeds),
        ridges=_float_sequence(args.ridges),
        selector_seed=args.selector_seed, pool_size=args.pool_size,
        null_replicates=args.null_replicates, null_seed=args.null_seed,
        coverage_constraint=args.coverage_constraint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for item in result["scenarios"]:
        print("[ROBUSTNESS] samples={} seed={} ridge={} observed_gain={:+.8f} "
              "null_mean={:+.8f} exploratory_p={:.4f}".format(
                  item["iid_covariance_samples"], item["covariance_seed"],
                  item["ridge"], item["observed"]["reference_minus_optimized"],
                  item["null_pipeline_matched"]["mean_gain"],
                  item["null_pipeline_matched"]["p_empirical_exploratory_upper"]))
    print(f"[REPORT] {args.out}")


if __name__ == "__main__":
    main()
