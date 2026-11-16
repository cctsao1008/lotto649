"""Retrospective comparison of exact and Monte Carlo covariance for Issue #8."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import sqrt
import platform
from pathlib import Path

import numpy as np

from lotto7x7.coverage_v06 import exact_coverage
from lotto7x7.portfolio.deficit_compensation_v025 import REFERENCE, load_csv_history
from lotto7x7.portfolio.exact_covariance_v025 import RULE, SCHEMA, U, cov_numerator
from lotto7x7.portfolio.multivariate_audit_v025 import (
    candidate_pool, covariance_model, distance, encode_features, iid_tickets,
    legacy_covariance_model, registry, sample_covariance,
)
from lotto7x7.portfolio.multivariate_robustness_v025 import _historical_tuples, compare_portfolio
from lotto7x7.portfolio.shape_deficit_v025 import load_exact_baseline

CUTOFF = "2026-11-06"
LAST_DRAW = 115000103
HISTORY_DRAWS = 2497
COVARIANCE_DRAWS = 8192
COVARIANCE_SEED = 649
SELECTOR_SEED = 649
POOL_SIZE = 300
RIDGE = 0.25
NULL_REPLICATES = 99
FIRST_NULL_SEED = 1650
CONSTRAINT = "reference-3plus"
SCORE_TOLERANCE = 1e-12
FIXTURE = Path("data/reference/shape_class_counts.json")
EXACT = Path("data/research/v025_exact_covariance/v025_full_163_exact_covariance_13983816.json")
ARMS = ("legacy_mc", "matched_mc", "exact")


def json_sha256(value):
    return sha256(json.dumps(value, separators=(",", ":")).encode("ascii")).hexdigest()


def population_covariance(report, names, theory):
    dimension = len(names)
    if (report["schema"] != SCHEMA or report["universe"] != U
            or report["reference_taxonomy_rule"] != RULE
            or report["feature_count"] != dimension
            or tuple(report["feature_registry"]) != names):
        raise ValueError("exact covariance registry differs from the legacy feature registry")
    sums = report["sums"]
    upper = report["raw_second_moment_upper"]
    if (len(sums) != dimension or any(type(x) is not int or x < 0 for x in sums)
            or len(upper) != dimension
            or any(len(row) != dimension - i
                   or any(type(x) is not int or x < 0 for x in row)
                   for i, row in enumerate(upper))):
        raise ValueError("malformed integer population moments")
    if not np.allclose(np.asarray(sums) / U, theory, rtol=0, atol=1e-14):
        raise ValueError("exact covariance means differ from the legacy fair means")
    covariance = np.empty((dimension, dimension))
    for i in range(dimension):
        for j in range(i, dimension):
            covariance[i, j] = covariance[j, i] = cov_numerator(report, i, j) / U**2
    if np.any(np.diag(covariance) <= 0):
        raise ValueError("population covariance contains a nonpositive variance")
    return covariance


def benchmark_models(baseline, moments):
    names, theory, catalog = registry(baseline)
    population = population_covariance(moments, names, theory)
    sample = sample_covariance(catalog, COVARIANCE_DRAWS, COVARIANCE_SEED)
    variance = np.diag(population)
    standardized = population / np.sqrt(np.outer(variance, variance))
    spectrum = np.linalg.eigvalsh(standardized)
    if spectrum.min() < -1e-10:
        raise ValueError("the exact standardized covariance is not positive semidefinite")
    legacy = legacy_covariance_model(names, theory, catalog, sample,
                                     COVARIANCE_DRAWS, COVARIANCE_SEED, RIDGE)
    matched = covariance_model(names, theory, catalog, sample, variance, RIDGE, {
        "source": "legacy sample covariance with exact population standard deviations",
        "samples": COVARIANCE_DRAWS, "seed": COVARIANCE_SEED,
    })
    exact = covariance_model(names, theory, catalog, population, variance, RIDGE, {
        "source": "exact population covariance and standard deviations", "universe": U,
    })
    diagnostics = {
        "exact_rank_at_1e_10": int(np.count_nonzero(spectrum > 1e-10)),
        "standardized_covariance_relative_frobenius_error": float(
            np.linalg.norm((sample - population) / np.sqrt(np.outer(variance, variance)))
            / np.linalg.norm(standardized)),
        "sample_zero_variance_features": [
            name for name, var in zip(names, sample.diagonal()) if var == 0
        ],
    }
    return dict(zip(ARMS, (legacy, matched, exact))), diagnostics


def evaluate_history(records, models, coverage_counts):
    exact = models["exact"]
    historical_sum = encode_features([row[2] for row in records], exact["catalog"]).sum(axis=0)
    result = {}
    for arm, model in models.items():
        outcome = compare_portfolio(records, model, pool_size=POOL_SIZE,
                                    selector_seed=SELECTOR_SEED, coverage_constraint=CONSTRAINT)
        tickets = outcome["selected_tickets"]
        if exact_coverage(tickets)["at_least_counts"] != coverage_counts:
            raise AssertionError("selected portfolio broke exact 3+/4+/5+/6 coverage parity")
        augmented = historical_sum + encode_features(tickets, exact["catalog"]).sum(axis=0)
        outcome["score_on_exact_metric"] = distance(augmented / (len(records) + 4), exact)
        result[arm] = outcome
    score = result["exact"]["score_on_exact_metric"]
    return {
        "arms": result,
        "matched_mc_minus_exact": result["matched_mc"]["score_on_exact_metric"] - score,
        "legacy_mc_minus_exact": result["legacy_mc"]["score_on_exact_metric"] - score,
    }


def calibrate(observed, null):
    count = sum(value >= observed - SCORE_TOLERANCE for value in null)
    n = len(null)
    p = count / n
    z = 1.959963984540054
    scale = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / scale
    half_width = z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / scale
    return {
        "observed": observed,
        "null_mean": float(np.mean(null)),
        "null_median": float(np.median(null)),
        "null_quantiles_2_5_and_97_5_pct": np.quantile(null, [0.025, 0.975]).tolist(),
        "null_at_least_observed": count,
        "null_replicates": n,
        "exploratory_upper_tail_p": (1 + count) / (n + 1),
        "tail_proportion_wilson_95_ci": [max(0.0, center - half_width),
                                         min(1.0, center + half_width)],
        "p_resolution": 1 / (n + 1),
    }


def benchmark(csv_dir, fixture, exact_path):
    records, csv_hash = load_csv_history(csv_dir, CUTOFF)
    if len(records) != HISTORY_DRAWS or records[-1][:2] != (CUTOFF, LAST_DRAW):
        raise ValueError("history must match the original 2,497-draw benchmark cutoff")
    baseline, fixture_hash = load_exact_baseline(fixture)
    moment_bytes = exact_path.read_bytes()
    models, diagnostics = benchmark_models(baseline, json.loads(moment_bytes))
    coverage_counts = exact_coverage(REFERENCE)["at_least_counts"]
    observed = evaluate_history(records, models, coverage_counts)
    null = []
    for seed in range(FIRST_NULL_SEED, FIRST_NULL_SEED + NULL_REPLICATES):
        synthetic = _historical_tuples(iid_tickets(len(records), seed))
        null.append({"seed": seed, **evaluate_history(synthetic, models, coverage_counts)})

    null_primary = [item["matched_mc_minus_exact"] for item in null]
    primary = calibrate(observed["matched_mc_minus_exact"], null_primary)
    arm_calibration = {
        arm: calibrate(observed["arms"][arm]["reference_minus_optimized"],
                       [item["arms"][arm]["reference_minus_optimized"] for item in null])
        for arm in ARMS
    }
    selection_agreement = {
        arm: sum(item["arms"][arm]["selected_tickets"] == item["arms"]["exact"]["selected_tickets"]
                 for item in null)
        for arm in ARMS[:-1]
    }
    return {
        "schema": "lotto649.v025.matched_covariance_benchmark.v1",
        "scope": "retrospective descriptive optimization under a fixed configuration",
        "history": {
            "draws": len(records), "cutoff": CUTOFF,
            "first_draw": records[0][1], "last_draw": LAST_DRAW,
            "input_csv_sha256": csv_hash, "used_history_sha256": json_sha256(records),
        },
        "config": {
            "covariance_draws": COVARIANCE_DRAWS, "covariance_seed": COVARIANCE_SEED,
            "ridge": RIDGE, "selector_seed": SELECTOR_SEED, "pool_size": POOL_SIZE,
            "candidate_pool_sha256": json_sha256(candidate_pool(POOL_SIZE, SELECTOR_SEED)),
            "coverage_constraint": CONSTRAINT, "exact_main_ball_coverage_counts": coverage_counts,
            "search": "legacy greedy, bounded feasibility fallback, up to 3 one-swap passes",
            "reference": REFERENCE, "shape_fixture_sha256": fixture_hash,
            "exact_population_moments_sha256": sha256(moment_bytes).hexdigest(),
            "null_first_seed": FIRST_NULL_SEED, "null_replicates": NULL_REPLICATES,
            "score_tolerance": SCORE_TOLERANCE,
            "python": platform.python_version(), "numpy": np.__version__,
        },
        "model_diagnostics": diagnostics,
        "models": {arm: model["meta"] for arm, model in models.items()},
        "observed": observed,
        "primary_exact_metric_comparison": primary,
        "descriptive_status": (
            "UNUSUALLY_LARGE_POSITIVE_GAIN_IN_EXPLORATORY_CALIBRATION"
            if primary["observed"] > SCORE_TOLERANCE and primary["exploratory_upper_tail_p"] <= 0.05
            else "NO_EVIDENCE_OF_UNUSUALLY_LARGE_POSITIVE_GAIN"
        ),
        "null_reference_gain_by_arm": arm_calibration,
        "null_portfolio_agreement_with_exact": selection_agreement,
        "null_primary_outcomes": {
            "exact_lower": sum(value > SCORE_TOLERANCE for value in null_primary),
            "tied": sum(abs(value) <= SCORE_TOLERANCE for value in null_primary),
            "matched_mc_lower": sum(value < -SCORE_TOLERANCE for value in null_primary),
        },
        "null_by_seed": null,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, default=Path("data/csv"))
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--exact-moments", type=Path, default=EXACT)
    parser.add_argument("--out", type=Path, default=Path("results/v025_matched_covariance.json"))
    args = parser.parse_args(argv)
    report = benchmark(args.csv_dir, args.fixture, args.exact_moments)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, separators=(",", ":")) + "\n", encoding="utf-8")
    primary = report["primary_exact_metric_comparison"]
    print(f"[MATCHED] gain={primary['observed']:+.10g}; "
          f"empirical upper-tail p={primary['exploratory_upper_tail_p']:.3f}")
    print(f"[REPORT] {args.out}")


if __name__ == "__main__":
    main()
