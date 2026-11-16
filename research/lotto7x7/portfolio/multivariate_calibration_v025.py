"""Finite-sample fair-IID covariance, residual, and entropy diagnostics."""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import combinations
import json
from math import comb
from pathlib import Path

import numpy as np

from research.lotto7x7.portfolio.coverage_audit_v06 import _wilson
from research.lotto7x7.portfolio.deficit_compensation_v025 import load_csv_history
from research.lotto7x7.portfolio.matched_covariance_v025 import (
    CUTOFF, EXACT, FIXTURE, HISTORY_DRAWS, LAST_DRAW, RIDGE, U, json_sha256, population_covariance,
)
from research.lotto7x7.portfolio.multivariate_audit_v025 import covariance_model, encode_features, registry
from research.lotto7x7.portfolio.shape_deficit_v025 import load_exact_baseline

NULL_REPLAYS = 1999
NULL_SEED = 7649
VALIDATION_REPLAYS = 199
VALIDATION_SEED = 8649
PAIR_POSITIONS = np.asarray(list(combinations(range(6), 2)), dtype=int).T
LABEL_PAIRS = np.triu_indices(49, k=1)
PAIR_INDEX = np.full((49, 49), -1, dtype=int)
PAIR_INDEX[LABEL_PAIRS] = np.arange(comb(49, 2))


def entropy(counts):
    probabilities = np.asarray(counts, dtype=float)
    probabilities = probabilities[probabilities > 0]
    probabilities /= probabilities.sum()
    return float(-np.dot(probabilities, np.log2(probabilities)))


def population_entropy(model):
    shape = {
        name: entropy(model["theory"][[s.startswith(name + "_") for s in model["names"]]])
        for name in ("component", "partition", "edge")
    }
    return {
        "labels": entropy(np.ones(49)), "pairs": entropy(np.ones(comb(49, 2))),
        "gaps": entropy([comb(49 - gap, 5) for gap in range(1, 45)]), **shape,
    }


def features(tickets, model):
    numbers = np.asarray(tickets, dtype=int)
    matrix = encode_features(numbers, model["catalog"])
    numbers = np.sort(numbers, axis=1)
    flat = numbers - 1
    left, right = PAIR_POSITIONS
    pairs = PAIR_INDEX[flat[:, left], flat[:, right]]
    gaps = np.diff(numbers, axis=1)
    return matrix, pairs, gaps


def covariance_distance(standardized_residual, n, model):
    rotated = model["basis"].T @ standardized_residual
    return float(np.dot(rotated / model["eigenvalues"], rotated) / n)


def statistics(matrix, pairs, gaps, model):
    total = matrix.sum(axis=0)
    n = len(matrix)
    residual = (total - n * model["theory"]) / model["sd"]
    stats = {
        "covariance_distance": covariance_distance(residual, n, model),
        "max_abs_standardized_residual": float(np.abs(residual).max() / np.sqrt(n)),
        "labels": entropy(total[:49]),
        "pairs": entropy(np.bincount(pairs.ravel(), minlength=comb(49, 2))),
        "gaps": entropy(np.bincount(gaps.ravel(), minlength=45)),
    }
    start = 52
    for name, classes in model["catalog"].items():
        end = start + len(classes)
        stats[name] = entropy(total[start:end])
        start = end
    return stats


def simulate_tickets(rng, n):
    keys = rng.random((n, 49))
    return np.sort(np.argpartition(keys, 5, axis=1)[:, :6] + 1, axis=1)


def calibrate(records, model):
    tickets = np.asarray([r[2] for r in records], dtype=int)
    n = len(tickets)
    if n < 4:
        raise ValueError("at least four historical draws are required")
    boundaries = [i * n // 4 for i in range(5)]
    segments = {"full": slice(0, n), **{
        f"quarter_{i + 1}": slice(a, b)
        for i, (a, b) in enumerate(zip(boundaries, boundaries[1:]))
    }}

    def evaluate(tickets):
        matrix, pairs, gaps = features(tickets, model)
        return {name: statistics(matrix[segment], pairs[segment], gaps[segment], model)
                for name, segment in segments.items()}

    observed = evaluate(tickets)
    names = list(observed["full"])
    simulated = np.empty((NULL_REPLAYS, len(segments), len(names)))
    rng = np.random.default_rng(NULL_SEED)
    for i in range(NULL_REPLAYS):
        result = evaluate(simulate_tickets(rng, n))
        simulated[i] = [list(item.values()) for item in result.values()]

    exact_entropy = population_entropy(model)
    tests = len(segments) * len(names)
    rows = {}
    for j, (segment_name, segment) in enumerate(segments.items()):
        endpoints = {}
        for k, name in enumerate(names):
            value = observed[segment_name][name]
            reference = simulated[:, j, k]
            upper_count = int(np.count_nonzero(reference >= value - 1e-12))
            lower_count = int(np.count_nonzero(reference <= value + 1e-12))
            tail_count = min(lower_count, upper_count) if name in exact_entropy else upper_count
            multiplier = 2 if name in exact_entropy else 1
            p = min(1., multiplier * (tail_count + 1) / (NULL_REPLAYS + 1))
            mean = float(reference.mean())
            endpoints[name] = {
                "observed": value, "iid_mean": mean, "observed_minus_iid_mean": value - mean,
                "iid_quantiles_2_5_and_97_5_pct": np.quantile(reference, [0.025, 0.975]).tolist(),
                "exploratory_p": p, "bonferroni_p": min(1., tests * p),
                "upper_tail_count": upper_count, "lower_tail_count": lower_count,
                "upper_tail_wilson_95": _wilson(upper_count, NULL_REPLAYS),
                "lower_tail_wilson_95": _wilson(lower_count, NULL_REPLAYS),
            }
            if name in exact_entropy:
                endpoints[name].update({"population_entropy_bits": exact_entropy[name],
                                        "estimated_plugin_bias_bits": mean - exact_entropy[name]})
        rows[segment_name] = {
            "draws": segment.stop - segment.start, "first": records[segment.start][:2],
            "last": records[segment.stop - 1][:2], "endpoints": endpoints,
        }

    full_primary = simulated[:, 0, 0]
    rng = np.random.default_rng(VALIDATION_SEED)
    rejected = 0
    for _ in range(VALIDATION_REPLAYS):
        matrix = encode_features(simulate_tickets(rng, n), model["catalog"])
        standardized = (matrix.sum(axis=0) - n * model["theory"]) / model["sd"]
        stat = covariance_distance(standardized, n, model)
        p = (1 + np.count_nonzero(full_primary >= stat - 1e-12)) / (NULL_REPLAYS + 1)
        rejected += int(p <= 0.05)

    totals = encode_features(tickets, model["catalog"]).sum(axis=0)
    z = (totals - n * model["theory"]) / (np.sqrt(n) * model["sd"])
    ranked = sorted(zip(model["names"], totals, n * model["theory"], z),
                    key=lambda r: (-abs(r[3]), r[0]))
    return {
        "endpoints": rows,
        "tests_in_bonferroni_family": tests,
        "primary_size_validation": {
            "nominal_alpha": 0.05, "replays": VALIDATION_REPLAYS, "seed": VALIDATION_SEED,
            "rejections": rejected, "rejection_rate": rejected / VALIDATION_REPLAYS,
            "wilson_95_ci_conditional_on_null_reference": _wilson(rejected, VALIDATION_REPLAYS),
        },
        "ranked_marginal_residuals": [
            {"feature": name, "observed_sum": int(actual), "expected_sum": float(expected),
             "standardized_residual": float(stat)}
            for name, actual, expected, stat in ranked
        ],
        "descriptive_decision": (
            "EXPLORATORY_PRIMARY_ANOMALY"
            if rows["full"]["endpoints"]["covariance_distance"]["bonferroni_p"] <= 0.05
            else "NO_EVIDENCE_IN_PRIMARY_CALIBRATION"
        ),
    }


def report(csv_dir, fixture, moments_path):
    records, csv_hash = load_csv_history(csv_dir, CUTOFF)
    if len(records) != HISTORY_DRAWS or records[-1][:2] != (CUTOFF, LAST_DRAW):
        raise ValueError("history must match the original 2,497-draw cutoff")
    baseline, fixture_hash = load_exact_baseline(fixture)
    raw = moments_path.read_bytes()
    names, theory, catalog = registry(baseline)
    covariance = population_covariance(json.loads(raw), names, theory)
    model = covariance_model(names, theory, catalog, covariance, np.diag(covariance), RIDGE, {
        "source": "exact population covariance", "universe": U,
        "exact_covariance_rank": int(np.linalg.matrix_rank(covariance)),
    })
    return {
        "schema": "lotto649.v025.multivariate_calibration.v1",
        "scope": "descriptive diagnostics with exploratory retrospective null tests",
        "history": {"draws": len(records), "last_date": CUTOFF, "last_draw": LAST_DRAW,
                    "input_csv_sha256": csv_hash, "used_history_sha256": json_sha256(records)},
        "model": {**model["meta"], "fixture_sha256": fixture_hash,
                  "exact_moments_sha256": sha256(raw).hexdigest()},
        "config": {"null_replays": NULL_REPLAYS, "null_seed": NULL_SEED,
                   "numpy": np.__version__, "p_resolution": 1 / (NULL_REPLAYS + 1)},
        **calibrate(records, model),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, default=Path("data/csv"))
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--exact-moments", type=Path, default=EXACT)
    parser.add_argument("--out", type=Path, default=Path("results/v025_multivariate_calibration.json"))
    args = parser.parse_args(argv)
    result = report(args.csv_dir, args.fixture, args.exact_moments)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print("[CALIBRATION]", result["descriptive_decision"])
    print("[REPORT]", args.out)


if __name__ == "__main__":
    main()
