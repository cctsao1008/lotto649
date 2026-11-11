"""v0.29 P3A: structural conditional-redundancy certificate, not forecasting.

The v0.28 baseline is L3 shape + exact Sum + predeclared Log Product bin.
Joint(Sum, LogBin) is determined by the last two; L1/L2 are nested in L3.
Find legal equal-baseline tickets with unequal long-range image descriptors.
This establishes algebraic non-identity only, not information about future draws.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import prod
from pathlib import Path

import numpy as np

from .geometry_v029 import catalog
from .geometry_structure_v029 import PAIR_FEATURES, _features_and_l3, _mapping_labels
from .geometry_convergence_v029 import _validate_draws, audit_repository_inputs
from .log_product_v027 import _grid

SCHEMA = "lotto649.geometry_conditional_ablation.v0.29.p3a.v1"
TICKETS_CERTIFICATE = (
    (3, 10, 19, 23, 31, 46),
    (4, 8, 12, 27, 34, 47),
)
DEFAULT_IID_SAMPLE = 50000
DEFAULT_SEED = 649


def log_grid() -> dict:
    """Use the same exact integer Product bin boundaries as v0.28 M0."""
    return _grid(49, 6, "0.5")


def _descriptors(numbers: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact v0.28 baseline key, three pair-geometry values, sorted tickets."""
    draws = _validate_draws(numbers)
    labels = _mapping_labels("sequential", 0)
    features, l3 = _features_and_l3(draws, labels)
    products = np.prod(draws.astype(np.int64), axis=1, dtype=np.int64)
    thresholds = np.asarray(log_grid()["integer_product_thresholds"], dtype=np.int64)
    bins = np.searchsorted(thresholds, products, side="right") - 1
    if np.any(bins < 0) or np.any(bins >= len(thresholds) - 1):
        raise AssertionError("exact product outside frozen Log Product bin grid")
    sums = draws.sum(axis=1, dtype=np.int32)
    key = np.stack((l3.astype(np.int64), sums, bins), axis=1)
    return key, features[:, :3].astype(np.int64), draws


def _certificate() -> dict:
    # _validate_draws deliberately requires >=25 records; pad only for this
    # algebraic fixture, never to train a statistical model.
    a = np.asarray(TICKETS_CERTIFICATE, dtype=np.int16)
    padded = np.vstack((a, np.tile(a[0], (23, 1))))
    keys, features, _ = _descriptors(padded)
    if not np.array_equal(keys[0], keys[1]):
        raise AssertionError("P3A exact Sum + LogBin + L3 collision certificate changed")
    if np.array_equal(features[0], features[1]):
        raise AssertionError("two tickets do not distinguish long-range geometry")
    expected = {"l3": int(keys[0, 0]), "sum": 132, "log_bin": int(keys[0, 2])}
    if expected["log_bin"] != 20 or int(features[0, 0]) != 172 or int(features[1, 0]) != 258:
        raise AssertionError("fixed strict collision geometry changed")
    return {
        "same_v028_baseline": expected,
        "baseline_definition": "L3 edge class, exact Sum, integer Log Product bin; Joint is derived from the latter two",
        "tickets": [
            {"numbers": list(a[i]), "exact_product": int(prod(a[i])),
             "pair_geometry": dict(zip(PAIR_FEATURES, map(int, features[i])))}
            for i in (0, 1)
        ],
        "structural_nonidentity_proven": True,
        "limitation": "Algebraic non-identity does not establish conditional probability anomaly or next-draw predictability.",
    }


def _stratum_audit(numbers: object) -> dict:
    """Count observed same-baseline groups and residual pair-feature variation.

    Within-group variance is intentionally NOT interpreted as a predictive
    skill estimate, calibrated p-value, or expected future repair gain.
    """
    keys, features, draws = _descriptors(numbers)
    groups = {}
    repeated = 0
    variable = 0
    differing_features = [0, 0, 0]
    first_example = None
    for i in range(len(draws)):
        key = tuple(map(int, keys[i]))
        f = tuple(map(int, features[i]))
        if key not in groups:
            groups[key] = [i, 1, list(f), list(f), [0, 0, 0]]
        else:
            item = groups[key]
            item[1] += 1
            for j in range(3):
                item[2][j] = min(item[2][j], f[j])
                item[3][j] = max(item[3][j], f[j])
    for key, (first_idx, count, low, high, _) in groups.items():
        if count >= 2:
            repeated += 1
            different = [low[j] != high[j] for j in range(3)]
            if any(different):
                variable += 1
                differing_features = [x + int(y) for x, y in zip(differing_features, different)]
                if first_example is None:
                    first_example = {
                        "baseline_key_l3_sum_log_bin": list(key),
                        "first_ticket": [int(v) for v in draws[first_idx]],
                        "region_pair_feature_min": dict(zip(PAIR_FEATURES, low)),
                        "region_pair_feature_max": dict(zip(PAIR_FEATURES, high)),
                        "group_draw_count": count,
                    }
    return {
        "sample_draws": int(len(draws)),
        "observed_baseline_cells": len(groups),
        "baseline_cells_with_multiple_draws": repeated,
        "repeated_cells_with_structural_variation": variable,
        "repeated_cells_variable_by_feature": dict(zip(PAIR_FEATURES, differing_features)),
        "first_descriptive_example": first_example,
        "warnings": [
            "Collision fractions depend on sample size and the sparse stratification.",
            "Repeated historical cells are not independent and have no predictive interpretation.",
            "The exact v0.28 Sum/Log/Joint/L3 mapping is conditioned upon, not replaced.",
        ],
    }


def analyze_conditional_ablation(history: object, *,
                                 iid_sample: int = DEFAULT_IID_SAMPLE,
                                 seed: int = DEFAULT_SEED) -> dict:
    if type(iid_sample) is not int or not 25 <= iid_sample <= 500000:
        raise ValueError("iid_sample must be an integer in 25..500000")
    if type(seed) is not int or not 0 <= seed < 2 ** 63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    audited = _validate_draws(history)
    if catalog()["definition_sha256"] != (
        "5aa5f2820c3287fde1d548de8c90a46b1aed95ed0b7b0f8b9a84c2ea39c82f47"
    ):
        raise ValueError("fixed geometric definition changed")
    rng = np.random.default_rng(seed)
    iid = rng.permuted(np.broadcast_to(
        np.arange(1, 50, dtype=np.int16), (iid_sample, 49)), axis=1)[:, :6]
    grid = log_grid()
    grid_info = {
        "edge_start": grid["edge_start"], "edge_end": grid["edge_end"],
        "width": grid["width"], "bin_count": grid["bin_count"],
        "integer_thresholds_sha256": sha256(
            json.dumps(grid["integer_product_thresholds"], separators=(",", ":")).encode("ascii")
        ).hexdigest(),
    }
    return {
        "schema": SCHEMA,
        "decision": "STRUCTURAL_NONIDENTITY_PROVEN_PREDICTIVE_NO_GO",
        "geometry_definition_sha256": catalog()["definition_sha256"],
        "baseline": "v0.28 M0: L1/L2/L3 nested Shape block; exact Sum; LogProductBin; Joint(Sum,LogProductBin)",
        "log_grid": grid_info,
        "mapping": "sequential",
        "certificate": _certificate(),
        "historical_descriptive_collision_audit": _stratum_audit(audited),
        "fair_iid_descriptive_collision_audit": _stratum_audit(iid),
        "iid_sampling": {"seed": seed, "sample_size": iid_sample,
                         "method": "independent row-wise permutation of range(1,50), six distinct numbers per draw"},
        "explicit_nonclaims": [
            "No hypothesis test of predictive advantage is performed by P3A.",
            "The finite-sample collision rate is not a p-value or estimate of future odds.",
            "Pair distances are not algebraically determined by L3+Sum+LogBin, but are deterministic functions of the exact six numbers.",
            "Sum+Log Product Joint is a deterministic combination of the two existing objectives.",
            "P2A and P2B whole-search tests did not reject fair IID; this ablation does not overturn those results.",
            "Do not add a fifth candidate selection objective unless prospective evidence is independently demonstrated.",
        ],
        "next_step": "P3B counterfactual distance-repair can be analyzed only as descriptive optimization with an IID candidate control; no automatic paid or shadow ticket changes.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-root", type=Path, default=Path("data/csv"))
    parser.add_argument("--bin-root", type=Path, default=Path("data/bin"))
    parser.add_argument("--iid-sample", type=int, default=DEFAULT_IID_SAMPLE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    if not output.is_relative_to(Path("results").resolve()):
        raise ValueError("P3A report path must be under gitignored results/")
    history, gate = audit_repository_inputs(args.csv_root, args.bin_root)
    report = analyze_conditional_ablation(
        history, iid_sample=args.iid_sample, seed=args.seed)
    report["source_gate"] = gate
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii", newline="\n") as fp:
        json.dump(report, fp, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        fp.write("\n")
    h = report["historical_descriptive_collision_audit"]
    r = report["fair_iid_descriptive_collision_audit"]
    print("PASS: historical source gate and fixed baseline collision certificate")
    print(f"DATA: {gate['draws']} verified historical draws; "
          f"{args.iid_sample} independent fair-IID demonstration draws")
    print(f"BASELINE STRATA: history {h['baseline_cells_with_multiple_draws']} repeated, "
          f"{h['repeated_cells_with_structural_variation']} structurally variable; "
          f"IID {r['baseline_cells_with_multiple_draws']} repeated, "
          f"{r['repeated_cells_with_structural_variation']} structurally variable")
    print("DECISION: STRUCTURAL_NONIDENTITY_PROVEN_PREDICTIVE_NO_GO")
    print(f"REPORT: {output}")


if __name__ == "__main__":
    main()
