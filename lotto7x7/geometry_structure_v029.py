"""v0.29 P2B: graph-edge controls and pair-distance convergence vs fair IID.

All features and null endpoints are defined before looking at draw outcomes.
Graph edges and cycle rank are already represented by existing L3 taxonomy;
only predeclared longer-range pair features are tested as structural extensions.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .geometry_v029 import POINTS, catalog, mapping_positions
from .geometry_convergence_v029 import (
    LAYOUTS, _check_config, _prefixes, _validate_draws, audit_repository_inputs,
)
from .shape_taxonomy_v015 import classify_numbers

SCHEMA = "lotto649.geometry_convergence.v0.29.p2b.v1"
PAIR_FEATURES = ("pair_squared_distance_sum", "pair_midrange_3_to_18", "pair_far_25_plus")
GRAPH_CONTROLS = ("cardinal_edges", "diagonal_edges")
FEATURES = PAIR_FEATURES + GRAPH_CONTROLS
DEFAULT_REPLICATES = 199
I6, J6 = np.triu_indices(6, 1)
I49, J49 = np.triu_indices(49, 1)


def _pair_weights(squared_distances: np.ndarray) -> np.ndarray:
    """Five fixed pair weights; last two are exactly represented in L3."""
    d = squared_distances
    return np.stack((d, (d > 2) & (d <= 18), d >= 25, d == 1, d == 2),
                    axis=-1).astype(np.float64)


def _fair_inclusion(k: int) -> float:
    num = 1
    den = 1
    for i in range(k):
        num *= 6 - i
        den *= 49 - i
    return num / den


def exact_pair_moments() -> dict[str, dict[str, float]]:
    """Exact mean/variance for pair-sum U-statistics over 6 of 49.

    The second moment partitions two distinct edge pairs into those sharing
    one vertex (three selected points) and disjoint pairs (four points).
    """
    xy = np.asarray(POINTS, dtype=np.int16)
    delta = xy[I49] - xy[J49]
    d2 = np.sum(delta.astype(np.int32) ** 2, axis=1)
    matrix = _pair_weights(d2)
    p2, p3, p4 = (_fair_inclusion(k) for k in (2, 3, 4))
    result = {}
    for index, name in enumerate(FEATURES):
        h = matrix[:, index]
        total = float(h.sum())
        squares = float(np.dot(h, h))
        incident = np.bincount(I49, weights=h, minlength=49) + np.bincount(
            J49, weights=h, minlength=49)
        shared = (float(np.dot(incident, incident)) - 2 * squares) / 2
        all_pairs = (total ** 2 - squares) / 2
        disjoint = all_pairs - shared
        mean = p2 * total
        second = p2 * squares + 2 * p3 * shared + 2 * p4 * disjoint
        variance = second - mean ** 2
        if not np.isfinite(variance) or variance <= 0:
            raise AssertionError(f"invalid exact fair variance: {name}")
        result[name] = {"mean": mean, "variance": variance,
                        "standard_deviation": float(np.sqrt(variance)),
                        "pair_weight_sum_49_cells": total}
    return result


def _mapping_labels(method: str, seed: int) -> np.ndarray:
    """Translate a number to its corresponding fixed image cell (1..49)."""
    layout = mapping_positions(method, seed)
    return np.array([r * 7 + c + 1 for r, c in layout], dtype=np.uint8)


def _features_and_l3(draws: np.ndarray, labels: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    mapped = labels[draws - 1]
    y = ((mapped.astype(np.int16) - 1) // 7)
    x = ((mapped.astype(np.int16) - 1) % 7)
    dy = y[:, I6] - y[:, J6]
    dx = x[:, I6] - x[:, J6]
    d2 = dy.astype(np.int32) ** 2 + dx.astype(np.int32) ** 2
    features = _pair_weights(d2).sum(axis=1)
    topology = classify_numbers(mapped)
    if not np.array_equal(features[:, 3], topology["cardinal_edges"]):
        raise AssertionError("cardinal edge count differs from frozen L3")
    if not np.array_equal(features[:, 4], topology["diagonal_edges"]):
        raise AssertionError("diagonal edge count differs from frozen L3")
    return features, topology["edge_code"]


def _within_l3_descriptive(features: np.ndarray, codes: np.ndarray) -> dict:
    """Illustrate distinct pair geometry within an identical L3 class.

    This is descriptive non-identity, NOT evidence of conditional predictive
    information or an independent new anomaly.
    """
    distinct = np.unique(codes)
    repeated, variable = 0, 0
    example = None
    for code in distinct:
        indexes = np.flatnonzero(codes == code)
        if len(indexes) < 2:
            continue
        repeated += 1
        values = features[indexes, 0]
        if np.ptp(values) > 0:
            variable += 1
            if example is None:
                lo = int(indexes[np.argmin(values)])
                hi = int(indexes[np.argmax(values)])
                example = {"l3_code": int(code), "draw_index_zero_based": [lo, hi],
                           "pair_squared_distance_sums": [float(features[lo, 0]),
                                                          float(features[hi, 0])]}
    return {"distinct_observed_l3_classes": int(len(distinct)),
            "l3_classes_with_multiple_draws": repeated,
            "repeated_l3_classes_with_distance_variation": variable,
            "first_observed_variation_example": example,
            "interpretation": "Not algebraically determined by L3; still may be nonpredictive."}


def _evaluate(draws: np.ndarray, prefixes: list[int],
              moments: dict[str, dict[str, float]], *, details: bool) -> dict:
    mean = np.array([moments[n]["mean"] for n in FEATURES])
    scale = np.sqrt(np.array([moments[n]["variance"] for n in FEATURES]))
    max_pair, max_graph = 0., 0.
    by_mapping = {}
    for method, seed in LAYOUTS:
        labels = _mapping_labels(method, seed)
        name = method if method != "random" else f"random:{seed}"
        values, codes = _features_and_l3(draws, labels)
        running = np.cumsum(values, axis=0, dtype=np.float64)
        rows = []
        for n in prefixes:
            observed_means = running[n - 1] / n
            z = np.sqrt(n) * (observed_means - mean) / scale
            max_pair = max(max_pair, float(np.max(np.abs(z[:3]))))
            max_graph = max(max_graph, float(np.max(np.abs(z[3:]))))
            if details:
                rows.append({"draws": n, "metrics": {
                    feature: {"mean": float(observed_means[i]),
                              "mean_minus_fair": float(observed_means[i] - mean[i]),
                              "standardized_mean_residual": float(z[i])}
                    for i, feature in enumerate(FEATURES)}})
        if details:
            by_mapping[name] = {
                "prefixes": rows, "l3_redundancy_audit": _within_l3_descriptive(values, codes),
                "graph_controls_are_already_encoded_in_l3": True,
                "graph_cycle_rank_is_determined_by_l3": True,
            }
    return {"maximum_absolute_pair_z": max_pair,
            "maximum_absolute_graph_control_z": max_graph,
            "by_mapping": by_mapping}


def analyze_structure(numbers: object, *, iid_replicates: int = DEFAULT_REPLICATES,
                      seed: int = 649) -> dict:
    _check_config(iid_replicates, seed)
    draws = _validate_draws(numbers)
    n = len(draws)
    prefixes = _prefixes(n)
    moments = exact_pair_moments()
    actual = _evaluate(draws, prefixes, moments, details=True)
    rng = np.random.default_rng(seed)
    null_pair = np.empty(iid_replicates)
    null_graph = np.empty(iid_replicates)
    universe = np.arange(1, 50, dtype=np.int16)
    for index in range(iid_replicates):
        synthetic = rng.permuted(np.broadcast_to(universe, (n, 49)), axis=1)[:, :6]
        outcome = _evaluate(synthetic, prefixes, moments, details=False)
        null_pair[index] = outcome["maximum_absolute_pair_z"]
        null_graph[index] = outcome["maximum_absolute_graph_control_z"]
    exceed_pair = int(np.count_nonzero(null_pair >= actual["maximum_absolute_pair_z"]))
    exceed_graph = int(np.count_nonzero(
        null_graph >= actual["maximum_absolute_graph_control_z"]))
    p_pair = (exceed_pair + 1) / (iid_replicates + 1)
    p_graph = (exceed_graph + 1) / (iid_replicates + 1)
    return {
        "schema": SCHEMA,
        "decision": "EXPLORATORY_DESCRIPTIVE_ONLY_NOT_PREDICTIVE",
        "source_geometry_sha256": catalog()["definition_sha256"],
        "draws": n, "seed": seed, "iid_replicates": iid_replicates,
        "features": {
            "tested_structural_pair_features": list(PAIR_FEATURES),
            "existing_l3_graph_controls": list(GRAPH_CONTROLS),
            "graph_cycle_rank_redundancy": "cycle_rank = cardinal_edges + diagonal_edges - 6 + connected_components; no new L3 dimension",
            "repeated_l3_pair_distance_variation": "descriptive only, not a test of predictability",
        },
        "theory": moments,
        "prefix_lengths": prefixes,
        "mapping_ids": list(actual["by_mapping"]),
        "observed": actual,
        "same_pipeline_iid": {
            "replicates": iid_replicates,
            "simulation_method": "independent row-wise uniform permutation, first six",
            "search_family": "4 fixed mappings x 7 fixed prefix lengths x 3 pair/2 graph metrics",
            "pair_max_z_empirical_p": p_pair,
            "graph_control_max_z_empirical_p": p_graph,
            "pair_two_endpoint_bonferroni_p": min(1., 2 * p_pair),
            "graph_two_endpoint_bonferroni_p": min(1., 2 * p_graph),
            "minimum_resolvable_p": 1 / (iid_replicates + 1),
            "pair_null_quantiles_50_90_95_99": np.quantile(
                null_pair, [0.5, 0.9, 0.95, 0.99]).tolist(),
            "graph_null_quantiles_50_90_95_99": np.quantile(
                null_graph, [0.5, 0.9, 0.95, 0.99]).tolist(),
        },
        "limitations": [
            "P2B addresses predeclared pair-distance features; within-L3 variation does not prove predictive value.",
            "Graph edge counts and cycle rank are already determined by L3 topology and are negative-control dimensions.",
            "Same-pipeline null calibrates only the specified P2B family; historical design choices were previously inspected.",
            "The two P2B endpoints use Bonferroni adjustment; P2A and other prior experiments are separate exploratory families.",
            "No exact conditional distribution against Sum/Log/Joint is claimed; that is a distinct later ablation.",
            "Under fair independent draws, historical deficits do not force future correction.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-root", type=Path, default=Path("data/csv"))
    parser.add_argument("--bin-root", type=Path, default=Path("data/bin"))
    parser.add_argument("--iid-reps", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=649)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    path = args.out.resolve()
    if not path.is_relative_to(Path("results").resolve()):
        raise ValueError("P2B reports must be written under results/")
    historical, gate = audit_repository_inputs(args.csv_root, args.bin_root)
    result = analyze_structure(historical, iid_replicates=args.iid_reps, seed=args.seed)
    result["source_gate"] = gate
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="ascii", newline="\n") as f:
        json.dump(result, f, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        f.write("\n")
    iid = result["same_pipeline_iid"]
    print(f"PASS: {len(historical)} verified historical draws; "
          f"{iid['replicates']} same-pipeline IID controls")
    print(f"PAIR: max|Z|={result['observed']['maximum_absolute_pair_z']:.5f}; "
          f"familywise p={iid['pair_max_z_empirical_p']:.5f}; "
          f"2-endpoint adjusted={iid['pair_two_endpoint_bonferroni_p']:.5f}")
    print(f"L3 GRAPH CONTROL: max|Z|={result['observed']['maximum_absolute_graph_control_z']:.5f}; "
          f"familywise p={iid['graph_control_max_z_empirical_p']:.5f}; "
          f"2-endpoint adjusted={iid['graph_two_endpoint_bonferroni_p']:.5f}")
    print("DECISION: EXPLORATORY_DESCRIPTIVE_ONLY_NOT_PREDICTIVE")
    print(f"REPORT: {path}")


if __name__ == "__main__":
    main()
