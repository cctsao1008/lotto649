"""v0.30 P2A: chronological distribution-repair ranking versus fair IID.

An observed historical repair score is not a calibrated draw probability.
All feature definitions, mapping seeds and fair moments are frozen in v0.29/P0.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .distribution_dynamics_v030 import exact_null_dynamics, _uniform_history
from .geometry_convergence_v029 import _validate_draws, audit_repository_inputs
from .geometry_repair_v029 import P0_SHA256
from .geometry_structure_v029 import PAIR_FEATURES, _mapping_labels
from .geometry_v029 import catalog

SCHEMA = "lotto649.v030.distribution_prediction.p2a.v1"
DEFAULT_SEED = 649
MAPPINGS = (("sequential", 0), ("snake", 0), ("random", 649), ("random", 650))
ARMS = ("compensation", *PAIR_FEATURES, "reverse_compensation",
        "snake_compensation", "random649_compensation", "random650_compensation")
PAIR_I, PAIR_J = np.triu_indices(6, 1)


def feature_matrix(draws: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute frozen P0 non-L3 pair features without slow L3 classification."""
    values = np.asarray(draws)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("expected (N, 6) tickets")
    mapped = labels[values - 1].astype(np.int16) - 1
    y, x = mapped // 7, mapped % 7
    dy = y[:, PAIR_I] - y[:, PAIR_J]
    dx = x[:, PAIR_I] - x[:, PAIR_J]
    distances = dy.astype(np.int32) ** 2 + dx.astype(np.int32) ** 2
    return np.column_stack((
        distances.sum(axis=1),
        ((distances > 2) & (distances <= 18)).sum(axis=1),
        (distances >= 25).sum(axis=1),
    )).astype(np.float64)


def midrank_percentile(actual: float, reference: np.ndarray) -> float:
    """Midrank against unconditional fair reference tickets; ties count half."""
    values = np.asarray(reference, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() \
            or not np.isfinite(actual):
        raise ValueError("reference scores must be nonempty finite 1D values")
    return float((np.count_nonzero(values < actual)
                  + 0.5 * np.count_nonzero(values == actual)) / len(values))


def _standardized(draws: np.ndarray, labels: np.ndarray,
                  mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (feature_matrix(draws, labels) - mean) / scale


def _score_arms(residuals: dict[str, np.ndarray], tickets: np.ndarray,
                labels: dict[str, np.ndarray], mean: np.ndarray,
                scale: np.ndarray, whitener: np.ndarray) -> dict[str, np.ndarray]:
    """Use a fixed past-only residual to score legal candidates."""
    standard = _standardized(tickets, labels["sequential"], mean, scale)
    residual = residuals["sequential"]
    whitened = standard @ whitener
    primary = -(whitened @ (residual @ whitener))
    scores = {"compensation": primary,
              "reverse_compensation": -primary}
    for i, name in enumerate(PAIR_FEATURES):
        scores[name] = -standard[:, i] * residual[i]
    for mapping_name, arm in (("snake", "snake_compensation"),
                              ("random649", "random649_compensation"),
                              ("random650", "random650_compensation")):
        other = _standardized(tickets, labels[mapping_name], mean, scale)
        scores[arm] = -(other @ whitener) @ (residuals[mapping_name] @ whitener)
    return scores


def _path_rankings(draws: np.ndarray, *, starts: np.ndarray, references: int,
                   rng: np.random.Generator, theory: dict,
                   labels: dict[str, np.ndarray], details: bool) -> dict:
    mean = np.asarray(theory["exact_mean"], dtype=np.float64)
    scale = np.asarray(theory["feature_scale"], dtype=np.float64)
    whitener = np.asarray(theory["fixed_whitener"], dtype=np.float64)
    # History at prediction origin t uses draws[:t], never draws[t].
    running = {name: np.cumsum(
        _standardized(draws, lab, mean, scale), axis=0, dtype=np.float64)
        for name, lab in labels.items()}
    totals = {name: 0.0 for name in ARMS}
    rows = []
    for t in starts:
        past = {name: path[t - 1] for name, path in running.items()}
        reference_tickets = _uniform_history(rng, references)
        tickets = np.concatenate((draws[t:t + 1], reference_tickets), axis=0)
        scores = _score_arms(past, tickets, labels, mean, scale, whitener)
        percentiles = {arm: midrank_percentile(values[0], values[1:])
                       for arm, values in scores.items()}
        for arm, value in percentiles.items():
            totals[arm] += value
        if details:
            rows.append({"target_index_zero_based": int(t),
                         "past_draw_count": int(t),
                         "actual_numbers": [int(x) for x in draws[t]],
                         "actual_compensation_score": float(scores["compensation"][0]),
                         "percentiles": percentiles})
    result = {"mean_percentile": {name: totals[name] / len(starts) for name in ARMS}}
    if details:
        result["chronological_rows"] = rows
    return result


def analyze_prediction(numbers: object, *, start_draw: int = 500,
                       step: int = 5, references: int = 64,
                       iid_reps: int = 39, seed: int = DEFAULT_SEED) -> dict:
    """Run identical ranking and arm searches on observed and IID histories."""
    if type(start_draw) is not int or not 25 <= start_draw:
        raise ValueError("start_draw must be an integer >= 25")
    if type(step) is not int or not 1 <= step <= 100:
        raise ValueError("step must be an integer in 1..100")
    if type(references) is not int or not 8 <= references <= 4096:
        raise ValueError("references must be an integer in 8..4096")
    if type(iid_reps) is not int or not 3 <= iid_reps <= 1999:
        raise ValueError("iid_reps must be an integer in 3..1999")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    draws = _validate_draws(numbers)
    if start_draw >= len(draws):
        raise ValueError("start_draw must precede the last observed draw")
    if catalog()["definition_sha256"] != P0_SHA256:
        raise ValueError("frozen v0.29 geometry definition changed")
    if tuple(PAIR_FEATURES) != ("pair_squared_distance_sum", "pair_midrange_3_to_18",
                                 "pair_far_25_plus"):
        raise ValueError("P0 pair-feature order has changed")
    theory = exact_null_dynamics()
    labels = {method if method != "random" else f"random{mapping_seed}":
              _mapping_labels(method, mapping_seed)
              for method, mapping_seed in MAPPINGS}
    starts = np.arange(start_draw, len(draws), step, dtype=np.int32)
    config = {"schema": SCHEMA, "seed": seed, "start_draw": start_draw,
              "step": step, "references_per_target": references,
              "iid_replicates": iid_reps, "mapping_ids": list(labels),
              "arms": list(ARMS), "geometry_sha256": P0_SHA256,
              "rank_ties": "midrank", "reference_sampling": "unconditional fair 6/49",
              "theory": "v0.30 P0 exact fair moments and v0.29 ridge whitening"}
    config_sha = sha256(json.dumps(config, sort_keys=True, separators=(",", ":"))
                        .encode("ascii")).hexdigest()
    spawned = np.random.SeedSequence(seed).spawn(2 * iid_reps + 1)
    observed = _path_rankings(draws, starts=starts, references=references,
                              rng=np.random.default_rng(spawned[0]),
                              theory=theory, labels=labels, details=True)
    null = np.empty((iid_reps, len(ARMS)), dtype=np.float64)
    for i in range(iid_reps):
        fair_history = _uniform_history(np.random.default_rng(spawned[2*i + 1]),
                                        len(draws))
        null_row = _path_rankings(fair_history, starts=starts, references=references,
                                  rng=np.random.default_rng(spawned[2*i + 2]),
                                  theory=theory, labels=labels, details=False)
        null[i] = [null_row["mean_percentile"][arm] for arm in ARMS]
    observed_effect = np.asarray([observed["mean_percentile"][arm] - 0.5
                                  for arm in ARMS])
    null_effect = null - 0.5
    scale = null_effect.std(axis=0, ddof=1)
    if not (np.isfinite(scale).all() and np.all(scale > 0)):
        raise ValueError("degenerate IID controls: increase repetitions")
    null_max = np.max(np.abs(null_effect / scale), axis=1)
    obs_max = float(np.max(np.abs(observed_effect / scale)))
    primary_p = (1 + int(np.count_nonzero(null_effect[:, 0] >= observed_effect[0]))) \
        / (iid_reps + 1)
    global_p = (1 + int(np.count_nonzero(null_max >= obs_max))) / (iid_reps + 1)
    return {
        "schema": SCHEMA, "research_issue": 32,
        "decision": "EXPLORATORY_ONLY_NO_PROSPECTIVE_PREDICTIVE_GO",
        "data_use": "retrospective_previously_inspected_not_blind",
        "draws": len(draws), "evaluated_next_draws": len(starts),
        "frozen_configuration": config, "configuration_sha256": config_sha,
        "primary": {"arm": "compensation", "fair_iid_expected_percentile": 0.5,
                    "observed_mean_percentile": float(observed_effect[0] + 0.5),
                    "observed_effect_above_fair": float(observed_effect[0]),
                    "one_sided_upper_tail_p": primary_p},
        "all_arms": {arm: {"observed_mean_percentile":
                            float(observed["mean_percentile"][arm]),
                            "observed_minus_fair_half": float(observed_effect[i]),
                            "iid_null_sd": float(scale[i])}
                     for i, arm in enumerate(ARMS)},
        "matched_full_pipeline_iid": {
            "replicates": iid_reps,
            "minimum_resolvable_p": 1 / (iid_reps + 1),
            "primary_null_quantiles_05_50_95": np.quantile(
                null[:, 0], [0.05, 0.5, 0.95]).tolist(),
            "familywise_max_absolute_standardized_deviation": obs_max,
            "familywise_empirical_p": global_p,
            "all_arm_null_mean_percentiles": {arm: null[:, i].tolist()
                                              for i, arm in enumerate(ARMS)},
            "whole_pipeline": "same origins, reference draws, mappings, score arms, and rank ties",
        },
        "per_draw_observed": observed["chronological_rows"],
        "limitations": [
            "Previously inspected historical draws cannot establish independent prospective skill.",
            "Midrank percentiles compare scores, not calibrated per-ticket probabilities.",
            "No future candidate is filtered by its unknown Sum/L3/LogBin; P1 is descriptive only.",
            "Ablations and reverse sign are correlated; the global maximum is IID calibrated.",
            "Small IID pilot repetitions have limited p-value resolution.",
            "No purchased tickets, immutable research seals, source CSV/BIN, or manifest are modified.",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-root", type=Path, default=Path("data/csv"))
    parser.add_argument("--bin-root", type=Path, default=Path("data/bin"))
    parser.add_argument("--start-draw", type=int, default=500)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--references", type=int, default=64)
    parser.add_argument("--iid-reps", type=int, default=39)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.out.resolve()
    if not output.is_relative_to(Path("results").resolve()):
        raise ValueError("P2A reports must be written beneath the ignored results/ tree")
    if output.exists():
        raise FileExistsError(f"P2A output already exists: {output}")
    draws, provenance = audit_repository_inputs(args.csv_root, args.bin_root)
    report = analyze_prediction(draws, start_draw=args.start_draw,
                                step=args.step, references=args.references,
                                iid_reps=args.iid_reps, seed=args.seed)
    report["source_gate"] = provenance
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii", newline="\n") as fp:
        json.dump(report, fp, sort_keys=True, indent=2,
                  allow_nan=False, ensure_ascii=True)
        fp.write("\n")
    print(f"PASS P2A: {len(draws)} audited draws; "
          f"{len(report['per_draw_observed'])} chronological targets; "
          f"{args.iid_reps} full IID paths")
    print(f"PRIMARY: mean rank={report['primary']['observed_mean_percentile']:.6f}, "
          f"one-sided IID p={report['primary']['one_sided_upper_tail_p']:.5f}")
    print(f"FAMILYWISE: p={report['matched_full_pipeline_iid']['familywise_empirical_p']:.5f}")
    print(f"DECISION: {report['decision']}")
    print(f"REPORT: {output}")


if __name__ == "__main__":
    main()
