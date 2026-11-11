"""v0.30 P0 — distribution dynamics and conditional compensation under fair IID.

A shrinking sample-average residual and a historically selectable 'repair'
ticket are not evidence that the next random ticket compensates deficits.

Three fixed v0.29 P3B pair-distance features, EXACT unconditional fair
mean/covariance and its fixed whitening, are reused without re-estimation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .geometry_convergence_v029 import (
    _check_config, _prefixes, _validate_draws, audit_repository_inputs,
)
from .geometry_repair_v029 import exact_structural_covariance, P0_SHA256
from .geometry_structure_v029 import (
    PAIR_FEATURES, _features_and_l3, _mapping_labels,
)
from .geometry_v029 import catalog

SCHEMA = "lotto649.v030.distribution_dynamics.martingale.v1"
DEFAULT_IID_REPLICATES = 199
DEFAULT_SEED = 649
MAX_REPLICATES = 1999


def exact_null_dynamics() -> dict:
    """No history fitting: theory of the same fixed regularized P3B features."""
    original = exact_structural_covariance()
    mean = np.asarray(original["mean"], dtype=float)
    covariance = np.asarray(original["covariance"], dtype=float)
    scale = np.sqrt(np.diag(covariance))
    whitening = np.asarray(original["whitener"], dtype=float)
    correlation = covariance / np.outer(scale, scale)
    # For row-vector z=(phi-mu)/sd @ W, Cov(z)=W^T Corr W.
    transformed = whitening.T @ correlation @ whitening
    transformed = (transformed + transformed.T) / 2
    if not np.all(np.linalg.eigvalsh(transformed) > 0):
        raise AssertionError("fixed whitened-feature fair covariance is not SPD")
    if not np.allclose(transformed, (transformed.T), atol=1e-12):
        raise AssertionError("unexpected transformed covariance asymmetry")
    fair_energy_per_draw = float(np.trace(transformed))
    return {
        "feature_names": list(PAIR_FEATURES),
        "exact_mean": mean.tolist(),
        "exact_covariance": covariance.tolist(),
        "feature_scale": scale.tolist(),
        "fixed_whitener": whitening.tolist(),
        "whitened_fair_covariance": transformed.tolist(),
        "fair_quadratic_energy_increment": fair_energy_per_draw,
        "martingale_definition": "M_n = ||sum(z_1..z_n)||^2 - n*trace(C_z)",
        "compensation_definition": "A_next = -R_n dot z_(n+1); fair conditional mean exactly 0",
        "conditional_normalized_energy_identity": (
            "E[D_(n+1)|history] = (n*D_n + trace(C_z))/(n+1), "
            "with D_n=||R_n||^2/n"
        ),
        "important_nonclaim": (
            "Normalized discrepancy has a conditional tendency toward its "
            "stationary mean trace(C_z); no individual 6/49 draw is forced to "
            "compensate historical deficits."
        ),
    }


def _z(draws: np.ndarray, theory: dict) -> np.ndarray:
    features, _ = _features_and_l3(
        draws, _mapping_labels("sequential", 0)
    )
    mean = np.asarray(theory["exact_mean"], dtype=float)
    sd = np.asarray(theory["feature_scale"], dtype=float)
    whitening = np.asarray(theory["fixed_whitener"], dtype=float)
    return ((features[:, :3] - mean) / sd) @ whitening


def _path_statistics(
    draws: np.ndarray, theory: dict, prefixes: list[int], *, details: bool
) -> dict:
    z = _z(draws, theory)
    n = len(z)
    C = np.asarray(theory["whitened_fair_covariance"], dtype=float)
    fair_trace = float(theory["fair_quadratic_energy_increment"])
    cumulative = np.cumsum(z, axis=0, dtype=float)
    previous = np.concatenate([np.zeros((1, z.shape[1])), cumulative[:-1]], axis=0)
    raw_correction = -np.einsum("ij,ij->i", previous, z)
    variance = np.einsum("ni,ij,nj->n", previous, C, previous)
    valid = variance > 1e-24
    normalized = np.zeros(n, dtype=float)
    np.divide(raw_correction, np.sqrt(np.maximum(variance, 0)),
              out=normalized, where=valid)
    # On draw 1, the past residual is the zero vector and no past-conditioned
    # directional alignment can be defined; it is excluded by valid=False.
    valid_cum = np.cumsum(valid.astype(np.int32), dtype=np.int32)
    aligned_cum = np.cumsum(normalized, dtype=float)
    energy = np.einsum("ij,ij->i", cumulative, cumulative)
    size = np.arange(1, n + 1, dtype=float)
    D = energy / size
    max_alignment = 0.
    max_energy = 0.
    rows: list[dict] = []
    for length in prefixes:
        ix = length - 1
        v = int(valid_cum[ix])
        normalized_alignment_sum = float(aligned_cum[ix] / np.sqrt(v)) if v else 0.
        current_energy = float(D[ix])
        max_alignment = max(max_alignment, abs(normalized_alignment_sum))
        max_energy = max(max_energy, abs(current_energy - fair_trace))
        if details:
            rows.append({
                "draw_count": int(length),
                "valid_past_conditioned_steps": v,
                "sum_of_standardized_alignment_over_sqrt_steps":
                    normalized_alignment_sum,
                "mean_standardized_alignment":
                    float(aligned_cum[ix] / v) if v else None,
                "observed_energy_D_n": current_energy,
                "fair_unconditional_expected_D_n": fair_trace,
                "observed_corrected_energy_martingale_M_n":
                    float(energy[ix] - length * fair_trace),
                "iid_expected_next_raw_squared_energy_increment": fair_trace,
                "iid_expected_next_raw_compensation": 0.0,
                "iid_expected_next_D": (
                    length * current_energy + fair_trace
                ) / (length + 1),
                "iid_expected_next_D_change": (
                    fair_trace - current_energy
                ) / (length + 1),
                "observed_prefix_ending_raw_correction": float(raw_correction[ix]),
            })
    result = {
        "primary_max_absolute_prefix_alignment_z": max_alignment,
        "secondary_max_absolute_prefix_energy_minus_trace": max_energy,
    }
    if details:
        result.update({
            "prefix_diagnostics": rows,
            "observed_past_conditioned_correction_sign_rate": float(
                np.count_nonzero((raw_correction > 0) & valid)
                / max(1, np.count_nonzero(valid))
            ),
            "final_cumulative_residual": cumulative[-1].tolist(),
            "final_feature_energy": float(energy[-1]),
            "theoretical_corrected_energy_martingale_mean": 0.0,
            "full_history_number_of_alignment_steps": int(valid_cum[-1]),
            "empirical_alignment_observation_warning": (
                "An alignment sign-rate > 0.5 is not itself a p-value; "
                "conditional distributions can be asymmetric."
            ),
        })
    return result


def _uniform_history(rng: np.random.Generator, n: int) -> np.ndarray:
    # Each row is an independent uniform random subset of 6 from 49.
    numbers = np.arange(1, 50, dtype=np.int16)
    return rng.permuted(np.broadcast_to(numbers, (n, 49)), axis=1)[:, :6]


def analyze_dynamics(
    numbers: object, *, iid_replicates: int = DEFAULT_IID_REPLICATES,
    seed: int = DEFAULT_SEED
) -> dict:
    _check_config(iid_replicates, seed)
    if iid_replicates > MAX_REPLICATES:
        raise ValueError(f"v0.30 P0 limits iid_replicates to {MAX_REPLICATES}")
    draws = _validate_draws(numbers)
    if catalog()["definition_sha256"] != P0_SHA256:
        raise ValueError("v0.29 frozen P0 geometry unexpectedly changed")
    prefixes = _prefixes(len(draws))
    theory = exact_null_dynamics()
    observed = _path_statistics(draws, theory, prefixes, details=True)
    null_alignment = np.empty(iid_replicates, dtype=float)
    null_energy = np.empty(iid_replicates, dtype=float)
    rng = np.random.default_rng(seed)
    for index in range(iid_replicates):
        outcome = _path_statistics(
            _uniform_history(rng, len(draws)), theory, prefixes, details=False
        )
        null_alignment[index] = outcome["primary_max_absolute_prefix_alignment_z"]
        null_energy[index] = outcome["secondary_max_absolute_prefix_energy_minus_trace"]
    primary = (
        1 + int(np.count_nonzero(
            null_alignment >= observed["primary_max_absolute_prefix_alignment_z"]
        ))
    ) / (iid_replicates + 1)
    secondary = (
        1 + int(np.count_nonzero(
            null_energy >= observed["secondary_max_absolute_prefix_energy_minus_trace"]
        ))
    ) / (iid_replicates + 1)
    return {
        "schema": SCHEMA,
        "research_issue": 32,
        "decision": "MARTINGALE_NULL_BASELINE_NO_PREDICTIVE_GO",
        "draws": len(draws),
        "seed": seed,
        "fixed_mapping": "sequential",
        "exact_null": theory,
        "prefix_lengths": prefixes,
        "observed": observed,
        "matched_entire_search_iid": {
            "replicates": iid_replicates,
            "synthetic_method": "independent per-draw without-replacement uniform six of 49",
            "search_family": "fixed sequential mapping, three frozen P3B features, "
                             "frozen chronological prefixes, two global endpoints",
            "primary_alignment_empirical_p": primary,
            "secondary_energy_empirical_p": secondary,
            "two_endpoint_bonferroni_primary_p": min(1., 2 * primary),
            "two_endpoint_bonferroni_secondary_p": min(1., 2 * secondary),
            "minimum_resolvable_p": 1 / (iid_replicates + 1),
            "primary_null_quantiles_50_90_95_99":
                np.quantile(null_alignment, [0.5, 0.9, 0.95, 0.99]).tolist(),
            "secondary_null_quantiles_50_90_95_99":
                np.quantile(null_energy, [0.5, 0.9, 0.95, 0.99]).tolist(),
        },
        "limits": [
            "Analytic compensation identities assume IID independent fair six-of-49 draws, "
            "not merely historical distribution fit.",
            "P0 uses previously explored v0.29 geometric features; even adjusted "
            "P0 p-values are exploratory across the wider v0.24–v0.30 program.",
            "Secondary energy deviation is globally calibrated by IID whole-history "
            "simulation, not by assuming a Gaussian distribution for D_n.",
            "Positive historical counterfactual repair does not imply the next draw "
            "has negative alignment with historical residual.",
            "No purchase, betting optimization, shadow seal change, M0 edit "
            "or historical source update is performed.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-root", type=Path, default=Path("data/csv"))
    parser.add_argument("--bin-root", type=Path, default=Path("data/bin"))
    parser.add_argument("--iid-reps", type=int, default=DEFAULT_IID_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    if not output.is_relative_to(Path("results").resolve()):
        raise ValueError("v0.30 reports must be written inside the ignored results/ tree")
    draws, provenance = audit_repository_inputs(args.csv_root, args.bin_root)
    report = analyze_dynamics(draws, iid_replicates=args.iid_reps, seed=args.seed)
    report["source_gate"] = provenance
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii", newline="\n") as fp:
        json.dump(report, fp, sort_keys=True, indent=2, allow_nan=False, ensure_ascii=True)
        fp.write("\n")
    control = report["matched_entire_search_iid"]
    print(f"PASS P0: {len(draws)} source-verified draws; "
          f"{control['replicates']} matched IID full paths")
    print(f"ALIGNMENT max|Z|={report['observed']['primary_max_absolute_prefix_alignment_z']:.5f}"
          f" empirical p={control['primary_alignment_empirical_p']:.5f}"
          f" two-endpoint adjusted={control['two_endpoint_bonferroni_primary_p']:.5f}")
    print(f"ENERGY max|D-tr(C)|={report['observed']['secondary_max_absolute_prefix_energy_minus_trace']:.5f}"
          f" empirical p={control['secondary_energy_empirical_p']:.5f}"
          f" two-endpoint adjusted={control['two_endpoint_bonferroni_secondary_p']:.5f}")
    print("DECISION: MARTINGALE_NULL_BASELINE_NO_PREDICTIVE_GO")
    print(f"REPORT: {output}")


if __name__ == "__main__":
    main()
