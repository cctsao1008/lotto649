"""v0.29 P2A: audit historical geometric-region convergence against fair IID.

No predictive model, purchased ticket, or outcome-based geometric feature tuning.
Both primary and secondary global tests replay the same predeclared search on
independent synthetic six-without-replacement histories.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .geometry_v029 import REGIONS, REGION_SETS, catalog, mapping_positions
from .geometry_theory_v029 import UNIVERSE, occupancy_numerators, pair_covariance

SCHEMA = "lotto649.geometry_convergence.v0.29.p2a.v1"
REGION_NAMES = tuple(REGIONS)
LAYOUTS = (("sequential", 0), ("snake", 0), ("random", 649), ("random", 650))
PREFIX_GRID = (100, 250, 500, 1000, 1500, 2000)
DEFAULT_REPLICATES = 199
P0_DEFINITION_SHA256 = "5aa5f2820c3287fde1d548de8c90a46b1aed95ed0b7b0f8b9a84c2ea39c82f47"


def _check_config(iid_replicates: int, seed: int) -> None:
    if type(iid_replicates) is not int or not 1 <= iid_replicates <= 9999:
        raise ValueError("iid_replicates must be an integer in 1..9999")
    if type(seed) is not int or not 0 <= seed < 2 ** 63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    if catalog()["definition_sha256"] != P0_DEFINITION_SHA256:
        raise ValueError("P0 feature geometry fingerprint differs from frozen P2A protocol")


def _validate_draws(numbers: object) -> np.ndarray:
    draws = np.asarray(numbers)
    if (draws.ndim != 2 or draws.shape[1] != 6 or len(draws) < 25
            or not np.issubdtype(draws.dtype, np.integer)
            or np.issubdtype(draws.dtype, np.bool_)):
        raise ValueError("expected at least 25 integer rows with six main numbers")
    if np.any(draws < 1) or np.any(draws > 49):
        raise ValueError("main numbers must be 1..49")
    ordered = np.sort(draws.astype(np.int16), axis=1)
    if np.any(np.diff(ordered, axis=1) == 0):
        raise ValueError("six main numbers in each draw must be distinct")
    return ordered


def _prefixes(n: int) -> list[int]:
    return sorted({n, *(p for p in PREFIX_GRID if p <= n)})


def _mapping_matrices() -> dict[str, np.ndarray]:
    """A number-to-region membership matrix for each frozen bijective layout."""
    matrices = {}
    for method, seed in LAYOUTS:
        mapping = mapping_positions(method, seed)
        name = method if method != "random" else f"random:{seed}"
        matrices[name] = np.array(
            [[int(cell in REGION_SETS[r]) for r in REGION_NAMES] for cell in mapping],
            dtype=np.uint8,
        )
    return matrices


def _null_parameters() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = np.array([occupancy_numerators(name) for name in REGION_NAMES], dtype=float)
    probabilities /= UNIVERSE
    means = (probabilities * np.arange(7)[None, :]).sum(axis=1)
    variances = np.array([pair_covariance(name, name) for name in REGION_NAMES])
    if np.any(variances <= 0) or not np.isfinite(variances).all():
        raise AssertionError("invalid theoretical null variances")
    return probabilities, means, variances


def _mapped_signature(draws: np.ndarray, incidence: np.ndarray,
                      prefixes: list[int], probabilities: np.ndarray,
                      means: np.ndarray, variances: np.ndarray,
                      *, details: bool) -> dict:
    """Evaluate every region and every prefix, with no post-selection of bins."""
    hit_counts = incidence[draws - 1].sum(axis=1).astype(np.uint8)
    _, regions = hit_counts.shape
    if regions != len(REGION_NAMES) or np.any(hit_counts > 6):
        raise AssertionError("invalid discrete region occupancy")
    cumulative = np.cumsum(hit_counts, axis=0, dtype=np.int64)
    # Histograms explicitly retain the seven K=0..6 categories, even when rare.
    indicators = (hit_counts[:, :, None] == np.arange(7)[None, None, :]).astype(np.int16)
    cumulative_bins = np.cumsum(indicators, axis=0, dtype=np.int64)
    primary_max, secondary_max = 0.0, 0.0
    observations = []
    for prefix in prefixes:
        historical_means = cumulative[prefix - 1] / prefix
        z = np.sqrt(prefix) * (historical_means - means) / np.sqrt(variances)
        actual_counts = cumulative_bins[prefix - 1]
        frequency = actual_counts / prefix
        tv = 0.5 * np.abs(frequency - probabilities).sum(axis=1)
        primary_max = max(primary_max, float(np.abs(z).max()))
        secondary_max = max(secondary_max, float(tv.max()))
        if details:
            observations.append({
                "draws": prefix,
                "regions": {
                    region: {
                        "mean": float(historical_means[k]),
                        "mean_minus_fair": float(historical_means[k] - means[k]),
                        "standardized_mean_residual": float(z[k]),
                        "counts_k0_to_k6": [int(v) for v in actual_counts[k]],
                        "total_variation": float(tv[k]),
                    }
                    for k, region in enumerate(REGION_NAMES)
                },
            })
    return {
        "maximum_absolute_standardized_mean_residual": primary_max,
        "maximum_total_variation": secondary_max,
        "prefixes": observations if details else None,
    }


def _evaluate(draws: np.ndarray, prefixes: list[int], matrices: dict[str, np.ndarray],
              probabilities: np.ndarray, means: np.ndarray, variances: np.ndarray,
              *, details: bool) -> dict:
    by_mapping = {}
    max_z, max_tv = 0.0, 0.0
    for name, incidence in matrices.items():
        result = _mapped_signature(draws, incidence, prefixes, probabilities, means,
                                   variances, details=details)
        max_z = max(max_z, result["maximum_absolute_standardized_mean_residual"])
        max_tv = max(max_tv, result["maximum_total_variation"])
        if details:
            by_mapping[name] = result
    return {"max_abs_z": max_z, "max_tv": max_tv, "by_mapping": by_mapping}


def analyze_convergence(numbers: object, *, iid_replicates: int = DEFAULT_REPLICATES,
                        seed: int = 649) -> dict:
    """Same-pipeline fair-IID calibration over all regions, maps and time cutoffs."""
    _check_config(iid_replicates, seed)
    actual = _validate_draws(numbers)
    n = len(actual)
    prefixes = _prefixes(n)
    matrices = _mapping_matrices()
    probabilities, means, variances = _null_parameters()
    observed = _evaluate(actual, prefixes, matrices, probabilities, means,
                         variances, details=True)
    rng = np.random.default_rng(seed)
    null_max_z, null_max_tv = np.empty(iid_replicates), np.empty(iid_replicates)
    numbers_universe = np.arange(1, 50, dtype=np.int16)
    for replication in range(iid_replicates):
        # Each row is a separate uniformly permuted universe; take six distinct balls.
        simulated = rng.permuted(np.broadcast_to(numbers_universe, (n, 49)), axis=1)[:, :6]
        score = _evaluate(simulated, prefixes, matrices, probabilities, means,
                          variances, details=False)
        null_max_z[replication] = score["max_abs_z"]
        null_max_tv[replication] = score["max_tv"]
    primary_exceed = int(np.count_nonzero(null_max_z >= observed["max_abs_z"]))
    secondary_exceed = int(np.count_nonzero(null_max_tv >= observed["max_tv"]))
    p_primary = (primary_exceed + 1) / (iid_replicates + 1)
    p_secondary = (secondary_exceed + 1) / (iid_replicates + 1)
    return {
        "schema": SCHEMA,
        "decision": "EXPLORATORY_DESCRIPTIVE_ONLY_NOT_PREDICTIVE",
        "geometry_definition_sha256": catalog()["definition_sha256"],
        "number_of_draws": n,
        "fair_universe_size": UNIVERSE,
        "mapping_ids": list(matrices),
        "region_names": list(REGION_NAMES),
        "prefix_lengths": prefixes,
        "null_hypothesis": "independent uniform 6-of-49 tickets, no replacement within draws",
        "exact_region_probability_k0_to_k6": {
            name: [float(x) for x in probabilities[i]] for i, name in enumerate(REGION_NAMES)
        },
        "fair_mean_by_region": dict(zip(REGION_NAMES, map(float, means))),
        "fair_variance_by_region": dict(zip(REGION_NAMES, map(float, variances))),
        "observed": observed,
        "whole_pipeline_iid_control": {
            "replicates": iid_replicates,
            "seed": seed,
            "simulation_method": "numpy.Generator.permuted(range(1,50),axis=1) independently for each draw",
            "search_family": "4 predeclared layouts x 10 regions x all prefix lengths; two endpoints",
            "primary_endpoint": "maximum absolute region-mean z over complete search family",
            "secondary_endpoint": "maximum region-histogram total variation over complete search family",
            "primary_exceedances": primary_exceed,
            "secondary_exceedances": secondary_exceed,
            "primary_empirical_p": p_primary,
            "secondary_empirical_p": p_secondary,
            "primary_bonferroni_two_endpoint_p": min(1.0, 2 * p_primary),
            "secondary_bonferroni_two_endpoint_p": min(1.0, 2 * p_secondary),
            "minimum_resolvable_p": 1 / (iid_replicates + 1),
            "primary_null_quantiles_50_90_95_99": [float(x) for x in np.quantile(null_max_z, [0.5, 0.9, 0.95, 0.99])],
            "secondary_null_quantiles_50_90_95_99": [float(x) for x in np.quantile(null_max_tv, [0.5, 0.9, 0.95, 0.99])],
        },
        "limitations": [
            "Familywise comparisons account only for the frozen features, layouts and prefixes in this protocol.",
            "The geometry features and broader research family were chosen after earlier historical inspection; this is not an untouched prospective test.",
            "Prefix statistics are nested and highly dependent; simulated maxima retain this dependence.",
            "The two reported endpoint p-values are Bonferroni-adjusted; selection across earlier model families remains exploratory.",
            "A historical deficit does not force a future fair IID draw to correct that deficit.",
            "No candidate tickets, betting suggestions, or predictive GO decisions are generated.",
        ],
    }


def audit_repository_inputs(csv_root: Path, bin_root: Path) -> tuple[np.ndarray, dict]:
    """Fail closed against validated manifest, both BINs, flat annual CSVs and dates."""
    from scripts.export_lotto649_csv import generate_yearly_csv, verify_files
    from .temporal_dataset_v024 import load_csv_history

    csv_root, bin_root = Path(csv_root), Path(bin_root)
    expected_csv, annual_counts = generate_yearly_csv(bin_root)
    verify_files(csv_root, expected_csv)
    source = load_csv_history(csv_root, image_bin=bin_root / "lotto649_images_v2.bin")
    manifest_path = bin_root / "lotto649_source_manifest.json"
    manifest_content = manifest_path.read_bytes()
    manifest = json.loads(manifest_content)
    if (len(source.ids) != manifest["total_draws"] or
            int(source.ids[0]) != manifest["first_draw_id"] or
            int(source.ids[-1]) != manifest["last_draw_id"]):
        raise ValueError("history identities do not match source manifest")
    if int(source.dates[-1]) != int(manifest["cutoff"].replace("-", "")):
        raise ValueError("latest date does not match source manifest cutoff")
    fingerprints = {
        "manifest_sha256": sha256(manifest_content).hexdigest(),
        "history_bin_sha256": manifest["files"]["lotto649_history_v1.bin"]["sha256"],
        "image_bin_sha256": manifest["files"]["lotto649_images_v2.bin"]["sha256"],
        "csv_file_sha256": source.sources["files_sha256"],
    }
    return source.numbers, {
        "status": "PASS_REPOSITORY_MANIFEST_CSV_AND_BOTH_BINS",
        "provenance": "third-party historical archive with separately sourced appended draws; not independent official-source certification",
        "draws": len(source.ids),
        "first_draw_id": int(source.ids[0]),
        "last_draw_id": int(source.ids[-1]),
        "first_date": int(source.dates[0]),
        "last_date": int(source.dates[-1]),
        "annual_files": len(annual_counts),
        "fingerprints": fingerprints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-root", type=Path, default=Path("data/csv"))
    parser.add_argument("--bin-root", type=Path, default=Path("data/bin"))
    parser.add_argument("--iid-reps", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=649)
    parser.add_argument("--out", type=Path, required=True,
                        help="New JSON report under results/; existing files are never overwritten")
    args = parser.parse_args()
    output = args.out.resolve()
    if not output.is_relative_to(Path("results").resolve()):
        raise ValueError("P2A reports must be written under the untracked results directory")
    historical, evidence = audit_repository_inputs(args.csv_root, args.bin_root)
    report = analyze_convergence(historical, iid_replicates=args.iid_reps, seed=args.seed)
    report["source_gate"] = evidence
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    p = report["whole_pipeline_iid_control"]
    print(f"PASS: {evidence['draws']} validated historical draws; "
          f"{p['replicates']} matched IID histories; "
          f"primary adjusted p={p['primary_bonferroni_two_endpoint_p']:.6g}; "
          f"secondary adjusted p={p['secondary_bonferroni_two_endpoint_p']:.6g}")
    print(f"REPORT: {output}")
    print("DECISION: EXPLORATORY_DESCRIPTIVE_ONLY_NOT_PREDICTIVE")


if __name__ == "__main__":
    main()
