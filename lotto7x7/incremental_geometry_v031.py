"""v0.31 H2: past-only, regularized M0-derived vs geometric forecast ablation.

All models forecast the same 49 next-draw main-number inclusion marginals.
The M0-derived input compresses frozen L1/L2/L3, Sum, exact integer LogBin,
and their Sum x LogBin interaction. It is NOT the original v0.28 M0 selector.
The geometry input is the frozen v0.29 three pair-distance measurements.

The per-number Brier and Bernoulli log scores are proper for marginal forecasts,
not normalized 13,983,816-ticket joint forecasts or payout predictions.
Retrospective results are exploratory and do not establish predictive skill.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path

import numpy as np

from .distribution_dynamics_v030 import exact_null_dynamics, _uniform_history
from .distribution_prediction_v030 import feature_matrix
from .geometry_convergence_v029 import _validate_draws, audit_repository_inputs
from .geometry_repair_v029 import P0_SHA256
from .geometry_structure_v029 import _mapping_labels
from .geometry_v029 import catalog
from .log_product_v027 import _grid, theoretical_log_moments
from .shape_taxonomy_v015 import classify_numbers

SCHEMA = "lotto649.v031.incremental_geometry.h2.v1"
ARMS = ("iid", "m0_compressed", "geometry", "combined")
CONTRASTS = ("m0_vs_iid", "geometry_vs_iid", "combined_vs_iid", "combined_vs_m0")
FAIR_P = 6.0 / 49.0
ALPHA_DEFAULT = 64.0
DEFAULT_SEED = 649


def categorical_features(draws: np.ndarray, grid: dict, log_theory: dict) -> np.ndarray:
    """Frozen low-dimensional, injective representation of L3 plus Sum/LogBin.

    The nonlinear original M0 categorical *repair objective* is not fitted or
    silently substituted: this is a transparent, fixed linear-model proxy.
    """
    topo = classify_numbers(draws)
    parts = topo["partition_code"].astype(np.int64)
    shifts = np.array([15, 12, 9, 6, 3, 0], dtype=np.int64)
    component_sizes = ((parts[:, None] >> shifts[None, :]) & 7).astype(float)
    main_sum = draws.sum(axis=1).astype(float)
    z_sum = (main_sum - 150.0) / sqrt(1075.0)
    product = np.prod(draws.astype(np.int64), axis=1, dtype=np.int64)
    thresholds = np.asarray(grid["integer_product_thresholds"], dtype=np.int64)
    bins = np.searchsorted(thresholds, product, side="right") - 1
    if np.any(bins < 0) or np.any(bins >= len(thresholds) - 1):
        raise ValueError("an exact ticket product is outside the frozen log grid")
    width = 0.5
    edge0 = float(grid["edge_start"])
    center = (log_theory["mean"] - edge0) / width - 0.5
    log_scale = log_theory["stddev"] / width
    z_log_bin = (bins - center) / log_scale
    shape = np.column_stack((
        (topo["n_components"].astype(float) - 3.0) / 2.0,
        component_sizes / 6.0,
        topo["cardinal_edges"].astype(float) / 5.0,
        topo["diagonal_edges"].astype(float) / 5.0,
    ))
    return np.column_stack((
        np.ones(len(draws)), shape, z_sum, z_log_bin, z_sum * z_log_bin
    ))


def geometry_features(draws: np.ndarray, theory: dict) -> np.ndarray:
    """Reproduce the original frozen v0.29 ridge-whitened three-feature map."""
    m = np.asarray(theory["exact_mean"], dtype=float)
    sd = np.asarray(theory["feature_scale"], dtype=float)
    w = np.asarray(theory["fixed_whitener"], dtype=float)
    return ((feature_matrix(draws, _mapping_labels("sequential", 0)) - m) / sd) @ w


def model_features(draws: np.ndarray, grid: dict, log_theory: dict,
                   geometry_theory: dict) -> dict[str, np.ndarray]:
    m0 = categorical_features(draws, grid, log_theory)
    geo = geometry_features(draws, geometry_theory)
    if not np.all(np.isfinite(m0)) or not np.all(np.isfinite(geo)):
        raise ValueError("invalid fixed-feature transformation")
    return {"m0_compressed": m0,
            "geometry": np.column_stack((np.ones(len(draws)), geo)),
            "combined": np.column_stack((m0, geo))}


def legal_marginal_projection(raw: np.ndarray) -> np.ndarray:
    """Euclidean project onto 49-coordinate capped simplex sum(p)=6.

    This 6-of-49 hypersimplex is the convex hull of legal ticket indicators,
    although an actual full joint ticket law is not estimated by this model.
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 49 or not np.all(np.isfinite(raw)):
        raise ValueError("expected finite (N,49) marginal predictions")
    eps = 1e-7
    lower = np.min(raw - (1.0 - eps), axis=1) - 1.0
    upper = np.max(raw - eps, axis=1) + 1.0
    for _ in range(55):
        mid = (lower + upper) / 2.0
        p = np.clip(raw - mid[:, None], eps, 1.0 - eps)
        too_large = p.sum(axis=1) > 6.0
        lower = np.where(too_large, mid, lower)
        upper = np.where(too_large, upper, mid)
    projected = np.clip(raw - ((lower + upper) / 2.0)[:, None], eps, 1.0 - eps)
    if not np.allclose(projected.sum(axis=1), 6, atol=1e-9, rtol=0):
        raise AssertionError("not a valid six-of-49 inclusion vector")
    return projected


def forecast_marginals(features: np.ndarray, y: np.ndarray,
                       targets: np.ndarray, ridge: float) -> np.ndarray:
    """Train on (features[i-1] -> draw[i]) for i < target, never draw[target]."""
    if features.ndim != 2 or y.shape != (len(features), 49):
        raise ValueError("features/targets disagree")
    if not (np.all(np.isfinite(features)) and np.all(np.isfinite(y))):
        raise ValueError("nonfinite training data")
    if len(targets) < 1 or int(targets.min()) < 2 or int(targets.max()) >= len(y):
        raise ValueError("target indices out of range")
    if type(ridge) is not float or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be a positive finite float")
    x_train = features[:-1]
    y_train = y[1:] - FAIR_P
    # Freeze the same ridge, lag, and training horizon for every model and IID run.
    gram = np.cumsum(np.einsum("ni,nj->nij", x_train, x_train), axis=0)
    cross = np.cumsum(np.einsum("ni,nj->nij", x_train, y_train), axis=0)
    selected = targets - 2
    reg = ridge * np.eye(features.shape[1], dtype=np.float64)[None, :, :]
    coeff = np.linalg.solve(gram[selected] + reg, cross[selected])
    predicted = FAIR_P + np.einsum("nd,ndk->nk", features[targets - 1], coeff)
    return legal_marginal_projection(predicted)


def one_step_scores(draws: np.ndarray, targets: np.ndarray,
                    ridge: float, grid: dict, log_theory: dict,
                    geo_theory: dict, *, detailed: bool) -> dict:
    """Evaluate the same fair main-number marginals and proper scoring on all arms."""
    y = np.zeros((len(draws), 49), dtype=float)
    y[np.arange(len(draws))[:, None], draws - 1] = 1.0
    features = model_features(draws, grid, log_theory, geo_theory)
    forecast = {"iid": np.full((len(targets), 49), FAIR_P)}
    for name, arr in features.items():
        forecast[name] = forecast_marginals(arr, y, targets, ridge)
    actual = y[targets]
    brier = {name: np.mean((p - actual) ** 2, axis=1) for name, p in forecast.items()}
    cross_entropy = {
        name: -np.mean(actual * np.log(p) + (1.0 - actual) * np.log1p(-p), axis=1)
        for name, p in forecast.items()
    }
    avg_brier = {name: float(np.mean(brier[name])) for name in ARMS}
    contrasts = {
        "m0_vs_iid": avg_brier["iid"] - avg_brier["m0_compressed"],
        "geometry_vs_iid": avg_brier["iid"] - avg_brier["geometry"],
        "combined_vs_iid": avg_brier["iid"] - avg_brier["combined"],
        "combined_vs_m0": avg_brier["m0_compressed"] - avg_brier["combined"],
    }
    result = {"mean_brier": avg_brier,
              "mean_bernoulli_log_loss": {
                  name: float(np.mean(cross_entropy[name])) for name in ARMS},
              "brier_improvements": contrasts,
              "feature_dimensions": {name: int(features[name].shape[1]) for name in features},
              "forecasts_sum_to_six": all(np.allclose(p.sum(axis=1), 6., atol=1e-9)
                                          for p in forecast.values())}
    if detailed:
        result["per_target"] = [{
            "target_zero_based": int(t),
            "training_pairs": int(t - 1),
            "brier": {name: float(brier[name][i]) for name in ARMS},
            "bernoulli_log_loss": {
                name: float(cross_entropy[name][i]) for name in ARMS},
        } for i, t in enumerate(targets)]
        result["first_target_prediction_marginal_sum"] = {
            name: float(forecast[name][0].sum()) for name in ARMS}
    return result


def analyze_incremental_geometry(numbers: object, *, start_draw: int = 500,
                                 step: int = 10, iid_reps: int = 19,
                                 seed: int = DEFAULT_SEED,
                                 ridge: float = ALPHA_DEFAULT) -> dict:
    """Matched chronological ablation with full-history IID re-fitting."""
    if type(start_draw) is not int or start_draw < 50:
        raise ValueError("start_draw must be >= 50")
    if type(step) is not int or step < 1 or step > 100:
        raise ValueError("step must be in 1..100")
    if type(iid_reps) is not int or iid_reps < 3 or iid_reps > 1999:
        raise ValueError("iid_reps must be in 3..1999")
    if type(seed) is not int or not 0 <= seed < 2 ** 63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    if type(ridge) not in (float, int) or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be a positive finite number")
    draws = _validate_draws(numbers)
    if start_draw >= len(draws):
        raise ValueError("start_draw must precede final draw")
    if catalog()["definition_sha256"] != P0_SHA256:
        raise ValueError("frozen sequential geometry catalog mismatch")
    expected = ("pair_squared_distance_sum", "pair_midrange_3_to_18",
                "pair_far_25_plus")
    geo_theory = exact_null_dynamics()
    if tuple(geo_theory["feature_names"]) != expected:
        raise ValueError("frozen three-feature order changed")
    grid = _grid(49, 6, "0.5")
    log_theory = theoretical_log_moments()
    starts = np.arange(start_draw, len(draws), step, dtype=np.int32)
    ridge = float(ridge)
    config = {"schema": SCHEMA, "start_draw": start_draw, "step": step,
              "iid_replicates": iid_reps, "ridge_alpha": ridge,
              "seed": seed, "geometry_catalog_sha256": P0_SHA256,
              "mapping": "sequential", "lag_draws": 1,
              "m0_proxy": "fixed compressed L1/L2/L3 + Sum + exact LogBin + Sum*LogBin",
              "geometry": "exact P0 moments and ridge 0.05 whitened 3 pair features",
              "training": "expanding ridge with all available earlier (last draw, next draw) pairs",
              "calibration": "Euclidean capped-simplex projection to [0,1]^49, sum6",
              "forecast_target": "49 next-draw main-number inclusion marginals",
              "primary_endpoint": "mean_Brier(M0_compressed)-mean_Brier(Combined)",
              "secondary_endpoints": list(CONTRASTS[:-1]),
              "full_pipeline_IID": "independently sampled six-subsets, refit all arms at original origins"}
    config_sha = sha256(json.dumps(config,sort_keys=True,separators=(",", ":"))
                        .encode("ascii")).hexdigest()
    observed = one_step_scores(draws, starts, ridge, grid, log_theory,
                               geo_theory, detailed=True)
    null = np.empty((iid_reps, len(CONTRASTS)), dtype=float)
    ss = np.random.SeedSequence(seed).spawn(iid_reps)
    for i, seq in enumerate(ss):
        null_draws = _uniform_history(np.random.default_rng(seq), len(draws))
        result = one_step_scores(null_draws, starts, ridge, grid, log_theory,
                                 geo_theory, detailed=False)
        null[i] = [result["brier_improvements"][name] for name in CONTRASTS]
    obs = np.asarray([observed["brier_improvements"][key] for key in CONTRASTS])
    pvals = {k: (1. + int(np.count_nonzero(null[:, j] >= obs[j]))) / (iid_reps + 1)
             for j, k in enumerate(CONTRASTS)}
    nullmean = null.mean(axis=0)
    nullsd = null.std(axis=0, ddof=1)
    if not np.all(np.isfinite(nullsd)) or np.any(nullsd < 1e-16):
        raise ValueError("IID variability is insufficient for familywise calibration")
    standardized = (obs - nullmean) / nullsd
    nullstat = np.max((null - nullmean) / nullsd, axis=1)
    fam_p = (1. + int(np.count_nonzero(nullstat >= max(standardized)))) / (iid_reps + 1)
    return {"schema": SCHEMA, "research_issue": 31,
            "decision": "EXPLORATORY_ONLY_NO_PROSPECTIVE_PREDICTIVE_GO",
            "configuration": config, "configuration_sha256": config_sha,
            "draws": int(len(draws)), "targets": int(len(starts)),
            "observed": observed,
            "primary": {"name": "combined_vs_m0", "brier_improvement": float(obs[-1]),
                        "IID_one_sided_upper_tail_p": pvals["combined_vs_m0"]},
            "IID_full_pipeline": {
                "replicates": iid_reps,
                "null_contrast_matrix": null.tolist(),
                "contrast_order": list(CONTRASTS),
                "p_values_upper_tail": pvals,
                "null_means": dict(zip(CONTRASTS, map(float, nullmean))),
                "null_standard_deviations": dict(zip(CONTRASTS, map(float, nullsd))),
                "observed_null_centered_standardized": dict(zip(CONTRASTS, map(float, standardized))),
                "familywise_max_Z_upper_tail_p": float(fam_p),
                "minimum_p_resolution": 1. / (iid_reps + 1)},
            "limitations": [
                "Compressed M0-derived feature encoding is not the original M0 portfolio optimizer.",
                "Proper Brier/log score concerns each number's inclusion probability, not a full joint ticket probability.",
                "No pre-draw holdout: existing historical data and feature families were examined previously.",
                "IID Monte Carlo p-value resolution and null-mean estimation are limited by replicate count.",
                "Combined model has more features than each standalone model; ridge is fixed but model capacity differs.",
                "History is not proof of a mean-reverting physical lottery mechanism."],
            "original_seals_read_only": True}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-root", type=Path, default=Path("data/csv"))
    parser.add_argument("--bin-root", type=Path, default=Path("data/bin"))
    parser.add_argument("--start-draw", type=int, default=500)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--iid-reps", type=int, default=19)
    parser.add_argument("--seed", type=int, default=649)
    parser.add_argument("--ridge", type=float, default=ALPHA_DEFAULT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if not out.is_relative_to(Path("results").resolve()):
        raise ValueError("output must be within ignored results/ tree")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite: {out}")
    draws, gate = audit_repository_inputs(args.csv_root, args.bin_root)
    report = analyze_incremental_geometry(draws, start_draw=args.start_draw,
                                          step=args.step, iid_reps=args.iid_reps,
                                          seed=args.seed, ridge=args.ridge)
    report["source_gate"] = gate
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="ascii", newline="\n") as fp:
        json.dump(report, fp, indent=2, sort_keys=True,
                  ensure_ascii=True, allow_nan=False)
        fp.write("\n")
    for arm, val in report["observed"]["mean_brier"].items():
        print(f"{arm.upper()}: mean Brier={val:.9f}")
    print(f"PASS H2: {len(draws)} audited draws; {report['targets']} chronological targets; "
          f"{args.iid_reps} full IID histories")
    print(f"PRIMARY: combined-versus-M0 Brier gain={report['primary']['brier_improvement']:.9f}; "
          f"IID p={report['primary']['IID_one_sided_upper_tail_p']:.5f}")
    print(f"FAMILYWISE: p={report['IID_full_pipeline']['familywise_max_Z_upper_tail_p']:.5f}")
    print(f"DECISION: {report['decision']}")
    print(f"REPORT: {out}")


if __name__ == "__main__":
    main()
