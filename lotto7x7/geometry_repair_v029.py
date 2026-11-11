"""v0.29 P3B: counterfactual structural repair, with a full matched IID control.

P3B is mathematical portfolio optimization on a finite candidate sample.
The same procedure often repairs random IID histories; its repair gain is
not a forecast or evidence of physical mean reversion.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .geometry_convergence_v029 import _check_config, _validate_draws, audit_repository_inputs
from .geometry_structure_v029 import (
    PAIR_FEATURES, I49, J49, _fair_inclusion, _features_and_l3,
    _mapping_labels, _pair_weights, exact_pair_moments,
)
from .geometry_v029 import POINTS, catalog
from .geometry_redundancy_v029 import _certificate, TICKETS_CERTIFICATE

SCHEMA = "lotto649.geometry_counterfactual.v0.29.p3b.v1"
RIDGE = 0.05
POOL_DEFAULT = 1024
RANDOM_PORTFOLIOS_DEFAULT = 256
IID_REPLICATES_DEFAULT = 99
MAX_EXCHANGE_PASSES = 4
P0_SHA256 = "5aa5f2820c3287fde1d548de8c90a46b1aed95ed0b7b0f8b9a84c2ea39c82f47"


def exact_structural_covariance() -> dict:
    """Compute exact 3x3 covariance from pair-edge overlap identities."""
    xy = np.asarray(POINTS, dtype=np.int16)
    delta = xy[I49] - xy[J49]
    d2 = np.sum(delta.astype(np.int32) ** 2, axis=1)
    weights = _pair_weights(d2)[:, :3]
    incident = np.zeros((49, 3), dtype=float)
    np.add.at(incident, I49, weights)
    np.add.at(incident, J49, weights)
    totals = weights.sum(axis=0)
    equal_edge = weights.T @ weights
    shared_distinct = incident.T @ incident - 2 * equal_edge
    disjoint_distinct = np.outer(totals, totals) - equal_edge - shared_distinct
    p2, p3, p4 = (_fair_inclusion(k) for k in (2, 3, 4))
    mu = p2 * totals
    covariance = (p2 * equal_edge + p3 * shared_distinct + p4 * disjoint_distinct
                  - np.outer(mu, mu))
    covariance = (covariance + covariance.T) / 2
    eig = np.linalg.eigvalsh(covariance)
    if np.any(eig <= 0):
        raise AssertionError("exact structural covariance is not positive definite")
    p2b = exact_pair_moments()
    for index, name in enumerate(PAIR_FEATURES):
        if not np.isclose(p2b[name]["mean"], mu[index], atol=1e-9) or not np.isclose(
            p2b[name]["variance"], covariance[index, index], atol=1e-9
        ):
            raise AssertionError(f"exact covariance does not match P2B: {name}")
    sd = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(sd, sd)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation + RIDGE * np.eye(3))
    if np.any(eigenvalues <= 0):
        raise AssertionError("regularized correlation covariance is not invertible")
    whitening = (eigenvectors * (1 / np.sqrt(eigenvalues))) @ eigenvectors.T
    return {
        "feature_names": list(PAIR_FEATURES), "mean": mu.tolist(),
        "covariance": covariance.tolist(), "correlation": correlation.tolist(),
        "eigenvalues_unregularized": np.linalg.eigvalsh(correlation).tolist(),
        "ridge_added_to_correlation_diagonal": RIDGE,
        "whitener": whitening.tolist(),
        "method": "exact finite-population pair-edge overlap, stable normalized covariance",
    }


def _whitened_features(tickets: np.ndarray, mean: np.ndarray,
                       scale: np.ndarray, whitening: np.ndarray) -> np.ndarray:
    structural, _ = _features_and_l3(tickets, _mapping_labels("sequential", 0))
    return ((structural[:, :3] - mean) / scale) @ whitening


def _uniform_tickets(rng: np.random.Generator, count: int) -> np.ndarray:
    universe = np.arange(1, 50, dtype=np.int16)
    return np.sort(rng.permuted(np.broadcast_to(universe, (count, 49)), axis=1)[:, :6], axis=1)


def _unique_pool(rng: np.random.Generator, size: int) -> np.ndarray:
    """Exact pool size of distinct legal tickets from uniform 6-of-49 draws."""
    rows = []
    seen = set()
    while len(rows) < size:
        block = _uniform_tickets(rng, max(64, size - len(rows)))
        for row in block:
            key = tuple(map(int, row))
            if key not in seen:
                seen.add(key)
                rows.append(key)
                if len(rows) == size:
                    break
    return np.asarray(rows, dtype=np.int16)


def _bitmasks(tickets: np.ndarray) -> np.ndarray:
    return np.asarray([sum(1 << (int(n) - 1) for n in ticket)
                       for ticket in tickets], dtype=np.uint64)


def _random_disjoint_portfolios(masks: np.ndarray, rng: np.random.Generator,
                                count: int) -> np.ndarray:
    """A predeclared randomized greedy baseline using the SAME candidate pool."""
    output = np.empty((count, 4), dtype=np.int32)
    for i in range(count):
        for _ in range(5):
            chosen = []
            occupied = 0
            for index in rng.permutation(len(masks)):
                mask = int(masks[index])
                if mask & occupied == 0:
                    chosen.append(int(index))
                    occupied |= mask
                    if len(chosen) == 4:
                        break
            if len(chosen) == 4:
                output[i] = chosen
                break
        else:
            raise ValueError("candidate pool has too few four-disjoint portfolios")
    return output


def _distances(indices: np.ndarray, residual: np.ndarray,
               whitened: np.ndarray, n: int) -> np.ndarray:
    updated_residual = residual[None, :] + whitened[indices].sum(axis=1)
    return np.einsum("ij,ij->i", updated_residual, updated_residual) / (n + 4)


def _locally_optimize(indices: np.ndarray, masks: np.ndarray,
                      whitened: np.ndarray, residual: np.ndarray,
                      n: int) -> tuple[np.ndarray, float]:
    current = np.asarray(indices, dtype=np.int32).copy()
    current_score = float(_distances(current[None, :], residual, whitened, n)[0])
    for _ in range(MAX_EXCHANGE_PASSES):
        changed = False
        for slot in range(4):
            others = np.delete(current, slot)
            occupied = int(masks[others[0]]) | int(masks[others[1]]) | int(masks[others[2]])
            allowed = np.flatnonzero((masks & np.uint64(occupied)) == 0)
            if not len(allowed):
                raise AssertionError("no feasible swap: existing ticket must be feasible")
            fixed_residual = residual + whitened[others].sum(axis=0)
            totals = fixed_residual[None, :] + whitened[allowed]
            trial = np.einsum("ij,ij->i", totals, totals) / (n + 4)
            ix = int(np.argmin(trial))
            if float(trial[ix]) < current_score - 1e-12:
                current[slot] = allowed[ix]
                current_score = float(trial[ix])
                changed = True
        if not changed:
            break
    if len(set(map(int, current))) != 4:
        raise AssertionError("duplicate optimized portfolio tickets")
    selected = masks[current]
    if any(int(selected[i]) & int(selected[j]) for i in range(4) for j in range(i + 1, 4)):
        raise AssertionError("optimized tickets overlap")
    return current, current_score


def _selection(history: np.ndarray, rng: np.random.Generator, *,
               pool_size: int, random_portfolios: int,
               mean: np.ndarray, scale: np.ndarray, whitening: np.ndarray,
               include_tickets: bool) -> dict:
    n = len(history)
    residual = _whitened_features(history, mean, scale, whitening).sum(axis=0)
    before = float(np.dot(residual, residual) / n)
    pool = _unique_pool(rng, pool_size)
    pool_sha = sha256(pool.astype("<i2", copy=False).tobytes()).hexdigest()
    whitened = _whitened_features(pool, mean, scale, whitening)
    masks = _bitmasks(pool)
    single_distances = np.einsum("ij,ij->i",
                                residual[None, :] + whitened,
                                residual[None, :] + whitened) / (n + 1)
    single_idx = int(np.argmin(single_distances))
    random_sets = _random_disjoint_portfolios(masks, rng, random_portfolios)
    random_scores = _distances(random_sets, residual, whitened, n)
    original_idx = int(np.argmin(random_scores))
    best_idx, best_distance = _locally_optimize(random_sets[original_idx], masks,
                                               whitened, residual, n)
    # Report a fair *mathematical* random baseline with identical pool/constraints.
    median_random_gain = float(before - np.median(random_scores))
    optimized_gain = float(before - best_distance)
    output = {
        "baseline_n_scaled_distance": before,
        "best_single_append_gain": float(before - single_distances[single_idx]),
        "random_portfolio_gain_quantiles_0_25_50_75_100": [
            float(before - v) for v in np.quantile(random_scores, [1, .75, .5, .25, 0])
        ],
        "random_portfolio_gain_median": median_random_gain,
        "optimized_four_append_distance": best_distance,
        "optimized_four_append_gain": optimized_gain,
        "optimized_excess_over_random_median": optimized_gain - median_random_gain,
        "pool_sha256": pool_sha,
        "randomized_greedy_baseline": "first 4 pairwise-disjoint tickets from a random pool permutation; repeated for each baseline portfolio",
        "optimizer": "best randomized baseline + at most 4 deterministic coordinate-exchange passes with disjointness enforced",
        "optimizer_certification": "FINITE_POOL_HEURISTIC_NOT_GLOBAL_OPTIMUM",
    }
    if include_tickets:
        from .geometry_redundancy_v029 import _descriptors
        chosen = pool[best_idx]
        labels, features, _ = _descriptors(chosen if len(chosen) >= 25 else
             np.vstack((chosen, np.tile(chosen[0], (25 - len(chosen), 1)))))
        output["research_only_best_single_candidate"] = [int(v) for v in pool[single_idx]]
        output["research_only_four_ticket_candidate"] = [
            [int(v) for v in ticket] for ticket in chosen
        ]
        output["research_only_portfolio_baseline_classes"] = [
            {"l3": int(labels[i][0]), "sum": int(labels[i][1]),
             "log_bin": int(labels[i][2]),
             "pair_features": dict(zip(PAIR_FEATURES, [int(v) for v in features[i]]))}
            for i in range(4)
        ]
        output["candidate_notice"] = "NOT A PURCHASE RECOMMENDATION, NOT A FROZEN PROSPECTIVE TEST, NO IMPLIED IMPROVEMENT IN WIN CHANCES"
    return output


def analyze_counterfactual(history: object, *,
                           pool_size: int = POOL_DEFAULT,
                           random_portfolios: int = RANDOM_PORTFOLIOS_DEFAULT,
                           iid_replicates: int = IID_REPLICATES_DEFAULT,
                           seed: int = 649) -> dict:
    _check_config(iid_replicates, seed)
    if type(pool_size) is not int or not 128 <= pool_size <= 8192:
        raise ValueError("pool_size must be an integer in 128..8192")
    if type(random_portfolios) is not int or not 16 <= random_portfolios <= 4096:
        raise ValueError("random_portfolios must be an integer in 16..4096")
    if catalog()["definition_sha256"] != P0_SHA256:
        raise ValueError("P0 catalog has changed")
    draws = _validate_draws(history)
    exact = exact_structural_covariance()
    mean = np.asarray(exact["mean"])
    scale = np.sqrt(np.diag(np.asarray(exact["covariance"])))
    whitening = np.asarray(exact["whitener"])
    rng = np.random.default_rng(seed)
    observed = _selection(draws, rng, pool_size=pool_size,
                          random_portfolios=random_portfolios,
                          mean=mean, scale=scale, whitening=whitening,
                          include_tickets=True)
    fair_excess = np.empty(iid_replicates)
    fair_gain = np.empty(iid_replicates)
    for r in range(iid_replicates):
        iid_history = _uniform_tickets(rng, len(draws))
        null = _selection(iid_history, rng, pool_size=pool_size,
                          random_portfolios=random_portfolios,
                          mean=mean, scale=scale, whitening=whitening,
                          include_tickets=False)
        fair_excess[r] = null["optimized_excess_over_random_median"]
        fair_gain[r] = null["optimized_four_append_gain"]
    p_excess = (1 + int(np.count_nonzero(fair_excess >=
                         observed["optimized_excess_over_random_median"]))) / (iid_replicates + 1)
    p_gain = (1 + int(np.count_nonzero(
                     fair_gain >= observed["optimized_four_append_gain"]))) / (iid_replicates + 1)
    certificate = _certificate()
    example = np.asarray(TICKETS_CERTIFICATE, dtype=np.int16)
    example_vals = _whitened_features(example, mean, scale, whitening)
    hist_residual = _whitened_features(draws, mean, scale, whitening).sum(axis=0)
    example_d = np.einsum("ij,ij->i", hist_residual[None, :] + example_vals,
                          hist_residual[None, :] + example_vals) / (len(draws) + 1)
    certificate["descriptive_single_append_gains"] = {
        "ticket_A": observed["baseline_n_scaled_distance"] - float(example_d[0]),
        "ticket_B": observed["baseline_n_scaled_distance"] - float(example_d[1]),
    }
    return {
        "schema": SCHEMA,
        "decision": "DESCRIPTIVE_COUNTERFACTUAL_ONLY_NO_PREDICTIVE_GO",
        "geometry_catalog_sha256": P0_SHA256,
        "history_draws": len(draws),
        "objective": "n * (feature_mean - exact_fair_mean)^T R_reg_inverse (feature_mean - exact_fair_mean); three standardized pair statistics",
        "exact_null": exact,
        "control_design": {
            "seed": seed, "pool_size": pool_size,
            "randomized_greedy_portfolios_per_history": random_portfolios,
            "iid_replicates": iid_replicates,
            "random_histories_with_same_length_as_observed": True,
            "regenerate_pool_and_randomized_portfolios_per_history": True,
            "strictly_four_disjoint_six_number_tickets": True,
            "notes": "Same search and optimization performed for actual and every fair-IID history. Portfolio baselines are randomized greedy, not uniform over all feasible sets.",
        },
        "observed": observed,
        "matched_whole_pipeline_iid": {
            "excess_over_random_median_empirical_upper_tail_p": p_excess,
            "optimized_gain_empirical_upper_tail_p": p_gain,
            "two_endpoint_bonferroni_excess_p": min(1.0, 2 * p_excess),
            "two_endpoint_bonferroni_gain_p": min(1.0, 2 * p_gain),
            "minimum_resolvable_p": 1 / (iid_replicates + 1),
            "null_excess_quantiles_5_50_95": np.quantile(fair_excess, [.05, .5, .95]).tolist(),
            "null_optimized_gain_quantiles_5_50_95": np.quantile(fair_gain, [.05, .5, .95]).tolist(),
        },
        "same_existing_v028_class_different_geometry_certificate": certificate,
        "limitations": [
            "Selecting a portfolio to repair past discrepancies does not alter fair future winning odds.",
            "A highly optimized fair-IID history also yields apparent distribution-repair gains.",
            "This is a finite candidate-pool heuristic, not an exhaustive or certified global optimum.",
            "Unconditional pair features may correlate with Sum, Log, L3, and original number frequencies; no independent predictive weight is justified.",
            "Previous P2A/P2B maximum deviation tests did not reject fair IID; this retrospective design is not prospective evidence.",
            "The added four tickets are hypothetical in a counterfactual distribution, not chronologically drawn new outcomes.",
            "This report does not mutate paid tickets, v0.28 M0 candidates or the P3f preregistered shadow portfolio.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-root", type=Path, default=Path("data/csv"))
    parser.add_argument("--bin-root", type=Path, default=Path("data/bin"))
    parser.add_argument("--pool-size", type=int, default=POOL_DEFAULT)
    parser.add_argument("--random-portfolios", type=int, default=RANDOM_PORTFOLIOS_DEFAULT)
    parser.add_argument("--iid-reps", type=int, default=IID_REPLICATES_DEFAULT)
    parser.add_argument("--seed", type=int, default=649)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    destination = args.out.resolve()
    if not destination.is_relative_to(Path("results").resolve()):
        raise ValueError("P3B reports must be written under the gitignored results/ tree")
    historic, audit = audit_repository_inputs(args.csv_root, args.bin_root)
    report = analyze_counterfactual(
        historic, pool_size=args.pool_size, random_portfolios=args.random_portfolios,
        iid_replicates=args.iid_reps, seed=args.seed,
    )
    report["source_gate"] = audit
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="ascii", newline="\n") as fp:
        json.dump(report, fp, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        fp.write("\n")
    obs, control = report["observed"], report["matched_whole_pipeline_iid"]
    print(f"PASS: {len(historic)} audited draws; "
          f"{args.iid_reps} full IID re-optimizations; exact 3x3 fair covariance")
    print(f"REPAIR: four-ticket gain={obs['optimized_four_append_gain']:.7f}; "
          f"random median gain={obs['random_portfolio_gain_median']:.7f}; "
          f"excess={obs['optimized_excess_over_random_median']:.7f}")
    print(f"IID: excess upper-tail p={control['excess_over_random_median_empirical_upper_tail_p']:.5f}, "
          f"gain upper-tail p={control['optimized_gain_empirical_upper_tail_p']:.5f}")
    print("DECISION: DESCRIPTIVE_COUNTERFACTUAL_ONLY_NO_PREDICTIVE_GO")
    print(f"REPORT: {destination}")


if __name__ == "__main__":
    main()
