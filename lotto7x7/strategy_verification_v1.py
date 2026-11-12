"""Independent read-only audit of v0.30 P2B four-ticket strategy execution.

This module deliberately does not call the P2B candidate generator, optimizer,
feature scorer, or selection procedure.  It independently reconstructs their
versioned contract and then compares the report against its own calculations.
Replaying the algorithm is evidence of deterministic compliance, not proof of
its global optimality, forecast utility, or physical lottery fairness.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import comb
from pathlib import Path
from typing import Any

import numpy as np

from .geometry_convergence_v029 import _validate_draws, audit_repository_inputs
from .geometry_v029 import catalog

VERIFIER_SCHEMA = "lotto649.strategy_verification.v1"
SUPPORTED_SCHEMA = "lotto649.v030.portfolio_benchmark.p2b.v1"
GEOMETRY_SHA256 = "5aa5f2820c3287fde1d548de8c90a46b1aed95ed0b7b0f8b9a84c2ea39c82f47"
ARMS = ("random", "coverage", "repair", "hybrid")
FLOAT_ABS_TOL = 2e-8
SEARCH_TOL = 1e-12


def _canonical_sha(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def _ticket(row: object) -> tuple[int, ...]:
    if (not isinstance(row, list) or len(row) != 6
            or any(type(v) is not int or v < 1 or v > 49 for v in row)
            or row != sorted(set(row))):
        raise ValueError("ticket must be six ascending, distinct integers from 1..49")
    return tuple(row)


def _pool(rng: np.random.Generator, pool_size: int) -> np.ndarray:
    """Independent seed-contract replay (int16 6-of-49 draws; dedup first seen)."""
    universe = np.arange(1, 50, dtype=np.int16)
    found: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    while len(found) != pool_size:
        count = max(64, pool_size - len(found))
        draws = rng.permuted(np.broadcast_to(universe, (count, 49)), axis=1)[:, :6]
        for row in draws:
            key = tuple(sorted(int(v) for v in row))
            if key not in seen:
                found.append(key)
                seen.add(key)
            if len(found) == pool_size:
                break
    return np.asarray(found, dtype=np.int16)


def _masks(pool: np.ndarray) -> list[int]:
    return [sum(1 << (int(x) - 1) for x in line) for line in pool]


def _unique_count(indices: np.ndarray | list[int], masks: list[int]) -> int:
    occupied = 0
    for ix in indices:
        occupied |= masks[int(ix)]
    return occupied.bit_count()


def _theory() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent finite-population pair-covariance calculation.

    This intentionally avoids importing the v0.29/P2B feature scorer, weights,
    covariance routine, and whitening transform; only the frozen mapping and
    its registered SHA are taken from the research catalog.
    """
    points = np.asarray([(i // 7, i % 7) for i in range(49)], dtype=int)
    pairs = np.asarray([(a, b) for a in range(49) for b in range(a + 1, 49)])
    delta = points[pairs[:, 0]] - points[pairs[:, 1]]
    squared = np.sum(delta ** 2, axis=1)
    weights = np.column_stack((squared, (squared > 2) & (squared <= 18),
                               squared >= 25)).astype(float)
    incidence = np.zeros((49, 3), dtype=float)
    for endpoint in (0, 1):
        np.add.at(incidence, pairs[:, endpoint], weights)
    all_weights = weights.sum(axis=0)
    edge_equal = weights.T @ weights
    edge_shared = incidence.T @ incidence - 2 * edge_equal
    edge_disjoint = np.outer(all_weights, all_weights) - edge_equal - edge_shared
    def inc(k: int) -> float:
        return comb(6, k) / comb(49, k)
    mean = inc(2) * all_weights
    cov = (inc(2) * edge_equal + inc(3) * edge_shared
           + inc(4) * edge_disjoint - np.outer(mean, mean))
    cov = (cov + cov.T) / 2
    scale = np.sqrt(np.diag(cov))
    corr = cov / np.outer(scale, scale)
    e, v = np.linalg.eigh(corr + 0.05 * np.eye(3))
    if min(e) <= 0:
        raise AssertionError("invalid independent feature theory")
    whiten = (v * (1 / np.sqrt(e))) @ v.T
    return mean, scale, whiten


def _features(draws: np.ndarray, mean: np.ndarray,
              scale: np.ndarray, whiten: np.ndarray) -> np.ndarray:
    coordinates = np.stack(((draws.astype(np.int32) - 1) // 7,
                            (draws.astype(np.int32) - 1) % 7), axis=-1)
    features = np.zeros((len(draws), 3), dtype=float)
    for a in range(6):
        for b in range(a + 1, 6):
            squared = np.sum((coordinates[:, a] - coordinates[:, b]) ** 2, axis=1)
            features[:, 0] += squared
            features[:, 1] += (squared > 2) & (squared <= 18)
            features[:, 2] += squared >= 25
    return ((features - mean) / scale) @ whiten


def _energy(residual: np.ndarray, chosen: np.ndarray, features: np.ndarray,
            n: int) -> float:
    corrected = residual.copy()
    for ix in chosen:
        corrected += features[int(ix)]
    return float(corrected @ corrected / (n + 4))


def _disjoint(rng: np.random.Generator, masks: list[int]) -> np.ndarray:
    for _ in range(20):
        chosen: list[int] = []
        occupied = 0
        for ix in rng.permutation(len(masks)):
            i = int(ix)
            if not (masks[i] & occupied):
                chosen.append(i)
                occupied |= masks[i]
                if len(chosen) == 4:
                    return np.asarray(chosen, dtype=np.int32)
    raise ValueError("could not build four-disjoint baseline")


def _bounded_swaps(start: np.ndarray, masks: list[int], features: np.ndarray,
                   residual: np.ndarray, n: int, floor: int) -> np.ndarray:
    """Replay exact v0.30 four-pass, best-improvement coordinate-search rule."""
    chosen = start.copy()
    current = _energy(residual, chosen, features, n)
    for _ in range(4):
        changed = False
        for slot in range(4):
            other = [int(chosen[i]) for i in range(4) if i != slot]
            occupied = masks[other[0]] | masks[other[1]] | masks[other[2]]
            allowed = [i for i, bit in enumerate(masks)
                       if i not in other and (occupied | bit).bit_count() >= floor]
            if not allowed:
                raise AssertionError("no feasible candidate for current slot")
            best_index = min(allowed, key=lambda ix: _energy(
                residual, np.asarray(other + [ix]), features, n))
            best_value = _energy(residual, np.asarray(other + [best_index]), features, n)
            if best_value < current - SEARCH_TOL:
                chosen[slot] = best_index
                current = best_value
                changed = True
        if not changed:
            break
    return chosen


def _best_one_swap(indices: np.ndarray, masks: list[int], features: np.ndarray,
                   residual: np.ndarray, n: int, floor: int) -> dict[str, Any]:
    original = _energy(residual, indices, features, n)
    best = original
    best_swap: dict[str, int] | None = None
    for slot in range(4):
        original_ix = int(indices[slot])
        others = [int(indices[j]) for j in range(4) if j != slot]
        union = masks[others[0]] | masks[others[1]] | masks[others[2]]
        for ix in range(len(masks)):
            if ix in others or (union | masks[ix]).bit_count() < floor:
                continue
            score = _energy(residual, np.asarray(others + [ix]), features, n)
            if score < best - SEARCH_TOL:
                best = score
                best_swap = {"slot": slot, "from_pool_index": original_ix,
                             "to_pool_index": ix}
    return {"one_swap_improvement": float(max(0., original - best)),
            "best_one_swap": best_swap,
            "status": ("ONE_SWAP_STATIONARY" if best_swap is None
                       else "IMPROVABLE_WITHIN_FIXED_POOL"),
            "scope": "only the 4*pool_size one-slot neighborhood; not a global certificate"}


def verify_selection(report: dict, history: object,
                     *, source_gate: dict | None = None) -> dict:
    """Independently audit a P2B report; fail closed on missing/altered claims.

    This is a read-only strategy-compliance check. The bounded optimizer is
    re-executed on an independent score implementation, not self-validated.
    """
    checks: dict[str, bool] = {}
    failures: list[str] = []
    notes: list[str] = []
    arm_checks: dict[str, Any] = {}

    def test(key: str, condition: bool, detail: str) -> None:
        checks[key] = bool(condition)
        if not condition:
            failures.append(f"{key}: {detail}")

    try:
        if not isinstance(report, dict):
            raise ValueError("report must be an object")
        config = report["configuration"]
        if not isinstance(config, dict):
            raise ValueError("configuration must be a JSON object")
        test("schema", report.get("schema") == SUPPORTED_SCHEMA and
             config.get("schema") == SUPPORTED_SCHEMA, "unsupported report schema")
        test("config_digest", report.get("configuration_sha256") == _canonical_sha(config),
             "configuration hash mismatch")
        frozen = {"objective": "minimize ||R + sum(z(ticket))||^2 / (n+4)",
                  "coverage_objective": "maximize union size of 4 tickets",
                  "hybrid": "minimize counterfactual energy with union size floor",
                  "mapping": "sequential", "geometry_sha256": GEOMETRY_SHA256,
                  "candidate_pool": "exact finite uniform sample of distinct six-subsets",
                  "local_search_passes": 4}
        test("frozen_strategy", all(config.get(k) == v for k, v in frozen.items()),
             "strategy or geometry definition changed")
        test("geometry_catalog", catalog()["definition_sha256"] == GEOMETRY_SHA256,
             "local geometry catalog differs from required definition")
        n_pool, n_random, floor, n_reps, seed = (
            config[k] for k in ("pool_size", "random_portfolios", "min_hybrid_unique",
                                "iid_reps", "seed"))
        if (any(type(v) is not int for v in (n_pool, n_random, floor, n_reps, seed))
            or not (128 <= n_pool <= 4096 and 16 <= n_random <= 4096
                    and 6 <= floor <= 24 and 3 <= n_reps <= 199
                    and 0 <= seed < 2**63)):
            raise ValueError("unacceptable frozen selection parameters")
        draws = _validate_draws(history)
        n = len(draws)
        test("history_length", report.get("history_draws") == n,
             "research report and history sample sizes differ")
        if source_gate is not None:
            test("source_gate", report.get("source_gate") == source_gate,
                 "manifest/CSV/BIN/source provenance did not match report")
        else:
            notes.append("SOURCE_FINGERPRINT_NOT_VERIFIED: supply audited source_gate for full PASS")
            test("source_gate", False, "audited provenance not supplied")
        rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2 * n_reps + 1)[0])
        pool = _pool(rng, n_pool)
        pool_hash = sha256(pool.astype("<i2", copy=False).tobytes()).hexdigest()
        test("pool_digest", report["observed"].get("pool_sha256") == pool_hash,
             "candidate-pool seed, ordering, or SHA mismatch")
        random_sets = np.asarray([rng.choice(n_pool, size=4, replace=False)
                                  for _ in range(n_random)], dtype=np.int32)
        coverage = _disjoint(rng, _masks(pool))
        mask = _masks(pool)
        mu, scale, w = _theory()
        residual = _features(draws, mu, scale, w).sum(axis=0)
        features = _features(pool, mu, scale, w)
        before = float(residual @ residual / n)
        test("baseline_energy", np.isclose(before, report["observed"]["before_energy"],
             rtol=0., atol=FLOAT_ABS_TOL), "historical baseline energy wrong")
        best_baseline = min(random_sets,
                            key=lambda ix: _energy(residual, ix, features, n))
        repair = _bounded_swaps(best_baseline, mask, features, residual, n, 6)
        valid_seeds = [coverage]
        eligible = [ix for ix in random_sets if _unique_count(ix, mask) >= floor]
        if eligible:
            valid_seeds.append(min(eligible,
                                   key=lambda ix: _energy(residual, ix, features, n)))
        hybrid = min((_bounded_swaps(seed_start, mask, features, residual, n, floor)
                      for seed_start in valid_seeds),
                     key=lambda ix: _energy(residual, ix, features, n))
        selections = {"random": random_sets[0], "coverage": coverage,
                      "repair": repair, "hybrid": hybrid}
        for arm in ARMS:
            selected = selections[arm]
            expected_rows = [tuple(int(v) for v in pool[int(ix)]) for ix in selected]
            row = report["observed"]["arms"][arm]
            actual_rows = [_ticket(line) for line in row["research_only_tickets"]]
            valid = len(actual_rows) == 4 and len(set(actual_rows)) == 4
            test(f"{arm}.tickets_valid", valid, "four distinct sorted six-number tickets required")
            test(f"{arm}.selection_trace", actual_rows == expected_rows,
                 "selected tickets differ from reproducible frozen selection procedure")
            coverage_count = _unique_count(selected, mask)
            test(f"{arm}.coverage", row["unique_numbers"] == coverage_count and
                 row["duplicate_exposures"] == 24 - coverage_count and
                 (arm != "coverage" or coverage_count == 24) and
                 (arm != "hybrid" or coverage_count >= floor),
                 "reported coverage/strategy floor disagrees with independent count")
            after = _energy(residual, selected, features, n)
            gain = before - after
            test(f"{arm}.objective", np.isclose(after, row["counterfactual_energy_after_four"],
                rtol=0., atol=FLOAT_ABS_TOL) and np.isclose(
                gain, row["counterfactual_energy_repair_gain"],
                rtol=0., atol=FLOAT_ABS_TOL), "reported joint repair energy/gain is inconsistent")
            # Do not assert global optimality for bounded four-pass coordinate search.
            arm_checks[arm] = {"tickets": [list(x) for x in actual_rows],
                "unique_numbers": coverage_count,
                "independent_energy_after_four": after,
                "independent_joint_gain": gain,
                "quality": (_best_one_swap(selected, mask, features, residual, n,
                                           floor if arm == "hybrid" else 6)
                            if arm in ("repair", "hybrid") else
                            {"status": "NOT_OPTIMIZED_FOR_REPAIR",
                             "scope": "random or coverage-only selection"})}
        control = report["observed"]["random_portfolio_controls"]
        random_coverage = np.asarray([_unique_count(ix, mask) for ix in random_sets],
                                     dtype=np.float64)
        random_energies = np.asarray([_energy(residual, ix, features, n)
                                      for ix in random_sets], dtype=np.float64)
        random_gain = before - random_energies
        def same_array(expected: object, actual: np.ndarray) -> bool:
            try:
                return np.allclose(np.asarray(expected, dtype=float), actual,
                                   rtol=0., atol=FLOAT_ABS_TOL)
            except (TypeError, ValueError):
                return False
        test("random_controls", control["sample_count"] == n_random
             and np.isclose(control["coverage_mean"], float(random_coverage.mean()),
                            rtol=0., atol=FLOAT_ABS_TOL)
             and same_array(control["coverage_quantiles_05_50_95"],
                            np.quantile(random_coverage, [.05, .5, .95]))
             and np.isclose(control["repair_gain_median"],
                            float(before - np.median(random_energies)),
                            rtol=0., atol=FLOAT_ABS_TOL)
             and same_array(control["repair_gain_quantiles_05_50_95"],
                            np.quantile(random_gain, [.05, .5, .95])),
             "independent random-control budget, coverage or score differs")
    except (KeyError, TypeError, ValueError, IndexError, AssertionError,
            OverflowError, FloatingPointError) as exc:
        test("report_structure", False, f"could not fully validate report: {type(exc).__name__}: {exc}")

    return {"schema": VERIFIER_SCHEMA,
            "decision": "STRATEGY_COMPLIANCE_PASS" if all(checks.values()) and checks else "STRATEGY_COMPLIANCE_FAIL",
            "scope": "v0.30 P2B report; four arms and frozen candidate-pool optimization only",
            "checks": checks,
            "failures": failures,
            "per_arm": arm_checks,
            "limitations": [
                "This verifies compliance and reproducible finite-pool execution, not global optimum.",
                "A one-swap opportunity indicates search-quality headroom, not necessarily a contract failure.",
                "The P2B probability/prize subreports are not independently re-derived here; their separate exact-DP regressions remain necessary.",
                "Report SHA hashes are integrity checks, not pre-draw trusted timestamp attestations.",
                "No next-draw forecast advantage, enhanced winning chance, or purchase recommendation is inferred.",
            ] + notes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--csv-root", type=Path, default=Path("data/csv"))
    parser.add_argument("--bin-root", type=Path, default=Path("data/bin"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    dest = args.out.resolve()
    if not dest.is_relative_to(Path("results").resolve()):
        parser.error("verification reports must be created under ignored results/")
    if dest.exists():
        parser.error("verification report output already exists; overwrite prohibited")
    selection = json.loads(args.selection_report.read_text("utf-8"))
    draws, provenance = audit_repository_inputs(args.csv_root, args.bin_root)
    verification = verify_selection(selection, draws, source_gate=provenance)
    verification["selection_report_sha256"] = sha256(args.selection_report.read_bytes()).hexdigest()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x", encoding="ascii", newline="\n") as fp:
        json.dump(verification, fp, sort_keys=True, indent=2, allow_nan=False)
        fp.write("\n")
    print(f"{verification['decision']}: {len(verification['checks'])} checks; "
          f"{len(verification['failures'])} failures")
    for arm, row in verification["per_arm"].items():
        print(f"{arm.upper()}: coverage={row['unique_numbers']}, "
              f"gain={row['independent_joint_gain']:.8f}, "
              f"quality={row['quality']['status']}")
    print(f"REPORT: {dest}")
    return 0 if verification["decision"] == "STRATEGY_COMPLIANCE_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())