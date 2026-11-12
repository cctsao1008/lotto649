"""v0.30 P2B: auditable four-ticket coverage versus counterfactual repair.

A portfolio's legal six-hit probability is exactly 4 / C(49, 6) when its
four tickets are distinct; retrospective repair cannot improve that null.
The exact max-match distribution is calculated by a multivariate
hypergeometric generating-function dynamic program, not lottery simulation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
from math import comb
from pathlib import Path

import numpy as np

from .geometry_convergence_v029 import _validate_draws, audit_repository_inputs
from .geometry_repair_v029 import (
    P0_SHA256, _bitmasks, _unique_pool, _uniform_tickets,
    _whitened_features, exact_structural_covariance,
)
from .geometry_v029 import catalog
from .portfolio_prize_v030 import exact_prize_distribution

SCHEMA = "lotto649.v030.portfolio_benchmark.p2b.v1"
ARMS = ("random", "coverage", "repair", "hybrid")
MAX_PASSES = 4


def exact_max_match_distribution(tickets: object, *, universe: int = 49,
                                 draw_size: int = 6) -> dict:
    """Exact P(max matches=k) for a portfolio, independent fair k-of-N draw.

    Partition the ball universe by the set of tickets containing each ball.
    For each group, choose j of m balls with multiplicity C(m,j). The
    coefficient of x^draw_size tracks joint ticket hits exactly, including
    overlap-induced dependencies. No independence of ticket hits is assumed.
    """
    if type(universe) is not int or type(draw_size) is not int or not (
            2 <= draw_size < universe <= 49):
        raise ValueError("invalid finite universe")
    try:
        rows = tuple(tuple(x for x in row) for row in tickets)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid portfolio") from exc
    if len(rows) != 4 or len(set(rows)) != 4:
        raise ValueError("four distinct tickets are required")
    if any(len(row) != draw_size or len(set(row)) != draw_size or
           any(not isinstance(x, (int, np.integer)) or isinstance(x, (bool, np.bool_))
               or not 1 <= x <= universe for x in row)
           for row in rows):
        raise ValueError("tickets must contain distinct in-range ball numbers")
    if len(set(tuple(sorted(map(int, row))) for row in rows)) != 4:
        raise ValueError("ticket order must not create duplicate combinations")

    by_membership = Counter(
        sum((1 << i) for i, ticket in enumerate(rows) if ball in ticket)
        for ball in range(1, universe + 1)
    )
    states = {(0, (0, 0, 0, 0)): 1}
    for mask, size in sorted(by_membership.items()):
        nxt = defaultdict(int)
        for (used, hits), ways in states.items():
            for amount in range(min(size, draw_size - used) + 1):
                joint = tuple(hits[i] + (amount if mask & (1 << i) else 0)
                              for i in range(4))
                nxt[(used + amount, joint)] += ways * comb(size, amount)
        states = nxt
    histogram = [0] * (draw_size + 1)
    total_hits_weighted = 0
    for (used, hits), ways in states.items():
        if used == draw_size:
            histogram[max(hits)] += ways
            total_hits_weighted += ways * sum(hits)
    denominator = comb(universe, draw_size)
    if sum(histogram) != denominator:
        raise AssertionError("exact portfolio partition did not count all draws")
    expected_hits = 4 * draw_size * draw_size / universe
    if not np.isclose(total_hits_weighted / denominator, expected_hits, atol=1e-12):
        raise AssertionError("exact portfolio hit expectation mismatch")
    if histogram[-1] != 4:
        raise AssertionError("four unique tickets must give exactly four jackpot draws")
    return {
        "universe": universe,
        "draw_size": draw_size,
        "exact_total_outcomes": denominator,
        "max_match_exact_counts_0_to_6": histogram,
        "max_match_probabilities_0_to_6": [v / denominator for v in histogram],
        "at_least_probabilities": {
            str(k): sum(histogram[k:]) / denominator
            for k in range(1, draw_size + 1)
        },
        "expected_sum_of_main_ball_matches": expected_hits,
        "jackpot_probability": histogram[-1] / denominator,
        "method": "exact_multivariate_hypergeometric_membership_dp",
    }


def _coverage(indices: np.ndarray, masks: np.ndarray) -> int:
    occupied = 0
    for index in indices:
        occupied |= int(masks[int(index)])
    return occupied.bit_count()


def _before_after(indices: np.ndarray, residual: np.ndarray,
                  features: np.ndarray, n: int) -> tuple[float, float]:
    after = residual + features[indices].sum(axis=0)
    return (float(residual @ residual / n),
            float(after @ after / (n + 4)))


def _random_indices(rng: np.random.Generator, pool_size: int,
                    repetitions: int) -> np.ndarray:
    """Every row is a uniform 4-subset of identical candidate indices."""
    return np.asarray([rng.choice(pool_size, size=4, replace=False)
                       for _ in range(repetitions)], dtype=np.int32)


def _disjoint_start(rng: np.random.Generator, masks: np.ndarray) -> np.ndarray:
    """Construct exactly 24 unique numbers if such a greedy path exists."""
    for _ in range(20):
        chosen = []
        occupied = 0
        for ix in rng.permutation(len(masks)):
            mask = int(masks[int(ix)])
            if not mask & occupied:
                chosen.append(int(ix))
                occupied |= mask
                if len(chosen) == 4:
                    return np.asarray(chosen, dtype=np.int32)
    raise ValueError("candidate pool did not yield four disjoint tickets")


def _local_repair(start: np.ndarray, masks: np.ndarray, features: np.ndarray,
                  residual: np.ndarray, n: int, *, min_unique: int) -> np.ndarray:
    """Deterministic 4-pass finite-pool coordinate descent, no global claim."""
    current = start.astype(np.int32).copy()
    if _coverage(current, masks) < min_unique:
        raise ValueError("repair start violates portfolio coverage constraint")
    _, best = _before_after(current, residual, features, n)
    for _ in range(MAX_PASSES):
        changed = False
        for slot in range(4):
            others = np.delete(current, slot)
            occupied = 0
            for i in others:
                occupied |= int(masks[int(i)])
            allowed = np.flatnonzero([
                ix not in others and (occupied | int(mask)).bit_count() >= min_unique
                for ix, mask in enumerate(masks)
            ])
            if not len(allowed):
                raise AssertionError("current candidate inexplicably became infeasible")
            partial = residual + features[others].sum(axis=0)
            all_after = partial[None, :] + features[allowed]
            scores = np.einsum("ij,ij->i", all_after, all_after) / (n + 4)
            best_candidate = int(np.argmin(scores))
            if float(scores[best_candidate]) < best - 1e-12:
                current[slot] = int(allowed[best_candidate])
                best = float(scores[best_candidate])
                changed = True
        if not changed:
            break
    if len(set(map(int, current))) != 4 or _coverage(current, masks) < min_unique:
        raise AssertionError("local repair lost legal portfolio constraints")
    return current


def _selection(history: np.ndarray, rng: np.random.Generator, *,
               pool_size: int, random_portfolios: int, min_hybrid_unique: int,
               mean: np.ndarray, scale: np.ndarray,
               whitening: np.ndarray, details: bool) -> dict:
    n = len(history)
    residual = _whitened_features(history, mean, scale, whitening).sum(axis=0)
    pool = _unique_pool(rng, pool_size)
    masks = _bitmasks(pool)
    features = _whitened_features(pool, mean, scale, whitening)
    random_sets = _random_indices(rng, pool_size, random_portfolios)
    before = float(residual @ residual / n)
    random_after = residual[None, :] + features[random_sets].sum(axis=1)
    random_distances = np.einsum("ij,ij->i", random_after, random_after) / (n + 4)
    random_coverage = np.asarray([_coverage(p, masks) for p in random_sets])

    coverage_start = _disjoint_start(rng, masks)
    repair_start = random_sets[int(np.argmin(random_distances))]
    repair = _local_repair(repair_start, masks, features, residual, n, min_unique=6)
    hybrid_starts = [coverage_start]
    eligible = np.flatnonzero(random_coverage >= min_hybrid_unique)
    if len(eligible):
        hybrid_starts.append(random_sets[eligible[int(np.argmin(random_distances[eligible]))]])
    hybrid = min(
        (_local_repair(start, masks, features, residual, n,
                       min_unique=min_hybrid_unique) for start in hybrid_starts),
        key=lambda x: _before_after(x, residual, features, n)[1]
    )
    selected = {
        "random": random_sets[0],
        "coverage": coverage_start,
        "repair": repair,
        "hybrid": hybrid,
    }
    summary = {}
    for arm, indices in selected.items():
        if len(set(map(int, indices))) != 4:
            raise AssertionError(f"{arm} produced duplicate tickets")
        unique_count = _coverage(indices, masks)
        if arm == "coverage" and unique_count != 24:
            raise AssertionError("coverage arm failed to attain 24-number union")
        if arm == "hybrid" and unique_count < min_hybrid_unique:
            raise AssertionError("hybrid coverage constraint violated")
        _, after = _before_after(indices, residual, features, n)
        summary[arm] = {
            "unique_numbers": unique_count,
            "duplicate_exposures": 24 - unique_count,
            "counterfactual_energy_after_four": after,
            "counterfactual_energy_repair_gain": before - after,
        }
        if details:
            lines = pool[indices]
            summary[arm]["research_only_tickets"] = [list(map(int, row)) for row in lines]
            summary[arm]["exact_fair_max_main_ball_match"] = exact_max_match_distribution(lines)
            summary[arm]["exact_fair_special_ball_prizes"] = exact_prize_distribution(lines)
    output = {
        "pool_sha256": sha256(pool.astype("<i2", copy=False).tobytes()).hexdigest(),
        "before_energy": before,
        "arms": summary,
        "random_portfolio_controls": {
            "sample_count": random_portfolios,
            "coverage_mean": float(np.mean(random_coverage)),
            "coverage_quantiles_05_50_95": np.quantile(random_coverage, [.05, .5, .95]).tolist(),
            "repair_gain_median": float(before - np.median(random_distances)),
            "repair_gain_quantiles_05_50_95":
                np.quantile(before - random_distances, [.05, .5, .95]).tolist(),
        },
    }
    return output


def analyze_portfolios(numbers: object, *, pool_size: int = 512,
                       random_portfolios: int = 256,
                       min_hybrid_unique: int = 22,
                       iid_reps: int = 19, seed: int = 649) -> dict:
    """Compare 4 fixed arms on one candidate pool and matched IID histories."""
    if type(pool_size) is not int or not 128 <= pool_size <= 4096:
        raise ValueError("pool_size must be an integer in 128..4096")
    if type(random_portfolios) is not int or not 16 <= random_portfolios <= 4096:
        raise ValueError("random_portfolios must be an integer in 16..4096")
    if type(min_hybrid_unique) is not int or not 6 <= min_hybrid_unique <= 24:
        raise ValueError("min_hybrid_unique must be in 6..24")
    if type(iid_reps) is not int or not 3 <= iid_reps <= 199:
        raise ValueError("iid_reps must be an integer in 3..199")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    history = _validate_draws(numbers)
    if catalog()["definition_sha256"] != P0_SHA256:
        raise ValueError("frozen geometry definition changed")
    theory = exact_structural_covariance()
    mean = np.asarray(theory["mean"], dtype=float)
    scale = np.sqrt(np.diag(np.asarray(theory["covariance"], dtype=float)))
    whitening = np.asarray(theory["whitener"], dtype=float)

    config = {"schema": SCHEMA, "pool_size": pool_size,
              "random_portfolios": random_portfolios,
              "min_hybrid_unique": min_hybrid_unique,
              "iid_reps": iid_reps, "seed": seed,
              "mapping": "sequential", "geometry_sha256": P0_SHA256,
              "objective": "minimize ||R + sum(z(ticket))||^2 / (n+4)",
              "coverage_objective": "maximize union size of 4 tickets",
              "hybrid": "minimize counterfactual energy with union size floor",
              "local_search_passes": MAX_PASSES,
              "candidate_pool": "exact finite uniform sample of distinct six-subsets"}
    checksum = sha256(json.dumps(config, sort_keys=True, separators=(",", ":"))
                      .encode("ascii")).hexdigest()
    streams = np.random.SeedSequence(seed).spawn(2 * iid_reps + 1)
    observed = _selection(history, np.random.default_rng(streams[0]),
                          pool_size=pool_size, random_portfolios=random_portfolios,
                          min_hybrid_unique=min_hybrid_unique, mean=mean,
                          scale=scale, whitening=whitening, details=True)
    null = {arm: [] for arm in ARMS}
    for i in range(iid_reps):
        synthetic = _uniform_tickets(np.random.default_rng(streams[2 * i + 1]),
                                     len(history))
        simulated = _selection(
            synthetic, np.random.default_rng(streams[2 * i + 2]),
            pool_size=pool_size, random_portfolios=random_portfolios,
            min_hybrid_unique=min_hybrid_unique, mean=mean, scale=scale,
            whitening=whitening, details=False,
        )
        for arm in ARMS:
            null[arm].append(simulated["arms"][arm]["counterfactual_energy_repair_gain"])
    observed_excess = {
        arm: observed["arms"][arm]["counterfactual_energy_repair_gain"]
             - observed["random_portfolio_controls"]["repair_gain_median"]
        for arm in ARMS
    }
    return {
        "schema": SCHEMA,
        "research_issue": 32,
        "decision": "DESCRIPTIVE_PORTFOLIO_TRADEOFF_ONLY_NO_PREDICTIVE_GO",
        "history_draws": len(history),
        "configuration": config,
        "configuration_sha256": checksum,
        "observed": observed,
        "matched_whole_pipeline_iid": {
            "replicates": iid_reps,
            "minimum_resolvable_p": 1 / (iid_reps + 1),
            "gain_null_by_arm": {
                arm: {
                    "mean": float(np.mean(values)),
                    "quantiles_05_50_95": np.quantile(values, [.05, .5, .95]).tolist(),
                    "observed_gain_upper_tail_p": (
                        1 + sum(x >= observed["arms"][arm]["counterfactual_energy_repair_gain"]
                                for x in values)) / (iid_reps + 1),
                } for arm, values in null.items()
            },
        },
        "observed_excess_gain_over_random_median": observed_excess,
        "limitations": [
            "Four distinct tickets have the same exact six-hit probability, independent of coverage or repair.",
            "Main-ball match tails are calculated exactly, but bonus-ball prize categories and payout are not modeled.",
            "Coverage and repair are different objectives; optimizing either does not prove predictive advantage.",
            "All outcomes and features were examined historically; these are post-selection descriptive results.",
            "Finite random candidate pool and deterministic local search are heuristics, not global optimization certificates.",
            "Whole-pipeline IID calibration applies only to the predeclared retrospective structural gain endpoints.",
            "No bets, purchased tickets, frozen M0, P3f/P5A seals, or historical data are modified.",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-root", type=Path, default=Path("data/csv"))
    parser.add_argument("--bin-root", type=Path, default=Path("data/bin"))
    parser.add_argument("--pool-size", type=int, default=512)
    parser.add_argument("--random-portfolios", type=int, default=256)
    parser.add_argument("--min-hybrid-unique", type=int, default=22)
    parser.add_argument("--iid-reps", type=int, default=19)
    parser.add_argument("--seed", type=int, default=649)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.out.resolve()
    if not output.is_relative_to(Path("results").resolve()):
        raise ValueError("P2B report must be created beneath the ignored results/ directory")
    if output.exists():
        raise FileExistsError(f"P2B report already exists: {output}")
    source, provenance = audit_repository_inputs(args.csv_root, args.bin_root)
    report = analyze_portfolios(
        source, pool_size=args.pool_size, random_portfolios=args.random_portfolios,
        min_hybrid_unique=args.min_hybrid_unique,
        iid_reps=args.iid_reps, seed=args.seed,
    )
    report["source_gate"] = provenance
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii", newline="\n") as fp:
        json.dump(report, fp, indent=2, sort_keys=True,
                  ensure_ascii=True, allow_nan=False)
        fp.write("\n")
    print(f"PASS P2B: {report['history_draws']} audited draws; "
          f"{args.pool_size} candidate tickets; {args.iid_reps} whole IID histories")
    for arm in ARMS:
        row = report["observed"]["arms"][arm]
        tail = row["exact_fair_max_main_ball_match"]["at_least_probabilities"]
        print(f"{arm.upper()}: unique={row['unique_numbers']}, "
              f"repair_gain={row['counterfactual_energy_repair_gain']:.8f}, "
              f"P(max_hits>=3)={tail['3']:.8f}, "
              f"P(any_prize)={row['exact_fair_special_ball_prizes']['probability_any_prize']:.8f}")
    print(f"DECISION: {report['decision']}")
    print(f"REPORT: {output}")


if __name__ == "__main__":
    main()
