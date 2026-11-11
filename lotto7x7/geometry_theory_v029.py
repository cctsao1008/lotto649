"""v0.29 P1 exact geometric nulls and independent uniform six-of-49 simulation.

All combinations are unordered six-number subsets sampled without replacement.
P1 computes theory before looking at any historical outcomes.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import combinations
import json
from math import comb, sqrt
from pathlib import Path
from random import Random

from .geometry_v029 import REGIONS, REGION_SETS, catalog

UNIVERSE = comb(49, 6)
SCHEMA = "lotto649.geometry_exact_null.v0.29.p1.v1"
MC_SCHEMA = "lotto649.geometry_mc_check.v0.29.p1.v1"
PREDECLARED_PAIR = ("disk_r3", "border")


def _choose(n: int, k: int) -> int:
    return comb(n, k) if 0 <= k <= n else 0


def occupancy_numerators(region: str) -> list[int]:
    """Exact counts for k=0..6 occupied lattice cells inside a fixed region."""
    if region not in REGION_SETS:
        raise ValueError("unknown fixed geometric region")
    m = len(REGION_SETS[region])
    counts = [_choose(m, k) * _choose(49 - m, 6 - k) for k in range(7)]
    if sum(counts) != UNIVERSE:
        raise AssertionError("single-region fair mass not normalized")
    return counts


def pair_joint_numerators(first: str, second: str) -> list[list[int]]:
    """Exact P(K_A=i,K_B=j) numerator from a four-cell-set partition."""
    if first not in REGION_SETS or second not in REGION_SETS:
        raise ValueError("unknown fixed geometric region")
    a_set, b_set = REGION_SETS[first], REGION_SETS[second]
    both = len(a_set & b_set)
    only_a = len(a_set - b_set)
    only_b = len(b_set - a_set)
    neither = 49 - len(a_set | b_set)
    out = [[0] * 7 for _ in range(7)]
    for i in range(7):
        for j in range(7):
            numerator = 0
            for both_count in range(7):
                numerator += (
                    _choose(both, both_count)
                    * _choose(only_a, i - both_count)
                    * _choose(only_b, j - both_count)
                    * _choose(neither, 6 - i - j + both_count)
                )
            out[i][j] = numerator
    if sum(map(sum, out)) != UNIVERSE:
        raise AssertionError("joint fair mass not normalized")
    if [sum(row) for row in out] != occupancy_numerators(first):
        raise AssertionError("joint distribution's A marginal disagrees with exact theory")
    if [sum(out[i][j] for i in range(7)) for j in range(7)] != occupancy_numerators(second):
        raise AssertionError("joint distribution's B marginal disagrees with exact theory")
    return out


def pair_covariance(first: str, second: str) -> float:
    """Covariance under a uniform six-without-replacement draw."""
    if first not in REGION_SETS or second not in REGION_SETS:
        raise ValueError("unknown fixed geometric region")
    a = len(REGION_SETS[first])
    b = len(REGION_SETS[second])
    overlap = len(REGION_SETS[first] & REGION_SETS[second])
    p1 = 6 / 49
    p2 = (6 * 5) / (49 * 48)
    return overlap * p1 + (a * b - overlap) * p2 - (6 * a / 49) * (6 * b / 49)


def exact_reference() -> dict:
    """Stable exact probabilities; no training-data or outcome dependency."""
    region_names = list(REGIONS)
    exact = {}
    for name in region_names:
        count = len(REGION_SETS[name])
        exact[name] = {
            "lattice_cells": count,
            "numerators_k0_to_k6": occupancy_numerators(name),
            "denominator": UNIVERSE,
            "mean": 6 * count / 49,
            "variance": pair_covariance(name, name),
        }
    pairs = []
    for first, second in combinations(region_names, 2):
        pairs.append({
            "first": first, "second": second,
            "overlap_cells": len(REGION_SETS[first] & REGION_SETS[second]),
            "numerators_k_first_by_k_second": pair_joint_numerators(first, second),
            "denominator": UNIVERSE,
            "covariance": pair_covariance(first, second),
        })
    return {
        "schema": SCHEMA,
        "geometry_definition_sha256": catalog()["definition_sha256"],
        "sampling": "uniform six distinct cells without replacement",
        "universe_combinations": UNIVERSE,
        "region_count": len(region_names),
        "pair_count": len(pairs),
        "regions": exact,
        "pairs": pairs,
        "limitations": [
            "Exact finite-lattice probabilities, not continuous geometric area ratios.",
            "Region and pair occupancies are correlated, not independent forecasts.",
            "The fair geometric null is invariant to a fixed bijective number-to-cell mapping.",
            "Historical distribution deficits do not imply future catch-up behavior.",
        ],
    }


def monte_carlo_check(trials: int = 20000, seed: int = 649) -> dict:
    """Independent seeded fair drawings to cross-check exact P1 references."""
    if type(trials) is not int or not 1 <= trials <= 1000000:
        raise ValueError("trials must be an integer between 1 and 1000000")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    names = list(REGIONS)
    region_sets = [REGION_SETS[name] for name in names]
    counts = [[0] * 7 for _ in names]
    means = [0] * len(names)
    pair_counts = [[0] * 7 for _ in range(7)]
    pair_a = names.index(PREDECLARED_PAIR[0])
    pair_b = names.index(PREDECLARED_PAIR[1])
    rng = Random(seed)
    # Uniform subsets are sampled without replacement within each draw.
    for _ in range(trials):
        selected = {divmod(i, 7) for i in rng.sample(range(49), 6)}
        hits = [len(selected & mask) for mask in region_sets]
        for index, hit in enumerate(hits):
            counts[index][hit] += 1
            means[index] += hit
        pair_counts[hits[pair_a]][hits[pair_b]] += 1
    max_abs_z = 0.0
    worst = None
    for index, name in enumerate(names):
        probs = [n / UNIVERSE for n in occupancy_numerators(name)]
        for k, p in enumerate(probs):
            error = counts[index][k] / trials - p
            se = sqrt(p * (1 - p) / trials)
            z = abs(error) / se if se > 0 else 0.0
            if se == 0 and counts[index][k] != trials * p:
                raise AssertionError("impossible Monte Carlo event")
            if z > max_abs_z:
                max_abs_z = z
                worst = [name, k]
    expected_pair = pair_joint_numerators(*PREDECLARED_PAIR)
    if sum(map(sum, pair_counts)) != trials:
        raise AssertionError("Monte Carlo pair normalization failed")
    return {
        "schema": MC_SCHEMA,
        "geometry_definition_sha256": catalog()["definition_sha256"],
        "trials": trials,
        "seed": seed,
        "sampling": "random.Random(seed).sample(range(49),6); independent draws",
        "region_counts_k0_to_k6": dict(zip(names, counts)),
        "region_empirical_means": dict(zip(names, [s / trials for s in means])),
        "maximum_absolute_single_bin_z": max_abs_z,
        "worst_single_bin": worst,
        "predeclared_pair": {
            "regions": list(PREDECLARED_PAIR),
            "empirical_joint_counts": pair_counts,
            "exact_joint_numerators": expected_pair,
            "exact_denominator": UNIVERSE,
        },
        "caution": "MC diagnostic z is exploratory; the maximum across many dependent bins is not a calibrated discovery p-value.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=649)
    parser.add_argument("--theory-only", action="store_true")
    parser.add_argument("--out", type=Path, help="Create one new JSON report without overwriting")
    args = parser.parse_args()
    report = {
        "theory": exact_reference(),
        "simulation": None if args.theory_only else monte_carlo_check(args.trials, args.seed),
    }
    encoded = json.dumps(report, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    if args.out is None:
        # The complete joint table is large; standard output is a short summary.
        print(json.dumps({
            "theory_schema": report["theory"]["schema"],
            "geometry_definition_sha256": report["theory"]["geometry_definition_sha256"],
            "universe_combinations": UNIVERSE,
            "region_count": report["theory"]["region_count"],
            "pair_count": report["theory"]["pair_count"],
            "disk_r3_numerators": occupancy_numerators("disk_r3"),
            "disk_r3_probability_at_least_4": sum(occupancy_numerators("disk_r3")[4:]) / UNIVERSE,
            "mc_trials": None if args.theory_only else args.trials,
            "mc_max_abs_z": None if args.theory_only else report["simulation"]["maximum_absolute_single_bin_z"],
        }, indent=2))
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="ascii", newline="\n") as handle:
            handle.write(encoded)
        print(f"PASS: wrote P1 exact-null and MC report to {args.out}")


if __name__ == "__main__":
    main()
