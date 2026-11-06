"""Lotto649 transformed-state and hidden-periodicity study v0.1.0.

Read-only input: data/normalized/lotto649_base.npy
Outputs: data/sequence/report.txt, report.json, lag_scan.csv, recurrences.csv

Each sorted six-number combination is a state. The full-rank representation is
bijective; all other representations are lossy and cannot predict full tickets
just because their symbols exhibit a periodic-looking pattern.

Global permutation test scans every chosen representation and lag with a
single maximum statistic, preserving observed state/feature frequencies.
It tests temporal order only, NOT whether numbers are marginally uniform.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from math import comb, exp
from pathlib import Path

import numpy as np

VERSION = "0.1.0"
SOURCE = Path("data/normalized/lotto649_base.npy")
OUTPUT = Path("data/sequence")
COMBINATIONS = comb(49, 6)


def load_numbers(path: Path) -> tuple[np.ndarray, int, int]:
    draws = np.load(path, allow_pickle=False)
    required = {"date", "draw_id", "numbers"}
    if draws.dtype.names is None or not required.issubset(draws.dtype.names):
        raise ValueError(f"Missing NPY fields {sorted(required)}")
    nums = np.asarray(draws["numbers"], dtype=np.int16)
    if nums.ndim != 2 or nums.shape[1] != 6 or len(nums) < 30:
        raise ValueError("Expected at least 30 draws of shape (N,6)")
    nums = np.sort(nums, axis=1)
    if np.any(nums < 1) or np.any(nums > 49) or np.any(np.diff(nums, axis=1) == 0):
        raise ValueError("Invalid main six numbers")
    dates = np.asarray(draws["date"], dtype=np.int64)
    ids = np.asarray(draws["draw_id"], dtype=np.int64)
    if np.any(np.diff(dates) <= 0) or len(set(map(int, ids))) != len(ids):
        raise ValueError("Dates must be ascending and draw IDs unique")
    return nums, int(dates[0]), int(dates[-1])


def rank_colex(row) -> int:
    """Collision-free combinatorial rank, 0..C(49,6)-1."""
    return sum(comb(int(v) - 1, k) for k, v in enumerate(row, 1))


def representations(nums: np.ndarray) -> dict[str, np.ndarray]:
    # Values returned are class IDs; their numeric distance is NOT meaningful.
    rank = np.asarray([rank_colex(row) for row in nums], dtype=np.int64)
    diffs = np.diff(nums, axis=1)
    gap = [tuple(map(int, row)) for row in diffs]
    # All six numbers are covered by a position-independent residue histogram.
    residue = [tuple(np.bincount(row % 7, minlength=7).tolist()) for row in nums]
    feature_values = {
        "exact_set": rank.tolist(),
        "gap_pattern": gap,
        "sum": nums.sum(axis=1).tolist(),
        "odd_count": (nums % 2).sum(axis=1).tolist(),
        "mod7_histogram": residue,
    }
    out = {}
    for name, values in feature_values.items():
        _, inv = np.unique(np.asarray(values), return_inverse=True, axis=0) if name in ("gap_pattern", "mod7_histogram") else (None, None)
        if inv is None:
            _, inv = np.unique(values, return_inverse=True)
        out[name] = np.asarray(inv, dtype=np.int32)
    if len(np.unique(out["exact_set"])) != len(set(rank.tolist())):
        raise AssertionError("exact_set encoding collision")
    return out


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def recurrence_rows(nums):
    seen = {}
    rows = []
    ranks = [rank_colex(row) for row in nums]
    for i, r in enumerate(ranks):
        if r in seen:
            for previous in seen[r]:
                length = 0
                while i + length < len(ranks) and ranks[previous + length] == ranks[i + length]:
                    length += 1
                rows.append({"first_index_1based": previous + 1,
                             "repeat_index_1based": i + 1,
                             "distance_draws": i - previous,
                             "consecutive_equal_states": length,
                             "numbers": "-".join(f"{n:02d}" for n in nums[i])})
        seen.setdefault(r, []).append(i)
    return rows


def sample_stat(seq, lag, baseline):
    matches = int(np.count_nonzero(seq[:-lag] == seq[lag:]))
    pairs = len(seq) - lag
    expected = pairs * baseline
    # Standardization is only a ranking heuristic; significance is
    # calibrated by the entire permutation max-statistic distribution.
    denom = max(1.0, np.sqrt(expected * max(1 - baseline, 0.0)))
    return matches, expected, (matches - expected) / denom


def scan_all(features, max_lag):
    tables = []
    winner = None
    for name, seq in features.items():
        histogram = np.bincount(seq)
        n = len(seq)
        # Probability of equality for two distinct positions when shuffled.
        equality_rate = float(np.sum(histogram.astype(np.float64) *
                                     (histogram.astype(np.float64) - 1)) / (n * (n - 1)))
        for lag in range(1, max_lag + 1):
            matches, expected, z = sample_stat(seq, lag, equality_rate)
            row = {"representation": name, "lag": lag, "matches": matches,
                   "expected_under_permutation": round(expected, 5),
                   "standardized_deviation": round(z, 6)}
            tables.append(row)
            if winner is None or abs(z) > abs(winner["standardized_deviation"]):
                winner = row
    return tables, winner


def max_stat(features, max_lag):
    best = 0.0
    for seq in features.values():
        counts = np.bincount(seq).astype(np.float64)
        n = len(seq)
        q = float(np.sum(counts * (counts - 1)) / (n * (n - 1)))
        for lag in range(1, max_lag + 1):
            _, _, z = sample_stat(seq, lag, q)
            best = max(best, abs(z))
    return best


def permutation_test(features, max_lag, observed, repeats, seed):
    # Shared row permutation retains synchronous same-draw relationships.
    rng = np.random.default_rng(seed)
    n = len(next(iter(features.values())))
    null = []
    for _ in range(repeats):
        permutation = rng.permutation(n)
        sample = {key: values[permutation] for key, values in features.items()}
        null.append(max_stat(sample, max_lag))
    exceed = sum(score >= observed for score in null)
    return {
        "permutations": repeats, "seed": seed,
        "global_p_value": (exceed + 1) / (repeats + 1),
        "null_max_mean": round(float(np.mean(null)), 6),
        "null_max_p95": round(float(np.quantile(null, 0.95)), 6),
        "method": "shared-row permutation; global maximum across all transforms and lags",
        "limitation": "Only tests observed time ordering. Does not prove fair marginal distribution or hidden determinism."
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-lag", type=int, default=100)
    ap.add_argument("--permutations", type=int, default=199)
    ap.add_argument("--seed", type=int, default=649)
    args = ap.parse_args()
    nums, first, last = load_numbers(SOURCE)
    n = len(nums)
    if not 1 <= args.max_lag <= n // 3:
        ap.error(f"--max-lag must be between 1 and {n // 3}")
    if args.permutations < 19:
        ap.error("--permutations must be at least 19")

    print(f"Lotto649 Transformed Sequence Analyzer v{VERSION}")
    print(f"[PASS] {n:,} base draws | {first}–{last}")
    features = representations(nums)
    recurrence = recurrence_rows(nums)
    # Birthday approximation for repetitions among independent fair combinations.
    expected_collision_pairs = n * (n - 1) / (2 * COMBINATIONS)
    probability_at_least_one_approx = 1 - exp(-expected_collision_pairs)
    print(f"[INFO] exact collision pairs: {len(recurrence)}; fair expected: {expected_collision_pairs:.4f}")

    lag_rows, strongest = scan_all(features, args.max_lag)
    print(f"[PASS] {len(features)} representations x {args.max_lag} lags")
    print(f"[INFO] global permutation test: {args.permutations} repeats")
    calibrated = permutation_test(features, args.max_lag,
                                  abs(strongest["standardized_deviation"]),
                                  args.permutations, args.seed)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "recurrences.csv", recurrence)
    write_csv(OUTPUT / "lag_scan.csv", lag_rows)
    report = {
        "version": VERSION, "draws": n, "first_date": first, "last_date": last,
        "combinations": COMBINATIONS,
        "representations": {
            name: {"distinct_classes": len(np.unique(values)),
                   "lossless": name == "exact_set"} for name, values in features.items()
        },
        "exact_recurrence_pairs": len(recurrence),
        "expected_recurrence_pairs_fair": expected_collision_pairs,
        "probability_at_least_one_collision_fair_approx": probability_at_least_one_approx,
        "longest_observed_exact_run": max((r["consecutive_equal_states"] for r in recurrence), default=0),
        "lag_scan": {"max_lag": args.max_lag, "largest_absolute_standardized_deviation": strongest},
        "global_permutation": calibrated,
        "cautions": [
            "A repeated set does not establish a repeated draw sequence.",
            "An apparent cycle in a lossy feature does not identify the next full set.",
            "A fixed long period beyond the observation window is not identifiable.",
            "Global permutation p-value is exploratory and is not forecasting validation.",
            "No final number selection is performed."
        ]
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        f"Lotto649 Transformed Sequence Analyzer v{VERSION}",
        f"Draws: {n:,} | {first} - {last}",
        f"Complete six-number state space: {COMBINATIONS:,}",
        f"Exact repeated pairs: {len(recurrence)} | fair expected: {expected_collision_pairs:.4f}",
        f"Fair probability of >=1 exact repeat (approx): {probability_at_least_one_approx:.4%}",
        f"Longest observed consecutive exact repeat run: {report['longest_observed_exact_run']}",
        "", "Representations:"
    ]
    for key, value in report["representations"].items():
        lines.append(f"  {key:16s}: {value['distinct_classes']:5d} distinct classes | "
                     f"{'lossless' if value['lossless'] else 'lossy'}")
    lines.extend([
        "", f"Strongest candidate across lags 1..{args.max_lag}: "
        f"{strongest['representation']} lag={strongest['lag']}, "
        f"matches={strongest['matches']}, z={strongest['standardized_deviation']:+.4f}",
        f"Global shared-permutation p-value ({args.permutations}): {calibrated['global_p_value']:.6f}",
        "CAUTION: This explores temporal structure; it does not prove a cycle or predict future full combinations."
    ])
    (OUTPUT / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] {OUTPUT.resolve()}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
