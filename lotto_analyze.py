"""Lotto649 Statistical Analyzer v0.1.0.

Analyze independent base draws only. Bonus combinations are descriptive metadata,
NOT additional independent draws. No network calls or changes to source files.
Run from E:\\lotto649: python lotto_analyze.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, chi2, hypergeom

VERSION = "0.1.0"
BASE_PATH = Path("data/normalized/lotto649_base.npy")
BONUS_PATH = Path("data/normalized/lotto649_bonus.npy")
ACTIVITIES_PATH = Path("data/normalized/activities.json")
OUTPUT = Path("data/analysis")
P = 6 / 49


def validate_base(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    required = {"draw_id", "date", "numbers", "special"}
    if draws.dtype.names is None or not required.issubset(draws.dtype.names):
        raise ValueError(f"Expected structured .npy fields {sorted(required)}, found {draws.dtype.names}")
    if len(draws) < 2:
        raise ValueError("At least two base draws required")
    nums = np.asarray(draws["numbers"], dtype=np.int16)
    special = np.asarray(draws["special"], dtype=np.int16)
    if nums.shape != (len(draws), 6):
        raise ValueError(f"Expected (N, 6) numbers, got {nums.shape}")
    if np.any((nums < 1) | (nums > 49)) or np.any((special < 1) | (special > 49)):
        raise ValueError("Number outside 1..49")
    if np.any(np.diff(np.sort(nums, axis=1), axis=1) == 0):
        raise ValueError("Duplicate main numbers within a draw")
    if np.any(np.any(nums == special[:, None], axis=1)):
        raise ValueError("Special number overlaps the main six")
    dates = draws["date"].astype(np.int64)
    ids = draws["draw_id"].astype(np.int64)
    if len(set(ids.tolist())) != len(ids) or len(set(dates.tolist())) != len(dates):
        raise ValueError("Duplicate draw ID or date")
    if np.any(np.diff(dates) <= 0):
        raise ValueError("Draws must be strictly ordered by date")
    for value in dates:
        datetime.strptime(str(int(value)), "%Y%m%d")
    return nums, special


def write_csv(path: Path, fields: list[str], rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze_frequency(nums: np.ndarray):
    n = len(nums)
    counts = np.bincount(nums.ravel(), minlength=50)[1:50].astype(int)
    expected = n * P
    standard_deviation = np.sqrt(n * P * (1 - P))
    z = (counts - expected) / standard_deviation
    pvals = [binomtest(int(c), n, P).pvalue for c in counts]
    # Bonferroni adjustment for checking 49 numbers simultaneously.
    rows = [dict(number=i + 1, count=int(counts[i]), rate=round(counts[i] / n, 6),
                 expected=round(expected, 4), z_score=round(float(z[i]), 4),
                 p_value=round(pvals[i], 8), p_bonferroni=round(min(1., pvals[i] * 49), 8),
                 significant_05_bonferroni=(pvals[i] * 49 < 0.05)) for i in range(49)]
    return rows, counts


def analyze_gaps(nums: np.ndarray):
    n = len(nums)
    rows = []
    for number in range(1, 50):
        hits = np.flatnonzero(np.any(nums == number, axis=1))
        gaps = np.diff(hits) - 1  # Draws WITHOUT the number between appearances.
        rows.append(dict(number=number, occurrences=int(len(hits)),
                         current_gap=int(n - 1 - hits[-1]) if len(hits) else n,
                         mean_gap=round(float(np.mean(gaps)), 3) if len(gaps) else "",
                         max_gap=int(np.max(gaps)) if len(gaps) else "",
                         last_hit_index=int(hits[-1]) if len(hits) else ""))
    return rows


def analyze_combinations(nums: np.ndarray):
    n = len(nums)
    odd_counts = np.count_nonzero(nums % 2, axis=1)
    low_counts = np.count_nonzero(nums <= 24, axis=1)
    consecutive_pairs = np.count_nonzero(np.diff(np.sort(nums, axis=1), axis=1) == 1, axis=1)
    totals = nums.sum(axis=1)
    span = nums.max(axis=1) - nums.min(axis=1)
    overlap = np.sum(np.any(nums[1:, :, None] == nums[:-1, None, :], axis=2), axis=1)
    presence = np.zeros((n, 49), dtype=np.int32)
    presence[np.arange(n)[:, None], nums - 1] = 1
    pair = presence.T @ presence
    pairs = [dict(n1=i + 1, n2=j + 1, count=int(pair[i, j]))
             for i in range(49) for j in range(i + 1, 49)]
    pairs.sort(key=lambda r: (-r["count"], r["n1"], r["n2"]))
    stats = {
        "odd_count_distribution": {str(i): int(np.sum(odd_counts == i)) for i in range(7)},
        "low_1_to_24_distribution": {str(i): int(np.sum(low_counts == i)) for i in range(7)},
        "consecutive_pair_distribution": {str(i): int(np.sum(consecutive_pairs == i)) for i in range(6)},
        "sum_min": int(totals.min()), "sum_mean": float(totals.mean()), "sum_max": int(totals.max()),
        "span_min": int(span.min()), "span_mean": float(span.mean()), "span_max": int(span.max()),
        "adjacent_overlap_distribution": {str(i): int(np.sum(overlap == i)) for i in range(7)},
        "adjacent_overlap_mean": float(overlap.mean()),
        "adjacent_overlap_expected": 36 / 49,
        "pair_expected_count_per_pair": n * 30 / (49 * 48),
        "pair_total": len(pairs),
    }
    return stats, pairs, overlap


def randomness_tests(counts: np.ndarray, overlap: np.ndarray, n: int):
    # Under 6-from-49 WITHOUT replacement, counts have covariance; the naive
    # multinomial chi-square (df=48) reference is NOT valid. The statistic is
    # approximately (49-1)*[49-1]/[49-6] times chi-square_48? Exact design-effect
    # correction for counts = (49-6)/(49-1) = 43/48 under multivariate hypergeometric.
    # Thus Pearson X² / (43/48) approximately follows chi-square(df=48).
    expected = n * P
    pearson = float(np.sum((counts - expected) ** 2 / expected))
    fpc = (49 - 6) / (49 - 1)  # finite-population correction
    adjusted = pearson / fpc
    # Adjacent draw overlap: Hypergeom(M=49, n=6, N=6), expectation 36/49.
    # Report observed counts and theoretical distribution; avoid claiming an
    # independence-test p-value from overlapping adjacent pairs.
    theoretical_overlap = {str(k): float(hypergeom.pmf(k, 49, 6, 6)) for k in range(7)}
    return {
        "frequency_pearson_chisq": pearson,
        "frequency_fpc": fpc,
        "frequency_adjusted_chisq": adjusted,
        "frequency_reference_df": 48,
        "frequency_adjusted_p_approx": float(chi2.sf(adjusted, 48)),
        "frequency_test_note": "Asymptotic chi-square with finite-population correction; not a predictor.",
        "adjacent_overlap_theoretical_prob": theoretical_overlap,
        "adjacent_overlap_observations": len(overlap),
    }


def describe_bonus():
    if not BONUS_PATH.exists():
        return {"status": "not_found"}, []
    b = np.load(BONUS_PATH, allow_pickle=False)
    needed = {"draw_id", "date", "activity_id", "numbers"}
    if b.dtype.names is None or not needed.issubset(b.dtype.names):
        raise ValueError("Bonus npy missing expected fields")
    names = json.loads(ACTIVITIES_PATH.read_text(encoding="utf-8")) if ACTIVITIES_PATH.exists() else {}
    rows = []
    for activity_id in sorted(set(b["activity_id"].astype(int).tolist())):
        subset = b[b["activity_id"] == activity_id]
        keys = [tuple([int(r["draw_id"])] + list(map(int, r["numbers"]))) for r in subset]
        rows.append(dict(activity_id=activity_id, activity=names.get(str(activity_id), "unknown"),
                         combinations=len(subset), distinct_draws=len(set(subset["draw_id"].tolist())),
                         repeated_combinations=len(keys) - len(set(keys))))
    return {"status": "descriptive_only", "records": len(b),
            "activity_types": len(rows), "note": "Bonus combinations are not treated as independent 6/49 draws."}, rows


def main():
    print(f"Lotto649 Statistical Analyzer v{VERSION}")
    if not BASE_PATH.exists():
        raise FileNotFoundError(f"Run lotto_normalize.py first: {BASE_PATH}")
    draws = np.load(BASE_PATH, allow_pickle=False)
    nums, special = validate_base(draws)
    n = len(draws)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frequency, counts = analyze_frequency(nums)
    gaps = analyze_gaps(nums)
    combos, pairs, overlap = analyze_combinations(nums)
    tests = randomness_tests(counts, overlap, n)
    bonus, bonus_rows = describe_bonus()

    write_csv(OUTPUT / "frequency.csv", list(frequency[0]), frequency)
    write_csv(OUTPUT / "gaps.csv", list(gaps[0]), gaps)
    write_csv(OUTPUT / "pairs.csv", list(pairs[0]), pairs)
    if bonus_rows:
        write_csv(OUTPUT / "bonus_activities.csv", list(bonus_rows[0]), bonus_rows)

    report = {
        "version": VERSION, "base_draws": n,
        "first_date": int(draws["date"][0]), "last_date": int(draws["date"][-1]),
        "theoretical_main_number_probability": P,
        "theoretical_mean_gap_between_hits": (1-P)/P,
        "significant_numbers_bonferroni": [r["number"] for r in frequency if r["significant_05_bonferroni"]],
        "combination": combos, "randomness": tests, "bonus": bonus,
        "special_number_frequency": {str(i): int(np.sum(special == i)) for i in range(1, 50)},
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    highest = sorted(frequency, key=lambda r: (-r["count"], r["number"]))[:10]
    lowest = sorted(frequency, key=lambda r: (r["count"], r["number"]))[:10]
    lines = [f"Lotto649 Statistical Analyzer v{VERSION}",
             f"Base draws: {n:,} | {report['first_date']} - {report['last_date']}",
             f"Expected appearances per number: {n*P:.2f}",
             "", "Top 10 historical frequency:"]
    lines += [f"  {r['number']:02d}: {r['count']:4d}  z={r['z_score']:+.2f}" for r in highest]
    lines += ["", "Bottom 10 historical frequency:"]
    lines += [f"  {r['number']:02d}: {r['count']:4d}  z={r['z_score']:+.2f}" for r in lowest]
    lines += ["", f"Adjusted frequency chi-square p (approx): {tests['frequency_adjusted_p_approx']:.6g}",
              f"Bonferroni significant numbers: {report['significant_numbers_bonferroni']}",
              f"Adjacent-draw mean overlap: {combos['adjacent_overlap_mean']:.4f} (expected {36/49:.4f})",
              "", "Top 10 historical pairs:"]
    lines += [f"  {r['n1']:02d}+{r['n2']:02d}: {r['count']}" for r in pairs[:10]]
    lines += ["", "CAUTION: Historical patterns do not imply future predictability.",
              "CAUTION: Bonus activity rows are excluded from base-draw statistical tests."]
    (OUTPUT / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[PASS] base draws: {n:,}; date range: {report['first_date']}–{report['last_date']}")
    print(f"[PASS] Frequency / Gap / Pair / Structure / Randomness")
    print(f"[INFO] Bonus: {bonus.get('records', 0):,} records; descriptive only")
    print(f"[INFO] Significant numbers (Bonferroni): {report['significant_numbers_bonferroni']}")
    print(f"[DONE] {OUTPUT.resolve()}")
    print("\n".join(lines[:7]))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        print(f"[FAIL] {exc}")
        raise SystemExit(1)
