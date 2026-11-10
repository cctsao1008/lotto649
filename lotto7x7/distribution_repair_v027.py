"""v0.27 P0: exact fair 6/49 sum theory and historical discrepancy baseline.

Research scope: descriptive distribution analysis only. This module deliberately
DOES NOT rank future tickets or infer an increased probability of any outcome.
The frozen primary discrepancy is squared L2 distance between *probabilities*.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
from hashlib import sha256
import json
from math import comb
from pathlib import Path

SCHEMA = "lotto7x7.distribution_repair.v0.27.p0"
METRIC = "squared_l2_probability_v1"
CSV_COLUMNS = ("date", "draw_id", "n1", "n2", "n3", "n4", "n5", "n6", "bonus")
BALLS = 49
PICKS = 6
TOTAL = comb(BALLS, PICKS)
LOW = PICKS * (PICKS + 1) // 2
HIGH = PICKS * (2 * BALLS - PICKS + 1) // 2


def exact_sum_counts(n: int = BALLS, k: int = PICKS) -> tuple[int, ...]:
    """Count all k-subsets of {1,...,n} by sum using exact integer DP.

    The returned vector includes zero-count sums below the possible minimum.
    Time: O(n*k^2*n); memory: O(k^2*n), no enumeration or Monte Carlo.
    """
    if not isinstance(n, int) or isinstance(n, bool) or not isinstance(k, int) or isinstance(k, bool):
        raise ValueError("n and k must be integers")
    if not (1 <= k <= n):
        raise ValueError("require 1 <= k <= n")
    maximum = k * (2 * n - k + 1) // 2
    dp = [[0] * (maximum + 1) for _ in range(k + 1)]
    dp[0][0] = 1
    for ball in range(1, n + 1):
        for chosen in range(min(ball, k), 0, -1):
            dst, src = dp[chosen], dp[chosen - 1]
            for subtotal in range(maximum, ball - 1, -1):
                dst[subtotal] += src[subtotal - ball]
    answer = tuple(dp[k])
    if sum(answer) != comb(n, k):
        raise AssertionError("exact theory does not normalize")
    return answer


def exact_theory() -> dict:
    """Exact 6/49 total-sum histogram with independent moment checks."""
    all_counts = exact_sum_counts()
    counts = all_counts[LOW:HIGH + 1]
    if len(counts) != 259 or counts[0] != 1 or counts[-1] != 1:
        raise AssertionError("sum support boundary mismatch")
    if sum(counts) != TOTAL:
        raise AssertionError("sum counts mismatch C(49, 6)")
    if counts != counts[::-1]:
        raise AssertionError("sum histogram is not symmetric about 150")
    weighted = sum((LOW + i) * count for i, count in enumerate(counts))
    if weighted != TOTAL * 150:
        raise AssertionError("sum mean is not exactly 150")
    second_centered = sum(((LOW + i) - 150) ** 2 * count for i, count in enumerate(counts))
    if second_centered != TOTAL * 1075:
        raise AssertionError("sum variance is not exactly 1075")
    # Independent finite-population sampling formula; not inferred from DP.
    population_variance = (BALLS * BALLS - 1) // 12
    formula_variance_numerator = PICKS * (BALLS - PICKS) * population_variance
    if formula_variance_numerator != 1075 * (BALLS - 1):
        raise AssertionError("finite-population variance identity failed")
    canonical = json.dumps({"support_min": LOW, "counts": counts},
                           separators=(",", ":"), sort_keys=True).encode("ascii")
    peak = max(counts)
    return {
        "universe": TOTAL,
        "support_min": LOW,
        "support_max": HIGH,
        "counts": list(counts),
        "mean": 150,
        "variance": 1075,
        "mode_sums": [LOW + i for i, value in enumerate(counts) if value == peak],
        "method": "exact integer subset-sum dynamic programming; without replacement",
        "checksum_sha256": sha256(canonical).hexdigest(),
        "checksum_protocol": "SHA256 of ASCII compact sorted-key JSON {counts:[...],support_min:21}",
    }


def squared_l2_probability(observed: list[int], theoretical: list[int]) -> float:
    """Fixed unweighted squared L2 between empirical and exact probabilities."""
    if len(observed) != len(theoretical) or not observed or any(x < 0 for x in observed + theoretical):
        raise ValueError("count histograms must have identical nonempty supports")
    n = sum(observed)
    total = sum(theoretical)
    if n <= 0 or total <= 0:
        raise ValueError("empty count histogram")
    return sum((a / n - b / total) ** 2 for a, b in zip(observed, theoretical))


def _prefix_sha256(ids, dates, numbers) -> str:
    """Canonical hash of only included chronological main-ball input rows.

    Future rows and per-file source-byte hashes have no effect on this hash.
    """
    digest = sha256()
    for draw_id, day, ticket in zip(ids, dates, numbers):
        row = [int(day), int(draw_id), *(int(x) for x in ticket)]
        digest.update((json.dumps(row, separators=(",", ":")) + "\n").encode("ascii"))
    return digest.hexdigest()


def baseline(csv_dir: Path, *, cutoff: str) -> dict:
    """Validate source CSV, freeze prefix at an existing draw date, audit sums.

    Full annual CSV files are validated, but metrics/input hash use only rows
    at or before cutoff. Post-cutoff outcomes never enter observed counts.
    """
    try:
        day = date.fromisoformat(cutoff)
    except (TypeError, ValueError) as exc:
        raise ValueError("cutoff must be a valid ISO YYYY-MM-DD date") from exc
    if day.isoformat() != cutoff:
        raise ValueError("cutoff must have canonical ISO YYYY-MM-DD formatting")

    # The existing temporal loader verifies row content, chronology, ROC year,
    # numbers and bonus. P0 adds an exact flat-CSV schema/width gate: the older
    # loader intentionally accepts extra fields, which is too loose here.
    folder = Path(csv_dir)
    for path in sorted(folder.glob("lotto649_????.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            if next(reader, None) != list(CSV_COLUMNS):
                raise ValueError(f"{path}: expected exact nine-column CSV schema")
            for line_number, row in enumerate(reader, start=2):
                if len(row) != len(CSV_COLUMNS):
                    raise ValueError(f"{path}:{line_number}: inconsistent CSV width")

    from .temporal_dataset_v024 import load_csv_history

    data = load_csv_history(folder)
    if data.dates is None:
        raise AssertionError("CSV mode must supply validated dates")
    cutoff_day = int(day.strftime("%Y%m%d"))
    index = [i for i, d in enumerate(data.dates) if int(d) == cutoff_day]
    if len(index) != 1:
        raise ValueError("cutoff must equal exactly one existing draw date")
    n = index[0] + 1
    numbers = data.numbers[:n]
    selected_ids = data.ids[:n]
    selected_dates = data.dates[:n]
    theory = exact_theory()
    observed = [0] * len(theory["counts"])
    for ticket in numbers:
        total_sum = sum(int(v) for v in ticket)
        if not LOW <= total_sum <= HIGH:
            raise AssertionError("validated ticket sum outside exact support")
        observed[total_sum - LOW] += 1
    if sum(observed) != n:
        raise AssertionError("historical sum histogram count mismatch")
    bins = []
    for i, (actual, possible) in enumerate(zip(observed, theory["counts"])):
        expected = n * possible / TOTAL
        residual = actual - expected
        bins.append({
            "sum": LOW + i,
            "theory_combinations": possible,
            "theory_probability": possible / TOTAL,
            "observed_count": actual,
            "observed_probability": actual / n,
            "expected_count": expected,
            "residual_count": residual,
            "deficit_count": -residual,
            "residual_probability": actual / n - possible / TOTAL,
        })
    empirical_mean = sum((LOW + i) * value for i, value in enumerate(observed)) / n
    empirical_variance = sum(((LOW + i) - empirical_mean) ** 2 * value
                             for i, value in enumerate(observed)) / n
    l2 = squared_l2_probability(observed, theory["counts"])
    return {
        "schema": SCHEMA,
        "stage": "P0_EXACT_THEORY_AND_HISTORICAL_BASELINE",
        "decision": "DESCRIPTIVE_BASELINE_PASS; PREDICTIVE_ADVANTAGE_NOT_EVALUATED",
        "metric": {"name": METRIC, "weights": "uniform per exact sum bin",
                   "normalization": "empirical and exact probabilities, not raw counts",
                   "zero_bin_policy": "no division by individual theoretical probabilities",
                   "parameter_tuning": "none; fixed before P1"},
        "source": {
            "csv_dir": str(Path(csv_dir).as_posix()),
            "cutoff": cutoff,
            "first_draw_date": date(int(selected_dates[0]) // 10000,
                                    int(selected_dates[0]) // 100 % 100,
                                    int(selected_dates[0]) % 100).isoformat(),
            "first_draw_id": int(selected_ids[0]),
            "last_draw_date": cutoff,
            "last_draw_id": int(selected_ids[-1]),
            "historical_draws": n,
            "validated_source_rows": len(data.ids),
            "post_cutoff_rows_excluded": len(data.ids) - n,
            "analysis_input_sha256": _prefix_sha256(selected_ids, selected_dates, numbers),
            "analysis_input_sha256_protocol":
                "SHA256 of UTF-8 JSON Lines [YYYYMMDD,draw_id,n1,n2,n3,n4,n5,n6] for included draws only; sorted balls, no whitespace",
            "source_file_sha256": data.sources["files_sha256"],
            "provenance": data.sources.get("origin_note"),
            "bonus": "validated by legacy loader; not a P0 feature",
        },
        "theory": theory,
        "observed": {
            "counts": observed,
            "mean": empirical_mean,
            "variance": empirical_variance,
            "squared_l2_probability": l2,
            "total_residual_count": sum(row["residual_count"] for row in bins),
            "top_10_deficit_bins": [v["sum"] for v in sorted(bins, key=lambda v: (-v["deficit_count"], v["sum"]))[:10]],
            "top_10_surplus_bins": [v["sum"] for v in sorted(bins, key=lambda v: (-v["residual_count"], v["sum"]))[:10]],
        },
        "bins": bins,
        "interpretation": "Distance is descriptive. IID sampling can converge without self-correction. No odds or next-draw predictions.",
    }


def markdown_report(report: dict) -> str:
    """Deterministic human-readable distribution chart and signed residuals."""
    src = report["source"]
    theory = report["theory"]
    obs = report["observed"]
    bins = report["bins"]
    lines = [
        "# Lotto649 v0.27 — P0 Exact Sum Baseline", "",
        f"- **Cutoff:** {src['cutoff']} / #{src['last_draw_id']}",
        f"- **Draws:** {src['historical_draws']}; excluded post-cutoff: {src['post_cutoff_rows_excluded']}",
        f"- **Historical prefix SHA-256:** `{src['analysis_input_sha256']}`",
        f"- **Theory SHA-256:** `{theory['checksum_sha256']}`",
        f"- **Theory:** C(49,6) = {theory['universe']:,}; sums {theory['support_min']}–{theory['support_max']}; mean {theory['mean']}; variance {theory['variance']}",
        f"- **Observed:** mean {obs['mean']:.6f}; population variance {obs['variance']:.6f}",
        f"- **Primary metric:** `{report['metric']['name']}` = {obs['squared_l2_probability']:.12g}",
        "- **Decision:** descriptive baseline only; predictive advantage not evaluated.",
        "", "## Distribution by 10-point sum buckets", "",
        "Buckets are descriptive visualization only; the L2 metric uses all 259 individual sum bins.", "",
        "| Sum range | Observed | IID expected | Observed − expected | Signed bar |",
        "|---:|---:|---:|---:|:---|",
    ]
    groups = []
    for lo in range(LOW, HIGH + 1, 10):
        subset = bins[lo - LOW: min(lo + 10, HIGH + 1) - LOW]
        count = sum(v["observed_count"] for v in subset)
        expected = sum(v["expected_count"] for v in subset)
        groups.append((lo, min(lo + 9, HIGH), count, expected))
    max_residual = max(abs(count - exp) for _, _, count, exp in groups)
    for lo, hi, count, exp in groups:
        difference = count - exp
        bar_width = round(14 * abs(difference) / max_residual) if max_residual else 0
        bar = ("+" if difference >= 0 else "−") + "█" * bar_width
        lines.append(f"| {lo}–{hi} | {count} | {exp:.2f} | {difference:+.2f} | {bar} |")
    lines.extend(["", "## Largest sum-bin deficits (expected − observed)", "",
                  "| Sum | Observed | IID expected | Deficit |", "|---:|---:|---:|---:|"])
    by_deficit = sorted(bins, key=lambda b: (-b["deficit_count"], b["sum"]))[:10]
    for row in by_deficit:
        lines.append(f"| {row['sum']} | {row['observed_count']} | {row['expected_count']:.3f} | {row['deficit_count']:+.3f} |")
    lines.extend(["", "## Largest sum-bin surpluses (observed − expected)", "",
                  "| Sum | Observed | IID expected | Surplus |", "|---:|---:|---:|---:|"])
    by_surplus = sorted(bins, key=lambda b: (-b["residual_count"], b["sum"]))[:10]
    for row in by_surplus:
        lines.append(f"| {row['sum']} | {row['observed_count']} | {row['expected_count']:.3f} | {row['residual_count']:+.3f} |")
    lines.extend(["", "## Reproduce", "",
                  "```powershell",
                  "& .\\.venv\\Scripts\\python.exe -m lotto7x7.distribution_repair_v027 "
                  f"--csv-dir data/csv --cutoff {src['cutoff']} "
                  "--out results/v027_p0_baseline.json --md-out results/v027_p0_baseline.md",
                  "```", "",
                  "Source-file SHA-256 values are embedded in JSON. The historical prefix hash excludes post-cutoff rows; source-file byte hashes may change when annual CSV files are extended.",
                  "The source is third-party archival data, not independently verified against Taiwan Lottery official records.",
                  "", "**Interpretation:** A declining discrepancy alone is compatible with ordinary IID convergence; predictive self-correction is not established in P0.", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, default=Path("data/csv"))
    parser.add_argument("--cutoff", required=True, help="existing draw date (YYYY-MM-DD); required to prevent lookahead")
    parser.add_argument("--out", type=Path, default=Path("results/v027_p0_baseline.json"))
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args(argv)
    json_path = args.out
    md_path = args.md_out or args.out.with_suffix(".md")
    if json_path.resolve() == md_path.resolve() or json_path.exists() or md_path.exists():
        raise FileExistsError("report paths must be distinct and unused; never overwrite experiment reports")
    result = baseline(args.csv_dir, cutoff=args.cutoff)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    md_path.write_text(markdown_report(result), encoding="utf-8")
    print(f"[P0 PASS] {result['source']['historical_draws']} draws through {result['source']['cutoff']} #{result['source']['last_draw_id']}")
    print(f"[THEORY] {result['theory']['universe']:,} combinations; mean=150; variance=1075; SHA256={result['theory']['checksum_sha256']}")
    print(f"[LOSS] {METRIC}={result['observed']['squared_l2_probability']:.12g}")
    print(f"[DATA] prefix SHA256={result['source']['analysis_input_sha256']}")
    print(f"[REPORT] {json_path} / {md_path}")
    print("[SCOPE] Descriptive baseline only; predictive self-correction not yet evaluated")


if __name__ == "__main__":
    main()
