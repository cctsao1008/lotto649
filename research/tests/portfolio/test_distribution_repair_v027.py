"""v0.27 P0 exact theory and chronological baseline regression tests."""
from __future__ import annotations

import csv
from itertools import combinations
from math import comb
from pathlib import Path

import pytest

from lotto7x7.distribution_repair_v027 import (
    LOW, HIGH, TOTAL, METRIC, baseline, exact_sum_counts, exact_theory,
    markdown_report, squared_l2_probability, _prefix_sha256,
)


def test_exact_global_sum_theory_and_checksum():
    t = exact_theory()
    assert t["universe"] == 13_983_816 == comb(49, 6)
    assert (t["support_min"], t["support_max"]) == (21, 279)
    assert len(t["counts"]) == 259 and t["counts"][0] == t["counts"][-1] == 1
    assert t["counts"] == list(reversed(t["counts"]))
    assert sum(t["counts"]) == TOTAL
    assert sum((LOW + i) * c for i, c in enumerate(t["counts"])) == 150 * TOTAL
    assert sum(((LOW + i) - 150) ** 2 * c for i, c in enumerate(t["counts"])) == 1075 * TOTAL
    assert t["mode_sums"] == [150]
    assert t == exact_theory()
    assert len(t["checksum_sha256"]) == 64


@pytest.mark.parametrize("n,k", [(6, 1), (7, 2), (8, 3), (10, 4)])
def test_integer_dp_against_independent_small_exhaustive(n, k):
    histogram = exact_sum_counts(n, k)
    brute = [0] * len(histogram)
    for ticket in combinations(range(1, n + 1), k):
        brute[sum(ticket)] += 1
    assert histogram == tuple(brute)
    assert sum(histogram) == comb(n, k)


@pytest.mark.parametrize("n,k", [(0, 1), (7, 0), (5, 6), (5, True), (3.0, 2)])
def test_invalid_theory_parameters(n, k):
    with pytest.raises(ValueError):
        exact_sum_counts(n, k)


def test_primary_l2_metric_and_zero_bins():
    assert squared_l2_probability([1, 1], [1, 1]) == 0.0
    assert squared_l2_probability([2, 0], [1, 1]) == pytest.approx(.5)
    assert squared_l2_probability([0, 2, 0], [1, 1, 0]) == pytest.approx(.5)
    for a, b in (([], []), ([0], [1]), ([1], [0]), ([1, 2], [1]), ([-1], [1])):
        with pytest.raises(ValueError):
            squared_l2_probability(a, b)


def write_annual(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("date", "draw_id", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"))
        writer.writerows(rows)


ROWS = [
    ("2026/10/30", 115000101, 4, 22, 25, 30, 34, 37, 48),
    ("2026/11/03", 115000102, 2, 23, 24, 25, 29, 39, 31),
    ("2026/11/06", 115000103, 5, 9, 26, 32, 41, 48, 23),
    ("2026/11/10", 115000104, 1, 2, 3, 4, 5, 6, 7),
]


def test_realistic_csv_baseline_cutoff_no_lookahead(tmp_path: Path):
    path = tmp_path / "lotto649_2026.csv"
    write_annual(path, ROWS[:3])
    early = baseline(tmp_path, cutoff="2026-11-06")
    assert early["stage"] == "P0_EXACT_THEORY_AND_HISTORICAL_BASELINE"
    assert early["metric"]["name"] == METRIC
    assert early["source"]["historical_draws"] == 3
    assert early["source"]["last_draw_id"] == 115000103
    assert early["source"]["post_cutoff_rows_excluded"] == 0
    assert len(early["bins"]) == 259
    assert sum(row["observed_count"] for row in early["bins"]) == 3
    assert sum(row["theory_combinations"] for row in early["bins"]) == TOTAL
    assert sum(row["deficit_count"] for row in early["bins"]) == pytest.approx(0, abs=1e-12)
    assert early["observed"]["total_residual_count"] == pytest.approx(0, abs=1e-12)
    assert early["observed"]["squared_l2_probability"] > 0
    assert "PREDICTIVE_ADVANTAGE_NOT_EVALUATED" in early["decision"]
    assert "Distribution by 10-point sum buckets" in markdown_report(early)

    # Appending a future draw must not change any descriptive P0 metric,
    # prefix hash or included bin, even though raw source-file hash changes.
    write_annual(path, ROWS)
    later = baseline(tmp_path, cutoff="2026-11-06")
    assert later["source"]["historical_draws"] == 3
    assert later["source"]["validated_source_rows"] == 4
    assert later["source"]["post_cutoff_rows_excluded"] == 1
    assert later["observed"] == early["observed"]
    assert later["bins"] == early["bins"]
    assert later["source"]["analysis_input_sha256"] == early["source"]["analysis_input_sha256"]
    assert later["source"]["source_file_sha256"] != early["source"]["source_file_sha256"]
    assert markdown_report(later) == markdown_report(baseline(tmp_path, cutoff="2026-11-06"))


def test_cutoff_requires_real_draw_date_and_valid_iso(tmp_path: Path):
    write_annual(tmp_path / "lotto649_2026.csv", ROWS[:3])
    for cutoff in ("2026/11/06", "2026-11-05", "2026-11-10", "2026-13-01", "2026-10-29"):
        with pytest.raises(ValueError):
            baseline(tmp_path, cutoff=cutoff)


def test_bad_chronology_bonus_duplicate_and_year_rejected(tmp_path: Path):
    file = tmp_path / "lotto649_2026.csv"
    for rows in (
        ROWS[:3][::-1],
        [ROWS[0], ROWS[0], ROWS[2]],
        [ROWS[0][:-1] + (ROWS[0][2],), *ROWS[1:3]],
        [("2025/10/30", *ROWS[0][1:]), *ROWS[1:3]],
        [ROWS[1], ROWS[0], ROWS[2]],
    ):
        write_annual(file, rows)
        with pytest.raises(ValueError):
            baseline(tmp_path, cutoff="2026-11-06")


def test_prefix_sha_is_sensitive_to_included_data_only():
    ids = [115000101, 115000102]
    dates = [20261030, 20261103]
    numbers = [(4, 22, 25, 30, 34, 37), (2, 23, 24, 25, 29, 39)]
    digest = _prefix_sha256(ids, dates, numbers)
    assert len(digest) == 64
    assert digest != _prefix_sha256(ids, dates, [numbers[0], (2, 23, 24, 25, 29, 40)])


def test_extra_csv_header_or_columns_rejected(tmp_path: Path):
    path = tmp_path / "lotto649_2026.csv"
    write_annual(path, ROWS[:3])
    path.write_text(path.read_text(encoding="utf-8").replace("bonus\n", "bonus,unexpected\n", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="nine-column"):
        baseline(tmp_path, cutoff="2026-11-06")
    write_annual(path, ROWS[:3])
    with path.open("a", encoding="utf-8") as file:
        file.write("2026/11/10,115000104,1,2,3,4,5,6,7,extra\n")
    with pytest.raises(ValueError, match="CSV width"):
        baseline(tmp_path, cutoff="2026-11-06")
