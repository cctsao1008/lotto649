"""Issue #15 P0: exact integer Log Product bins and frozen 2026 history."""
from __future__ import annotations

from bisect import bisect_right
import csv
from itertools import combinations
from math import comb, fsum, log, prod, sqrt
import json

import pytest

from lotto7x7.distribution_repair_v027 import baseline as sum_baseline
from lotto7x7.log_product_v027 import (
    DEFAULT_WIDTH, EXPECTED_FROZEN_PREFIX, SCHEMA, _grid,
    exact_log_theory, historical_baseline, main, markdown_report,
    product_bin, theoretical_log_moments,
)


@pytest.fixture(scope="session")
def theory():
    return exact_log_theory()


ROWS = [
    ("2026/10/30", 115000101, 4, 22, 25, 30, 34, 37, 48),
    ("2026/11/03", 115000102, 2, 23, 24, 25, 29, 39, 31),
    ("2026/11/06", 115000103, 5, 9, 26, 32, 41, 48, 23),
    ("2026/11/10", 115000104, 1, 2, 3, 4, 5, 6, 7),
]


def write_year(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("date", "draw_id", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"))
        writer.writerows(rows)


@pytest.fixture
def history_folder(tmp_path):
    write_year(tmp_path / "lotto649_2026.csv", ROWS[:3])
    return tmp_path


def test_full_exact_6_of_49_mass_fixed_bins_and_skew(theory):
    assert theory["universe"] == comb(49, 6) == 13_983_816
    assert theory["grid"]["bin_count"] == 34
    assert theory["grid"]["edge_start"] == "6.5"
    assert theory["grid"]["edge_end"] == "23.5"
    assert theory["grid"]["min_product"] == 720
    assert theory["grid"]["max_product"] == prod(range(44, 50))
    assert sum(theory["counts"]) == theory["universe"]
    assert theory["counts"][0] == 5
    assert theory["counts"][-1] == 2
    assert theory["counts"][23] == max(theory["counts"])
    assert theory["mode_bin_indices"] == [23]
    assert theory["strict_interior_local_peaks"] == 1
    assert sum(theory["probabilities"]) == pytest.approx(1, abs=1e-15)
    assert len(theory["checksum_sha256"]) == 64
    assert theory["moments"]["mean"] == pytest.approx(17.701927830164678, abs=1e-12)
    assert theory["moments"]["variance"] == pytest.approx(4.150073302017148, abs=1e-12)
    assert theory["moments"]["skewness"] == pytest.approx(-0.46564263743495726, abs=1e-12)
    assert theory["support_log_min"] == pytest.approx(log(720))
    assert theory["support_log_max"] == pytest.approx(log(prod(range(44, 50))))
    assert theory["counts"][product_bin(720, theory["grid"])] >= 1


@pytest.mark.parametrize("n,k", [(7, 2), (8, 3), (9, 4), (12, 5)])
def test_independent_small_exhaustive_product_bin_count_and_analytic_moments(n, k):
    expected = exact_log_theory(n, k)
    grid = expected["grid"]
    histogram = [0] * grid["bin_count"]
    all_logs = []
    for seq in combinations(range(1, n + 1), k):
        p = prod(seq)
        L = log(p)
        all_logs.append(L)
        # Independent brute force classification uses float log edges, rather
        # than the production integer threshold cutoffs.
        candidates = [float(v) for v in grid["edges"]]
        index = bisect_right(candidates, L) - 1
        assert index == product_bin(p, grid)
        histogram[index] += 1
    assert histogram == expected["counts"]
    assert sum(histogram) == comb(n, k)
    center = fsum(all_logs) / len(all_logs)
    variance = fsum((x - center) ** 2 for x in all_logs) / len(all_logs)
    third = fsum((x - center) ** 3 for x in all_logs) / len(all_logs)
    moments = theoretical_log_moments(n, k)
    assert center == pytest.approx(moments["mean"], abs=1e-12)
    assert variance == pytest.approx(moments["variance"], abs=1e-12)
    assert third == pytest.approx(moments["third_central_moment"], abs=1e-12)
    assert third / variance**1.5 == pytest.approx(moments["skewness"], abs=1e-12)


def test_deterministic_theory_integer_edges_and_boundary_closure(theory):
    grid = theory["grid"]
    assert theory["checksum_sha256"] == exact_log_theory()["checksum_sha256"]
    cuts = grid["integer_product_thresholds"]
    assert cuts == sorted(set(cuts))
    assert all(0 < p for p in cuts)
    assert len(cuts) == grid["bin_count"] + 1
    for i, cutoff in enumerate(cuts[1:-1], start=1):
        assert product_bin(cutoff - 1, grid) == i - 1
        assert product_bin(cutoff, grid) == i
    with pytest.raises(ValueError):
        product_bin(719, grid)
    with pytest.raises(ValueError):
        product_bin(grid["max_product"] + 1, grid)
    with pytest.raises(ValueError):
        product_bin(True, grid)


def test_quantile_bins_order_and_count_mass(theory):
    rows = theory["quantile_brackets"]
    previous = -1
    for p, row in rows.items():
        index = row["bin_index"]
        assert previous <= index
        previous = index
        assert 0 <= index < 34
        assert float(row["log_lower"]) < float(row["log_upper_exclusive"])
        rank = row["rank_at_or_above"]
        left = sum(theory["counts"][:index])
        right = left + theory["counts"][index]
        assert left < rank <= right


@pytest.mark.parametrize("n,k,width", [
    (3, 2, "0.5"), (50, 6, "0.5"), (49, 0, "0.5"),
    (49, 49, "0.5"), (49, 6, "0"), (49, 6, "-1"),
    (49, 6, "NaN"), (49, 6, "0.01"),
    (True, 6, "0.5"), (49.0, 6, "0.5"),
])
def test_invalid_grid_or_moments_parameters(n, k, width):
    with pytest.raises(ValueError):
        _grid(n, k, width)


def test_frozen_historical_log_p0_and_cutoff_no_lookahead(theory, history_folder):
    sum_p0 = sum_baseline(history_folder, cutoff="2026-11-06")
    expected_prefix = sum_p0["source"]["analysis_input_sha256"]
    before = historical_baseline(history_folder, theory=theory,
                                 expected_prefix_sha256=expected_prefix)
    assert before["schema"] == SCHEMA
    assert before["source"]["historical_draws"] == 3
    assert before["source"]["last_draw_id"] == 115000103
    assert before["source"]["analysis_input_sha256"] == expected_prefix
    assert before["sum_baseline_reference"]["historical_input_sha256"] == expected_prefix
    assert before["metric"]["name"] == "squared_l2_probability_v1"
    assert len(before["bins"]) == 34
    assert sum(before["observed"]["counts"]) == 3
    assert sum(row["theory_combinations"] for row in before["bins"]) == 13_983_816
    assert sum(row["residual_count"] for row in before["bins"]) == pytest.approx(0, abs=1e-12)
    assert before["observed"]["squared_l2_probability"] > 0
    assert before["theory"]["checksum_sha256"] == theory["checksum_sha256"]
    assert "Log Product P0" in markdown_report(before)
    assert "not evidence" in markdown_report(before)

    # Future data can change only metadata about validated source rows and
    # annual raw-file hashes, not any included-bin count, loss or prefix hash.
    write_year(history_folder / "lotto649_2026.csv", ROWS)
    after = historical_baseline(history_folder, theory=theory,
                                expected_prefix_sha256=expected_prefix)
    assert before["observed"] == after["observed"]
    assert before["bins"] == after["bins"]
    assert before["theory"] == after["theory"]
    assert before["source"]["analysis_input_sha256"] == after["source"]["analysis_input_sha256"]
    assert before["source"]["source_file_sha256"] != after["source"]["source_file_sha256"]
    assert after["source"]["post_cutoff_rows_excluded"] == 1


def test_reject_wrong_frozen_hash_and_cutoff(theory, history_folder):
    with pytest.raises(ValueError, match="prefix SHA256"):
        historical_baseline(history_folder, theory=theory,
                            expected_prefix_sha256=EXPECTED_FROZEN_PREFIX)
    with pytest.raises(ValueError, match="cutoff"):
        historical_baseline(history_folder, cutoff="2026-11-05", theory=theory)
    with pytest.raises(ValueError, match="cutoff"):
        historical_baseline(history_folder, cutoff="2026-11-10", theory=theory)


def test_reject_corrupt_or_wrong_year_csv(theory, history_folder):
    bad = history_folder / "lotto649_2026.csv"
    original = bad.read_text(encoding="utf-8")
    bad.write_text(original.replace("2026/11/03", "2025/11/03", 1), encoding="utf-8")
    with pytest.raises(ValueError):
        historical_baseline(history_folder, theory=theory)


def test_cli_report_input_gate_and_overwrite(history_folder, tmp_path, capsys):
    prefix = sum_baseline(history_folder, cutoff="2026-11-06")["source"]["analysis_input_sha256"]
    out = tmp_path / "log_baseline.json"
    md = tmp_path / "log_baseline.md"
    args = ["--csv-dir", str(history_folder), "--cutoff", "2026-11-06",
            "--expected-prefix-sha256", prefix, "--out", str(out), "--md-out", str(md)]
    main(args)
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["schema"] == SCHEMA
    assert result["source"]["analysis_input_sha256"] == prefix
    assert result["theory"]["universe"] == 13_983_816
    assert "quantile brackets" in md.read_text(encoding="utf-8")
    assert "[LOG-P0 PASS]" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        main(args)
