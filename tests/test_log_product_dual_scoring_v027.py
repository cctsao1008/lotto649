"""Issue #15 P1: exact rational signs for correlated Sum/Log marginal repair."""
from __future__ import annotations

import csv
import json
from math import isclose
from pathlib import Path

import pytest

from lotto7x7.distribution_repair_v027 import baseline, squared_l2_probability
from lotto7x7.distribution_repair_scoring_v027 import score_p0
from lotto7x7.log_product_v027 import historical_baseline
from lotto7x7.log_product_dual_scoring_v027 import (
    SCHEMA, dual_score, exact_gain_numerators, main, markdown_report,
    score_ticket, verify_log_p0,
)


@pytest.mark.parametrize("history,theory", [
    ([1, 2, 0], [1, 1, 2]),
    ([10, 0, 1, 4], [3, 7, 2, 0]),
    ([1, 1, 1], [10, 10, 10]),
    ([0, 100, 0], [1, 1, 1]),
    ([2, 4, 6, 3, 0], [13, 9, 7, 4, 1]),
])
def test_exact_gain_integer_numerators_against_independent_brute_l2(history, theory):
    result = exact_gain_numerators(history, theory)
    assert len(result["numerators"]) == len(history)
    before = squared_l2_probability(history, theory)
    for j, numerator in enumerate(result["numerators"]):
        copy = history[:]
        copy[j] += 1
        brute = before - squared_l2_probability(copy, theory)
        exact = numerator / result["denominator"]
        assert exact == pytest.approx(brute, abs=1e-15)
        if numerator:
            assert (exact > 0) == (numerator > 0)


@pytest.mark.parametrize("hist,theory", [
    ([], []), ([0], [1]), ([2], [0]), ([2], [1, 1]),
    ([True], [1]), ([1.0], [1]), ([-1], [1]), ([1], [-1]),
])
def test_invalid_exact_gain_histograms(hist, theory):
    with pytest.raises(ValueError):
        exact_gain_numerators(hist, theory)


@pytest.fixture(scope="module")
def fixture_dual(tmp_path_factory):
    folder = tmp_path_factory.mktemp("lotto_dual_hist")
    path = folder / "lotto649_2026.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("date", "draw_id", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"))
        writer.writerows([
            ("2026/10/30", 115000101, 4, 22, 25, 30, 34, 37, 48),
            ("2026/11/03", 115000102, 2, 23, 24, 25, 29, 39, 31),
            ("2026/11/06", 115000103, 5, 9, 26, 32, 41, 48, 23),
        ])
    sum_p0 = baseline(folder, cutoff="2026-11-06")
    sum_p1 = score_p0(sum_p0)
    log_p0 = historical_baseline(folder, cutoff="2026-11-06")
    result = dual_score(sum_p1, log_p0)
    return sum_p1, log_p0, result


def test_independent_dual_signs_and_exact_fair_mass(fixture_dual):
    sum_p1, log_p0, report = fixture_dual
    assert report["schema"] == SCHEMA
    assert report["source"]["last_draw_id"] == 115000103
    assert report["source"]["analysis_input_sha256"] == log_p0["source"]["analysis_input_sha256"]
    assert len(report["sum_bins"]) == 259
    assert len(report["log_bins"]) == 34
    assert sum(report["qualifying"]["per_sum_counts"]) == report["qualifying"]["dual_positive_ticket_count"]
    assert sum(report["qualifying"]["per_log_bin_counts"]) == report["qualifying"]["dual_positive_ticket_count"]
    assert report["conditional_iid"]["dual_positive_fair_probability"] == pytest.approx(
        report["qualifying"]["dual_positive_ticket_count"] / 13_983_816
    )
    for row in report["sum_bins"]:
        exact = row["exact_gain_numerator"] / report["exact_rational"]["sum_common_denominator"]
        assert exact == pytest.approx(row["gain"], abs=1e-15)
        assert row["exact_positive"] == (exact > 0)
    for row in report["log_bins"]:
        exact = row["exact_gain_numerator"] / report["exact_rational"]["log_common_denominator"]
        assert exact == pytest.approx(row["gain"], abs=1e-15)
        assert row["exact_positive"] == (exact > 0)
    assert "The two features are correlated" in markdown_report(report)
    assert "No four-disjoint portfolio" in markdown_report(report)
    for row in report["qualifying"]["lex_first_examples"]:
        assert score_ticket(report, row)["dual_positive"]
    with pytest.raises(ValueError):
        score_ticket(report, [1, 1, 2, 3, 4, 5])


def test_fixed_tickets_are_scored_without_claiming_a_new_purchase(fixture_dual):
    _, _, report = fixture_dual
    tickets = (
        (4, 9, 17, 24, 29, 42), (10, 13, 14, 27, 28, 34),
        (6, 12, 19, 20, 44, 48), (3, 7, 18, 33, 40, 47),
        (6, 15, 23, 29, 39, 40),
    )
    for t in tickets:
        score = score_ticket(report, t)
        assert score["sum"] == sum(t)
        assert score["dual_positive"] == (score["sum_exact_positive"] and score["log_exact_positive"])
        assert score["integer_product"] > 0
        assert score["sum_loss_before"] - score["sum_loss_after"] == pytest.approx(score["sum_gain"], abs=1e-14)
        assert score["log_loss_before"] - score["log_loss_after"] == pytest.approx(score["log_gain"], abs=1e-14)


def test_reject_wrong_prefix_and_corrupt_log_histogram(fixture_dual):
    import copy
    sum_p1, log_p0, _ = fixture_dual
    altered = copy.deepcopy(log_p0)
    altered["source"]["analysis_input_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source mismatch"):
        verify_log_p0(altered, sum_p1)
    altered = copy.deepcopy(log_p0)
    altered["observed"]["counts"][0] += 1
    # Completeness check occurs after exact theory recomputation.
    with pytest.raises(ValueError, match="historical counts"):
        verify_log_p0(altered, sum_p1)


def test_cli_examples_and_never_overwrites(fixture_dual, tmp_path, capsys):
    sum_p1, log_p0, _ = fixture_dual
    sum_path = tmp_path / "sum.json"
    log_path = tmp_path / "log.json"
    sum_path.write_text(json.dumps(sum_p1), encoding="utf-8")
    log_path.write_text(json.dumps(log_p0), encoding="utf-8")
    out = tmp_path / "dual.json"
    md = tmp_path / "dual.md"
    args = ["--sum-p1", str(sum_path), "--log-p0", str(log_path),
            "--out", str(out), "--md-out", str(md),
            "--ticket", "4,9,17,24,29,42"]
    main(args)
    result = json.loads(out.read_text(encoding="utf-8"))
    assert len(result["ticket_examples"]) == 1
    assert len(result["input_sum_p1_sha256"]) == 64
    assert len(result["input_log_p0_sha256"]) == 64
    assert "[DUAL-P1 PASS]" in capsys.readouterr().out
    assert "Log Product bin-by-bin" in md.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        main(args)
