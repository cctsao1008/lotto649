"""v0.27 P1 exact conditional IID and no-lookahead regression tests."""
from __future__ import annotations

import csv
from itertools import combinations
from math import comb, isclose
from pathlib import Path

import pytest

from lotto7x7.distribution_repair_v027 import baseline, exact_theory, squared_l2_probability
from lotto7x7.distribution_repair_scoring_v027 import (
    SCHEMA, conditional_repair, first_ticket_with_sum, score_p0, score_ticket,
    markdown_report, main,
)


def test_synthetic_exact_gain_against_independent_brute_force():
    history = [1, 2, 0, 3, 1]
    theory = [1, 3, 2, 5, 1]
    report = conditional_repair(history, theory)
    before = squared_l2_probability(history, theory)
    assert report["loss_before"] == before
    for i, row in enumerate(report["rows_by_index"]):
        augmented = history.copy()
        augmented[i] += 1
        after = squared_l2_probability(augmented, theory)
        assert row["loss_after"] == pytest.approx(after, abs=1e-15)
        assert row["gain"] == pytest.approx(before - after, abs=1e-15)
        expected_t = (sum(history) + 1) * theory[i] - sum(theory) * history[i]
        assert row["rank_integer"] == expected_t
    assert report == conditional_repair(history, theory)


def test_iid_null_weighted_exactly_not_uniform_over_sum_bins():
    history = [3, 0, 1, 0]
    theory = [1, 8, 2, 1]
    report = conditional_repair(history, theory)
    rows = report["rows_by_index"]
    null = report["iid_null"]
    total = sum(theory)
    expected = sum(r["gain"] * p for r, p in zip(rows, theory)) / total
    assert null["expected_gain"] == pytest.approx(expected, abs=1e-16)
    assert null["expected_gain_closed_form"] == pytest.approx(expected, rel=1e-12, abs=1e-14)
    assert null["stddev_gain"] ** 2 == pytest.approx(
        sum(p * (r["gain"] - expected) ** 2 for r, p in zip(rows, theory)) / total,
        abs=1e-17,
    )
    assert sum(atom["probability"] for atom in null["law_by_exact_rank_integer"]) == pytest.approx(1)
    for row in rows:
        t = row["rank_integer"]
        ge = sum(p for r, p in zip(rows, theory) if r["rank_integer"] >= t) / total
        lt = sum(p for r, p in zip(rows, theory) if r["rank_integer"] < t) / total
        eq = sum(p for r, p in zip(rows, theory) if r["rank_integer"] == t) / total
        assert row["iid_conditional"]["probability_greater_equal"] == pytest.approx(ge)
        assert row["iid_conditional"]["midrank_percentile"] == pytest.approx(lt + eq / 2)


def test_exact_integer_ties_not_floating_point_ties():
    x = conditional_repair([1, 0, 0], [1, 1, 1])
    a, b, c = x["rows_by_index"]
    assert b["rank_integer"] == c["rank_integer"]
    assert b["rank"] == c["rank"] == 1
    assert a["rank"] == 3
    assert b["iid_conditional"]["probability_equal"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("sum_value", [21, 22, 37, 70, 100, 150, 200, 250, 278, 279])
def test_lexicographically_first_representative(sum_value):
    result = first_ticket_with_sum(sum_value)
    assert len(result) == 6 and len(set(result)) == 6
    assert tuple(sorted(result)) == result
    assert 1 <= min(result) <= max(result) <= 49
    assert sum(result) == sum_value


@pytest.mark.parametrize("sum_value", [20, 280, 150.5, True, "150", None])
def test_bad_representative_sum(sum_value):
    with pytest.raises(ValueError):
        first_ticket_with_sum(sum_value)


def test_representatives_agree_with_independent_exhaustive_small_prefix():
    assert first_ticket_with_sum(21) == (1, 2, 3, 4, 5, 6)
    assert first_ticket_with_sum(279) == (44, 45, 46, 47, 48, 49)
    # For near-minimum sums the independent enumeration remains tiny.
    for wanted in range(21, 35):
        first = next(x for x in combinations(range(1, 50), 6) if sum(x) == wanted)
        assert first_ticket_with_sum(wanted) == first


def test_bad_histogram_inputs():
    for observed, theory in (([], []), ([1], []), ([0], [1]), ([1], [0]),
                             ([1], [1, 2]), ([-1], [1]), ([1.5], [1]), ([True], [1])):
        with pytest.raises(ValueError):
            conditional_repair(observed, theory)


ROWS = [
    ("2026/10/30", 115000101, 4, 22, 25, 30, 34, 37, 48),
    ("2026/11/03", 115000102, 2, 23, 24, 25, 29, 39, 31),
    ("2026/11/06", 115000103, 5, 9, 26, 32, 41, 48, 23),
    ("2026/11/10", 115000104, 1, 2, 3, 4, 5, 6, 7),
]


def _write_history(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        out = csv.writer(file)
        out.writerow(["date", "draw_id", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"])
        out.writerows(rows)


def test_p0_integration_ties_scores_and_future_exclusion(tmp_path):
    f = tmp_path / "lotto649_2026.csv"
    _write_history(f, ROWS[:3])
    original = baseline(tmp_path, cutoff="2026-11-06")
    result = score_p0(original)
    assert result["schema"] == SCHEMA
    assert result["source"]["historical_draws"] == 3
    assert result["source"]["last_draw_id"] == 115000103
    assert result["baseline_loss"] == original["observed"]["squared_l2_probability"]
    assert len(result["sums"]) == 259
    assert result["iid_null"]["expected_gain"] != 0
    assert sum(v["theory_combinations"] for v in result["sums"]) == comb(49, 6)
    ordered = sorted(result["sums"], key=lambda r: (-r["rank_integer"], r["sum"]))
    assert [r["sum"] for r in result["ranking"]["top"]] == [r["sum"] for r in ordered[:10]]
    assert [r["sum"] for r in result["ranking"]["bottom"]] == [r["sum"] for r in ordered[-10:][::-1]]
    assert all(r["gain"] == pytest.approx(r["loss_before"] - r["loss_after"], abs=1e-15)
               for r in result["sums"])

    # Same-sum tickets are EXACTLY tied, independent of individual numbers.
    t1 = score_ticket(result, (10, 15, 20, 25, 35, 45))
    t2 = score_ticket(result, (1, 16, 25, 30, 35, 43))
    assert t1["sum"] == t2["sum"] == 150
    assert t1["gain"] == t2["gain"]
    assert t1["rank"] == t2["rank"]
    with pytest.raises(ValueError):
        score_ticket(result, (1, 1, 2, 3, 4, 5))

    _write_history(f, ROWS)
    frozen = score_p0(baseline(tmp_path, cutoff="2026-11-06"))
    assert frozen["sums"] == result["sums"]
    assert frozen["iid_null"] == result["iid_null"]
    assert frozen["source"] == result["source"]
    assert "Highest hypothetical repair scores" in markdown_report(result)
    assert "No P2 four-ticket optimization" in markdown_report(result)


def test_p0_integrity_checks(tmp_path):
    f = tmp_path / "lotto649_2026.csv"
    _write_history(f, ROWS[:3])
    report = baseline(tmp_path, cutoff="2026-11-06")
    import copy
    for mutation in (
        lambda x: x["metric"].update(name="pearson"),
        lambda x: x["theory"].update(checksum_sha256="broken"),
        lambda x: x["observed"]["counts"].__setitem__(0, 19),
        lambda x: x["source"].update(analysis_input_sha256="broken"),
        lambda x: x["source"].update(last_draw_date="2026-11-10"),
        lambda x: x["observed"].update(squared_l2_probability=0.0),
    ):
        bad = copy.deepcopy(report)
        mutation(bad)
        with pytest.raises(ValueError):
            score_p0(bad)


def test_invalid_top_k(tmp_path):
    f = tmp_path / "lotto649_2026.csv"
    _write_history(f, ROWS[:3])
    source = baseline(tmp_path, cutoff="2026-11-06")
    for k in (0, 260, -2, True):
        with pytest.raises(ValueError):
            score_p0(source, top_k=k)


def test_cli_generates_both_reports_without_overwrite(tmp_path, capsys):
    folder = tmp_path / "dataset"
    folder.mkdir()
    _write_history(folder / "lotto649_2026.csv", ROWS[:3])
    p0 = baseline(folder, cutoff="2026-11-06")
    import json
    input_file = tmp_path / "baseline.json"
    input_file.write_text(json.dumps(p0), encoding="utf-8")
    path = tmp_path / "p1.json"
    md = tmp_path / "p1.md"
    args = ["--p0", str(input_file), "--out", str(path), "--md-out", str(md),
            "--ticket", "10,15,20,25,35,45", "--top-k", "5"]
    main(args)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert len(saved["ranking"]["top"]) == 5
    assert len(saved["ticket_examples"]) == 1
    assert saved["ticket_examples"][0]["sum"] == 150
    assert len(saved["input_p0_file_sha256"]) == 64
    assert md.exists() and "IID P(G > 0)" in md.read_text(encoding="utf-8")
    assert "[P1 PASS]" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        main(args)
