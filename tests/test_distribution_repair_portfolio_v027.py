"""v0.27 P2: four-disjoint portfolio, certificates, exact official awards."""
from __future__ import annotations

import copy
import csv
import json
from math import comb
from pathlib import Path

import pytest

from lotto7x7.coverage_v06 import exact_coverage
from lotto7x7.deficit_compensation_v025 import REFERENCE
from lotto7x7.distribution_repair_v027 import baseline
from lotto7x7.distribution_repair_scoring_v027 import score_p0
from lotto7x7.distribution_repair_portfolio_v027 import (
    SCHEMA, _try_target_sums, _sample_fallback, markdown_report, main,
    optimize, validate_four_disjoint, verify_p1,
)
from lotto7x7.official_prize_v026 import PRIZE_ORDER, official_prize_coverage


@pytest.fixture
def p1_history(tmp_path: Path):
    folder = tmp_path / "data"
    folder.mkdir()
    path = folder / "lotto649_2026.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["date", "draw_id", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"])
        writer.writerows([
            ("2026/10/30", 115000101, 4, 22, 25, 30, 34, 37, 48),
            ("2026/11/03", 115000102, 2, 23, 24, 25, 29, 39, 31),
            ("2026/11/06", 115000103, 5, 9, 26, 32, 41, 48, 23),
        ])
    return score_p0(baseline(folder, cutoff="2026-11-06"))


def test_disjoint_witness_exact_four_152_and_all_numbers_distinct():
    tickets = [
        (6, 15, 23, 29, 39, 40),
        (5, 7, 9, 38, 46, 47),
        (4, 21, 27, 28, 31, 41),
        (11, 12, 19, 32, 35, 43),
    ]
    chosen = validate_four_disjoint([sorted(t) for t in tickets])
    assert len({x for t in chosen for x in t}) == 24
    assert [sum(t) for t in chosen] == [152] * 4
    assert all(len(t) == 6 for t in chosen)


def test_seeded_witness_algorithm_reproducible_primary_and_diverse():
    targets = [(152,) * 4, (152, 116, 153, 164)]
    for sums in targets:
        first, diag = _try_target_sums(sums, seed=649, restarts=10, trials=1000)
        assert first is not None
        assert [sum(t) for t in first] == list(sums)
        assert len({x for row in first for x in row}) == 24
        assert (first, diag) == _try_target_sums(sums, seed=649, restarts=10, trials=1000)


@pytest.mark.parametrize("bad", [
    [], [[1, 2, 3, 4, 5, 6]] * 4,
    [[1, 2, 3, 4, 5, 6], [1, 7, 8, 9, 10, 11],
     [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23]],
    [[2, 1, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12],
     [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]],
    [[1, 2, 3, 4, 5, 50], [7, 8, 9, 10, 11, 12],
     [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]],
])
def test_reject_invalid_portfolios(bad):
    with pytest.raises(ValueError):
        validate_four_disjoint(bad)


def test_reject_tampered_p1_histogram_metric_and_rank(p1_history):
    assert len(verify_p1(p1_history)) == 259
    for mutation in (
        lambda x: x["sums"][0].update(rank_integer=123456789),
        lambda x: x["sums"][1].update(gain=10),
        lambda x: x["sums"][0].update(observed_count=99),
        lambda x: x["metric"].update(name="pearson"),
        lambda x: x["source"].update(analysis_input_sha256="invalid"),
        lambda x: x["theory"].update(checksum_sha256="invalid"),
        lambda x: x["sums"].reverse(),
    ):
        changed = copy.deepcopy(p1_history)
        mutation(changed)
        with pytest.raises(ValueError):
            verify_p1(changed)


def test_optimizer_valid_exact_coverage_parity_and_objective(p1_history):
    result = optimize(p1_history, seed=649, restarts=30, trials=1000, samples=100)
    assert result["schema"] == SCHEMA
    assert result["source"]["cutoff"] == "2026-11-06"
    assert result["optimization"]["no_joint_append_to_history"]
    reference = official_prize_coverage(REFERENCE)
    ref_main = exact_coverage(REFERENCE)
    for key in ("primary", "distinct_sums_sensitivity"):
        item = result["optimization"][key]
        tickets = validate_four_disjoint(item["tickets"])
        assert len({v for t in tickets for v in t}) == 24
        gains = [p1_history["sums"][sum(ticket) - 21]["gain"] for ticket in tickets]
        ranks = [p1_history["sums"][sum(ticket) - 21]["rank_integer"] for ticket in tickets]
        assert item["objective"]["gain_sum"] == pytest.approx(sum(gains), abs=1e-16)
        assert item["objective"]["exact_integer_objective"] == sum(ranks)
        assert item["objective"]["exact_integer_objective"] <= item["objective"]["exact_integer_upper_bound"]
        if item["certificate"]["globally_optimal_proven"]:
            assert item["objective"]["gap_in_exact_integer_units"] == 0
            assert item["certificate"]["status"] == "GLOBAL_OPTIMUM_CERTIFIED"
        else:
            assert item["certificate"]["status"] == "HEURISTIC_UNCERTIFIED"
        assert item["coverage"]["official_probability"] == reference["any_prize_probability"]
        assert item["coverage"]["exact_official_eight_tier_parity"]
        assert item["coverage"]["exact_main_ball_coverage_parity"]
        got = official_prize_coverage(tickets)
        for tier in PRIZE_ORDER:
            assert got["any_ticket_by_tier"][tier]["outcomes"] == reference["any_ticket_by_tier"][tier]["outcomes"]
        assert exact_coverage(tickets)["max_hits_counts_0_to_6"] == ref_main["max_hits_counts_0_to_6"]
        assert got["total_outcomes"] == comb(49, 6) * 43
    assert result["optimization"]["distinct_sums_sensitivity"]["unique_ticket_sums"] == 4
    assert result["optimization"]["diversity_tradeoff"]["coverage_probability_change"] == 0


def test_determinism_different_sums_and_fallback_labels(p1_history):
    a = optimize(p1_history, seed=649, restarts=30, trials=1000, samples=75)
    b = optimize(p1_history, seed=649, restarts=30, trials=1000, samples=75)
    assert a == b
    result = optimize(p1_history, seed=649, restarts=0, samples=30)
    for name in ("primary", "distinct_sums_sensitivity"):
        portfolio = result["optimization"][name]
        assert portfolio["certificate"]["globally_optimal_proven"] == (
            portfolio["objective"]["exact_integer_objective"] ==
            portfolio["objective"]["exact_integer_upper_bound"]
        )
        assert portfolio["certificate"]["search"]["random_portfolios"] == 30
        assert len({n for ticket in portfolio["tickets"] for n in ticket}) == 24


def test_invalid_optimizer_budgets(p1_history):
    for kw in ({"seed": -1}, {"restarts": -1}, {"trials": 0},
               {"samples": 0}, {"seed": True}):
        with pytest.raises(ValueError):
            optimize(p1_history, **kw)


def test_cli_reports_and_never_overwrites(tmp_path, p1_history, capsys):
    inp = tmp_path / "p1.json"
    inp.write_text(json.dumps(p1_history), encoding="utf-8")
    out = tmp_path / "p2.json"
    md = tmp_path / "p2.md"
    args = ["--p1", str(inp), "--out", str(out), "--md-out", str(md),
            "--seed", "649", "--restarts", "20", "--trials", "1000"]
    main(args)
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["source"]["historical_draws"] == 3
    assert len(result["input_p1_file_sha256"]) == 64
    assert "## Diversity trade-off" in md.read_text(encoding="utf-8")
    assert "[P2 PRIMARY]" in capsys.readouterr().out
    assert "no four-draw append" in markdown_report(result)
    with pytest.raises(FileExistsError):
        main(args)
