"""v0.26 P2: enforce identical official coverage and leakage-free replay."""
from random import Random

import pytest

import lotto7x7.strategy_ab_v026 as mod
from lotto7x7.deficit_compensation_v025 import REFERENCE
from lotto7x7.official_prize_v026 import official_prize_coverage


def test_a_is_disjoint_and_relabeling_parity():
    a = mod.disjoint_portfolio(649)
    assert a == mod.disjoint_portfolio(649)
    assert a != mod.disjoint_portfolio(650)
    assert len({n for row in a for n in row}) == 24
    parity = mod.assert_official_coverage_parity(a, REFERENCE)
    assert parity["exact_official_prize_event_counts_equal"]
    assert parity["exact_main_ball_coverage_equal"]
    assert parity["any_official_prize_probability"] == pytest.approx(
        0.1227314589619088)


def test_coverage_constraint_cannot_be_weakened():
    a = mod.disjoint_portfolio(649)
    overlapping = list(REFERENCE)
    overlapping[1] = (4, 13, 14, 27, 28, 34)
    with pytest.raises(AssertionError):
        mod.assert_official_coverage_parity(a, overlapping)
    with pytest.raises(ValueError):
        mod.disjoint_portfolio(-1)


def test_official_bonus_used_in_realized_award():
    a = tuple(range(1, 7))
    assert mod.draw_awards([a], (1, 2, 10, 11, 12, 13), 3)["tiers"] == ["seventh"]
    assert mod.draw_awards([a], (1, 2, 10, 11, 12, 13), 14)["tiers"] == []
    assert mod.draw_awards([a], (1, 2, 3, 11, 12, 13), 14)["tiers"] == ["regular"]
    assert mod.draw_awards([a], (1, 2, 3, 11, 12, 13), 4)["tiers"] == ["sixth"]


def test_exact_paired_discordance():
    assert mod.paired_exact_p(0, 0) == 1.0
    assert mod.paired_exact_p(3, 0) == 0.25
    assert mod.paired_exact_p(2, 2) == 1.0


def test_chronological_nonoverlap_and_no_future_training(monkeypatch):
    from datetime import date, timedelta
    rng = Random(11)
    records = []
    bonuses = {}
    start = date(2010, 1, 1)
    for i in range(280):
        drawn = rng.sample(range(1, 50), 7)
        main = tuple(sorted(drawn[:6]))
        current = ((start + timedelta(days=i)).isoformat(), i + 1, main)
        records.append(current)
        bonuses[(current[0], current[1])] = drawn[6]
    call_ends = []

    def selector(train, model, *, pool_size, seed, coverage_constraint):
        assert coverage_constraint == "reference-3plus"
        call_ends.append(train[-1][1])
        return {"tickets": [list(t) for t in REFERENCE]}

    monkeypatch.setattr(mod, "select_multivariate_portfolio", selector)
    report = mod.chronological_walkforward(
        records, bonuses, {}, train_min=256, horizon=12, stride=12,
        max_folds=2, selector_seed=649, pool_size=60)
    assert call_ends == [256, 268]
    assert report["aggregate"]["draws"] == 24
    assert len(report["folds"]) == 2
    assert report["folds"][0]["test_begin_id"] == 257
    assert report["folds"][1]["test_begin_id"] == 269
    assert report["aggregate"]["a_wins"] <= 24
    assert report["aggregate"]["b_wins"] <= 24
    assert report["iid_theoretical_any_prize_probability_both"] == pytest.approx(
        official_prize_coverage(REFERENCE)["any_prize_probability"])
    assert report["exploratory_historical_not_prospective"]


def test_invalid_fold_specification():
    with pytest.raises(ValueError):
        mod.chronological_walkforward([(None, i, REFERENCE[0]) for i in range(270)],
                                      {}, {}, train_min=256, horizon=20, stride=10)


def test_bonus_csv_alignment(tmp_path):
    from datetime import date
    csv = tmp_path / "lotto649_2010.csv"
    csv.write_text("date,draw_id,n1,n2,n3,n4,n5,n6,bonus\n"
                   "2010-01-01,1,1,2,3,4,5,6,7\n", encoding="utf-8")
    records = [("2010-01-01", 1, (1, 2, 3, 4, 5, 6))]
    assert mod.checked_bonus_by_draw(tmp_path, records) == {
        ("2010-01-01", 1): 7}
    csv.write_text("date,draw_id,n1,n2,n3,n4,n5,n6,bonus\n"
                   "2010-01-01,1,1,2,3,4,5,6,6\n", encoding="utf-8")
    with pytest.raises(ValueError):
        mod.checked_bonus_by_draw(tmp_path, records)
