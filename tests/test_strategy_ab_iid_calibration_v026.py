"""v0.26 P3 pipeline-matched IID calibration regression tests."""
from datetime import date, timedelta
from random import Random

import pytest

import lotto7x7.strategy_ab_iid_calibration_v026 as cal
import lotto7x7.strategy_ab_v026 as ab
from lotto7x7.deficit_compensation_v025 import REFERENCE


def fake_timeline(n=270):
    base = date(2012, 1, 1)
    rng = Random(44)
    rec = []
    bonus = {}
    for i in range(n):
        drawn = rng.sample(range(1, 50), 7)
        day = (base + timedelta(days=i)).isoformat()
        rec.append((day, i + 1, tuple(sorted(drawn[:6]))))
        bonus[(day, i + 1)] = drawn[6]
    return rec, bonus


def test_independent_iid_full_draws_reproducible_and_valid():
    records, _ = fake_timeline(270)
    a, ba = cal.iid_full_history(records, seed=2649)
    b, bb = cal.iid_full_history(records, seed=2649)
    c, _ = cal.iid_full_history(records, seed=2650)
    assert a == b and ba == bb and a != c
    assert len(a) == len(records) and len(ba) == len(records)
    assert all(r[:2] == original[:2] for r, original in zip(a, records))
    assert all(len(set(r[2])) == 6 and
               ba[(r[0], r[1])] not in r[2] for r in a)
    assert any(r[2] != original[2] for r, original in zip(a, records))


def test_empirical_null_tails_include_observed_pseudocount():
    r = cal.empirical_calibration(3, (-3, 0, 1, 3, 4))
    assert r["p_empirical_upper_exploratory"] == pytest.approx(3/6)
    assert r["p_empirical_lower_exploratory"] == pytest.approx(5/6)
    assert r["p_empirical_two_sided_exploratory"] == pytest.approx(4/6)
    assert r["p_resolution"] == pytest.approx(1/6)
    with pytest.raises(ValueError):
        cal.empirical_calibration(3, ())
    with pytest.raises(ValueError):
        cal.iid_full_history([], seed=1)
    with pytest.raises(ValueError):
        cal.iid_full_history([("x", 1, ())], seed=-1)


def test_complete_synthetic_replay_retrains_and_preserves_folds(monkeypatch):
    records, observed_bonus = fake_timeline()
    calls = []
    def selector(train, model, *, pool_size, seed, coverage_constraint):
        calls.append((train[-1][1], train[0][2], len(train)))
        assert coverage_constraint == "reference-3plus"
        return {"tickets": [list(t) for t in REFERENCE]}
    monkeypatch.setattr(ab, "select_multivariate_portfolio", selector)
    report = cal.calibrate(records, observed_bonus, {},
                           null_replicates=2, null_seed=2649,
                           train_min=256, horizon=7, stride=7,
                           max_folds=2, pool_size=40)
    obs = report["observed"]
    assert obs["folds"] == 2 and obs["tested_draws"] == 14
    assert obs["b_minus_a"] == obs["b_wins"] - obs["a_wins"]
    assert len(report["null_replay_summaries"]) == 2
    assert report["null_calibration"]["replicates"] == 2
    assert report["null_calibration"]["p_resolution"] == pytest.approx(1/3)
    assert [x[0] for x in calls] == [256, 263, 256, 263, 256, 263]
    assert [x["history_seed"] for x in report["null_replay_summaries"]] == [2650, 2651]
    assert all(x["draws"] == 14 for x in report["null_replay_summaries"])
    assert report["strategy_design"]["theoretical_official_prize_coverage_locked"]
    assert "not prospectively" in str(report["limitations"]).lower() or report["limitations"]


def test_replay_does_not_use_observed_main_ball_outcomes(monkeypatch):
    records, bonuses = fake_timeline()
    calls = []
    def selector(train, model, *, pool_size, seed, coverage_constraint):
        calls.append(tuple(row[2] for row in train[:2]))
        return {"tickets": [list(t) for t in REFERENCE]}
    monkeypatch.setattr(ab, "select_multivariate_portfolio", selector)
    cal.calibrate(records, bonuses, {}, null_replicates=1, null_seed=987,
                  train_min=256, horizon=7, stride=7, max_folds=1,
                  pool_size=40)
    assert len(calls) == 2
    assert calls[0] != calls[1]  # independently generated null training history


def test_bad_replicates_rejected():
    records, bonuses = fake_timeline()
    for kwargs in ({"null_replicates": 0}, {"null_replicates": 1001},
                   {"null_seed": -2}):
        with pytest.raises(ValueError):
            cal.calibrate(records, bonuses, {}, **kwargs)
