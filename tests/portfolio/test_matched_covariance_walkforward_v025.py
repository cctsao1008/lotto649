"""Tests for chronological selection, bonus scoring, and paired replays."""
from pathlib import Path

import pytest

from lotto7x7 import matched_covariance_walkforward_v025 as wf
from lotto7x7.deficit_compensation_v025 import REFERENCE


def test_walkforward_training_excludes_holdout_and_scores_bonus(monkeypatch):
    records = [
        ("2026-01-01", 1, (4, 9, 1, 2, 3, 5)),
        ("2026-01-02", 2, (4, 9, 1, 2, 3, 5)),
        ("2026-01-03", 3, (4, 9, 1, 2, 3, 5)),
    ]
    bonuses = {row[:2]: 17 for row in records}
    alternate = (
        (19, 20, 21, 22, 23, 24),
        (25, 26, 27, 28, 29, 30),
        (31, 32, 33, 34, 35, 36),
        (37, 38, 39, 40, 41, 42),
    )
    called = []

    def select(train, arm, **kwargs):
        called.append((arm, tuple(row[1] for row in train)))
        return {"tickets": REFERENCE if arm == "exact" else alternate}

    monkeypatch.setattr(wf, "select_multivariate_portfolio", select)
    models = {arm: arm for arm in wf.ARMS}
    result = wf.walkforward(records, bonuses, models, train_min=1, horizon=1, fold_count=2)
    assert called == (
        [(arm, (1,)) for arm in wf.ARMS] + [(arm, (1, 2)) for arm in wf.ARMS]
    )
    assert result["aggregate"]["exact_minus_matched_mc"] == 2
    assert result["aggregate"]["paired"] == {
        "exact_only": 2, "matched_mc_only": 0, "both": 0, "neither": 0,
    }
    with pytest.raises(ValueError, match="chronological"):
        wf.walkforward(records[::-1], bonuses, models, train_min=1, horizon=1, fold_count=2)
    with pytest.raises(ValueError, match="full holdout"):
        wf.walkforward(records, bonuses, models, train_min=1, horizon=2, fold_count=2)


def test_two_real_pipeline_iid_replays_match_legacy_and_fold_counts(monkeypatch):
    monkeypatch.setattr(wf, "NULL_REPLICATES", 2)
    result = wf.report(Path("data/csv"), wf.FIXTURE, wf.EXACT)
    aggregate = result["observed"]["aggregate"]
    assert aggregate["winning_draws"]["history_blind"] == 22
    assert aggregate["winning_draws"]["legacy_mc"] == 25
    assert aggregate["tested_draws"] == 200
    assert len(result["observed"]["folds"]) == 8
    assert result["null_calibration"]["replicates"] == 2
    for item in result["null_calibration"]["null_by_seed"]:
        assert sum(item["paired"].values()) == 200
        assert item["tested_draws"] == 200
        assert set(item["winning_draws"]) == set(wf.WIN_ARMS)
