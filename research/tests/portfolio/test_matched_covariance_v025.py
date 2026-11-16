"""Regression checks for the matched covariance comparison."""
import json
from pathlib import Path

import numpy as np
import pytest

from research.lotto7x7 import matched_covariance_v025 as matched
from research.lotto7x7.shape_deficit_v025 import load_exact_baseline


def test_exact_moments_match_legacy_registry():
    baseline, _ = load_exact_baseline(matched.FIXTURE)
    names, theory, _ = matched.registry(baseline)
    moments = json.loads(matched.EXACT.read_text(encoding="utf-8"))
    covariance = matched.population_covariance(moments, names, theory)

    assert covariance.shape == (163, 163)
    assert covariance[49, 49] == 1075.0
    np.testing.assert_array_equal(covariance, covariance.T)
    np.testing.assert_allclose(covariance[:, :49].sum(axis=1), 0, atol=1e-12)

    moments["feature_registry"][:2] = reversed(moments["feature_registry"][:2])
    with pytest.raises(ValueError, match="feature registry"):
        matched.population_covariance(moments, names, theory)


def test_paired_benchmark_reproduces_legacy_and_contract(monkeypatch):
    monkeypatch.setattr(matched, "NULL_REPLICATES", 2)
    report = matched.benchmark(Path("data/csv"), matched.FIXTURE, matched.EXACT)
    assert report["history"]["draws"] == 2497
    assert report["history"]["last_draw"] == 115000103
    assert report["model_diagnostics"]["exact_rank_at_1e_10"] == 142
    assert len(report["model_diagnostics"]["sample_zero_variance_features"]) == 18

    arms = report["observed"]["arms"]
    legacy = arms["legacy_mc"]
    assert legacy["reference_minus_optimized"] == pytest.approx(0.00013428910911708614)
    assert legacy["selected_tickets"] == [
        [6, 9, 17, 25, 36, 44],
        [7, 12, 18, 29, 45, 47],
        [10, 13, 16, 19, 27, 42],
        [24, 28, 32, 37, 39, 48],
    ]
    assert report["observed"]["matched_mc_minus_exact"] == 0
    assert arms["matched_mc"]["selected_tickets"] == arms["exact"]["selected_tickets"]
    assert report["config"]["exact_main_ball_coverage_counts"] == {
        "3": 1040096, "4": 55216, "5": 1036, "6": 4,
    }
    assert [item["seed"] for item in report["null_by_seed"]] == [1650, 1651]
    for item in report["null_by_seed"]:
        assert set(item["arms"]) == set(matched.ARMS)
        assert all(arm["coverage_floor_pass"] for arm in item["arms"].values())
    assert sum(report["null_primary_outcomes"].values()) == 2


def test_exploratory_calibration_counts_ties():
    result = matched.calibrate(0, [-1, -1e-13, 0, 1])
    assert result["null_at_least_observed"] == 3
    assert result["exploratory_upper_tail_p"] == 0.8
    assert result["p_resolution"] == 0.2
    lo, hi = result["tail_proportion_wilson_95_ci"]
    assert lo < 0.75 < hi
