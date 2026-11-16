"""Regression tests for finite-sample multivariate diagnostics."""
import json
from math import comb, log2
from pathlib import Path

import numpy as np
import pytest

from research.lotto7x7 import multivariate_calibration_v025 as calibration


def test_gap_distribution_and_entropy_anchors():
    weights = [comb(49 - gap, 5) for gap in range(1, 45)]
    assert sum(weights) == calibration.U
    assert calibration.entropy([0, 1, 1, 0]) == pytest.approx(1)
    assert calibration.entropy(np.ones(49)) == pytest.approx(log2(49))


def test_small_replay_preserves_blocks_and_multiplicity(monkeypatch):
    monkeypatch.setattr(calibration, "NULL_REPLAYS", 3)
    monkeypatch.setattr(calibration, "VALIDATION_REPLAYS", 2)
    result = calibration.report(Path("data/csv"), calibration.FIXTURE, calibration.EXACT)
    assert result["tests_in_bonferroni_family"] == 40
    segments = result["endpoints"]
    assert segments["full"]["draws"] == 2497
    assert sum(segment["draws"] for name, segment in segments.items() if name != "full") == 2497
    assert result["primary_size_validation"]["replays"] == 2
    for segment in segments.values():
        for row in segment["endpoints"].values():
            assert row["bonferroni_p"] == min(1, 40 * row["exploratory_p"])
            assert row["upper_tail_count"] <= 3
    pair = segments["full"]["endpoints"]["pairs"]
    assert pair["population_entropy_bits"] == pytest.approx(log2(comb(49, 2)))
    assert pair["estimated_plugin_bias_bits"] <= 0


def test_archived_calibration_can_resolve_familywise_threshold():
    path = Path("research/artifacts/v025_exact_covariance/v025_multivariate_calibration_20261116.json")
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["config"]["null_replays"] == 1999
    assert 2 * result["config"]["p_resolution"] * result["tests_in_bonferroni_family"] <= 0.05
    assert result["primary_size_validation"]["replays"] == 199
    assert len(result["ranked_marginal_residuals"]) == 163
