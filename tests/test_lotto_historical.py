"""Phase 5: historical ingestion, chronological isolation, and scoring tests.

Tests use fabricated archive records; they do not need the official file.
All source-code comments and runtime messages are English-only.
"""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from lotto_historical import (
    CANDIDATES, choose_from_validation, lag_log_probs,
    load_official_main_draws, moving_block_ci, run_historical,
)
from lotto_neural import IID_LOGP
from lotto_synthetic import generate
from lotto_temporal import fit_beta, lag_mean_log_score


def make_official_fixture(path: Path, draws: np.ndarray) -> None:
    """Construct an official-layout fixture with unsorted main numbers."""
    dtype = np.dtype([
        ("draw_id", "<u4"), ("date", "<u4"),
        ("numbers", "u1", (6,)), ("special", "u1"),
    ])
    records = np.empty(len(draws), dtype=dtype)
    for i, row in enumerate(draws):
        numbers = list(map(int, row))
        records[i]["draw_id"] = 100000001 + i
        dt = date(2018, 1, 1) + timedelta(days=3 * i)
        records[i]["date"] = int(dt.strftime("%Y%m%d"))
        records[i]["numbers"] = row[::-1]
        records[i]["special"] = next(v for v in range(1, 50) if v not in numbers)
    np.save(path, records, allow_pickle=False)


class HistoricalValidationTests(unittest.TestCase):
    def setUp(self):
        self.draws = generate(count=120, mode="iid", seed=2026)[0]

    def test_official_records_are_sorted_without_using_special(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.npy"
            make_official_fixture(path, self.draws)
            numbers, dates, ids = load_official_main_draws(path)
            np.testing.assert_array_equal(numbers, self.draws)
            self.assertTrue(np.all(np.diff(dates) > 0))
            self.assertEqual(len(np.unique(ids)), 120)

    def test_reject_invalid_special_and_duplicate_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.npy"
            make_official_fixture(path, self.draws)
            fixture = np.load(path, allow_pickle=False)
            fixture["special"][0] = fixture["numbers"][0][0]
            np.save(path, fixture, allow_pickle=False)
            with self.assertRaises(ValueError):
                load_official_main_draws(path)
            make_official_fixture(path, self.draws)
            fixture = np.load(path, allow_pickle=False)
            fixture["date"][4] = fixture["date"][3]
            np.save(path, fixture, allow_pickle=False)
            with self.assertRaises(ValueError):
                load_official_main_draws(path)

    def test_validation_selection_tie_favors_iid(self):
        scores = {"iid": IID_LOGP, "static": IID_LOGP,
                  "lag": IID_LOGP, "mlp": IID_LOGP}
        self.assertEqual(choose_from_validation(scores), "iid")
        scores["lag"] = IID_LOGP + 0.001
        self.assertEqual(choose_from_validation(scores), "lag")
        with self.assertRaises(ValueError):
            choose_from_validation({"iid": IID_LOGP})

    def test_block_bootstrap_reproducible_and_handles_constant_series(self):
        values = np.array([0.02] * 30)
        ci = moving_block_ci(values, samples=100, block_size=7)
        self.assertAlmostEqual(ci[0], 0.02)
        self.assertAlmostEqual(ci[1], 0.02)
        noise = np.linspace(-0.2, 0.1, 53)
        self.assertEqual(moving_block_ci(noise, samples=80, seed=101),
                         moving_block_ci(noise, samples=80, seed=101))
        with self.assertRaises(ValueError):
            moving_block_ci(np.array([]))

    def test_lag_per_draw_scores_match_reference(self):
        draws = generate(count=120, mode="iid", seed=17)[0].astype(np.int64)
        fitted_beta = fit_beta(draws, train_end=72)
        idx = np.arange(96, 120)
        p = lag_log_probs(draws, idx, fitted_beta)
        self.assertAlmostEqual(p.mean(), lag_mean_log_score(draws, idx, fitted_beta), places=12)

    def test_complete_historical_replay_is_read_only_and_exploratory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.npy"
            make_official_fixture(path, self.draws)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            report = run_historical(source=path, seed=11, window=3,
                                    hidden=8, epochs=2, patience=2,
                                    batch_size=32, bootstrap_samples=30)
            after = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(before, after)
            self.assertEqual(report["source_sha256"], before)
            self.assertEqual(report["research_status"],
                             "EXPLORATORY_HISTORICAL_REUSE_NOT_PROSPECTIVE_BLIND")
            self.assertEqual(report["rows"], 120)
            self.assertEqual(report["splits"]["train"]["indices"], [3, 71])
            self.assertEqual(report["splits"]["validation"]["indices"], [72, 95])
            self.assertEqual(report["splits"]["test"]["indices"], [96, 119])
            self.assertEqual(set(report["test_models"]), set(CANDIDATES))
            self.assertEqual(report["validation_selected"],
                             choose_from_validation(report["validation_scores"]))
            self.assertTrue(all(np.isfinite(v["test_lift_nats"])
                                for v in report["test_models"].values()))
            self.assertEqual(sum(x["draws"] for x in report["test_by_year"]), 24)


if __name__ == "__main__":
    unittest.main()
