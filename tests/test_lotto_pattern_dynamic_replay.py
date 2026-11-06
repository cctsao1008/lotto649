"""End-to-end Phase 7C dynamic-image protocol tests on synthetic archives."""
from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from lotto_image_repr import encode_draws
from lotto_pattern_dynamics import METHODS, DynamicPatternLibrary
from lotto_pattern_dynamic_replay import (
    bootstrap_mean_ci, export_csv, run_dynamic_replay, score_period,
    shuffle_training_by_year,
)


def write_synthetic_archive(path: Path, draws: np.ndarray) -> None:
    """Write data-layout-compatible artificial draws; never real lottery records."""
    dtype = np.dtype([
        ("date", "<u4"), ("draw_id", "<u4"),
        ("numbers", "u1", (6,)), ("special", "u1"),
    ])
    records = np.zeros(len(draws), dtype=dtype)
    for i, six in enumerate(draws):
        day = date(2018, 1, 1) + timedelta(days=3 * i)
        records[i]["date"] = int(day.strftime("%Y%m%d"))
        records[i]["draw_id"] = 107000001 + i
        records[i]["numbers"] = six[::-1]
        records[i]["special"] = next(x for x in range(1, 50) if x not in six)
    np.save(path, records, allow_pickle=False)


class DynamicReplayTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "synthetic_draws.npy"
        rng = np.random.default_rng(999)
        self.draws = np.stack([np.sort(rng.choice(49, 6, replace=False) + 1)
                               for _ in range(120)]).astype(np.int64)
        write_synthetic_archive(self.path, self.draws)

    def experiment(self, **kwargs):
        config = {
            "source": self.path, "year": 2018, "seed": 101,
            "train_end": 72, "validation_end": 96,
            "window": 3, "neighbors": 3, "last_weight": 0.2,
            "null_samples": 2, "validation_null_samples": 2,
            "permutations": 9, "bank_shuffles": 1,
            "bootstrap_repeats": 20, "bootstrap_block": 4,
        }
        config.update(kwargs)
        return run_dynamic_replay(**config)

    def test_circular_bootstrap_reproducible_and_finite(self):
        scores = np.linspace(-1.0, 1.0, 24)
        a = bootstrap_mean_ci(scores, seed=3, repeats=30, block=4)
        b = bootstrap_mean_ci(scores, seed=3, repeats=30, block=4)
        self.assertEqual(a, b)
        self.assertLessEqual(a[0], a[1])
        self.assertTrue(np.isfinite(a).all())

    def test_year_preserving_bank_shuffle_never_modifies_source(self):
        masks = encode_draws(self.draws)
        dates = np.asarray([20180101] * 60 + [20190101] * 60)
        shuffled = shuffle_training_by_year(
            masks, dates, train_end=72, seed=42)
        self.assertEqual(shuffled.shape, (72, 7, 7))
        for year in (2018, 2019):
            positions = np.flatnonzero(dates[:72] // 10000 == year)
            self.assertEqual(
                sorted(tuple(x.ravel()) for x in shuffled[positions]),
                sorted(tuple(x.ravel()) for x in masks[positions]))
        np.testing.assert_array_equal(masks, encode_draws(self.draws))

    def test_main_replay_returns_dynamic_similarity_and_diversity(self):
        file_hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        report = self.experiment()
        self.assertEqual(report["source_sha256"], file_hash)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(),
                         file_hash)
        self.assertEqual(report["evaluated_draws"], 24)
        self.assertEqual(report["split"]["train"], [3, 71])
        self.assertEqual(report["split"]["validation"], [72, 95])
        self.assertEqual(report["split"]["test"], [96, 119])
        self.assertEqual(report["primary_objective"],
                         "7x7_pattern_similarity_above_conditional_iid")
        self.assertIn(report["validation_selected"], METHODS)
        self.assertEqual(set(report["evaluation_summary"]), set(METHODS))
        self.assertEqual(len(report["individual_draws"]), 24)
        for method in METHODS:
            item = report["evaluation_summary"][method]
            self.assertEqual(item["draws"], 24)
            self.assertTrue(np.isfinite(item["mean_chance_adjusted"]["composite"]))
            self.assertEqual(item["diversity"]["draws"], 24)
            self.assertEqual(len(item["bootstrap_95_percent_interval"]), 2)
            self.assertTrue(0 <= item["diversity"]["top_pattern_fraction"] <= 1)

    def test_forecast_records_are_valid_and_separate_similarity_from_six_hit(self):
        report = self.experiment(bank_shuffles=0)
        for draw in report["individual_draws"]:
            self.assertEqual(sum(row.count("#") for row in draw["actual_pattern"]), 6)
            for method in METHODS:
                pred = draw["methods"][method]
                self.assertEqual(sum(row.count("#") for row in pred["predicted_pattern"]), 6)
                self.assertEqual(len(pred["predicted_main_six"]), 6)
                self.assertTrue(0 <= pred["raw_similarity"]["composite"] <= 1)
                self.assertTrue(np.isfinite(
                    pred["adjusted_similarity"]["composite"]))
        self.assertIsNone(
            report["training_transition_shuffle"]["shuffled_train_mean_adjusted_score"])

    def test_shuffled_context_and_train_bank_controls_are_reported(self):
        result = self.experiment()
        self.assertEqual(result["training_transition_shuffle"]["count"], 1)
        self.assertEqual(len(result["training_transition_shuffle"]["scores"]), 1)
        self.assertEqual(set(result["past_frame_order_scramble"]),
                         {"centroid_flow", "analog_fusion"})
        for row in result["past_frame_order_scramble"].values():
            self.assertTrue(np.isfinite(row["original_minus_scrambled"]))

    def test_synthetic_cycle_analog_recovers_all_images_when_one_neighbor(self):
        cycle = np.asarray([
            [1, 2, 3, 4, 5, 6],
            [8, 9, 10, 11, 12, 13],
            [15, 16, 17, 18, 19, 20],
            [22, 23, 24, 25, 26, 27],
        ])
        write_synthetic_archive(self.path, np.tile(cycle, (30, 1)))
        result = self.experiment(neighbors=1, last_weight=0.0,
                                 bank_shuffles=0)
        self.assertEqual(
            result["evaluation_summary"]["analog_fusion"][
                "exact_six_matches_secondary"], 24)
        self.assertAlmostEqual(
            result["evaluation_summary"]["analog_fusion"][
                "mean_raw"]["composite"], 1.0, places=12)

    def test_small_csv_contains_all_dynamic_methods(self):
        report = self.experiment(bank_shuffles=0)
        dest = self.path.parent / "phase7c.csv"
        export_csv(report, dest)
        with dest.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 24 * len(METHODS))
        self.assertEqual(set(r["method"] for r in rows), set(METHODS))
        for row in rows:
            self.assertEqual(len(row["predicted_pattern"].split("/")), 7)
            self.assertIn("adjusted_shape_d4", row)

    def test_same_target_mutation_leaves_score_period_predictions_unchanged(self):
        actual_images = encode_draws(self.draws)
        lib = DynamicPatternLibrary(actual_images, train_end=72,
                                    window=3, neighbors=3)
        t = 96
        original_prediction = lib.predict(t, method="analog_fusion")
        adjusted_images = actual_images.copy()
        adjusted_images[t] = encode_draws([1, 2, 3, 4, 5, 6])
        other_lib = DynamicPatternLibrary(adjusted_images, train_end=72,
                                          window=3, neighbors=3)
        np.testing.assert_array_equal(
            original_prediction, other_lib.predict(t, method="analog_fusion"))
        summary, audit, _ = score_period(
            lib, actual_images, np.asarray([t, t + 1]),
            methods=("analog_fusion",), null_samples=2, seed=13)
        self.assertEqual(summary["analog_fusion"]["draws"], 2)
        self.assertEqual(audit, [])

    def test_invalid_configurations_fail_closed(self):
        with self.assertRaises(ValueError):
            self.experiment(null_samples=1)
        with self.assertRaises(ValueError):
            self.experiment(permutations=1)
        with self.assertRaises(ValueError):
            self.experiment(year=2025)
        with self.assertRaises(ValueError):
            self.experiment(bank_shuffles=-1)
        with self.assertRaises(ValueError):
            bootstrap_mean_ci(np.asarray([1.0]), seed=3)


if __name__ == "__main__":
    unittest.main()
