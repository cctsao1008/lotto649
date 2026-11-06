"""Phase 7B train-only pattern libraries, causal replay and audit tests."""
from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from lotto_image_repr import encode_draws
from lotto_pattern_models import METHODS, PatternLibrary
from lotto_pattern_replay import export_csv, run_replay, split_indices


def write_fixture(path: Path, draws: np.ndarray) -> None:
    dtype = np.dtype([("date", "<u4"), ("draw_id", "<u4"),
                      ("numbers", "u1", (6,)), ("special", "u1")])
    arr = np.zeros(len(draws), dtype=dtype)
    for i, six in enumerate(draws):
        day = date(2018, 1, 1) + timedelta(days=3 * i)
        arr[i]["date"] = int(day.strftime("%Y%m%d"))
        arr[i]["draw_id"] = 107000001 + i
        arr[i]["numbers"] = six[::-1]
        arr[i]["special"] = next(x for x in range(1, 50) if x not in six)
    np.save(path, arr, allow_pickle=False)


class PatternReplayTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "synthetic_history.npy"
        rng = np.random.default_rng(2027)
        self.draws = np.stack([np.sort(rng.choice(49, 6, replace=False) + 1)
                               for _ in range(120)])
        write_fixture(self.path, self.draws)

    def evaluate(self, **kw):
        args = {
            "source": self.path, "year": 2018, "window": 3,
            "train_end": 72, "validation_end": 96,
            "clusters": 4, "null_samples": 3,
            "validation_null_samples": 2, "seed": 37,
        }
        args.update(kw)
        return run_replay(**args)

    def test_original_split_indices_are_stable(self):
        a, b, c = split_indices(120, window=3, train_end=72, validation_end=96)
        self.assertEqual((a[0], a[-1]), (3, 71))
        self.assertEqual((b[0], b[-1]), (72, 95))
        self.assertEqual((c[0], c[-1]), (96, 119))

    def test_train_only_members_and_cluster_transitions(self):
        lib = PatternLibrary(encode_draws(self.draws), train_end=72,
                             window=3, clusters=4)
        self.assertEqual(len(lib.bank_targets), 69)
        self.assertEqual(int(lib.bank_targets[-1]), 71)
        self.assertEqual(int(lib.cluster_sizes.sum()), 72)
        self.assertEqual(int(lib.transitions.sum()), 71)
        for cluster in lib.cluster_report():
            self.assertLess(sum(sum(row) for row in cluster["representative_image"]), 7)
            self.assertEqual(sum(sum(row) for row in cluster["representative_image"]), 6)

    def test_same_target_outcome_is_never_prediction_input(self):
        index = 96
        original = encode_draws(self.draws)
        changed_draws = self.draws.copy()
        changed_draws[index] = [1, 2, 3, 4, 5, 6]
        if np.array_equal(changed_draws[index], self.draws[index]):
            changed_draws[index] = [2, 3, 4, 5, 6, 7]
        modified = encode_draws(changed_draws)
        lib_a = PatternLibrary(original, train_end=72, window=3, clusters=4)
        lib_b = PatternLibrary(modified, train_end=72, window=3, clusters=4)
        for name in METHODS:
            np.testing.assert_array_equal(
                lib_a.predict(index, method=name),
                lib_b.predict(index, method=name))

    def test_replay_uses_similarity_as_primary_and_fixed_test_period(self):
        original_hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        r = self.evaluate()
        self.assertEqual(r["evaluated_draws"], 24)
        self.assertEqual(r["source_sha256"], original_hash)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(),
                         original_hash)
        self.assertEqual(r["split"]["evaluation"], [96, 119])
        self.assertIn(r["validation_selected"], METHODS)
        self.assertEqual(set(r["evaluation_summary"]), set(METHODS))
        self.assertEqual(r["primary_score"], "mean_chance_adjusted_composite")
        for name in METHODS:
            s = r["evaluation_summary"][name]
            self.assertEqual(s["draws"], 24)
            self.assertTrue(np.isfinite(s["mean_chance_adjusted_composite"]))
            self.assertTrue(0 <= s["mean_similarity"]["composite"] <= 1)
            self.assertTrue(0 <= s["mean_iid_expected_for_predictions"]["composite"] <= 1)
        self.assertEqual(len(r["individual_draws"]), 24)

    def test_artificial_predictable_cycle_yields_exact_shape_forecasts(self):
        cycle = np.array([
            [1, 2, 3, 4, 5, 6],
            [8, 9, 10, 11, 12, 13],
            [15, 16, 17, 18, 19, 20],
            [22, 23, 24, 25, 26, 27],
        ])
        write_fixture(self.path, np.tile(cycle, (30, 1)))
        report = self.evaluate()
        analog = report["evaluation_summary"]["temporal_analog"]
        self.assertAlmostEqual(analog["mean_similarity"]["composite"], 1.0, places=12)
        self.assertEqual(analog["exact_six_matches_secondary"], 24)
        self.assertGreater(analog["mean_chance_adjusted_composite"], 0.0)

    def test_replay_and_controls_reproducible(self):
        first = self.evaluate()
        second = self.evaluate()
        self.assertEqual(first, second)

    def test_export_csv_contains_pattern_and_null_comparison(self):
        report = self.evaluate()
        dest = self.path.parent / "replay.csv"
        export_csv(report, dest)
        with dest.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 24 * len(METHODS))
        self.assertIn("shape_d4", rows[0])
        self.assertIn("iid_expected_similarity", rows[0])
        self.assertIn("chance_adjusted_similarity", rows[0])
        self.assertEqual(len(rows[0]["actual_pattern"].split("/")), 7)

    def test_missing_year_and_bad_settings_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "absent"):
            self.evaluate(year=2025)
        with self.assertRaises(ValueError):
            self.evaluate(null_samples=1)
        with self.assertRaises(ValueError):
            self.evaluate(clusters=1)
        with self.assertRaises(ValueError):
            split_indices(120, window=3, train_end=96, validation_end=120)

    def test_non_year_filtered_evaluation_uses_full_test_range(self):
        r = self.evaluate(year=None)
        self.assertEqual(r["year"], "all_test_years")
        self.assertEqual(r["evaluated_draws"], 24)

    def test_outputs_are_always_valid_reversible_six_pixel_images(self):
        r = self.evaluate()
        for draw in r["individual_draws"]:
            self.assertEqual(sum(row.count("#") for row in draw["actual_pattern"]), 6)
            for method in METHODS:
                p = draw["methods"][method]
                self.assertEqual(sum(row.count("#") for row in p["predicted_pattern"]), 6)
                self.assertEqual(len(p["predicted_main_six"]), 6)
                self.assertTrue(0 <= p["similarity"]["shape_d4"] <= 1)


if __name__ == "__main__":
    unittest.main()
