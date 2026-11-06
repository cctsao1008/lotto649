"""Phase 7 controlled image-learning tests on artificial draw histories."""
from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch

from lotto_image_experiment import (MODEL_NAMES, SpatialCNN, TemporalCNN,
                                    export_csv, make_model_and_features,
                                    run_image_experiment)
from lotto_image_repr import temporal_windows


def write_fixture(path: Path, draws: np.ndarray) -> None:
    """Write a synthetic official-layout file, not a fetched lottery result."""
    dtype = np.dtype([("date", "<u4"), ("draw_id", "<u4"),
                      ("numbers", "u1", (6,)), ("special", "u1")])
    arr = np.zeros(len(draws), dtype=dtype)
    for i, numbers in enumerate(draws):
        day = date(2018, 1, 1) + timedelta(days=3 * i)
        arr[i]["date"] = int(day.strftime("%Y%m%d"))
        arr[i]["draw_id"] = 107000001 + i
        arr[i]["numbers"] = numbers[::-1]
        arr[i]["special"] = next(n for n in range(1, 50) if n not in numbers)
    np.save(path, arr, allow_pickle=False)


class ImageExperimentTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "synthetic_official_layout.npy"
        rng = np.random.default_rng(811)
        self.draws = np.stack([np.sort(rng.choice(49, 6, replace=False) + 1)
                               for _ in range(120)]).astype(np.int64)
        write_fixture(self.path, self.draws)

    def run_short(self, **kwargs):
        settings = {
            "source": self.path, "year": 2018,
            "models": ("iid", "flat_mlp", "cnn_sequential", "cnn_dct",
                       "cnn_fft", "cnn_temporal", "cnn_permuted"),
            "window": 3, "hidden": 8, "epochs": 2,
            "patience": 2, "batch_size": 32,
            "train_end": 72, "validation_end": 96,
            "seed": 55, "mapping_seed": 123,
        }
        settings.update(kwargs)
        return run_image_experiment(**settings)

    def test_all_model_inputs_produce_canonical_49_logits(self):
        for name in MODEL_NAMES:
            if name == "iid":
                continue
            model, x, y = make_model_and_features(
                self.draws, name=name, window=3, hidden=8,
                mapping_seed=123)
            self.assertEqual(len(x), 117)
            self.assertEqual(len(y), 117)
            np.testing.assert_array_equal(y.numpy(), self.draws[3:])
            logits = model(x[:2])
            self.assertEqual(tuple(logits.shape), (2, 49))
            self.assertTrue(torch.isfinite(logits).all())

    def test_every_feature_pipeline_is_causal_for_target(self):
        modified = self.draws.copy()
        modified[3] = [1, 2, 3, 4, 5, 6]
        if np.array_equal(self.draws[3], modified[3]):
            modified[3] = [2, 3, 4, 5, 6, 7]
        for name in MODEL_NAMES:
            if name == "iid":
                continue
            _, x0, y0 = make_model_and_features(
                self.draws, name=name, window=3, hidden=8,
                mapping_seed=649)
            _, x1, y1 = make_model_and_features(
                modified, name=name, window=3, hidden=8,
                mapping_seed=649)
            np.testing.assert_array_equal(x0[0].numpy(), x1[0].numpy())
            self.assertFalse(np.array_equal(y0[0].numpy(), y1[0].numpy()))
            self.assertFalse(np.array_equal(x0[1].numpy(), x1[1].numpy()))

    def test_spatial_and_temporal_models_reject_wrong_shapes(self):
        with self.assertRaises(ValueError):
            SpatialCNN(channels=0)
        with self.assertRaises(ValueError):
            TemporalCNN(window=0)
        with self.assertRaises(ValueError):
            SpatialCNN(channels=3)(torch.zeros((2, 3, 5, 5)))
        with self.assertRaises(ValueError):
            TemporalCNN(window=3)(torch.zeros((2, 1, 7, 7)))

    def test_all_models_run_chronologically_and_report_exact_six(self):
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        report = self.run_short()
        after = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.assertEqual(before, after)
        self.assertEqual(report["source_sha256"], before)
        self.assertEqual(report["split"]["train"], [3, 71])
        self.assertEqual(report["split"]["validation"], [72, 95])
        self.assertEqual(report["split"]["evaluation"], [96, 119])
        self.assertEqual(report["evaluated_draws"], 24)
        self.assertEqual(set(report["summary_by_model"]), set(report["models"]))
        self.assertIn(report["validation_selected"], report["models"])
        for name in report["models"]:
            s = report["summary_by_model"][name]
            self.assertTrue(0 <= s["exact_six_matches"] <= 24)
            self.assertEqual(s["evaluated_draws"], 24)
            self.assertTrue(np.isfinite(s["mean_log_score_lift_vs_iid_nats"]))
        for row in report["individual_draws"]:
            self.assertEqual(set(row["models"]), set(report["models"]))
            for data in row["models"].values():
                self.assertEqual(data["result"],
                                 "PASS" if data["exact_six_match"] else "FAIL")
                self.assertEqual(
                    data["exact_six_match"],
                    data["predicted_main_six"] == row["actual_main_six"])

    def test_export_csv_has_one_row_per_model_per_draw(self):
        result = self.run_short(models=("iid", "cnn_sequential"))
        destination = self.path.parent / "phase7.csv"
        export_csv(result, destination)
        with destination.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 48)
        self.assertEqual({row["model"] for row in rows},
                         {"iid", "cnn_sequential"})
        for row in rows:
            self.assertEqual(row["result"],
                             "PASS" if row["exact_six_match"] == "True" else "FAIL")

    def test_synthetic_constant_signal_is_recovered_as_exact_six(self):
        # Positive control with artificial constant numbers, not real lottery data.
        constant = np.tile(np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64), (120, 1))
        write_fixture(self.path, constant)
        result = self.run_short(models=("iid", "cnn_sequential"),
                                seed=999, epochs=5, patience=3)
        self.assertEqual(
            result["summary_by_model"]["cnn_sequential"]["exact_six_matches"], 24)
        self.assertEqual(result["summary_by_model"]["iid"]["exact_six_matches"], 0)

    def test_validation_only_iid_baseline_and_missing_year(self):
        result = self.run_short(models=("iid",))
        self.assertEqual(result["validation_selected"], "iid")
        self.assertAlmostEqual(
            result["summary_by_model"]["iid"]["mean_log_score_lift_vs_iid_nats"],
            0.0)
        with self.assertRaisesRegex(ValueError, "absent"):
            self.run_short(year=2025, models=("iid",))

    def test_invalid_model_configuration_refused(self):
        with self.assertRaises(ValueError):
            self.run_short(models=("cnn_sequential",))
        with self.assertRaises(ValueError):
            self.run_short(models=("iid", "iid"))
        with self.assertRaises(ValueError):
            make_model_and_features(self.draws, name="unknown",
                                    window=3, hidden=8, mapping_seed=649)


if __name__ == "__main__":
    unittest.main()
