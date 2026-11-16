"""Version 0.4 convergence and temporal predictability audit regressions."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from lotto7x7.bin_dataset import Draw, encode_bin
from lotto7x7.core import encode
from lotto7x7.learning import image_windows, permute_history_windows
from lotto7x7.predictability_v04 import (
    METHODS, SCHEMA, _predict, _score, _synthetic_images, audit,
    paired_block_interval, train_audit_model, uniform_brier,
)
from lotto7x7.research import split


class PredictabilityAuditTests(unittest.TestCase):
    def setUp(self):
        patterns = (
            [1, 2, 3, 4, 5, 6], [1, 8, 15, 22, 29, 36],
            [1, 9, 17, 25, 33, 41], [2, 11, 14, 23, 37, 49],
        )
        self.images = np.stack([encode(np.asarray(patterns[i % 4]))
                                for i in range(100)])
        self.fit = dict(train_end=60, validation_end=80, window=3,
                        epochs=2, patience=2, batch_size=16,
                        embedding=8, hidden=12, seed=42)

    def test_uniform_baseline_is_exact_for_any_legal_six_point_mask(self):
        p = 6.0 / 49.0
        expected = float(np.mean((self.images - p)**2))
        self.assertAlmostEqual(uniform_brier(), p*(1-p), places=14)
        self.assertAlmostEqual(uniform_brier(), expected, places=14)

    def test_epoch_zero_and_loss_trace_are_recorded_before_training(self):
        model, epoch0, info = train_audit_model(
            self.images, method="ordered_cnn_gru", **self.fit)
        self.assertTrue(info["epoch_zero_included"])
        self.assertEqual(info["history"][0]["epoch"], 0)
        self.assertEqual(len(info["history"]), info["epochs_run"]+1)
        self.assertEqual(info["training_targets"], [3, 59])
        self.assertEqual(info["validation_targets"], [60, 79])
        self.assertGreaterEqual(info["best_epoch"], 0)
        self.assertLessEqual(info["best_epoch"], self.fit["epochs"])
        self.assertTrue(all(np.isfinite(row["training_loss"]) and
                            np.isfinite(row["validation_loss"])
                            for row in info["history"]))
        self.assertIsNot(model, epoch0)
        self.assertEqual(next(epoch0.parameters()).device.type, "cpu")

    def test_epoch_zero_weights_match_between_ordered_and_shuffled(self):
        _a, initial_ordered, _ = train_audit_model(
            self.images, method="ordered_cnn_gru", **self.fit)
        _b, initial_shuffled, _ = train_audit_model(
            self.images, method="shuffled_cnn_gru", **self.fit)
        for key, x in initial_ordered.state_dict().items():
            torch.testing.assert_close(x, initial_shuffled.state_dict()[key],
                                       rtol=0, atol=0)

    def test_shuffled_context_keeps_exact_past_multiset(self):
        x, y = image_windows(self.images, window=3)
        shuffled = permute_history_windows(
            x, np.arange(3, len(self.images)), seed=42 + 9001)
        np.testing.assert_array_equal(y.numpy(), self.images[3:])
        for ordered, permuted in zip(x.numpy(), shuffled.numpy()):
            self.assertEqual(sorted(v.tobytes() for v in ordered),
                             sorted(v.tobytes() for v in permuted))

    def test_future_test_images_never_change_training_weights(self):
        first, _, _ = train_audit_model(
            self.images, method="ordered_cnn_gru", **self.fit)
        altered = self.images.copy()
        altered[80:] = self.images[1]
        second, _, _ = train_audit_model(
            altered, method="ordered_cnn_gru", **self.fit)
        for key, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[key],
                                       rtol=0, atol=0)

    def test_constant_template_is_independent_of_input_history(self):
        model, initial, info = train_audit_model(
            self.images, method="constant_template", **self.fit)
        ids = np.arange(80, 86)
        for current in (model, initial):
            a = _predict(current, self.images, targets=ids,
                         method="constant_template", seed=42, window=3,
                         batch_size=2)
            changed = self.images.copy()
            changed[:80] = self.images[1]
            b = _predict(current, changed, targets=ids,
                         method="constant_template", seed=42, window=3,
                         batch_size=2)
            np.testing.assert_array_equal(a, b)
        self.assertEqual(info["history_order"], "no_history")

    def test_causal_predictions_ignore_current_and_future_images(self):
        model, _, _ = train_audit_model(
            self.images, method="ordered_cnn_gru", **self.fit)
        t = np.array([80])
        a = _predict(model, self.images, targets=t,
                     method="ordered_cnn_gru", seed=42, window=3,
                     batch_size=2)
        changed = self.images.copy()
        changed[80:] = self.images[2]
        b = _predict(model, changed, targets=t,
                     method="ordered_cnn_gru", seed=42, window=3,
                     batch_size=2)
        np.testing.assert_array_equal(a, b)

    def test_iid_control_is_repeatable_and_contains_six_pixels(self):
        a = _synthetic_images("iid", 100, 42)
        b = _synthetic_images("iid", 100, 42)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.all(a.sum(axis=(1, 2)) == 6))
        with self.assertRaises(ValueError):
            _synthetic_images("future", 100, 42)

    def test_periodic_control_requires_order_not_just_past_multiset(self):
        data = _synthetic_images("periodic", 100, 42, window=5)
        np.testing.assert_array_equal(data[:-5], data[5:])
        self.assertEqual(len({x.tobytes() for x in data}), 5)
        past, future = image_windows(data, window=5)
        # All historical windows contain the same five images, but their
        # next-target images vary across the five phases.
        multiset = sorted(im.tobytes() for im in past[0].numpy())
        for context in past.numpy():
            self.assertEqual(multiset,
                             sorted(im.tobytes() for im in context))
        self.assertEqual(len({im.tobytes() for im in future.numpy()}), 5)

    def test_scoring_reports_brier_and_legal_top_six_images(self):
        logits = np.zeros((5, 7, 7), dtype=np.float32)
        metric, delta, brier = _score(logits, self.images[:5])
        self.assertIsNone(delta)
        self.assertEqual(metric["unique_patterns"], 1)
        self.assertEqual(brier.shape, (5,))
        self.assertAlmostEqual(float(brier.mean()), 0.25, places=7)
        with self.assertRaises(ValueError):
            _score(logits[:2], self.images[:5])

    def test_paired_interval_preserves_zero_difference(self):
        output = paired_block_interval(np.zeros(20), iterations=10, seed=42)
        self.assertEqual(output["mean"], 0)
        self.assertEqual(output["interval_95"], [0, 0])
        with self.assertRaises(ValueError):
            paired_block_interval(np.array([1.0, np.nan, 1.0]))

    def test_report_pipeline_has_all_arms_and_no_test_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "small.bin"
            draws = [Draw(draw_id=107000001+i,
                          main=tuple((np.flatnonzero(mask)+1).tolist()),
                          date=20180101+i, source="synthetic")
                     for i, mask in enumerate(self.images)]
            source.write_bytes(encode_bin(draws))
            result = audit(
                source, seeds=1, control_seeds=1, control_samples=100,
                epochs=2, control_epochs=2, window=3, embedding=8,
                hidden=12, patience=2, baseline=2, bootstrap=8,
                batch_size=16, seed=42)
            self.assertEqual(result["schema"], SCHEMA)
            self.assertEqual(result["evaluation_draws"], 20)
            self.assertEqual(result["split"]["evaluation_targets"], [80, 99])
            self.assertEqual(set(result["summary"]), set(METHODS))
            self.assertEqual(set(result["synthetic_controls"]),
                             {"iid", "periodic"})
            self.assertEqual(len(result["per_seed"]), 1)
            self.assertEqual(result["per_seed"][0]["seed"], 42)
            self.assertEqual(result["config"]["loss"], "balanced_bce")
            self.assertIn("ordered_training_gain_over_epoch0_adjusted",
                          result["paired_intervals"])
            self.assertEqual(result["config"]["device"], "cpu")
            for method in METHODS:
                self.assertEqual(len(result["per_seed"][0]["methods"][method]
                                     ["convergence"]["history"]), 3)
                self.assertTrue(np.isfinite(result["summary"][method]
                                            ["mean_pixel_brier"]))
                self.assertIn("mean_iid_adjusted_similarity",
                              result["per_seed"][0]["methods"][method]["epoch0"])

    def test_invalid_audit_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            train_audit_model(self.images, method="invalid", **self.fit)
        with self.assertRaises(ValueError):
            train_audit_model(self.images, method="ordered_cnn_gru",
                              **(self.fit | {"epochs": 0}))
        with self.assertRaises(ValueError):
            audit(seeds=0)


if __name__ == "__main__":
    unittest.main()
