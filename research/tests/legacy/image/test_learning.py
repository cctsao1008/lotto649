"""Tests only for the new image-trained CNN/GRU research path."""
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch

from lotto7x7.core import encode, similarity
from lotto7x7.learning import (
    ImageSequenceLearner, autocorrelation, decode_image, image_windows,
    learned_replay, predict_images, shape_loss, train_model,
)
from lotto7x7.__main__ import gallery


class ImageLearningTests(unittest.TestCase):
    def setUp(self):
        self.patterns = np.stack([encode(np.array(x)) for x in (
            [1, 2, 3, 4, 5, 6],
            [1, 8, 15, 22, 29, 36],
            [1, 9, 17, 25, 33, 41],
            [2, 11, 14, 23, 37, 49],
        )])
        self.images = np.tile(self.patterns, (30, 1, 1))

    def test_cnn_and_temporal_gru_return_7x7_images(self):
        model = ImageSequenceLearner(embedding=8, hidden=12)
        values = model(torch.as_tensor(self.images[:15].reshape(5, 3, 7, 7), dtype=torch.float32))
        self.assertEqual(tuple(values.shape), (5, 7, 7))
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue(any(isinstance(layer, torch.nn.Conv2d)
                            for layer in model.encoder))
        self.assertTrue(isinstance(model.temporal, torch.nn.GRU))

    def test_causal_image_windows_exclude_target_pixels(self):
        x, y = image_windows(self.images, window=3)
        self.assertEqual(tuple(x.shape), (117, 3, 7, 7))
        self.assertEqual(tuple(y.shape), (117, 7, 7))
        changed = self.images.copy()
        changed[3] = self.patterns[2]
        x2, y2 = image_windows(changed, window=3)
        np.testing.assert_array_equal(x[0].numpy(), x2[0].numpy())
        self.assertFalse(np.array_equal(y[0].numpy(), y2[0].numpy()))

    def test_shape_autocorrelation_is_translation_invariant(self):
        a = self.patterns[0]
        moved = np.zeros((7, 7), dtype=np.float32)
        moved[1, :6] = a[0, :6]
        z1 = autocorrelation(torch.as_tensor(a[None], dtype=torch.float32))
        z2 = autocorrelation(torch.as_tensor(moved[None]))
        torch.testing.assert_close(z1, z2, rtol=1e-5, atol=1e-5)

    def test_loss_emphasizes_image_shape_and_has_valid_gradients(self):
        logits = torch.randn((3, 7, 7), requires_grad=True)
        targets = torch.as_tensor(self.patterns[:3], dtype=torch.float32)
        result = shape_loss(logits, targets)
        self.assertTrue(torch.isfinite(result))
        result.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_top_six_image_output_is_legal(self):
        model = ImageSequenceLearner()
        tensor = model(torch.zeros((1, 5, 7, 7)))[0]
        mask = decode_image(tensor)
        self.assertEqual(mask.shape, (7, 7))
        self.assertEqual(int(mask.sum()), 6)
        self.assertEqual(mask.dtype, np.uint8)

    def test_trained_network_learns_a_deterministic_image_sequence(self):
        model, history = train_model(
            self.images, train_end=72, validation_end=96,
            window=3, epochs=70, patience=15, batch_size=16,
            learning_rate=0.003, seed=42)
        guesses = predict_images(model, self.images,
                                 indices=np.arange(96, 120), window=3)
        mean_shape = float(np.mean([
            similarity(guess, actual)["shape"]
            for guess, actual in zip(guesses, self.images[96:])
        ]))
        self.assertGreater(mean_shape, 0.95)
        self.assertGreaterEqual(len({mask.tobytes() for mask in guesses}), 3)
        self.assertLessEqual(history["best_epoch"], 70)
        self.assertEqual(history["training_targets"], [3, 71])
        self.assertEqual(history["validation_targets"], [72, 95])

    def test_changing_current_or_future_image_does_not_change_prediction(self):
        model = ImageSequenceLearner(embedding=8, hidden=12)
        a = predict_images(model, self.images, indices=np.array([96]), window=3)
        changed = self.images.copy()
        changed[96:] = np.repeat(self.patterns[0][None], 24, axis=0)
        b = predict_images(model, changed, indices=np.array([96]), window=3)
        np.testing.assert_array_equal(a, b)

    def test_image_learning_experiment_reports_adjusted_iid_score(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "synthetic.npy"
            dtype = np.dtype([
                ("date", "<u4"), ("draw_id", "<u4"),
                ("numbers", "u1", (6,)), ("special", "u1"),
            ])
            rec = np.zeros(120, dtype=dtype)
            for i, mask in enumerate(self.images):
                numbers = np.asarray(np.flatnonzero(mask) + 1, dtype=np.uint8)
                rec[i]["date"] = int((date(2018, 1, 1)
                                       + timedelta(days=i*2)).strftime("%Y%m%d"))
                rec[i]["draw_id"] = 107000001 + i
                rec[i]["numbers"] = numbers
                rec[i]["special"] = next(n for n in range(1, 50) if n not in numbers)
            np.save(path, rec, allow_pickle=False)
            report, model = learned_replay(
                path, window=3, epochs=2, patience=2, batch_size=32,
                embedding=8, hidden=12, baseline=2, seed=19)
            self.assertEqual(report["schema"], "lotto7x7.learned_image.v1")
            self.assertEqual(report["evaluation_draws"], 24)
            self.assertEqual(len(report["rows"]), 24)
            self.assertIn("learned_image", report["summary"])
            self.assertIn("previous_image", report["summary"])
            for item in report["summary"].values():
                self.assertTrue(np.isfinite(item["mean_iid_adjusted_similarity"]))
            visual = gallery(report, count=3)
            self.assertIn("learned_image", visual)
            self.assertIn("<svg ", visual)

    def test_bad_model_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            ImageSequenceLearner(embedding=1)
        with self.assertRaises(ValueError):
            ImageSequenceLearner()(torch.zeros((1, 1, 7, 7)))
        with self.assertRaises(ValueError):
            decode_image(torch.zeros((6, 6)))
        with self.assertRaises(ValueError):
            image_windows(self.images, window=1)
        with self.assertRaises(ValueError):
            train_model(self.images, train_end=72,
                        validation_end=96, window=3, epochs=0)


if __name__ == "__main__":
    unittest.main()
