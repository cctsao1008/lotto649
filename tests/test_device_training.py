"""Optional CUDA training support without changing frozen v0.2 source code."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from lotto7x7.bin_dataset import Draw, encode_bin
from lotto7x7.core import encode
from lotto7x7.device_training import (
    infer_device_logits, select_device, train_device_model,
)
from lotto7x7.learning import train_model
from lotto7x7.loss_study_v03 import METHODS, study, train_treatment


class DeviceTrainingTests(unittest.TestCase):
    def setUp(self):
        frames = [
            [1, 2, 3, 4, 5, 6],
            [1, 8, 15, 22, 29, 36],
            [1, 9, 17, 25, 33, 41],
            [2, 11, 14, 23, 37, 49],
        ]
        self.images = np.stack([encode(np.array(frames[i % 4]))
                                for i in range(120)])
        self.args = dict(train_end=72, validation_end=96, window=3,
                         epochs=2, patience=2, batch_size=16,
                         embedding=8, hidden=12, seed=42)

    def test_cpu_and_auto_device_resolution(self):
        self.assertEqual(select_device("cpu"), torch.device("cpu"))
        self.assertEqual(select_device("auto").type,
                         "cuda" if torch.cuda.is_available() else "cpu")
        with self.assertRaises(ValueError):
            select_device("gpu")
        with patch("lotto7x7.device_training.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
                select_device("cuda")

    def test_original_device_cpu_matches_frozen_v02_weight_by_weight(self):
        expected, info = train_model(self.images, **self.args)
        actual, new_info = train_device_model(
            self.images, method="original_v02", device=torch.device("cpu"),
            **self.args)
        self.assertEqual(info["history"], new_info["history"])
        self.assertEqual(info["best_epoch"], new_info["best_epoch"])
        for key, weight in expected.state_dict().items():
            torch.testing.assert_close(weight, actual.state_dict()[key], atol=0, rtol=0)

    def test_balanced_device_cpu_matches_v03_weight_by_weight(self):
        expected, info = train_treatment(self.images, method="balanced_bce", **self.args)
        actual, new_info = train_device_model(
            self.images, method="balanced_bce", device=torch.device("cpu"),
            **self.args)
        self.assertEqual(info["history"], new_info["history"])
        for key, weight in expected.state_dict().items():
            torch.testing.assert_close(weight, actual.state_dict()[key], atol=0, rtol=0)

    def test_cpu_inference_causality_and_model_device_check(self):
        model, _ = train_device_model(
            self.images, method="projected_six_mass", device=torch.device("cpu"),
            **self.args)
        ids = np.arange(96, 101)
        a = infer_device_logits(model, self.images, indices=ids,
                                window=3, batch_size=2, device=torch.device("cpu"))
        self.assertEqual(a.shape, (5, 7, 7))
        changed = self.images.copy()
        changed[96:] = self.images[0]
        b = infer_device_logits(model, changed, indices=ids,
                                window=3, batch_size=2, device=torch.device("cpu"))
        np.testing.assert_array_equal(a[0], b[0])
        with self.assertRaises(ValueError):
            infer_device_logits(model, self.images, indices=np.array([1]),
                                window=3, batch_size=2, device=torch.device("cpu"))
        with self.assertRaises(ValueError):
            infer_device_logits(model, self.images, indices=ids,
                                window=3, batch_size=2, device=torch.device("cuda"))

    def test_study_device_cpu_keeps_baseline_scores(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "images.bin"
            draws = [Draw(draw_id=107000001+i,
                          main=tuple((np.flatnonzero(mask) + 1).tolist()),
                          date=20180101+i, source="synthetic")
                     for i, mask in enumerate(self.images)]
            path.write_bytes(encode_bin(draws))
            opts = dict(path=path, year=2018, window=3, epochs=2,
                        patience=2, batch_size=16, embedding=8, hidden=12,
                        baseline=2, bootstrap=10, seed=42)
            baseline = study(**opts)
            selected = study(**opts, device="cpu")
            self.assertEqual(baseline["summary"], selected["summary"])
            self.assertEqual(baseline["rows"], selected["rows"])
            self.assertEqual(selected["execution"]["actual_device"], "cpu")
            self.assertGreaterEqual(selected["execution"]["training_seconds"], 0)

    @unittest.skipUnless(torch.cuda.is_available(), "Requires NVIDIA CUDA")
    def test_cuda_end_to_end_smoke(self):
        models = [train_device_model(
            self.images, method=method, device=torch.device("cuda"),
            **self.args) for method in METHODS]
        for model, fit in models:
            self.assertEqual(next(model.parameters()).device.type, "cuda")
            logits = infer_device_logits(model, self.images,
                                         indices=np.arange(96, 101), window=3,
                                         batch_size=2, device=torch.device("cuda"))
            self.assertEqual(logits.shape, (5, 7, 7))
            self.assertTrue(np.isfinite(logits).all())
            self.assertGreaterEqual(fit["best_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
