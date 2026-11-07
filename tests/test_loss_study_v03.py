"""Version 0.3 loss/decoder consistency tests; v0.2 must remain immutable."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from lotto7x7.__main__ import gallery
from lotto7x7.bin_dataset import Draw, encode_bin
from lotto7x7.core import encode
from lotto7x7.learning import ImageSequenceLearner, decode_image, shape_loss
from lotto7x7.loss_study_v03 import (
    METHODS, SCHEMA, V02_BASELINE_COMMIT, loss_v03,
    paired_intervals, projected_six_probabilities, study,
    synthetic_control, train_treatment,
)


class LossAlignmentTests(unittest.TestCase):
    def setUp(self):
        patterns = [
            [1, 2, 3, 4, 5, 6], [1, 8, 15, 22, 29, 36],
            [1, 9, 17, 25, 33, 41], [2, 11, 14, 23, 37, 49],
        ]
        self.images = np.stack([encode(np.array(patterns[i % 4]))
                                for i in range(120)])

    def test_v02_training_and_diagnostics_sources_are_frozen(self):
        expected = {
            "learning.py": "41718499b9e8095b501099a8c009eaa903922d6d05e7a38edafd1e66d9b169e2",
            "diagnostics.py": "9ad0048ec7575679e13d4a425330996c4ad09c1bee6eed2218bc85c0960a3811",
        }
        root = Path(__file__).resolve().parents[1] / "lotto7x7"
        for name, original_sha in expected.items():
            text = (root / name).read_text(encoding="utf-8").replace("\r\n", "\n")
            actual_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            self.assertEqual(actual_sha, original_sha, name)
        self.assertEqual(V02_BASELINE_COMMIT,
                         "0102c561530522dd0abae4fd4aa0b8fccef81a9e")

    def test_original_method_is_exact_v02_training_loss(self):
        torch.manual_seed(19)
        logits = torch.randn((3, 7, 7), requires_grad=True)
        targets = torch.as_tensor(self.images[:3], dtype=torch.float32)
        a = shape_loss(logits, targets)
        b = loss_v03(logits, targets, method="original_v02")
        self.assertEqual(float(a.item()), float(b.item()))
        torch.testing.assert_close(torch.autograd.grad(a, logits, retain_graph=True)[0],
                                   torch.autograd.grad(b, logits)[0], rtol=0, atol=0)

    def test_projected_soft_pixels_sum_to_exactly_six(self):
        torch.manual_seed(23)
        for scaling in (0.1, 1.0, 8.0):
            logits = torch.randn((7, 7, 7), dtype=torch.float32) * scaling
            mass = projected_six_probabilities(logits).sum(dim=(1, 2))
            torch.testing.assert_close(mass, torch.full_like(mass, 6.0),
                                       atol=3e-5, rtol=0)

    def test_projected_probabilities_preserve_top_six_order(self):
        torch.manual_seed(31)
        logits = torch.randn((8, 7, 7))
        scores = projected_six_probabilities(logits)
        for before, after in zip(logits, scores):
            np.testing.assert_array_equal(decode_image(before), decode_image(after))
            self.assertTrue(bool(torch.all((after > 0) & (after < 1))))

    def test_implicit_projection_gradient_is_tangent_to_mass_constraint(self):
        x = torch.randn((2, 7, 7), dtype=torch.float64, requires_grad=True)
        probability = projected_six_probabilities(x)
        grad = torch.autograd.grad(probability.sum(), x)[0]
        self.assertLess(float(grad.abs().max()), 1e-8)

    def test_all_loss_variants_are_differentiable(self):
        obs = torch.as_tensor(self.images[:3], dtype=torch.float32)
        logits = torch.randn((3, 7, 7), requires_grad=True)
        for method in METHODS:
            result = loss_v03(logits, obs, method=method)
            self.assertTrue(bool(torch.isfinite(result)))
            derivative = torch.autograd.grad(result, logits, retain_graph=True)[0]
            self.assertTrue(bool(torch.isfinite(derivative).all()))
            self.assertGreater(float(derivative.abs().sum()), 0.0)

    def test_bad_image_and_variant_configuration_rejected(self):
        logits = torch.zeros((2, 7, 7))
        observed = torch.as_tensor(self.images[:2], dtype=torch.float32)
        with self.assertRaises(ValueError):
            projected_six_probabilities(logits, temperature=0)
        with self.assertRaises(ValueError):
            projected_six_probabilities(torch.zeros((2, 7, 8)))
        with self.assertRaises(ValueError):
            projected_six_probabilities(torch.full((2, 7, 7), float("nan")))
        with self.assertRaises(ValueError):
            loss_v03(logits, observed, method="invented")
        with self.assertRaises(ValueError):
            loss_v03(logits, torch.zeros_like(observed), method="balanced_bce")
        with self.assertRaises(ValueError):
            train_treatment(self.images, method="original_v02",
                            train_end=72, validation_end=96)
        with self.assertRaises(ValueError):
            train_treatment(self.images, method="balanced_bce",
                            train_end=72, validation_end=96, epochs=0)

    def test_alternative_training_uses_same_cnn_and_gru_structure(self):
        for method in METHODS[1:]:
            model, fit = train_treatment(
                self.images, method=method,
                train_end=72, validation_end=96,
                window=3, embedding=8, hidden=12,
                epochs=2, patience=2, batch_size=32, seed=17)
            self.assertIsInstance(model, ImageSequenceLearner)
            self.assertEqual(fit["training_targets"], [3, 71])
            self.assertEqual(fit["validation_targets"], [72, 95])
            self.assertLessEqual(fit["best_epoch"], 2)

    def test_bootstrap_is_paired_and_reproducible(self):
        scores = {name: [float(i + j) for i in range(10)]
                  for j, name in enumerate(METHODS)}
        x = paired_intervals(scores, iterations=60, block=3, seed=7)
        y = paired_intervals(scores, iterations=60, block=3, seed=7)
        self.assertEqual(x, y)
        self.assertAlmostEqual(x["balanced_bce_minus_original_v02"]["mean"], 1.0)
        with self.assertRaises(ValueError):
            paired_intervals({name: [] for name in METHODS})

    def test_end_to_end_bin_study_does_not_use_special_number(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "test.bin"
            draws = []
            for i, mask in enumerate(self.images):
                main = tuple((np.flatnonzero(mask) + 1).tolist())
                draws.append(Draw(draw_id=107000001+i, main=main,
                                  date=20180101+i, source="synthetic"))
            binary.write_bytes(encode_bin(draws))
            options = dict(path=binary, year=2018, window=3, epochs=2,
                           patience=2, batch_size=32, embedding=8, hidden=12,
                           baseline=2, bootstrap=15, seed=32)
            a = study(**options)
            b = study(**options)
            self.assertEqual(a["schema"], SCHEMA)
            self.assertEqual(a["evaluation_draws"], 24)
            self.assertEqual(list(a["summary"]), list(METHODS))
            self.assertEqual(a["summary"], b["summary"])
            self.assertEqual(a["baseline_v02_commit"], V02_BASELINE_COMMIT)
            self.assertIsNone(a["rows"][0]["date"])
            self.assertEqual(a["rows"][0]["year"], 2018)
            self.assertAlmostEqual(a["logit_diagnostics"]["projected_six_mass"]["effective_soft_mass"], 6.0, places=4)
            self.assertEqual(len(a["rows"]), 24)
            self.assertIn('"schema":', json.dumps(a))
            html = gallery(a, count=2)
            self.assertIn("projected_six_mass", html)
            self.assertEqual(html.count("<svg "), 8)

    def test_synthetic_iid_control_has_no_hardcoded_positive_conclusion(self):
        iid = synthetic_control(kind="iid", n=100, epochs=2, seed=45)
        periodic = synthetic_control(kind="periodic", n=100, epochs=2, seed=45)
        self.assertEqual(iid["kind"], "iid")
        self.assertEqual(periodic["kind"], "periodic")
        self.assertEqual(set(iid["summary"]), set(METHODS))
        self.assertEqual(set(periodic["summary"]), set(METHODS))
        for result in (iid, periodic):
            self.assertEqual(result["evaluation_draws"], 20)
            for method in METHODS:
                item = result["summary"][method]
                self.assertTrue(np.isfinite(item["adjusted_mean"]))
                self.assertTrue(np.isfinite(item["mean_image_shape_similarity"]))


if __name__ == "__main__":
    unittest.main()
