"""Regression tests for image-logit collapse and temporal ablation diagnostics."""
from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import torch

from lotto7x7.bin_dataset import Draw, encode_bin
from lotto7x7.core import encode
from lotto7x7.diagnostics import (
    ConstantTemplate, SCHEMA, METHODS, decode_logits, diagnose,
    evaluate_predictions, infer_logits, inference_order_control,
    latent_diagnostics, logit_diagnostics, paired_block_intervals,
    train_constant_template,
)
from lotto7x7.learning import ImageSequenceLearner, permute_history_windows, train_model
from lotto7x7.__main__ import gallery


class ImageDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.patterns = np.stack([
            encode(np.array(v)) for v in (
                [1, 2, 3, 4, 5, 6],
                [1, 8, 15, 22, 29, 36],
                [1, 9, 17, 25, 33, 41],
                [2, 11, 14, 23, 37, 49],
            )
        ])
        self.images = np.tile(self.patterns, (30, 1, 1))

    def test_template_never_reads_history_values(self):
        model = ConstantTemplate()
        a = model(torch.zeros(7, 3, 7, 7))
        b = model(torch.ones(7, 3, 7, 7))
        self.assertEqual(a.shape, (7, 7, 7))
        torch.testing.assert_close(a, b)
        self.assertEqual(list(model.parameters())[0].numel(), 49)

    def test_context_shuffle_is_deterministic_and_multiset_preserving(self):
        x = torch.as_tensor(self.images[:40].reshape(8, 5, 7, 7), dtype=torch.float32)
        ids = np.arange(5, 13)
        out = permute_history_windows(x, ids, seed=919)
        again = permute_history_windows(x, ids, seed=919)
        torch.testing.assert_close(out, again)
        for row in range(len(out)):
            original = sorted([v.numpy().tobytes() for v in x[row]])
            altered = sorted([v.numpy().tobytes() for v in out[row]])
            self.assertEqual(original, altered)
        # The sequence contains repeated frames, so also use distinct frames.
        x_unique = torch.as_tensor(self.images[:5][None], dtype=torch.float32)
        changed = permute_history_windows(x_unique, np.array([5]), seed=919)
        self.assertFalse(torch.equal(x_unique, changed))

    def test_shuffle_is_independent_of_batch_composition(self):
        x = torch.as_tensor(self.images[:30].reshape(6, 5, 7, 7), dtype=torch.float32)
        idx = np.arange(5, 11)
        together = permute_history_windows(x, idx, seed=44)
        pieces = torch.cat([permute_history_windows(x[j:j+1], idx[j:j+1], seed=44)
                            for j in range(len(x))])
        torch.testing.assert_close(together, pieces)

    def test_no_future_leakage_in_shuffled_inference(self):
        torch.manual_seed(123)
        model = ImageSequenceLearner(embedding=8, hidden=12)
        before = infer_logits(model, self.images, indices=np.array([96]),
                              window=5, history_order="permuted", permutation_seed=8)
        altered = self.images.copy()
        altered[96:] = self.patterns[0]
        after = infer_logits(model, altered, indices=np.array([96]),
                             window=5, history_order="permuted", permutation_seed=8)
        np.testing.assert_array_equal(before, after)

    def test_logit_movement_can_collapse_to_same_top_six(self):
        logits = np.full((4, 7, 7), -3.0)
        logits[:, 0, :6] = 2.0
        logits[:, 0, 6] = 1.0
        logits[0, 1, 1] = -2.0
        logits[1, 1, 1] = 0.0
        logits[2, 1, 1] = -1.0
        logits[3, 1, 1] = 0.5
        stats = logit_diagnostics(logits)
        self.assertEqual(stats["unique_patterns"]["unique"], 1)
        self.assertGreater(stats["mean_pixel_logit_std"], 0)
        self.assertEqual(stats["always_selected_numbers"], [1, 2, 3, 4, 5, 6])
        self.assertEqual(stats["top_patterns"][0]["count"], 4)
        self.assertGreater(stats["min_top6_top7_margin"], 0)

    def test_top_six_margin_detects_boundary_ties(self):
        logits = np.zeros((1, 7, 7), dtype=np.float32)
        stats = logit_diagnostics(logits)
        self.assertEqual(stats["mean_top6_top7_margin"], 0.0)
        self.assertEqual(stats["unique_patterns"]["unique"], 1)
        self.assertEqual(int(decode_logits(logits)[0].sum()), 6)

    def test_training_permutation_mode_is_explicit_and_validated(self):
        with self.assertRaisesRegex(ValueError, "history_order"):
            train_model(self.images, train_end=72, validation_end=96,
                        window=3, epochs=2, history_order="not_a_real_mode")
        model, trace = train_model(
            self.images, train_end=72, validation_end=96, window=3,
            embedding=8, hidden=12, epochs=2, patience=2, batch_size=32,
            seed=19, history_order="permuted")
        self.assertEqual(trace["history_order"], "permuted")
        self.assertEqual(trace["training_targets"], [3, 71])
        self.assertIsInstance(model, ImageSequenceLearner)

    def test_template_fit_has_same_loss_metric_and_target_boundaries(self):
        model, trace = train_constant_template(
            self.images, train_end=72, validation_end=96, window=3,
            epochs=2, patience=2, batch_size=32, seed=42)
        self.assertIsInstance(model, ConstantTemplate)
        self.assertEqual(trace["training_targets"], [3, 71])
        self.assertEqual(trace["validation_targets"], [72, 95])
        self.assertEqual(trace["objective"], "same_shape_loss_as_cnn_gru")

    def test_order_sensitivity_measures_both_logits_and_masks(self):
        torch.manual_seed(15)
        model = ImageSequenceLearner(embedding=8, hidden=12)
        result = inference_order_control(
            model, self.images, indices=np.arange(96, 110),
            window=5, batch_size=7, seed=42)
        self.assertGreaterEqual(result["mean_absolute_logit_delta"], 0)
        self.assertGreaterEqual(result["changed_top6_fraction"], 0)
        self.assertLessEqual(result["changed_top6_fraction"], 1)
        self.assertGreaterEqual(result["mean_common_selected_pixels"], 0)

    def test_latent_diagnostics_detect_encoder_vs_hidden_state(self):
        model = ImageSequenceLearner(embedding=8, hidden=12)
        metrics = latent_diagnostics(
            model, self.images, indices=np.arange(96, 101), window=5,
            batch_size=3, history_order="permuted", permutation_seed=19)
        self.assertTrue(metrics["available"])
        for key in ("final_frame_embedding_mean_component_std",
                    "gru_final_hidden_mean_component_std"):
            self.assertTrue(np.isfinite(metrics[key]))
        self.assertFalse(latent_diagnostics(
            ConstantTemplate(), self.images, indices=np.array([96]), window=5
        )["available"])

    def test_paired_intervals_return_exact_zero_for_identical_scores(self):
        values = {name: [0.2, 0.1, -0.1, 0.0]*3 for name in METHODS}
        result = paired_block_intervals(values, iterations=40, seed=4)
        for interval in result.values():
            self.assertEqual(interval["mean"], 0.0)
            self.assertEqual(interval["interval_95"], [0.0, 0.0])

    def test_evaluation_uses_identical_random_controls_for_all_models(self):
        logits = np.zeros((4, 7, 7))
        by_method = {name: logits.copy() for name in METHODS}
        indices = np.arange(96, 100)
        result, _, values = evaluate_predictions(
            self.images, indices, by_method, baseline=3, seed=10,
            include_rows=False)
        self.assertEqual(len(result), 3)
        self.assertEqual(values[METHODS[0]], values[METHODS[1]])
        self.assertEqual(values[METHODS[0]], values[METHODS[2]])

    def test_diagnostic_report_is_fully_reproducible_and_renders_gallery(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"synthetic.bin"
            draws = []
            for i, mask in enumerate(self.images):
                ids = tuple((np.flatnonzero(mask.ravel()) + 1).tolist())
                draws.append(Draw(draw_id=107000001+i, main=ids,
                                  date=20180101+i, source="synthetic"))
            path.write_bytes(encode_bin(draws))
            params = dict(path=path, window=3, epochs=2, patience=2,
                          embedding=8, hidden=12, batch_size=32,
                          baseline=2, bootstrap=20, seed=17)
            a = diagnose(**params)
            b = diagnose(**params)
            self.assertEqual(a["schema"], SCHEMA)
            self.assertIn('"schema":', json.dumps(a))
            self.assertEqual(a["evaluation_draws"], 24)
            self.assertEqual(len(a["rows"]), 24)
            self.assertEqual(list(a["summary"]), list(METHODS))
            self.assertEqual(a["summary"], b["summary"])
            self.assertEqual(a["logit_diagnostics"], b["logit_diagnostics"])
            self.assertIsNone(a["rows"][0]["date"])
            self.assertEqual(a["rows"][0]["year"], 2018)
            self.assertIn("training", a)
            self.assertIn("fixed_weights_history_order_control", a)
            html = gallery(a, count=2)
            self.assertEqual(html.count("<svg "), 8)
            self.assertIn("shuffled_cnn_gru", html)

    def test_invalid_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            ConstantTemplate()(torch.zeros(1, 1, 7, 7))
        with self.assertRaises(ValueError):
            logit_diagnostics(np.zeros((2, 49)))
        with self.assertRaises(ValueError):
            permute_history_windows(torch.zeros(2, 4, 7, 7), np.array([4]), seed=1)
        with self.assertRaises(ValueError):
            infer_logits(ImageSequenceLearner(), self.images,
                         indices=np.array([120]), window=3)
        with self.assertRaises(ValueError):
            paired_block_intervals({name: [] for name in METHODS}, iterations=10)


if __name__ == "__main__":
    unittest.main()
