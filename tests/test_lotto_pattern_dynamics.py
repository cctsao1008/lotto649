"""Tests for causal six-pixel dynamic forecasting and pattern diversity."""
from __future__ import annotations

import unittest

import numpy as np

from lotto_image_repr import encode_draws
from lotto_pattern_dynamics import (
    METHODS, DynamicPatternLibrary, diversity, shape_class,
    translate_in_bounds,
)
from lotto_pattern_similarity import image_key


class DynamicPatternTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(903)
        self.draws = np.stack(
            [np.sort(rng.choice(49, 6, replace=False) + 1)
             for _ in range(100)])
        self.images = encode_draws(self.draws)
        self.library = DynamicPatternLibrary(
            self.images, train_end=60, window=3, neighbors=4)

    def test_translation_preserves_six_without_wrapping(self):
        first = encode_draws([1, 2, 3, 8, 9, 15])
        moved = translate_in_bounds(first, np.array([3, 2]))
        self.assertEqual(int(moved.sum()), 6)
        self.assertEqual(shape_class(first), shape_class(moved))
        self.assertFalse(np.array_equal(first, moved))
        boundary = translate_in_bounds(first, np.array([-100, -100]))
        self.assertEqual(int(boundary.sum()), 6)

    def test_center_shift_never_wraps_or_clips(self):
        corner = encode_draws([1, 2, 3, 8, 9, 15])
        shifted = translate_in_bounds(corner, np.array([200, 200]))
        self.assertEqual(int(shifted.sum()), 6)
        self.assertEqual(shape_class(corner), shape_class(shifted))
        with self.assertRaises(ValueError):
            translate_in_bounds(corner, np.array([float("nan"), 0]))

    def test_shape_class_is_invariant_under_rotation_and_translation(self):
        first = self.images[0]
        self.assertEqual(shape_class(first), shape_class(np.rot90(first)))
        self.assertEqual(shape_class(first), shape_class(np.fliplr(first)))
        translated = translate_in_bounds(first, np.array([1, -1]))
        self.assertEqual(shape_class(first), shape_class(translated))

    def test_diversity_catches_fixed_pattern_collapse(self):
        fixed = np.repeat(self.images[0][None], 40, axis=0)
        values = diversity(fixed)
        self.assertEqual(values["unique_pixel_patterns"], 1)
        self.assertEqual(values["unique_shape_classes_d4_translate"], 1)
        self.assertEqual(values["top_pattern_fraction"], 1.0)
        self.assertEqual(values["effective_pattern_count"], 1.0)
        varied = diversity(self.images)
        self.assertGreater(varied["unique_pixel_patterns"], 2)
        self.assertLess(varied["top_pattern_fraction"], 1.0)

    def test_all_dynamic_outputs_are_valid_masks(self):
        for t in range(60, 100):
            for method in METHODS:
                predicted = self.library.predict(t, method=method)
                self.assertEqual(len(image_key(predicted)), 6)
                self.assertEqual(int(predicted.sum()), 6)
                self.assertEqual(predicted.shape, (7, 7))
                self.assertTrue(np.all((predicted == 0) | (predicted == 1)))

    def test_target_mutation_does_not_change_same_date_forecast(self):
        changed = self.images.copy()
        changed[60] = encode_draws([1, 2, 3, 4, 5, 6])
        if np.array_equal(changed[60], self.images[60]):
            changed[60] = encode_draws([7, 8, 9, 10, 11, 12])
        modified = DynamicPatternLibrary(
            changed, train_end=60, window=3, neighbors=4)
        for method in METHODS:
            np.testing.assert_array_equal(
                self.library.predict(60, method=method),
                modified.predict(60, method=method))

    def test_mutating_future_test_draw_cannot_change_prior_forecast(self):
        changed = self.images.copy()
        changed[70:] = np.repeat(self.images[0][None], 30, axis=0)
        modified = DynamicPatternLibrary(
            changed, train_end=60, window=3, neighbors=4)
        for method in METHODS:
            np.testing.assert_array_equal(
                self.library.predict(69, method=method),
                modified.predict(69, method=method))

    def test_scrambled_context_only_uses_earlier_observed_frames(self):
        changed = self.images.copy()
        changed[60] = encode_draws([1, 2, 3, 4, 5, 6])
        modified = DynamicPatternLibrary(
            changed, train_end=60, window=3, neighbors=4)
        for method in METHODS:
            np.testing.assert_array_equal(
                self.library.predict(60, method=method,
                                     scramble_context=True, scramble_seed=5),
                modified.predict(60, method=method,
                                 scramble_context=True, scramble_seed=5))
        np.testing.assert_array_equal(
            self.library.predict(65, method="analog_fusion",
                                 scramble_context=True, scramble_seed=9),
            self.library.predict(65, method="analog_fusion",
                                 scramble_context=True, scramble_seed=9))

    def test_train_bank_override_does_not_mutate_original_history(self):
        original = self.images.copy()
        alternative = original[:60][::-1].copy()
        modified = DynamicPatternLibrary(
            self.images, train_end=60, window=3, neighbors=4,
            bank_override=alternative)
        self.assertEqual(modified.bank_images.shape, (60, 7, 7))
        np.testing.assert_array_equal(self.images, original)
        np.testing.assert_array_equal(modified.images, original)
        for t in range(60, 65):
            self.assertEqual(int(modified.predict(
                t, method="analog_fusion").sum()), 6)

    def test_synthetic_four_pattern_cycle_is_predicted_perfectly(self):
        cycle = np.stack([
            encode_draws([1, 2, 3, 4, 5, 6]),
            encode_draws([8, 9, 10, 11, 12, 13]),
            encode_draws([15, 16, 17, 18, 19, 20]),
            encode_draws([22, 23, 24, 25, 26, 27]),
        ])
        images = np.tile(cycle, (25, 1, 1))
        lib = DynamicPatternLibrary(
            images, train_end=60, window=3,
            neighbors=1, last_weight=0.0)
        for t in range(60, 100):
            np.testing.assert_array_equal(
                lib.predict(t, method="analog_fusion"),
                images[t])

    def test_invalid_configurations_are_rejected(self):
        with self.assertRaises(ValueError):
            DynamicPatternLibrary(self.images, train_end=60, window=1)
        with self.assertRaises(ValueError):
            DynamicPatternLibrary(self.images, train_end=60, window=3,
                                  neighbors=0)
        with self.assertRaises(ValueError):
            DynamicPatternLibrary(self.images, train_end=60, window=3,
                                  last_weight=1.0)
        with self.assertRaises(ValueError):
            self.library.predict(50, method="analog_fusion")
        with self.assertRaises(ValueError):
            self.library.predict(70, method="old_mlp")
        with self.assertRaises(ValueError):
            diversity(np.zeros((2, 7, 7), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
