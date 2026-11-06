"""Phase 7B metric tests: similarity is not equivalent to exact-six hits."""
from __future__ import annotations

import unittest

import numpy as np

from lotto_image_repr import encode_draws
from lotto_pattern_similarity import (WEIGHTS, conditional_chance_similarity,
                                      image_key, random_six_images, signature,
                                      similarity)


class PatternSimilarityTests(unittest.TestCase):
    def setUp(self):
        self.a = encode_draws([1, 2, 3, 8, 9, 15])
        self.b = encode_draws([25, 26, 27, 32, 33, 39])

    def test_identical_images_have_full_similarity(self):
        values = similarity(self.a, self.a)
        self.assertAlmostEqual(sum(WEIGHTS.values()), 1.0)
        for value in values.values():
            self.assertAlmostEqual(value, 1.0, places=12)

    def test_shape_match_can_have_zero_common_numbers(self):
        values = similarity(self.a, self.b)
        self.assertEqual(values["pixel_jaccard"], 0.0)
        self.assertEqual(values["shape_d4"], 1.0)
        self.assertTrue(0.0 < values["composite"] < 1.0)

    def test_rotation_reflection_are_shape_matches(self):
        transform = np.fliplr(np.rot90(self.a, 2))
        values = similarity(self.a, transform)
        self.assertAlmostEqual(values["shape_d4"], 1.0)
        self.assertGreater(values["composite"], 0.0)

    def test_score_is_symmetric_and_bounded(self):
        rng = np.random.default_rng(31)
        draws = np.stack([np.sort(rng.choice(49, 6, replace=False) + 1)
                          for _ in range(25)])
        images = encode_draws(draws)
        for a, b in zip(images[:-1], images[1:]):
            ab, ba = similarity(a, b), similarity(b, a)
            self.assertEqual(set(ab), set(ba))
            for name in ab:
                self.assertTrue(0 <= ab[name] <= 1, (name, ab[name]))
                self.assertAlmostEqual(ab[name], ba[name], places=12)

    def test_random_control_is_reproducible_and_legal(self):
        x = random_six_images(count=64, seed=99)
        y = random_six_images(count=64, seed=99)
        np.testing.assert_array_equal(x, y)
        self.assertEqual(x.shape, (64, 7, 7))
        self.assertTrue(np.all(x.sum(axis=(1, 2)) == 6))

    def test_chance_baseline_depends_on_predicted_shape(self):
        controls = random_six_images(count=100, seed=45)
        aa = conditional_chance_similarity(self.a, controls)
        bb = conditional_chance_similarity(self.b, controls)
        self.assertTrue(0 <= aa["composite"] <= 1)
        self.assertTrue(0 <= bb["composite"] <= 1)
        self.assertFalse(np.isclose(aa["composite"], 1.0))
        self.assertFalse(np.isclose(aa["composite"], bb["composite"]))

    def test_invalid_masks_rejected(self):
        for im in (np.zeros((7, 7)), np.ones((7, 7)), np.zeros((6, 7)),
                   np.full((7, 7), np.nan)):
            with self.assertRaises(ValueError):
                image_key(im)
            with self.assertRaises(ValueError):
                similarity(self.a, im)
        with self.assertRaises(ValueError):
            random_six_images(count=0, seed=649)
        with self.assertRaises(ValueError):
            conditional_chance_similarity(self.a, np.zeros((1, 7, 6)))

    def test_signature_is_finite_and_fixed_size(self):
        z = signature(self.a)
        self.assertEqual(z.shape, (27,))
        self.assertTrue(np.isfinite(z).all())
        np.testing.assert_array_equal(z, signature(self.a))


if __name__ == "__main__":
    unittest.main()
