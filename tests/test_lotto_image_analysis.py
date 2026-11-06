"""Phase 7 descriptive image diagnostics and independent IID controls."""
from __future__ import annotations

import unittest

import numpy as np

from lotto_image_analysis import analyze_draws, image_features
from lotto_image_repr import MAPPINGS, encode_draws


class ImageAnalysisTests(unittest.TestCase):
    def test_geometry_for_six_contiguous_pixels(self):
        mask = encode_draws(np.array([1, 2, 3, 4, 5, 6]))
        features = image_features(mask)
        self.assertEqual(features["four_neighbor_pairs"], 5)
        self.assertEqual(features["diagonal_neighbor_pairs"], 0)
        self.assertEqual(features["row_occupancy"], [6, 0, 0, 0, 0, 0, 0])
        self.assertEqual(features["column_occupancy"], [1, 1, 1, 1, 1, 1, 0])

    def test_each_mapping_has_reproducible_iid_control(self):
        rng = np.random.default_rng(7)
        draws = np.stack([np.sort(rng.choice(49, 6, replace=False) + 1)
                          for _ in range(36)])
        for mapping in MAPPINGS:
            a = analyze_draws(draws, mapping=mapping, control_samples=120, seed=99)
            b = analyze_draws(draws, mapping=mapping, control_samples=120, seed=99)
            self.assertEqual(a, b)
            self.assertEqual(a["draws"], 36)
            self.assertEqual(a["iid_control_draws"], 120)
            self.assertAlmostEqual(a["iid_theoretical_mean_consecutive_overlap"], 36/49)
            self.assertTrue(np.isfinite(a["observed_mean_four_neighbor_pairs"]))

    def test_invalid_masks_and_control_counts_rejected(self):
        with self.assertRaises(ValueError):
            image_features(np.zeros((7, 7)))
        with self.assertRaises(ValueError):
            image_features(np.ones((7, 7)))
        with self.assertRaises(ValueError):
            analyze_draws(np.array([[1, 2, 3, 4, 5, 6]]), control_samples=10)
        with self.assertRaises(ValueError):
            analyze_draws(np.tile([1, 2, 3, 4, 5, 6], (3, 1)),
                          control_samples=0)


if __name__ == "__main__":
    unittest.main()
