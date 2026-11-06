"""Tests of Phase 7C year-preserving time-order permutation control."""
from __future__ import annotations

import unittest

import numpy as np

from lotto_image_repr import encode_draws
from lotto_pattern_temporal_audit import (
    LAGS, lag_correlation, normalized_signatures, permutation_audit,
    shuffle_within_year,
)


class TemporalPermutationAuditTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(22)
        draws = np.stack([rng.choice(49, 6, replace=False) + 1
                          for _ in range(90)])
        self.images = encode_draws(draws)
        self.years = np.array([2018] * 30 + [2019] * 30 + [2020] * 30)
        self.dates = self.years * 10000 + 101

    def test_standardized_signatures_fixed_27_dims(self):
        z = normalized_signatures(self.images, train_end=50)
        self.assertEqual(z.shape, (90, 27))
        self.assertTrue(np.all(np.isfinite(z)))
        self.assertTrue(np.all(np.linalg.norm(z, axis=1) <= 1.000001))

    def test_standardization_is_not_fit_on_test_images(self):
        base = normalized_signatures(self.images, train_end=50)
        altered = self.images.copy()
        altered[60:] = np.repeat(self.images[0][None], 30, axis=0)
        changed = normalized_signatures(altered, train_end=50)
        np.testing.assert_array_equal(base[:50], changed[:50])

    def test_shuffle_preserves_each_year_image_multiset(self):
        z = normalized_signatures(self.images, train_end=50)
        shuffled = shuffle_within_year(
            z, self.years, start=20, end=80,
            rng=np.random.default_rng(444))
        np.testing.assert_array_equal(shuffled[:20], z[:20])
        np.testing.assert_array_equal(shuffled[80:], z[80:])
        for year in (2018, 2019, 2020):
            index = np.flatnonzero((self.years == year)
                                   & (np.arange(90) >= 20)
                                   & (np.arange(90) < 80))
            np.testing.assert_array_equal(
                np.sort(shuffled[index], axis=0),
                np.sort(z[index], axis=0))

    def test_lag_excludes_calendar_year_boundaries(self):
        z = normalized_signatures(self.images, train_end=50)
        _, count = lag_correlation(z, self.years, start=0, end=90, lag=1)
        self.assertEqual(count, 87)
        _, count2 = lag_correlation(z, self.years, start=30, end=60, lag=3)
        self.assertEqual(count2, 27)

    def test_constant_within_year_is_not_mistaken_for_temporal_signal(self):
        sample = np.repeat(
            encode_draws([1, 2, 3, 4, 5, 6])[None], 90, axis=0)
        r = permutation_audit(
            sample, self.dates, train_end=60,
            segments={"test": (60, 90)},
            lags=(1,), permutations=19, seed=11)
        lag = r["segments"]["test"]["lags"]["1"]
        self.assertAlmostEqual(lag["observed_mean_cosine"], 0.0)
        self.assertAlmostEqual(lag["shuffled_mean_cosine"], 0.0)
        self.assertAlmostEqual(lag["empirical_two_sided_tail"], 1.0)

    def test_block_regime_has_greater_positive_correlation_than_shuffle(self):
        a = encode_draws([1, 2, 3, 4, 5, 6])
        b = encode_draws([8, 14, 20, 26, 32, 38])
        blocks = np.stack([a] * 45 + [b] * 45)
        years = np.full(90, 2018 * 10000 + 101)
        result = permutation_audit(
            blocks, years, train_end=60,
            segments={"training": (0, 60)},
            lags=(1,), permutations=19, seed=99)
        lag = result["segments"]["training"]["lags"]["1"]
        self.assertGreater(lag["observed_mean_cosine"],
                           lag["shuffled_mean_cosine"])

    def test_two_sided_tail_detects_negative_or_positive_departure(self):
        a = encode_draws([1, 2, 3, 4, 5, 6])
        b = encode_draws([8, 14, 20, 26, 32, 38])
        alternating = np.stack([a if i % 2 else b for i in range(90)])
        years = np.full(90, 20180101)
        report = permutation_audit(
            alternating, years, train_end=60,
            segments={"test": (60, 90)},
            lags=(1,), permutations=19, seed=3)
        lag = report["segments"]["test"]["lags"]["1"]
        self.assertLess(lag["observed_mean_cosine"],
                        lag["shuffled_mean_cosine"])
        self.assertLess(lag["empirical_two_sided_tail"], 0.3)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            normalized_signatures(self.images, train_end=95)
        z = normalized_signatures(self.images, train_end=50)
        with self.assertRaises(ValueError):
            lag_correlation(z, self.years, start=0, end=10, lag=10)
        with self.assertRaises(ValueError):
            permutation_audit(self.images, self.dates, train_end=60,
                              segments={"all": (0, 90)}, permutations=1)
        with self.assertRaises(ValueError):
            shuffle_within_year(z, self.years, start=50, end=40,
                                rng=np.random.default_rng(4))


if __name__ == "__main__":
    unittest.main()
