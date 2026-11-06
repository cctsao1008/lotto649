"""Run from repository root: python -m unittest discover -s tests -v"""
import itertools
import math
import unittest
import numpy as np
from lotto_probability import FixedCardinalityDistribution, fair_log_prob
from lotto_synthetic import generate


class ProbabilityTests(unittest.TestCase):
    def test_uniform_lottery(self):
        dist = FixedCardinalityDistribution(np.zeros(49), 6)
        self.assertEqual(math.comb(49, 6), 13_983_816)
        self.assertAlmostEqual(dist.log_prob([3, 11, 18, 27, 35, 46]), fair_log_prob(), places=12)
        np.testing.assert_allclose(dist.marginals(), 6 / 49, atol=1e-12)
        self.assertAlmostEqual(dist.marginals().sum(), 6, places=12)

    def test_exact_small_space(self):
        logits = np.array([-1.2, 0.3, 1.1, 0.2, 2.0, -0.5, 0.7, -0.8])
        dist = FixedCardinalityDistribution(logits, 3)
        combos = list(itertools.combinations(range(1, 9), 3))
        p = np.array([dist.prob(s) for s in combos])
        self.assertAlmostEqual(p.sum(), 1, places=12)
        marg = [sum(float(v) for s, v in zip(combos, p) if i in s) for i in range(1, 9)]
        np.testing.assert_allclose(dist.marginals(), marg, atol=1e-12)

    def test_logits_shift(self):
        weights = np.linspace(-3, 2, 12)
        d1 = FixedCardinalityDistribution(weights, 4)
        d2 = FixedCardinalityDistribution(weights + 100, 4)
        self.assertAlmostEqual(d1.log_prob([1, 4, 7, 11]), d2.log_prob([1, 4, 7, 11]), places=11)

    def test_weighted_sampler(self):
        weights = np.zeros(8)
        weights[0] = 3
        dist = FixedCardinalityDistribution(weights, 3)
        rng = np.random.default_rng(1234)
        draws = [dist.sample(rng) for _ in range(1200)]
        self.assertTrue(all(len(set(row)) == 3 and 1 <= min(row) <= max(row) <= 8 for row in draws))
        self.assertAlmostEqual(sum(1 in row for row in draws) / len(draws), dist.marginals()[0], delta=0.06)

    def test_invalid_input(self):
        d = FixedCardinalityDistribution(np.zeros(49), 6)
        for subset in ([1, 1, 2, 3, 4, 5], [0, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5], [1.2, 2, 3, 4, 5, 6]):
            with self.assertRaises(ValueError):
                d.log_prob(subset)


class SyntheticTests(unittest.TestCase):
    def test_reproducible_and_legal(self):
        for mode in ("iid", "fixed_bias", "periodic_bias", "deterministic_cycle", "change_point"):
            with self.subTest(mode=mode):
                a = generate(count=120, mode=mode, seed=649, period=8)
                b = generate(count=120, mode=mode, seed=649, period=8)
                for left, right in zip(a, b):
                    np.testing.assert_array_equal(left, right)
                numbers = a[0]
                self.assertEqual(numbers.shape, (120, 6))
                self.assertTrue(np.all((numbers >= 1) & (numbers <= 49)))
                self.assertTrue(np.all(np.diff(numbers, axis=1) > 0))

    def test_injected_period(self):
        nums, phase, _ = generate(count=100, mode="deterministic_cycle", seed=5, period=7)
        np.testing.assert_array_equal(nums[:-7], nums[7:])
        self.assertEqual(len(set(map(tuple, nums[:7].tolist()))), 7)
        self.assertEqual(phase[0], phase[7])

    def test_known_bias_has_positive_expected_score(self):
        nums, _, logits = generate(count=250, mode="fixed_bias", seed=42, strength=2)
        dist = FixedCardinalityDistribution(logits[0], 6)
        mean_gain = float(np.mean([dist.log_prob(row) - fair_log_prob() for row in nums]))
        self.assertGreater(mean_gain, 0.1)


if __name__ == "__main__":
    unittest.main()
