"""Phase 4 temporal synthetic and non-leakage tests."""
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    import torch
    from lotto_temporal import (IID_LOGP, expected_overlap, fit_beta,
                                generate_lag_echo, lag_mean_log_score,
                                overlap_distribution, overlaps,
                                temporal_experiment)
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch required for Phase 4")
class TemporalControlTests(unittest.TestCase):
    def test_overlap_distribution_exact_normalization(self):
        for beta in (-4, -2, 0, 2, 4):
            p, log_z = overlap_distribution(beta)
            self.assertAlmostEqual(p.sum(), 1, places=12)
            self.assertTrue(math.isfinite(log_z))
        p, z = overlap_distribution(0)
        self.assertAlmostEqual(z, math.log(math.comb(49, 6)), places=12)
        self.assertAlmostEqual(expected_overlap(0), 36 / 49, places=12)

    def test_overlap_increases_with_bias(self):
        self.assertLess(expected_overlap(-2), expected_overlap(0))
        self.assertLess(expected_overlap(0), expected_overlap(2))

    def test_legal_draws_and_reproducibility(self):
        a = generate_lag_echo(count=250, seed=123, strength=1.8)
        b = generate_lag_echo(count=250, seed=123, strength=1.8)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (250, 6))
        self.assertTrue(np.all((a >= 1) & (a <= 49)))
        self.assertTrue(np.all(np.diff(a, axis=1) > 0))

    def test_empirical_conditional_overlap(self):
        draws = generate_lag_echo(count=3000, seed=1234, strength=2.0)
        self.assertAlmostEqual(overlaps(draws).mean(), expected_overlap(2), delta=0.12)

    def test_train_only_beta_recovery(self):
        draws = generate_lag_echo(count=2200, seed=42, strength=2.0)
        estimate = fit_beta(draws, train_end=1500)
        self.assertAlmostEqual(estimate, 2.0, delta=0.3)
        # Modifying the held-out tail must NOT change the fitted parameter.
        changed = draws.copy()
        changed[1500:] = np.tile(np.array([[1, 2, 3, 4, 5, 6]], dtype=np.uint8), (700, 1))
        self.assertAlmostEqual(estimate, fit_beta(changed, train_end=1500), places=12)

    def test_lag_model_log_score_oracle_advantage(self):
        draws = generate_lag_echo(count=2500, seed=649, strength=2.0)
        test_ids = np.arange(2000, 2500)
        fitted = fit_beta(draws, 1500)
        self.assertGreater(lag_mean_log_score(draws, test_ids, fitted) - IID_LOGP, 0.5)
        # Under IID-style beta=0, score must equal fair baseline exactly.
        self.assertAlmostEqual(lag_mean_log_score(draws, test_ids, 0), IID_LOGP, places=12)

    def test_bad_inputs(self):
        for strength in (float("nan"), -1, float("inf")):
            with self.assertRaises(ValueError):
                generate_lag_echo(count=100, seed=1, strength=strength)
        with self.assertRaises(ValueError):
            generate_lag_echo(count=2, seed=1)
        with self.assertRaises(ValueError):
            overlap_distribution(float("inf"))
        with self.assertRaises(ValueError):
            fit_beta(generate_lag_echo(count=50, seed=1), train_end=5)

    def test_oracle_not_passed_to_neural_and_summary(self):
        observed = []

        def fake_evaluate(*, data, seed, **kwargs):
            with np.load(data, allow_pickle=False) as npz:
                self.assertEqual(npz.files, ["numbers"])
                self.assertEqual(npz["numbers"].shape, (100, 6))
            observed.append(seed)
            return {"dataset_sha256": "abc",
                    "train_range": [5, 59],
                    "val_range": [60, 79],
                    "test_range": [80, 99],
                    "models": {
                        "static": {"test_lift_nats": -0.01, "best_epoch": 1},
                        "mlp": {"test_lift_nats": 0.02, "best_epoch": 2},
                    }}

        with patch("lotto_temporal.evaluate", side_effect=fake_evaluate):
            result = temporal_experiment(seeds=2, start_seed=10000, count=100,
                                         epochs=2, bootstrap_samples=30)
        self.assertEqual(observed, [10000, 10000, 10001, 10001])
        self.assertEqual(len(result["rows"]), 4)
        self.assertEqual(set(result["summary"]), {"iid", "lag_echo"})
        self.assertAlmostEqual(result["summary"]["lag_echo"]["mlp"]["mean_nats"], 0.02)
        self.assertTrue(all("fitted_lag_beta" in r for r in result["rows"]))


if __name__ == "__main__":
    unittest.main()
