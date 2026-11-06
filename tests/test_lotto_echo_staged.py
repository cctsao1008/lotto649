"""Phase 4.2: MLE parity, validation-safe staging and causal train split."""
import math
import unittest

import numpy as np
import torch

from lotto_echo_staged import fit_staged_residual, run, trial
from lotto_neural import chronological_split, make_observations, mean_log_score
from lotto_synthetic import generate
from lotto_temporal import fit_beta, generate_lag_echo, lag_mean_log_score
from lotto_torch_decoder import FixedCardinalityDecoder


class EchoStagedTests(unittest.TestCase):
    def test_train_only_mle_not_affected_by_test_draws(self):
        draws = generate_lag_echo(count=120, seed=11, strength=2.0)
        betahat = fit_beta(draws, 72)
        changed = draws.copy()
        changed[72:] = np.tile([1, 2, 3, 4, 5, 6], (48, 1))
        self.assertAlmostEqual(betahat, fit_beta(changed, 72), places=12)

    def test_epoch_zero_is_exact_frozen_mle_distribution(self):
        draws = generate_lag_echo(count=100, seed=15, strength=1.4)
        history, target, _ = make_observations(draws.astype(np.int64), 3)
        train, val, test = chronological_split(len(draws), 3)
        ids = lambda indices: torch.from_numpy(indices - 3)
        beta = fit_beta(draws, int(train[-1]) + 1)
        model, info = fit_staged_residual(
            beta, window=3, hidden=8,
            x_train=history[ids(train)], y_train=target[ids(train)],
            x_val=history[ids(val)], y_val=target[ids(val)],
            seed=22, epochs=1, patience=1, batch_size=32,
            min_delta=1e6,
        )
        self.assertFalse(info["selected_residual"])
        self.assertEqual(info["best_epoch"], 0)
        self.assertFalse(model.echo_gain.requires_grad)
        calculated = mean_log_score(
            model, FixedCardinalityDecoder(),
            history[ids(test)], target[ids(test)], batch_size=32)
        analytic = lag_mean_log_score(draws, test, beta)
        self.assertAlmostEqual(calculated, analytic, delta=1e-5)

    def test_staged_residual_cannot_rewrite_frozen_gain(self):
        draws = generate_lag_echo(count=100, seed=21, strength=2.0)
        history, targets, _ = make_observations(draws.astype(np.int64), 3)
        train, val, _ = chronological_split(len(draws), 3)
        ids = lambda positions: torch.from_numpy(positions - 3)
        fitted_beta = fit_beta(draws, int(train[-1]) + 1)
        model, info = fit_staged_residual(
            fitted_beta, window=3, hidden=8,
            x_train=history[ids(train)], y_train=targets[ids(train)],
            x_val=history[ids(val)], y_val=targets[ids(val)],
            seed=33, epochs=2, patience=2, batch_size=32,
        )
        self.assertEqual(float(model.echo_gain.item()), float(torch.tensor(fitted_beta)))
        self.assertGreaterEqual(info["validation_improvement_nats"], 0)
        self.assertTrue(math.isfinite(info["best_val_log_score"]))

    def test_complete_trial_and_output_fields(self):
        draws = generate(count=100, mode="iid", seed=18)[0]
        out = trial(draws=draws, seed=18, window=3, hidden=8,
                    epochs=2, patience=2, batch_size=32)
        self.assertEqual(out["train_range"], [3, 59])
        self.assertEqual(out["val_range"], [60, 79])
        self.assertEqual(out["test_range"], [80, 99])
        for field in ("legacy_echo", "mle_echo", "staged_residual",
                      "mle_minus_legacy", "staged_minus_mle"):
            self.assertTrue(math.isfinite(out[field]))

    def test_run_single_seed_smoke(self):
        result = run(seeds=1, count=100, start_seed=40001,
                     modes=("iid",), window=3, hidden=8,
                     epochs=1, patience=1, batch_size=32,
                     bootstrap_samples=20)
        self.assertEqual(result["schema"], "lotto649.neural.phase4.staged.v1")
        self.assertEqual(len(result["rows"]), 1)
        self.assertIsNone(result["summary"]["iid"]["staged_residual"]["std_nats"])
        self.assertGreaterEqual(result["summary"]["iid"]["residual_selected_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
