"""Phase 4.1 Echo-aware models: exactness, gradients, causal input and execution."""
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from lotto_echo import EchoOnlyModel, EchoResidualMLP, run
from lotto_neural import make_observations
from lotto_temporal import generate_lag_echo, overlap_distribution
from lotto_torch_decoder import FixedCardinalityDecoder


class EchoInductiveBiasTests(unittest.TestCase):
    def test_exact_lag_echo_model_probability(self):
        draws = generate_lag_echo(count=60, seed=42, strength=2.0)
        x, y, _ = make_observations(draws, window=5)
        model = EchoOnlyModel(window=5)
        with torch.no_grad():
            model.echo_gain.fill_(2.0)
        actual = FixedCardinalityDecoder()(model(x), y).detach().numpy()
        _, log_z = overlap_distribution(2.0)
        overlaps = np.sum(np.any(
            draws[5:, :, None] == draws[4:-1, None, :], axis=2), axis=1)
        expected = 2.0 * overlaps - log_z
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)

    def test_causal_last_draw_connection(self):
        model = EchoOnlyModel(window=3)
        with torch.no_grad():
            model.echo_gain.fill_(1.5)
        history = torch.zeros(2, 3 * 49)
        history[:, 0] = 1.0      # older history should not be copied
        history[0, -49 + 7] = 1.0
        history[1, -49 + 10] = 1.0
        output = model(history)
        self.assertAlmostEqual(output[0, 7].item(), 1.5)
        self.assertAlmostEqual(output[1, 10].item(), 1.5)
        self.assertEqual(output[0, 0].item(), 0.0)

    def test_models_initialize_at_iid(self):
        x = torch.randn(4, 3 * 49)
        for model in (EchoOnlyModel(3), EchoResidualMLP(3, hidden=8)):
            self.assertTrue(torch.allclose(model(x), torch.zeros(4, 49)))

    def test_scalar_and_neural_gradient(self):
        draws = generate_lag_echo(count=70, seed=31, strength=1.8)
        x, y, _ = make_observations(draws, window=3)
        model = EchoResidualMLP(3, hidden=8)
        loss = -FixedCardinalityDecoder()(model(x[:8]), y[:8]).mean()
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        self.assertGreater(abs(model.echo_gain.grad.item()), 0.01)

    def test_unsafe_shapes(self):
        with self.assertRaises(ValueError):
            EchoOnlyModel(0)
        with self.assertRaises(ValueError):
            EchoOnlyModel(5)(torch.zeros(2, 49))

    def test_run_with_mocked_base_does_not_pass_oracle(self):
        def fake_evaluate(*, data, seed, **kwargs):
            with np.load(data, allow_pickle=False) as archive:
                self.assertEqual(archive.files, ["numbers"])
                self.assertEqual(archive["numbers"].shape, (100, 6))
            return {"dataset_sha256": "mock_sha",
                    "train_range": [5, 59], "val_range": [60, 79],
                    "test_range": [80, 99],
                    "models": {
                        "static": {"test_lift_nats": -0.001},
                        "mlp": {"test_lift_nats": 0.002},
                    }}

        with patch("lotto_echo.evaluate", side_effect=fake_evaluate):
            output = run(seeds=1, draws=100, modes=("lag_echo",),
                         epochs=1, patience=1, batch_size=32,
                         bootstrap_samples=20)
        self.assertEqual(len(output["rows"]), 1)
        self.assertEqual(output["schema"], "lotto649.neural.phase4.echo.v1")
        self.assertTrue(math.isfinite(output["rows"][0]["echo_only"]))
        self.assertTrue(math.isfinite(output["rows"][0]["echo_residual"]))
        self.assertEqual(output["rows"][0]["mlp"], 0.002)
        self.assertIsNone(output["summary"]["lag_echo"]["echo_only"]["std_nats"])


if __name__ == "__main__":
    unittest.main()
