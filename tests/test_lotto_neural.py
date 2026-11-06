"""Phase 2 unit tests: causal feature construction, data isolation and training.

Install torch for training tests. Run:
    python -m unittest discover -s tests -v
"""
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    from lotto_neural import (IID_LOGP, SmallMLP, StaticLogitModel,
                              chronological_split, evaluate, load_draws,
                              make_observations)
    from lotto_synthetic import generate
    from lotto_torch_decoder import FixedCardinalityDecoder
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is needed for Phase 2")
class NeuralLearningTests(unittest.TestCase):
    def setUp(self):
        self.draws = generate(count=120, mode="iid", seed=17)[0].astype(np.int64)

    def test_history_features_never_include_target(self):
        x, y, origins = make_observations(self.draws, window=3)
        self.assertEqual(tuple(x.shape), (117, 147))
        self.assertEqual(tuple(y.shape), (117, 6))
        np.testing.assert_array_equal(origins, np.arange(3, 120))
        np.testing.assert_array_equal(y[0].numpy(), self.draws[3])
        for history_idx in range(3):
            active = np.flatnonzero(x[0, history_idx * 49:(history_idx + 1) * 49]) + 1
            np.testing.assert_array_equal(active, self.draws[history_idx])

    def test_chronological_partition_and_past_context(self):
        train, val, test = chronological_split(120, window=3)
        self.assertLess(train[-1], val[0])
        self.assertLess(val[-1], test[0])
        self.assertEqual(len(set(train) | set(val) | set(test)), 117)
        self.assertEqual(set(train) & set(val), set())
        self.assertEqual(set(val) & set(test), set())
        # First validation example legitimately reads historical train draws.
        self.assertEqual(val[0] - 1, train[-1])

    def test_models_start_at_fair_iid(self):
        x, y, _ = make_observations(self.draws, window=3)
        d = FixedCardinalityDecoder()
        for model in (StaticLogitModel(), SmallMLP(3, hidden=8)):
            scores = d(model(x[:5]), y[:5])
            self.assertTrue(torch.allclose(scores, torch.full_like(scores, IID_LOGP), atol=1e-5))

    def test_gradients_are_finite(self):
        x, y, _ = make_observations(self.draws, window=3)
        model = SmallMLP(3, hidden=8)
        loss = -FixedCardinalityDecoder()(model(x[:6]), y[:6]).mean()
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_oracle_payload_is_not_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.npz"
            np.savez(path, numbers=self.draws.astype(np.uint8),
                     phases=np.asarray([{"hidden": "oracle"}], dtype=object),
                     logits=np.asarray([{"hidden": "oracle"}], dtype=object))
            np.testing.assert_array_equal(load_draws(path), self.draws)

    def test_reject_invalid_draws(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = self.draws.copy()
            bad[0] = [1, 1, 2, 3, 4, 5]
            path = Path(tmp) / "bad.npz"
            np.savez(path, numbers=bad)
            with self.assertRaises(ValueError):
                load_draws(path)

    def test_complete_experiment_produces_finite_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixed_bias.npz"
            numbers, phases, logits = generate(count=100, mode="fixed_bias",
                                              seed=11, strength=2.0)
            np.savez(path, numbers=numbers, phases=phases, logits=logits)
            report = evaluate(data=path, seed=77, window=3, hidden=6,
                              epochs=2, patience=2, batch_size=32)
            self.assertEqual(report["draws"], 100)
            self.assertEqual(report["schema"], "lotto649.neural.phase2.v1")
            self.assertEqual(set(report["models"]), {"static", "mlp"})
            self.assertEqual(report["train_range"], [3, 59])
            self.assertEqual(report["val_range"], [60, 79])
            self.assertEqual(report["test_range"], [80, 99])
            for model in report["models"].values():
                self.assertTrue(math.isfinite(model["test_log_score"]))
                self.assertTrue(math.isfinite(model["test_lift_nats"]))


if __name__ == "__main__":
    unittest.main()
