"""Phase 3: seed-level summaries, diagnostic integrity and oracle isolation."""
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
    from lotto_robustness import aggregate, row_scores, run, summarize
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch required for Phase 3")
class RobustnessTests(unittest.TestCase):
    def test_summary_and_bootstrap_are_deterministic(self):
        first = summarize([-1.0, 0.0, 1.0, 2.0], bootstrap_samples=200)
        second = summarize([-1.0, 0.0, 1.0, 2.0], bootstrap_samples=200)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean_nats"], 0.5)
        self.assertEqual(first["positive_fraction"], 0.5)
        self.assertEqual(first["replicates"], 4)
        self.assertLessEqual(first["bootstrap_95pct_ci_mean_nats"][0], 0.5)
        self.assertGreaterEqual(first["bootstrap_95pct_ci_mean_nats"][1], 0.5)

    def test_one_seed_has_no_fake_confidence_interval(self):
        result = summarize([0.002])
        self.assertIsNone(result["std_nats"])
        self.assertIsNone(result["bootstrap_95pct_ci_mean_nats"])
        self.assertEqual(result["positive_fraction"], 1.0)
        with self.assertRaises(ValueError):
            summarize([])

    def test_best_of_two_metric_exposes_test_selection_optimism(self):
        result = row_scores({"models": {
            "static": {"test_lift_nats": 0.2},
            "mlp": {"test_lift_nats": -0.5}}})
        self.assertEqual(result["best_of_two_test_diagnostic"], 0.2)
        self.assertAlmostEqual(result["mlp_minus_static"], -0.7)

    def test_aggregate_keeps_failed_runs(self):
        rows = [{"mode": "iid", "static": -0.1, "mlp": 0.02,
                 "mlp_minus_static": 0.12, "best_of_two_test_diagnostic": 0.02},
                {"mode": "iid", "static": -0.03, "mlp": -0.01,
                 "mlp_minus_static": 0.02, "best_of_two_test_diagnostic": -0.01}]
        result = aggregate(rows, bootstrap_samples=100)
        self.assertAlmostEqual(result["iid"]["mlp"]["mean_nats"], 0.005)
        self.assertEqual(result["iid"]["mlp"]["positive_fraction"], 0.5)
        self.assertLess(result["iid"]["best_of_two_test_diagnostic"]["mean_nats"], 0.02)

    def test_seed_pairing_and_oracle_never_reaches_evaluator(self):
        observed = []

        def fake_evaluate(*, data, seed, **kwargs):
            with np.load(data, allow_pickle=False) as ar:
                self.assertEqual(ar.files, ["numbers"])
                n = len(ar["numbers"])
                self.assertEqual(tuple(ar["numbers"].shape), (90, 6))
            observed.append((seed, n))
            return {"train_range": [5, 53], "val_range": [54, 71],
                    "test_range": [72, 89],
                    "dataset_sha256": "abc",
                    "models": {"static": {"test_lift_nats": 0.01, "best_epoch": 1},
                               "mlp": {"test_lift_nats": -0.01, "best_epoch": 2}}}

        with patch("lotto_robustness.evaluate", side_effect=fake_evaluate):
            result = run(seeds=2, start_seed=700, draws=90,
                         epochs=2, bootstrap_samples=30)
        self.assertEqual(observed, [(700, 90), (700, 90), (701, 90), (701, 90)])
        self.assertEqual(len(result["rows"]), 4)
        self.assertEqual(set(result["summary"]), {"iid", "fixed_bias"})
        self.assertEqual([r["mode"] for r in result["rows"]],
                         ["iid", "fixed_bias", "iid", "fixed_bias"])

    def test_reject_invalid_mode_or_replicates(self):
        with self.assertRaises(ValueError):
            run(seeds=0)
        with self.assertRaises(ValueError):
            run(seeds=1, modes=("iid", "iid"))
        with self.assertRaises(ValueError):
            run(seeds=1, modes=("periodic_bias",))


if __name__ == "__main__":
    unittest.main()
