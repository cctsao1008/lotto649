"""v0.30 P2A causal score, IID controls, and fail-closed CLI regression."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from lotto7x7 import distribution_prediction_v030 as p2
from lotto7x7.distribution_dynamics_v030 import exact_null_dynamics, _z, _uniform_history
from lotto7x7.geometry_structure_v029 import _features_and_l3, _mapping_labels


class TestDistributionPredictionV030P2(unittest.TestCase):
    @staticmethod
    def history(n=75, seed=101):
        return _uniform_history(np.random.default_rng(seed), n)

    def test_fast_features_match_frozen_p0_and_whitening(self):
        draws = self.history(26)
        theory = exact_null_dynamics()
        labels = _mapping_labels("sequential", 0)
        fast = p2.feature_matrix(draws, labels)
        old, _ = _features_and_l3(draws, labels)
        np.testing.assert_array_equal(fast, old[:, :3])
        standardized = (fast - theory["exact_mean"]) / theory["feature_scale"]
        np.testing.assert_allclose(standardized @ theory["fixed_whitener"],
                                   _z(draws, theory), atol=1e-12)

    def test_midrank_exact_tie_and_direction(self):
        self.assertEqual(0.5, p2.midrank_percentile(2, np.array([2, 2, 2])))
        self.assertEqual(0.625, p2.midrank_percentile(2, np.array([1, 2, 2, 2])))
        self.assertEqual(0.0, p2.midrank_percentile(0, np.array([1, 2])))
        self.assertEqual(1.0, p2.midrank_percentile(3, np.array([1, 2])))
        with self.assertRaises(ValueError):
            p2.midrank_percentile(0, np.array([]))

    def test_score_sign_exact_and_reverse_control(self):
        theory = exact_null_dynamics()
        labels = {method if method != "random" else f"random{seed}":
                  _mapping_labels(method, seed) for method, seed in p2.MAPPINGS}
        draws = self.history(30)
        mean = np.asarray(theory["exact_mean"])
        sd = np.asarray(theory["feature_scale"])
        w = np.asarray(theory["fixed_whitener"])
        residuals = {name: p2._standardized(draws, lab, mean, sd).sum(axis=0)
                     for name, lab in labels.items()}
        scores = p2._score_arms(residuals, draws[:3], labels, mean, sd, w)
        actual_z = p2._standardized(draws[:3], labels["sequential"], mean, sd) @ w
        past_z = residuals["sequential"] @ w
        np.testing.assert_allclose(scores["compensation"], -(actual_z @ past_z))
        np.testing.assert_allclose(scores["reverse_compensation"],
                                   -scores["compensation"])

    def test_causality_future_mutation_does_not_change_past_score(self):
        history = self.history()
        altered = history.copy()
        altered[41:] = self.history(len(history) - 41, seed=999)
        labels = _mapping_labels("sequential", 0)
        theory = exact_null_dynamics()
        mu = np.asarray(theory["exact_mean"])
        sd = np.asarray(theory["feature_scale"])
        w = np.asarray(theory["fixed_whitener"])
        before = p2._standardized(history[:40], labels, mu, sd).sum(axis=0) @ w
        after = p2._standardized(altered[:40], labels, mu, sd).sum(axis=0) @ w
        np.testing.assert_array_equal(before, after)
        candidate_z = p2._standardized(history[40:41], labels, mu, sd) @ w
        self.assertAlmostEqual(float((-candidate_z @ before).item()), float((-candidate_z @ after).item()))

    def test_incorrect_history_and_configuration_fail_closed(self):
        clean = self.history()
        with self.assertRaises(ValueError):
            p2.analyze_prediction(clean, start_draw=100)
        with self.assertRaises(ValueError):
            p2.analyze_prediction(clean, references=0)
        broken = clean.copy()
        broken[5, 1] = broken[5, 0]
        with self.assertRaises(ValueError):
            p2.analyze_prediction(broken, start_draw=25, iid_reps=3)
        reversed_history = clean[::-1].copy()
        # Reversing chronology changes the research result; it is not invariant.
        a = p2.analyze_prediction(clean, start_draw=30, step=7,
                                  references=8, iid_reps=3, seed=649)
        b = p2.analyze_prediction(reversed_history, start_draw=30, step=7,
                                  references=8, iid_reps=3, seed=649)
        self.assertNotEqual(a["per_draw_observed"][0]["actual_numbers"],
                            b["per_draw_observed"][0]["actual_numbers"])

    def test_iid_full_pipeline_reproducible_and_auditable(self):
        kwargs = dict(start_draw=35, step=8, references=16, iid_reps=5, seed=649)
        first = p2.analyze_prediction(self.history(), **kwargs)
        second = p2.analyze_prediction(self.history(), **kwargs)
        self.assertEqual(json.dumps(first, sort_keys=True),
                         json.dumps(second, sort_keys=True))
        self.assertEqual("EXPLORATORY_ONLY_NO_PROSPECTIVE_PREDICTIVE_GO",
                         first["decision"])
        self.assertEqual(5, first["matched_full_pipeline_iid"]["replicates"])
        self.assertEqual(len(range(35, 75, 8)), len(first["per_draw_observed"]))
        self.assertGreaterEqual(first["primary"]["one_sided_upper_tail_p"], 1/6)
        self.assertLessEqual(first["primary"]["one_sided_upper_tail_p"], 1)
        self.assertEqual(8, len(first["all_arms"]))
        self.assertEqual("unconditional fair 6/49",
                         first["frozen_configuration"]["reference_sampling"])

    def test_fingerprint_failure_does_not_create_report(self):
        Path("results").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir="results") as folder:
            filename = Path(folder) / "report.json"
            with patch.object(p2, "audit_repository_inputs",
                              side_effect=ValueError("source fingerprint mismatch")):
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    p2.main(["--out", str(filename)])
            self.assertFalse(filename.exists())

    def test_cli_exclusive_create_and_source_gate(self):
        Path("results").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir="results") as folder:
            filename = Path(folder) / "report.json"
            with patch.object(p2, "audit_repository_inputs",
                              return_value=(self.history(), {"status": "PASS"})):
                p2.main(["--out", str(filename), "--start-draw", "30",
                         "--step", "8", "--references", "8", "--iid-reps", "3"])
                result = json.loads(filename.read_text("ascii"))
                self.assertEqual("PASS", result["source_gate"]["status"])
                with self.assertRaises(FileExistsError):
                    p2.main(["--out", str(filename), "--start-draw", "30",
                             "--step", "8", "--references", "8", "--iid-reps", "3"])


if __name__ == "__main__":
    unittest.main()
