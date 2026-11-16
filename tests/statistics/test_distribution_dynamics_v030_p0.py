"""v0.30 P0: exact fair martingale identities and full-path IID controls."""
import json
import unittest

import numpy as np

from lotto7x7 import distribution_dynamics_v030 as dynamics
from lotto7x7 import geometry_repair_v029 as p3b
from lotto7x7 import geometry_convergence_v029 as p2a
from lotto7x7 import geometry_structure_v029 as p2b


class TestDistributionDynamicsV030P0(unittest.TestCase):
    @staticmethod
    def history(n=130, seed=91):
        return dynamics._uniform_history(np.random.default_rng(seed), n)

    def test_exact_reference_cannot_fit_history(self):
        theory = dynamics.exact_null_dynamics()
        original = p3b.exact_structural_covariance()
        self.assertEqual(list(p2b.PAIR_FEATURES), theory["feature_names"])
        self.assertEqual(original["mean"], theory["exact_mean"])
        self.assertEqual(original["covariance"], theory["exact_covariance"])
        self.assertTrue(theory["fair_quadratic_energy_increment"] > 0)

    def test_transformed_covariance_is_exact_and_near_identity(self):
        theory = dynamics.exact_null_dynamics()
        C = np.array(theory["whitened_fair_covariance"])
        corr = np.asarray(p3b.exact_structural_covariance()["correlation"])
        eig = np.linalg.eigvalsh(corr)
        ridge = p3b.RIDGE
        self.assertTrue(np.all(np.linalg.eigvalsh(C) > 0))
        self.assertTrue(np.allclose(C, C.T))
        self.assertAlmostEqual(float(np.trace(C)),
                               float(np.sum(eig / (eig + ridge))), places=10)
        self.assertTrue(0 < np.trace(C) < 3)

    def test_uniform_iid_samples_legal_six_without_replacement(self):
        samples = self.history(200)
        self.assertEqual((200, 6), samples.shape)
        self.assertTrue(np.all((samples >= 1) & (samples <= 49)))
        sorted_rows = np.sort(samples, axis=1)
        self.assertTrue(np.all(np.diff(sorted_rows, axis=1) > 0))

    def test_empirical_innovations_agree_with_exact_moments(self):
        history = self.history(20000)
        theory = dynamics.exact_null_dynamics()
        z = dynamics._z(history, theory)
        C = np.asarray(theory["whitened_fair_covariance"])
        se = np.sqrt(np.diag(C) / len(history))
        self.assertTrue(np.all(np.abs(z.mean(axis=0)) < 5 * se))
        self.assertTrue(np.allclose(np.cov(z, rowvar=False), C, atol=.06))

    def test_conditional_energy_martingale_is_algebraically_correct(self):
        history = self.history()
        theory = dynamics.exact_null_dynamics()
        z = dynamics._z(history, theory)
        S = np.cumsum(z, axis=0)
        E = np.einsum("ij,ij->i", S, S)
        trace = theory["fair_quadratic_energy_increment"]
        M = E - np.arange(1, len(E) + 1) * trace
        direct = (2 * np.einsum("ij,ij->i", S[:-1], z[1:])
                  + np.einsum("ij,ij->i", z[1:], z[1:]) - trace)
        self.assertTrue(np.allclose(np.diff(M), direct, atol=1e-9))
        self.assertAlmostEqual(E[0], np.dot(z[0], z[0]))

    def test_conditional_next_energy_identity_and_zero_alignment(self):
        history = self.history()
        theory = dynamics.exact_null_dynamics()
        rows = dynamics._path_statistics(history, theory, [25, 100, 130],
                                         details=True)["prefix_diagnostics"]
        expected_trace = theory["fair_quadratic_energy_increment"]
        for row in rows:
            n = row["draw_count"]
            current = row["observed_energy_D_n"]
            self.assertAlmostEqual(
                row["iid_expected_next_D"],
                (n * current + expected_trace) / (n + 1),
            )
            self.assertAlmostEqual(
                row["iid_expected_next_D_change"],
                (expected_trace - current) / (n + 1),
            )
            self.assertEqual(0.0, row["iid_expected_next_raw_compensation"])
            self.assertAlmostEqual(
                row["observed_corrected_energy_martingale_M_n"],
                n * (current - expected_trace),
            )

    def test_alignment_has_predictable_conditional_mean_zero_under_iid(self):
        theory = dynamics.exact_null_dynamics()
        C = np.asarray(theory["whitened_fair_covariance"])
        rng = np.random.default_rng(42)
        z = dynamics._z(dynamics._uniform_history(rng, 30000), theory)
        anchor = np.asarray([3.0, -8.0, 1.5])
        alignment = -(z @ anchor)
        predicted_var = float(anchor @ C @ anchor)
        self.assertLess(abs(float(np.mean(alignment))),
                        5 * np.sqrt(predicted_var / len(z)))
        self.assertLess(abs(float(np.var(alignment, ddof=1)) - predicted_var),
                        .08 * predicted_var)

    def test_whole_search_iid_determinism(self):
        history = self.history()
        a = dynamics.analyze_dynamics(history, iid_replicates=3, seed=649)
        b = dynamics.analyze_dynamics(history, iid_replicates=3, seed=649)
        self.assertEqual(a, b)
        self.assertEqual(a["decision"], "MARTINGALE_NULL_BASELINE_NO_PREDICTIVE_GO")
        self.assertEqual(a["draws"], len(history))
        self.assertEqual(a["matched_entire_search_iid"]["minimum_resolvable_p"], .25)
        self.assertTrue(len(a["observed"]["prefix_diagnostics"]) >= 2)
        self.assertIsInstance(json.dumps(a, allow_nan=False), str)

    def test_two_endpoint_bonferroni_adjustment(self):
        r = dynamics.analyze_dynamics(self.history(100), iid_replicates=9)
        null = r["matched_entire_search_iid"]
        for p, adjusted in (
            ("primary_alignment_empirical_p", "two_endpoint_bonferroni_primary_p"),
            ("secondary_energy_empirical_p", "two_endpoint_bonferroni_secondary_p"),
        ):
            self.assertAlmostEqual(null[adjusted], min(1., 2 * null[p]))
            self.assertTrue(.1 <= null[p] <= 1)

    def test_prefix_is_unchanged_if_later_draw_changes(self):
        history = self.history(160)
        changed = history.copy()
        changed[-1] = [1, 2, 3, 4, 5, 6]
        theory = dynamics.exact_null_dynamics()
        a = dynamics._path_statistics(history, theory, [100, 160], details=True)
        b = dynamics._path_statistics(changed, theory, [100, 160], details=True)
        self.assertEqual(a["prefix_diagnostics"][0], b["prefix_diagnostics"][0])

    def test_repeated_non_iid_history_does_not_crash_or_predict(self):
        x = np.tile(np.arange(1, 7), (100, 1))
        out = dynamics.analyze_dynamics(x, iid_replicates=2)
        self.assertEqual(out["decision"], "MARTINGALE_NULL_BASELINE_NO_PREDICTIVE_GO")
        self.assertGreater(out["observed"]["secondary_max_absolute_prefix_energy_minus_trace"], 1)

    def test_fail_closed_malformed_input_and_bad_config(self):
        h = self.history(50)
        invalid = (h[:10], h[:, :5], h.astype(float))
        for item in invalid:
            with self.assertRaises(ValueError):
                dynamics.analyze_dynamics(item, iid_replicates=1)
        duplicate = h.copy()
        duplicate[0, 1] = duplicate[0, 0]
        with self.assertRaises(ValueError):
            dynamics.analyze_dynamics(duplicate, iid_replicates=1)
        for invalid_r in (0, 2000, True, 1.5):
            with self.assertRaises(ValueError):
                dynamics.analyze_dynamics(h, iid_replicates=invalid_r)
        with self.assertRaises(ValueError):
            dynamics.analyze_dynamics(h, iid_replicates=2, seed=-1)


if __name__ == "__main__":
    unittest.main()
