"""No-network tests for v0.29 P2A convergence signature and matched IID null."""
import unittest

import numpy as np

from lotto7x7 import geometry_convergence_v029 as p2
from lotto7x7 import geometry_theory_v029 as p1
from lotto7x7 import geometry_v029 as p0


class TestGeometryConvergenceV029P2A(unittest.TestCase):
    @staticmethod
    def synthetic_history(n=120, seed=101):
        rng = np.random.default_rng(seed)
        return rng.permuted(np.broadcast_to(np.arange(1, 50), (n, 49)), axis=1)[:, :6]

    def test_predeclared_search_is_frozen(self):
        self.assertEqual(10, len(p2.REGION_NAMES))
        self.assertEqual(4, len(p2.LAYOUTS))
        self.assertEqual(2, len([m for m, _ in p2.LAYOUTS if m == "random"]))
        self.assertEqual(p0.catalog()["definition_sha256"], p2.P0_DEFINITION_SHA256)
        self.assertEqual([100, 250, 500, 1000, 1500, 2000, 2498], p2._prefixes(2498))

    def test_simulation_is_reproducible(self):
        first = p2.analyze_convergence(self.synthetic_history(), iid_replicates=7, seed=649)
        second = p2.analyze_convergence(self.synthetic_history(), iid_replicates=7, seed=649)
        self.assertEqual(first, second)
        self.assertEqual(first["number_of_draws"], 120)
        self.assertEqual(first["decision"], "EXPLORATORY_DESCRIPTIVE_ONLY_NOT_PREDICTIVE")
        self.assertEqual(len(first["observed"]["by_mapping"]), 4)
        self.assertEqual(first["whole_pipeline_iid_control"]["minimum_resolvable_p"], 0.125)

    def test_observed_histograms_and_theory_normalize(self):
        report = p2.analyze_convergence(self.synthetic_history(), iid_replicates=3)
        for name in report["mapping_ids"]:
            for prefix_row in report["observed"]["by_mapping"][name]["prefixes"]:
                n = prefix_row["draws"]
                for region, row in prefix_row["regions"].items():
                    self.assertEqual(n, sum(row["counts_k0_to_k6"]))
                    self.assertAlmostEqual(
                        1, sum(report["exact_region_probability_k0_to_k6"][region])
                    )
                    self.assertLessEqual(row["total_variation"], 1)

    def test_pairwise_geometry_theory_is_consistent_with_p1(self):
        _, means, variances = p2._null_parameters()
        for k, region in enumerate(p2.REGION_NAMES):
            self.assertAlmostEqual(means[k], 6 * len(p0.REGIONS[region]) / 49)
            self.assertAlmostEqual(variances[k], p1.pair_covariance(region, region))

    def test_previous_prefix_not_affected_by_future_draw(self):
        numbers = self.synthetic_history(120)
        modified = numbers.copy()
        modified[-1] = np.arange(1, 7)
        a = p2.analyze_convergence(numbers, iid_replicates=3)
        b = p2.analyze_convergence(modified, iid_replicates=3)
        for name in a["mapping_ids"]:
            self.assertEqual(
                a["observed"]["by_mapping"][name]["prefixes"][0],
                b["observed"]["by_mapping"][name]["prefixes"][0],
            )
        self.assertNotEqual(
            a["observed"]["by_mapping"]["sequential"]["prefixes"][-1],
            b["observed"]["by_mapping"]["sequential"]["prefixes"][-1],
        )

    def test_invalid_six_balls_and_options_rejected(self):
        history = self.synthetic_history(35)
        for invalid in (history[:20], history[:, :5], history.astype(float)):
            with self.assertRaises(ValueError):
                p2.analyze_convergence(invalid, iid_replicates=1)
        repeated = history.copy()
        repeated[0, 1] = repeated[0, 0]
        with self.assertRaises(ValueError):
            p2.analyze_convergence(repeated, iid_replicates=1)
        for count in (0, True, 10000, 1.2):
            with self.assertRaises(ValueError):
                p2.analyze_convergence(history, iid_replicates=count)
        with self.assertRaises(ValueError):
            p2.analyze_convergence(history, iid_replicates=1, seed=-1)

    def test_global_max_p_values_bonferroni_two_endpoints(self):
        report = p2.analyze_convergence(
            np.tile(np.arange(1, 7), (160, 1)), iid_replicates=19
        )
        baseline = report["whole_pipeline_iid_control"]
        self.assertGreater(report["observed"]["max_abs_z"], 3)
        self.assertGreater(report["observed"]["max_tv"], 0.05)
        for prefix in ("primary", "secondary"):
            raw = baseline[f"{prefix}_empirical_p"]
            adjusted = baseline[f"{prefix}_bonferroni_two_endpoint_p"]
            self.assertAlmostEqual(adjusted, min(1, 2 * raw))
            self.assertGreaterEqual(raw, 1 / 20)
            self.assertLessEqual(raw, 1)

    def test_no_predictions_or_tickets_in_report(self):
        report = p2.analyze_convergence(self.synthetic_history(30), iid_replicates=2)
        for forbidden in ("ticket", "prediction", "payout", "candidate", "prize"):
            self.assertNotIn(forbidden, report.keys())
        self.assertIn("historical deficit", " ".join(report["limitations"]))


if __name__ == "__main__":
    unittest.main()
