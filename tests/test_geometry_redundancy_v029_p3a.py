"""v0.29 P3A strict conditional structural non-identity and negative controls."""
import json
import unittest

import numpy as np

from lotto7x7 import geometry_redundancy_v029 as p3
from lotto7x7 import geometry_v029 as geom
from lotto7x7.log_product_v027 import _grid
from lotto7x7.shape_taxonomy_v015 import classify_numbers


class TestGeometryRedundancyV029P3A(unittest.TestCase):
    @staticmethod
    def historical_fixture():
        a, b = p3.TICKETS_CERTIFICATE
        return np.asarray([a, b] * 13, dtype=np.int16)

    def test_strict_collision_certificate(self):
        certificate = p3._certificate()
        self.assertTrue(certificate["structural_nonidentity_proven"])
        self.assertEqual(132, certificate["same_v028_baseline"]["sum"])
        self.assertEqual(20, certificate["same_v028_baseline"]["log_bin"])
        self.assertEqual(
            [172, 258],
            [row["pair_geometry"]["pair_squared_distance_sum"]
             for row in certificate["tickets"]],
        )
        self.assertEqual([18694860, 16568064],
                         [record["exact_product"] for record in certificate["tickets"]])
        self.assertIsInstance(json.dumps(certificate, allow_nan=False), str)
        for record in certificate["tickets"]:
            self.assertEqual(6, len(record["numbers"]))
            self.assertEqual(sorted(set(record["numbers"])), record["numbers"])

    def test_collision_also_has_same_nested_l1_l2_l3(self):
        rows = np.asarray(p3.TICKETS_CERTIFICATE, dtype=np.int16)
        scores = classify_numbers(rows)
        for name in ("n_components", "partition_code", "edge_code",
                     "cardinal_edges", "diagonal_edges"):
            self.assertEqual(int(scores[name][0]), int(scores[name][1]), name)

    def test_existing_fixed_log_product_grid_reused(self):
        expected = _grid(49, 6, "0.5")
        self.assertEqual(expected, p3.log_grid())
        self.assertEqual(20, p3._certificate()["same_v028_baseline"]["log_bin"])
        self.assertGreater(expected["bin_count"], 1)

    def test_baseline_key_includes_shape_sum_log_and_joint_is_derived(self):
        rows = np.asarray(self.historical_fixture())
        keys, features, numbers = p3._descriptors(rows)
        self.assertEqual((26, 3), keys.shape)
        self.assertEqual((26, 3), features.shape)
        self.assertEqual((26, 6), numbers.shape)
        self.assertTrue(np.array_equal(keys[0], keys[1]))
        self.assertNotEqual(int(features[0][0]), int(features[1][0]))
        self.assertEqual(132, int(keys[0][1]))
        self.assertEqual(20, int(keys[0][2]))

    def test_observed_same_baseline_group_with_different_geometry(self):
        report = p3._stratum_audit(self.historical_fixture())
        self.assertEqual(26, report["sample_draws"])
        self.assertEqual(1, report["observed_baseline_cells"])
        self.assertEqual(1, report["baseline_cells_with_multiple_draws"])
        self.assertEqual(1, report["repeated_cells_with_structural_variation"])
        self.assertGreaterEqual(
            report["repeated_cells_variable_by_feature"]["pair_squared_distance_sum"], 1)

    def test_identical_tickets_not_misclassified_as_variation(self):
        repeats = np.tile(np.arange(1, 7), (30, 1))
        report = p3._stratum_audit(repeats)
        self.assertEqual(1, report["baseline_cells_with_multiple_draws"])
        self.assertEqual(0, report["repeated_cells_with_structural_variation"])
        self.assertIsNone(report["first_descriptive_example"])

    def test_deterministic_seeded_fair_control(self):
        history = self.historical_fixture()
        a = p3.analyze_conditional_ablation(history, iid_sample=1000, seed=649)
        b = p3.analyze_conditional_ablation(history, iid_sample=1000, seed=649)
        self.assertEqual(a, b)
        self.assertEqual(a["decision"], "STRUCTURAL_NONIDENTITY_PROVEN_PREDICTIVE_NO_GO")
        self.assertEqual(a["iid_sampling"]["sample_size"], 1000)
        self.assertEqual(a["geometry_definition_sha256"],
                         geom.catalog()["definition_sha256"])
        self.assertEqual(26, a["historical_descriptive_collision_audit"]["sample_draws"])

    def test_future_data_change_does_not_change_certificate(self):
        history = self.historical_fixture()
        modified = history.copy()
        modified[-1] = np.arange(1, 7)
        a = p3.analyze_conditional_ablation(history, iid_sample=100, seed=649)
        b = p3.analyze_conditional_ablation(modified, iid_sample=100, seed=649)
        self.assertEqual(a["certificate"], b["certificate"])
        self.assertEqual(a["fair_iid_descriptive_collision_audit"],
                         b["fair_iid_descriptive_collision_audit"])

    def test_invalid_numbers_and_configs_are_rejected(self):
        history = self.historical_fixture()
        for invalid in (history[:20], history[:, :5], history.astype(float)):
            with self.assertRaises(ValueError):
                p3.analyze_conditional_ablation(invalid, iid_sample=30)
        duplicate = history.copy()
        duplicate[0, 1] = duplicate[0, 0]
        with self.assertRaises(ValueError):
            p3.analyze_conditional_ablation(duplicate, iid_sample=30)
        for bad in (0, 24, 500001, True, 1.5):
            with self.assertRaises(ValueError):
                p3.analyze_conditional_ablation(history, iid_sample=bad)
        with self.assertRaises(ValueError):
            p3.analyze_conditional_ablation(history, iid_sample=25, seed=-1)

    def test_never_treat_algebraic_variation_as_predictive_gain(self):
        result = p3.analyze_conditional_ablation(
            self.historical_fixture(), iid_sample=100)
        text = " ".join(result["explicit_nonclaims"])
        self.assertIn("No hypothesis test of predictive advantage", text)
        self.assertIn("P2A and P2B", text)
        self.assertIn("Do not add a fifth candidate selection objective", text)
        self.assertNotIn("selected_tickets", result)
        self.assertNotIn("forecast", result)


if __name__ == "__main__":
    unittest.main()
