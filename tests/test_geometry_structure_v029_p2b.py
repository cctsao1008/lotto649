"""P2B pair-distance exact finite-population null and graph L3 redundancy tests."""
import unittest

import numpy as np

from lotto7x7 import geometry_structure_v029 as p2b
from lotto7x7 import geometry_convergence_v029 as p2a
from lotto7x7 import geometry_v029 as geom


class TestGeometryStructureV029P2B(unittest.TestCase):
    @staticmethod
    def synthetic(n=160, seed=123):
        rng = np.random.default_rng(seed)
        return rng.permuted(
            np.broadcast_to(np.arange(1, 50, dtype=np.int16), (n, 49)),
            axis=1)[:, :6]

    def test_exact_pair_moments_and_known_means(self):
        moments = p2b.exact_pair_moments()
        self.assertEqual(set(moments), set(p2b.FEATURES))
        self.assertAlmostEqual(moments["pair_squared_distance_sum"]["mean"], 245.0)
        self.assertAlmostEqual(moments["pair_midrange_3_to_18"]["mean"], 7.88265306122449)
        self.assertAlmostEqual(moments["pair_far_25_plus"]["mean"], 4.362244897959184)
        self.assertAlmostEqual(moments["cardinal_edges"]["mean"], 1.0714285714285714)
        self.assertAlmostEqual(moments["diagonal_edges"]["mean"], 0.9183673469387755)
        self.assertAlmostEqual(moments["pair_squared_distance_sum"]["variance"],
                               5048.2238667900165, places=6)
        self.assertTrue(all(v["variance"] > 0 for v in moments.values()))

    def test_pair_feature_definition_is_fixed(self):
        self.assertEqual(3, len(p2b.PAIR_FEATURES))
        self.assertEqual(2, len(p2b.GRAPH_CONTROLS))
        self.assertEqual(15, len(p2b.I6))
        self.assertEqual(1176, len(p2b.I49))
        self.assertEqual(4, len(p2a.LAYOUTS))

    def test_graph_edge_features_match_existing_l3_all_mappings(self):
        draws = self.synthetic()
        for method, seed in p2a.LAYOUTS:
            labels = p2b._mapping_labels(method, seed)
            matrix, codes = p2b._features_and_l3(draws, labels)
            self.assertEqual((len(draws), 5), matrix.shape)
            self.assertEqual((len(draws),), codes.shape)
            self.assertTrue(np.all(np.sum(matrix[:, 3:], axis=1) <= 15))
            self.assertTrue(np.all(matrix[:, 0] >= 0))
            self.assertTrue(np.all(matrix[:, 1] >= 0))
            self.assertTrue(np.all(matrix[:, 2] >= 0))

    def test_distance_is_not_determined_by_l3(self):
        tickets = np.asarray([[1, 3, 5, 15, 17, 19],
                              [1, 7, 22, 28, 43, 49]], dtype=np.uint8)
        values, codes = p2b._features_and_l3(
            tickets, p2b._mapping_labels("sequential", 0))
        self.assertEqual(int(codes[0]), int(codes[1]))
        self.assertNotEqual(float(values[0, 0]), float(values[1, 0]))
        d = p2b._within_l3_descriptive(values, codes)
        self.assertEqual(d["l3_classes_with_multiple_draws"], 1)
        self.assertEqual(d["repeated_l3_classes_with_distance_variation"], 1)
        self.assertEqual(d["distinct_observed_l3_classes"], 1)

    def test_exact_null_mean_matches_independent_fair_monte_carlo(self):
        r = self.synthetic(20000, 611)
        vals, _ = p2b._features_and_l3(r, p2b._mapping_labels("sequential", 0))
        theory = p2b.exact_pair_moments()
        for idx, name in enumerate(p2b.FEATURES):
            se = np.sqrt(theory[name]["variance"] / len(r))
            self.assertLess(abs(np.mean(vals[:, idx]) - theory[name]["mean"]), 5 * se)

    def test_whole_pipeline_iid_reproducibility(self):
        history = self.synthetic()
        a = p2b.analyze_structure(history, iid_replicates=7, seed=649)
        b = p2b.analyze_structure(history, iid_replicates=7, seed=649)
        self.assertEqual(a, b)
        self.assertEqual(a["decision"], "EXPLORATORY_DESCRIPTIVE_ONLY_NOT_PREDICTIVE")
        self.assertEqual(a["same_pipeline_iid"]["minimum_resolvable_p"], 0.125)
        self.assertEqual(a["draws"], len(history))
        self.assertEqual(len(a["observed"]["by_mapping"]), 4)

    def test_prefix_invariance_under_future_modification(self):
        history = self.synthetic()
        changed = history.copy()
        changed[-1] = [1, 2, 3, 4, 5, 6]
        a = p2b.analyze_structure(history, iid_replicates=2)
        b = p2b.analyze_structure(changed, iid_replicates=2)
        for method in a["mapping_ids"]:
            self.assertEqual(a["observed"]["by_mapping"][method]["prefixes"][0],
                             b["observed"]["by_mapping"][method]["prefixes"][0])

    def test_bonferroni_uses_both_global_endpoints(self):
        history = np.tile(np.arange(1, 7), (160, 1))
        report = p2b.analyze_structure(history, iid_replicates=9)
        null = report["same_pipeline_iid"]
        self.assertGreater(report["observed"]["maximum_absolute_pair_z"], 3.0)
        for family in ("pair", "graph_control"):
            p = null[f"{family}_max_z_empirical_p"]
            adjusted_key = "pair_two_endpoint_bonferroni_p" if family == "pair" else "graph_two_endpoint_bonferroni_p"
            self.assertAlmostEqual(null[adjusted_key], min(1., 2 * p))
            self.assertGreaterEqual(p, .1)

    def test_invalid_inputs_fail_closed(self):
        history = self.synthetic()
        cases = [history[:, :5], history[:20], history.astype(float)]
        duplicate = history.copy()
        duplicate[0, 1] = duplicate[0, 0]
        cases.append(duplicate)
        for x in cases:
            with self.assertRaises(ValueError):
                p2b.analyze_structure(x, iid_replicates=1)
        with self.assertRaises(ValueError):
            p2b.analyze_structure(history, iid_replicates=True)
        with self.assertRaises(ValueError):
            p2b.analyze_structure(history, iid_replicates=0)
        with self.assertRaises(ValueError):
            p2b.analyze_structure(history, iid_replicates=1, seed=-1)

    def test_d4_cannot_create_pair_distance_anomaly(self):
        ticket = [4, 9, 17, 24, 29, 42]
        base = geom.geometry_features(ticket)
        canonical = p2b._mapping_labels("sequential", 0)
        vals, _ = p2b._features_and_l3(np.array([ticket]), canonical)
        self.assertEqual(float(vals[0, 0]),
                         sum(v * d for v, d in zip(
                             base["graph"]["pair_distance_squared_histogram"],
                             geom.DIST2_SUPPORT)))


if __name__ == "__main__":
    unittest.main()
