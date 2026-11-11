"""v0.29 P3B conditional counterfactual repair and matched IID regression suite."""
import json
import unittest

import numpy as np

from lotto7x7 import geometry_repair_v029 as p3b
from lotto7x7 import geometry_structure_v029 as p2b
from lotto7x7 import geometry_v029 as p0


class TestGeometryRepairV029P3B(unittest.TestCase):
    @staticmethod
    def fair_draws(n=80, seed=7):
        rng = np.random.default_rng(seed)
        return rng.permuted(
            np.broadcast_to(np.arange(1, 50, dtype=np.int16), (n, 49)), axis=1
        )[:, :6]

    def test_exact_covariance_matches_p2b_and_is_spd(self):
        theory = p3b.exact_structural_covariance()
        self.assertEqual(list(p2b.PAIR_FEATURES), theory["feature_names"])
        mean = np.asarray(theory["mean"])
        covariance = np.asarray(theory["covariance"])
        self.assertEqual((3, 3), covariance.shape)
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0))
        self.assertTrue(np.allclose(covariance, covariance.T))
        self.assertTrue(np.allclose(mean, [245.0, 7.88265306122449, 4.362244897959184]))
        exact = p2b.exact_pair_moments()
        for i, name in enumerate(p2b.PAIR_FEATURES):
            self.assertAlmostEqual(mean[i], exact[name]["mean"], places=10)
            self.assertAlmostEqual(covariance[i, i], exact[name]["variance"], places=9)

    def test_regularized_whitening_is_symmetric(self):
        t = p3b.exact_structural_covariance()
        w = np.asarray(t["whitener"])
        corr = np.asarray(t["correlation"])
        self.assertTrue(np.allclose(w, w.T, atol=1e-10))
        self.assertTrue(np.allclose(w @ (corr + p3b.RIDGE * np.eye(3)) @ w,
                                    np.eye(3), atol=1e-9))

    def test_full_pool_unique_and_uniform_lottery_rules(self):
        a = p3b._unique_pool(np.random.default_rng(649), 160)
        b = p3b._unique_pool(np.random.default_rng(649), 160)
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(160, len(set(tuple(map(int, r)) for r in a)))
        self.assertTrue(np.all(np.diff(a, axis=1) > 0))
        self.assertTrue(np.all((a >= 1) & (a <= 49)))

    def test_random_and_optimized_four_ticket_portfolios_are_disjoint(self):
        rng = np.random.default_rng(649)
        pool = p3b._unique_pool(rng, 160)
        masks = p3b._bitmasks(pool)
        random_sets = p3b._random_disjoint_portfolios(masks, rng, 25)
        for p in random_sets:
            self.assertEqual(len(set(map(int, p))), 4)
            self.assertEqual(24, len(set(int(n) for ticket in pool[p] for n in ticket)))
        theory = p3b.exact_structural_covariance()
        mean = np.asarray(theory["mean"])
        cov = np.asarray(theory["covariance"])
        w = np.asarray(theory["whitener"])
        normalized = p3b._whitened_features(pool, mean, np.sqrt(np.diag(cov)), w)
        residual = np.array([.5, -1.0, .2])
        seed_set = random_sets[0]
        initial = p3b._distances(seed_set[None, :], residual, normalized, 100)[0]
        best, objective = p3b._locally_optimize(seed_set, masks, normalized, residual, 100)
        self.assertEqual(24, len(set(int(n) for ticket in pool[best] for n in ticket)))
        self.assertLessEqual(objective, initial + 1e-12)

    def test_one_history_counterfactual_math(self):
        history = self.fair_draws()
        theory = p3b.exact_structural_covariance()
        rng = np.random.default_rng(649)
        o = p3b._selection(
            history, rng, pool_size=160, random_portfolios=32,
            mean=np.asarray(theory["mean"]),
            scale=np.sqrt(np.diag(np.asarray(theory["covariance"]))),
            whitening=np.asarray(theory["whitener"]), include_tickets=True,
        )
        self.assertEqual(24, len(set(
            n for ticket in o["research_only_four_ticket_candidate"] for n in ticket)))
        self.assertEqual(4, len(o["research_only_portfolio_baseline_classes"]))
        self.assertEqual(64, len(o["pool_sha256"]))
        self.assertGreaterEqual(
            o["optimized_four_append_gain"],
            o["random_portfolio_gain_quantiles_0_25_50_75_100"][-1] - 1e-10,
        )
        self.assertAlmostEqual(
            o["optimized_excess_over_random_median"],
            o["optimized_four_append_gain"] - o["random_portfolio_gain_median"],
        )
        self.assertTrue(0 <= len(o["research_only_best_single_candidate"]) == 6)

    def test_seeded_full_pipeline_iid_is_reproducible(self):
        history = self.fair_draws(60)
        settings = {"pool_size": 128, "random_portfolios": 16,
                    "iid_replicates": 3, "seed": 649}
        a = p3b.analyze_counterfactual(history, **settings)
        b = p3b.analyze_counterfactual(history, **settings)
        self.assertEqual(a, b)
        self.assertEqual(a["decision"], "DESCRIPTIVE_COUNTERFACTUAL_ONLY_NO_PREDICTIVE_GO")
        self.assertEqual(a["history_draws"], 60)
        self.assertEqual(a["control_design"]["iid_replicates"], 3)
        self.assertEqual(a["geometry_catalog_sha256"], p0.catalog()["definition_sha256"])
        self.assertEqual(a["matched_whole_pipeline_iid"]["minimum_resolvable_p"], .25)
        self.assertIsInstance(json.dumps(a, allow_nan=False), str)

    def test_two_endpoint_iid_control_adjustment(self):
        r = p3b.analyze_counterfactual(
            self.fair_draws(50), pool_size=128, random_portfolios=16,
            iid_replicates=9, seed=649)
        null = r["matched_whole_pipeline_iid"]
        for raw, corrected in (
            ("excess_over_random_median_empirical_upper_tail_p",
             "two_endpoint_bonferroni_excess_p"),
            ("optimized_gain_empirical_upper_tail_p",
             "two_endpoint_bonferroni_gain_p"),
        ):
            self.assertAlmostEqual(null[corrected], min(1, 2 * null[raw]))
            self.assertTrue(.1 <= null[raw] <= 1)

    def test_algebraic_certificate_remains_descriptive_only(self):
        r = p3b.analyze_counterfactual(
            self.fair_draws(35), pool_size=128, random_portfolios=16,
            iid_replicates=2)
        c = r["same_existing_v028_class_different_geometry_certificate"]
        self.assertTrue(c["structural_nonidentity_proven"])
        self.assertEqual(20, c["same_v028_baseline"]["log_bin"])
        self.assertEqual(2, len(c["descriptive_single_append_gains"]))
        self.assertEqual("FINITE_POOL_HEURISTIC_NOT_GLOBAL_OPTIMUM",
                         r["observed"]["optimizer_certification"])

    def test_forward_data_modification_does_not_change_exact_null(self):
        a = self.fair_draws(45)
        b = a.copy()
        b[-1] = np.arange(1, 7)
        settings = {"pool_size": 128, "random_portfolios": 16,
                    "iid_replicates": 2, "seed": 649}
        left = p3b.analyze_counterfactual(a, **settings)
        right = p3b.analyze_counterfactual(b, **settings)
        self.assertEqual(left["exact_null"], right["exact_null"])
        self.assertEqual(left["control_design"], right["control_design"])
        self.assertEqual(left["observed"]["pool_sha256"], right["observed"]["pool_sha256"])
        self.assertNotEqual(left["observed"]["baseline_n_scaled_distance"],
                            right["observed"]["baseline_n_scaled_distance"])

    def test_malformed_data_and_bad_settings_fail_closed(self):
        history = self.fair_draws(35)
        invalid = [history[:20], history[:, :5], history.astype(float)]
        duplicate = history.copy()
        duplicate[0, 1] = duplicate[0, 0]
        invalid.append(duplicate)
        for item in invalid:
            with self.assertRaises(ValueError):
                p3b.analyze_counterfactual(item, pool_size=128,
                                          random_portfolios=16, iid_replicates=1)
        for count in (127, 8193, True, 2.0):
            with self.assertRaises(ValueError):
                p3b.analyze_counterfactual(history, pool_size=count, random_portfolios=16,
                                          iid_replicates=1)
        for count in (15, 4097, False):
            with self.assertRaises(ValueError):
                p3b.analyze_counterfactual(history, pool_size=128,
                                          random_portfolios=count, iid_replicates=1)
        with self.assertRaises(ValueError):
            p3b.analyze_counterfactual(history, pool_size=128,
                                      random_portfolios=16, iid_replicates=0)


if __name__ == "__main__":
    unittest.main()
