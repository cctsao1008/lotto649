"""v0.30 P1: exact finite-population Sum sampler and full-baseline filtering."""
from collections import Counter
from itertools import combinations
from math import comb
import json
import unittest

import numpy as np

from lotto7x7 import conditional_monte_carlo_v030 as cmc
from lotto7x7 import distribution_dynamics_v030 as dyn
from lotto7x7.geometry_redundancy_v029 import _certificate


class TestConditionalMonteCarloV030P1(unittest.TestCase):
    @staticmethod
    def history(n=80):
        return dyn._uniform_history(np.random.default_rng(17), n)

    def test_dp_counts_equal_direct_enumeration_on_small_universe(self):
        for n in range(7, 11):
            for k in (2, 3, 4):
                by_sum = Counter(sum(c) for c in combinations(range(1, n + 1), k))
                for target in range(k, k * n + 1):
                    self.assertEqual(
                        by_sum[target],
                        cmc.completion_count(1, k, target, n),
                    )
                self.assertEqual(
                    comb(n, k),
                    sum(cmc.completion_count(1, k, target, n)
                        for target in range(k, k*n + 1))
                )

    def test_exact_sum_sampler_is_individually_uniform_on_small_universe(self):
        n, k, total = 9, 3, 15
        legal = {tuple(c) for c in combinations(range(1, n+1), k) if sum(c) == total}
        rng = np.random.default_rng(1008)
        rows = cmc.sample_uniform_fixed_sum(
            total, 8000, rng=rng, universe=n, pick=k
        )
        counts = Counter(tuple(map(int, row)) for row in rows)
        self.assertEqual(legal, set(counts))
        expected = len(rows) / len(legal)
        sigma = np.sqrt(expected * (1 - 1 / len(legal)))
        self.assertLess(max(abs(value - expected) for value in counts.values()),
                        6 * sigma)

    def test_exact_sum_132_has_nonzero_completions(self):
        n = cmc.completion_count(1, 6, 132, 49)
        self.assertEqual(144587, n)
        self.assertLess(n, comb(49, 6))
        self.assertEqual(0, cmc.completion_count(1, 6, 1, 49))
        self.assertEqual(0, cmc.completion_count(1, 6, 400, 49))

    def test_sum_condition_sampler_draws_legal_six_and_is_reproducible(self):
        a = cmc.sample_uniform_fixed_sum(
            132, 3000, rng=np.random.default_rng(649))
        b = cmc.sample_uniform_fixed_sum(
            132, 3000, rng=np.random.default_rng(649))
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(np.all(a.sum(axis=1) == 132))
        self.assertTrue(np.all(np.diff(a, axis=1) > 0))
        self.assertTrue(np.all((a >= 1) & (a <= 49)))

    def test_invalid_fixed_sum_sampler_parameters_rejected(self):
        for args in ((10, 5, 49, 6), (500, 5, 49, 6),
                     (132, 0, 49, 6), (132, 5, 49, 50),
                     (132, True, 49, 6), (132, 5, 9, 3)):
            target, count, n, k = args
            with self.assertRaises(ValueError):
                cmc.sample_uniform_fixed_sum(
                    target, count, rng=np.random.default_rng(649),
                    universe=n, pick=k,
                )
        with self.assertRaises(ValueError):
            cmc.completion_count(1, True, 15, 49)

    def test_p3a_full_baseline_reference_is_unchanged(self):
        cert = _certificate()
        self.assertEqual(132, cert["same_v028_baseline"]["sum"])
        self.assertEqual(20, cert["same_v028_baseline"]["log_bin"])
        self.assertEqual(2, len(cert["tickets"]))

    def test_full_condition_acceptance_keeps_l3_sum_logbin(self):
        history = self.history()
        report = cmc.conditional_ablation(history, proposals=10000, seed=649)
        self.assertEqual("UNBIASED_CONDITIONAL_STRUCTURE_DESCRIPTIVE_NOT_PREDICTIVE",
                         report["decision"])
        self.assertEqual(132, report["fixed_baseline"]["sum"])
        self.assertEqual(20, report["fixed_baseline"]["log_bin"])
        self.assertEqual(10000, report["sampler"]["exact_sum_proposals"])
        self.assertGreaterEqual(report["sampler"]["accepted_full_condition"], 25)
        self.assertLessEqual(
            report["sampler"]["accepted_distinct_combinations"],
            report["sampler"]["accepted_full_condition"],
        )
        self.assertIn("uniform", report["sampler"]["sampling_proof"])
        self.assertEqual(3, len(report["conditional_fair_structural_statistics"]))
        self.assertIsInstance(json.dumps(report, allow_nan=False), str)

    def test_seeded_conditional_fair_distribution_does_not_fit_history(self):
        history = self.history()
        modified = history.copy()
        modified[-1] = [1, 2, 3, 4, 5, 6]
        a = cmc.conditional_ablation(history, proposals=8000, seed=649)
        b = cmc.conditional_ablation(modified, proposals=8000, seed=649)
        self.assertEqual(a, b)
        self.assertEqual(a["history_draws_read_only"], len(history))

    def test_prior_ticket_ranks_are_descriptive_not_p_values(self):
        r = cmc.conditional_ablation(self.history(), proposals=8000)
        for ticket in r["prior_exploratory_certificate_rank"]:
            self.assertEqual(6, len(ticket["ticket"]))
            for value in ticket["conditional_feature_ranks"].values():
                self.assertTrue(0 <= value["descriptive_fraction_conditional_le"] <= 1)
                self.assertTrue(value["post_selected_not_a_hypothesis_p"])
        self.assertTrue(any("unknown" in x for x in r["limits"]))

    def test_rejection_sampler_distinct_full_class_and_no_purchase(self):
        r = cmc.conditional_ablation(self.history(), proposals=8000)
        sample = r["sampler"]
        self.assertGreaterEqual(sample["l3_only_accept_count"],
                                sample["accepted_full_condition"])
        self.assertGreaterEqual(sample["logbin_only_accept_count"],
                                sample["accepted_full_condition"])
        self.assertEqual(sample["total_legal_combinations_with_exact_sum"], 144587)
        self.assertNotIn("purchase", r)
        self.assertNotIn("forecast", r)

    def test_source_config_and_unattainable_sample_size_rejected(self):
        h = self.history()
        for count in (0, 999, 250001, True, 1.5):
            with self.assertRaises(ValueError):
                cmc.conditional_ablation(h, proposals=count)
        with self.assertRaises(ValueError):
            cmc.conditional_ablation(h, proposals=1000, seed=-1)
        with self.assertRaises(ValueError):
            cmc.conditional_ablation(h[:10], proposals=2000)


if __name__ == "__main__":
    unittest.main()
