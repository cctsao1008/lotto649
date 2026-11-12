"""Unit tests for P2B structural portfolio benchmarks and exact fair tails."""
from __future__ import annotations

from itertools import combinations
from math import comb
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from lotto7x7.geometry_repair_v029 import _uniform_tickets
from lotto7x7.portfolio_benchmark_v030 import (
    ARMS, _selection, analyze_portfolios, exact_max_match_distribution, main,
)
from lotto7x7.geometry_repair_v029 import exact_structural_covariance


class TestPortfolioBenchmarkV030P2B(unittest.TestCase):
    @staticmethod
    def small_history(seed=97, count=32):
        return _uniform_tickets(np.random.default_rng(seed), count)

    def test_exact_dp_matches_brute_force_small_universe(self):
        tickets = ((1, 2, 3), (1, 4, 5), (2, 6, 7), (3, 8, 9))
        report = exact_max_match_distribution(tickets, universe=10, draw_size=3)
        brute = [0, 0, 0, 0]
        for drawn in combinations(range(1, 11), 3):
            brute[max(len(set(drawn) & set(line)) for line in tickets)] += 1
        self.assertEqual(report["max_match_exact_counts_0_to_6"], brute)
        self.assertEqual(sum(brute), comb(10, 3))
        self.assertEqual(brute[-1], 4)

    def test_exact_jackpot_invariant_to_overlap(self):
        rng = np.random.default_rng(101)
        disjoint = [tuple(range(i, i + 6)) for i in (1, 7, 13, 19)]
        overlap = [tuple(sorted(rng.choice(49, 6, replace=False) + 1))
                   for _ in range(4)]
        outputs = [exact_max_match_distribution(t) for t in (disjoint, overlap)]
        for result in outputs:
            self.assertEqual(result["jackpot_probability"], 4 / comb(49, 6))
            self.assertEqual(result["expected_sum_of_main_ball_matches"], 144 / 49)
            self.assertEqual(result["exact_total_outcomes"], comb(49, 6))
        self.assertNotEqual(
            outputs[0]["max_match_probabilities_0_to_6"],
            outputs[1]["max_match_probabilities_0_to_6"],
        )

    def test_duplicate_and_illegal_portfolios_rejected(self):
        valid = [(1, 2, 3), (1, 4, 5), (2, 6, 7), (3, 8, 9)]
        for invalid in (
            [valid[0], valid[0], valid[2], valid[3]],
            [valid[0], (3, 2, 1), valid[2], valid[3]],
            [valid[0], (1, 1, 5), valid[2], valid[3]],
            [valid[0], (1, 4, 11), valid[2], valid[3]],
            [valid[0], (1.5, 4, 5), valid[2], valid[3]],
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                exact_max_match_distribution(invalid, universe=10, draw_size=3)

    def test_all_arms_and_coverage_floor(self):
        result = analyze_portfolios(
            self.small_history(), pool_size=128, random_portfolios=32,
            min_hybrid_unique=22, iid_reps=3, seed=649,
        )
        arms = result["observed"]["arms"]
        self.assertEqual(set(arms), set(ARMS))
        self.assertEqual(arms["coverage"]["unique_numbers"], 24)
        self.assertGreaterEqual(arms["hybrid"]["unique_numbers"], 22)
        for arm in ARMS:
            lines = arms[arm]["research_only_tickets"]
            self.assertEqual(len(set(map(tuple, lines))), 4)
            self.assertTrue(all(len(set(row)) == 6 for row in lines))
            self.assertAlmostEqual(
                arms[arm]["exact_fair_max_main_ball_match"]["jackpot_probability"],
                4 / comb(49, 6), places=15)
        self.assertEqual(result["matched_whole_pipeline_iid"]["replicates"], 3)
        self.assertEqual(result["decision"],
                         "DESCRIPTIVE_PORTFOLIO_TRADEOFF_ONLY_NO_PREDICTIVE_GO")

    def test_reproducible_independent_iid_pipeline(self):
        history = self.small_history()
        a = analyze_portfolios(history, pool_size=128, random_portfolios=32,
                               iid_reps=3, seed=650)
        b = analyze_portfolios(history, pool_size=128, random_portfolios=32,
                               iid_reps=3, seed=650)
        self.assertEqual(a, b)

    def test_local_objectives_do_not_worsen_seed_starts(self):
        history = self.small_history()
        exact = exact_structural_covariance()
        mean = np.array(exact["mean"])
        scale = np.sqrt(np.diag(np.asarray(exact["covariance"])))
        whitening = np.asarray(exact["whitener"])
        row = _selection(history, np.random.default_rng(99), pool_size=128,
                         random_portfolios=32, min_hybrid_unique=22,
                         mean=mean, scale=scale, whitening=whitening,
                         details=False)
        self.assertGreaterEqual(
            row["arms"]["repair"]["counterfactual_energy_repair_gain"],
            row["random_portfolio_controls"]["repair_gain_median"])
        self.assertGreaterEqual(row["arms"]["hybrid"]["unique_numbers"], 22)

    def test_invalid_configuration_and_mutated_catalog(self):
        history = self.small_history()
        for kwargs in ({"pool_size": 4}, {"random_portfolios": 2},
                       {"min_hybrid_unique": 25}, {"iid_reps": 0},
                       {"seed": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                analyze_portfolios(history, **kwargs)
        with patch("lotto7x7.portfolio_benchmark_v030.catalog",
                   return_value={"definition_sha256": "tampered"}):
            with self.assertRaises(ValueError):
                analyze_portfolios(history, pool_size=128, iid_reps=3)

    def test_cli_report_exclusive_create_and_provenance_gate(self):
        history = self.small_history()
        Path("results").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir="results") as directory:
            outfile = Path(directory) / "p2b.json"
            args = ["--pool-size", "128", "--random-portfolios", "32",
                    "--iid-reps", "3", "--out", str(outfile)]
            with patch("lotto7x7.portfolio_benchmark_v030.audit_repository_inputs",
                       return_value=(history, {"status": "TEST_SOURCE_GATE"})):
                main(args)
                self.assertTrue(outfile.exists())
                with self.assertRaises(FileExistsError):
                    main(args)
            with patch("lotto7x7.portfolio_benchmark_v030.audit_repository_inputs",
                       side_effect=ValueError("invalid source fingerprint")):
                newfile = outfile.parent / "no_report.json"
                with self.assertRaises(ValueError):
                    main(args[:-1] + [str(newfile)])
                self.assertFalse(newfile.exists())


if __name__ == "__main__":
    unittest.main()
