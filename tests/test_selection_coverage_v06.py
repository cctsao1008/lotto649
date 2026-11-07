"""v0.6 Phase 2/3 exact mathematics, selection and report validation."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from math import comb
import unittest

import numpy as np

from lotto7x7.coverage_v06 import (UNIVERSE, coverage_counts, exact_coverage,
                                   normalized_bets, single_bet_anchor)
from lotto7x7.coverage_audit_v06 import _popcount64, audit_selection, pool_coverage_estimate
from lotto7x7.bet_selection_v06 import select_bets
from lotto7x7.candidate_pool_v06 import create_shape_pool
from lotto7x7.shape_engine_v06 import bitmap


class ExactCoverageTests(unittest.TestCase):
    def test_one_ticket_hypergeometric_anchor(self):
        self.assertEqual(coverage_counts([(1,2,3,4,5,6)]), single_bet_anchor())
        self.assertEqual(sum(single_bet_anchor()), UNIVERSE)
        self.assertEqual(single_bet_anchor()[6], 1)

    def test_two_disjoint_bets_three_hit_inclusion_exclusion(self):
        ticket_a = (1,2,3,4,5,6)
        ticket_b = (7,8,9,10,11,12)
        result = exact_coverage([ticket_a, ticket_b])
        single_ge3 = sum(single_bet_anchor()[3:])
        intersection = comb(6,3)**2  # exactly 3 hits in each ticket
        self.assertEqual(result['at_least_counts']['3'], 2*single_ge3-intersection)
        self.assertEqual(result['at_least_counts']['6'], 2)

    def test_five_ticket_distribution_and_order_invariance(self):
        bets = [(1,2,3,4,5,6), (5,6,7,8,9,10), (11,12,13,14,15,16),
                (1,17,18,19,20,21), (22,23,24,25,26,27)]
        a = exact_coverage(bets)
        b = exact_coverage(bets[::-1])
        self.assertEqual(a['max_hits_counts_0_to_6'], b['max_hits_counts_0_to_6'])
        self.assertEqual(sum(a['max_hits_counts_0_to_6']), UNIVERSE)
        self.assertEqual(a['at_least_counts']['6'], 5)
        for h in (3,4,5):
            self.assertGreaterEqual(a['at_least_counts'][str(h)], a['at_least_counts'][str(h+1)])

    def test_reject_invalid_or_duplicate_tickets(self):
        for bad in ([], [list(range(1,7))]*2, [list(range(1,8))],
                    [(1,2,3,4,5,50)], [(1,2,3,4,5,True)]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                normalized_bets(bad)
        with self.assertRaises(ValueError):
            normalized_bets([tuple(range(i,i+6)) for i in range(1,7)])

    def test_vectorized_popcount_matches_python(self):
        rng = np.random.default_rng(88)
        x = rng.integers(0, 1<<49, size=1000, dtype=np.uint64)
        self.assertEqual(_popcount64(x).tolist(), [int(v).bit_count() for v in x])


class PortfolioPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(649)
        images = np.stack([bitmap(tuple(int(n) for n in rng.choice(49, 6, replace=False)+1))
                           for _ in range(145)])
        ids = np.arange(115000001, 115000146, dtype=np.int64)
        cls.pool = create_shape_pool(images, ids, seed=649)

    def test_all_strategies_one_to_five_bets_and_seed_stability(self):
        for count in (1, 3, 5):
            result = select_bets(self.pool, count=count, strategy='all', seed=123)
            self.assertEqual(set(result['strategies']),
                             {'random','shape-diverse','coverage-optimized'})
            for method in result['strategies'].values():
                self.assertEqual(len(method['bets']), count)
                self.assertEqual(len({row['pool_index'] for row in method['bets']}), count)
                self.assertEqual(method['exact_fair_coverage']['at_least_counts']['6'], count)
            if count == 3:
                self.assertEqual(result, select_bets(self.pool, count=count, strategy='all', seed=123))

    def test_no_target_input_and_pareto_audit(self):
        selection = select_bets(self.pool, count=3, seed=2026)
        audit = audit_selection(selection, samples=1000, seed=21)
        self.assertEqual(audit['research_pool_count'], 100)
        self.assertEqual(audit['ticket_count_per_strategy'], 3)
        self.assertGreaterEqual(len(audit['pareto_nondominated']), 1)
        for item in audit['strategies'].values():
            self.assertEqual(item['exact_fair_coverage']['at_least_counts']['6'], 3)
        realized = audit_selection(selection, samples=1000, seed=21, actual=(1,2,3,4,5,6))
        self.assertTrue(realized['realized_draw']['post_draw_only'])
        self.assertEqual(len(realized['strategies']['random']['realized_draw']['per_ticket_hits']), 3)

    def test_tampered_source_or_selected_ticket_is_rejected(self):
        selected = select_bets(self.pool, count=2, strategy='random')
        bad = deepcopy(selected)
        bad['pool_numbers'][0][0] = 49
        with self.assertRaises(ValueError):
            audit_selection(bad, samples=100)
        bad = deepcopy(selected)
        bad['strategies']['random']['bets'][0]['numbers'] = [1,2,3,4,5,6]
        with self.assertRaises(ValueError):
            audit_selection(bad, samples=100)

    def test_cli_bet_select_and_coverage_audit_commands(self):
        # Integration test: catches argparse invalid-choice regressions.
        with tempfile.TemporaryDirectory() as tmp:
            pool_file = Path(tmp) / "pool.json"
            bets_file = Path(tmp) / "bets.json"
            audit_file = Path(tmp) / "audit.json"
            pool_file.write_text(json.dumps(self.pool), encoding="utf-8")
            subprocess.run([sys.executable, "-m", "lotto7x7", "bet-select",
                            "--pool", str(pool_file), "--count", "2",
                            "--strategy", "all", "--out", str(bets_file)],
                           check=True, capture_output=True, text=True)
            subprocess.run([sys.executable, "-m", "lotto7x7", "coverage-audit",
                            "--bets", str(bets_file), "--samples", "200",
                            "--out", str(audit_file)],
                           check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(audit_file.read_text(encoding="utf-8"))
                             ["ticket_count_per_strategy"], 2)

    def test_pool_monte_carlo_is_reproducible_and_labeled(self):
        numbers = [row['numbers'] for row in self.pool['candidates']]
        a = pool_coverage_estimate(numbers, samples=200, seed=42)
        self.assertEqual(a, pool_coverage_estimate(numbers, samples=200, seed=42))
        self.assertEqual(a['exact_six_coverage'], 100/UNIVERSE)
        self.assertEqual(sum(a['max_hits_counts_0_to_6']), 200)
        self.assertEqual(a['method'], 'iid_uniform_monte_carlo')


if __name__ == '__main__':
    unittest.main()