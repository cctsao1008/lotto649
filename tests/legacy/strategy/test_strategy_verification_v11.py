"""Second-engine finite-prize and frozen M0-manifest regression tests."""
from __future__ import annotations

import copy
from itertools import combinations
from pathlib import Path
import unittest

from lotto7x7.strategy_verification_v11 import (
    _four_tickets, _m0_exact_log_bin, _m0_partition,
    independent_exact_prizes, verify_m0_frozen_manifest,
    verify_p2b_prize_report,
)

# From the original v0.30 full P2B 512-pool CI; not fitted to outcomes.
FROZEN_EXAMPLE = [
    [4, 13, 16, 25, 28, 40],
    [1, 12, 15, 24, 39, 44],
    [9, 10, 19, 23, 33, 36],
    [3, 16, 18, 25, 27, 32],
]
EXACT_512_HYBRID_WIN_COUNTS = [529088868, 69989192, 2221548, 4480, 0]
M0_TICKETS = [
    ((10, 27, 29, 35, 38, 47), 186, 'P2-1-1-1-1'),
    ((15, 25, 30, 32, 39, 49), 190, 'P3-1-1-1'),
    ((16, 18, 24, 40, 42, 46), 186, 'P3-2-1'),
    ((20, 23, 28, 31, 33, 43), 178, 'P2-2-1-1'),
]


def brute_small(lines, universe, draw_size):
    histogram = [0] * 5
    for bonus in range(1, universe + 1):
        for mains in combinations((v for v in range(1, universe + 1) if v != bonus), draw_size):
            drawn = set(mains)
            wins = sum(len(drawn.intersection(t)) >= 3 or
                       (len(drawn.intersection(t)) == 2 and bonus in t)
                       for t in lines)
            histogram[wins] += 1
    return histogram


class TestIndependentVerificationV11(unittest.TestCase):
    def test_01_frozen_actual_full_pilot_exact_award_histogram(self):
        result = independent_exact_prizes(FROZEN_EXAMPLE)
        self.assertEqual(result['exact_counts_winning_lines_0_to_4'], EXACT_512_HYBRID_WIN_COUNTS)
        self.assertAlmostEqual(result['probability_any_prize'], .1200976701159564, 13)
        self.assertEqual(result['total_equally_likely_outcomes'], 601304088)

    def test_02_small_universe_exhaustive_disjoint(self):
        lines = [[1,2,3],[4,5,6],[7,8,9],[2,4,7]]
        actual = independent_exact_prizes(lines, universe=9, draw_size=3)
        self.assertEqual(actual['exact_counts_winning_lines_0_to_4'], brute_small(lines, 9, 3))
        self.assertEqual(sum(actual['exact_counts_winning_lines_0_to_4']), 9*56)

    def test_03_small_universe_exhaustive_overlapping(self):
        lines = [[1,2,3],[1,2,4],[1,2,5],[1,2,6]]
        actual = independent_exact_prizes(lines, universe=9, draw_size=3)
        self.assertEqual(actual['exact_counts_winning_lines_0_to_4'], brute_small(lines, 9, 3))
        self.assertEqual(actual['exact_counts_winning_lines_0_to_4'][4], 0)

    def test_04_small_universe_nontrivial_special_dependency(self):
        lines = [[1,2,3,4],[1,2,5,6],[3,4,5,6],[2,3,6,8]]
        actual = independent_exact_prizes(lines, universe=9, draw_size=4)
        self.assertEqual(actual['exact_counts_winning_lines_0_to_4'], brute_small(lines, 9, 4))

    def test_05_single_ticket_prize_tier_marginals_invariant(self):
        first = independent_exact_prizes(FROZEN_EXAMPLE)
        other = independent_exact_prizes([
            [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 7],
            [1, 2, 3, 4, 5, 8], [1, 2, 3, 4, 5, 9]])
        self.assertEqual(first['expected_ticket_counts_by_tier'], other['expected_ticket_counts_by_tier'])
        self.assertEqual(first['expected_nominal_fixed_lower_tier_payout_ntd'],
                         other['expected_nominal_fixed_lower_tier_payout_ntd'])
        self.assertNotEqual(first['probability_any_prize'], other['probability_any_prize'])

    def test_06_jackpot_probability_four_over_universe(self):
        result = independent_exact_prizes(FROZEN_EXAMPLE)
        from math import comb
        self.assertAlmostEqual(result['expected_ticket_counts_by_tier']['jackpot'], 4 / comb(49, 6))

    def test_07_invalid_or_duplicate_tickets_rejected(self):
        for bad in (
            [[1,2,3,4,5,6]] * 4,
            [[1,2,3,4,5,50]] + FROZEN_EXAMPLE[1:],
            [[1,2,3,4,5,5]] + FROZEN_EXAMPLE[1:],
            [[2,1,3,4,5,6]] + FROZEN_EXAMPLE[1:],
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                independent_exact_prizes(bad)

    def test_08_m0_frozen_partition_and_logbin_calculation(self):
        self.assertEqual(len(set(v for row, _, _ in M0_TICKETS for v in row)), 24)
        for row, sum_expected, partition_expected in M0_TICKETS:
            self.assertEqual(sum(row), sum_expected)
            self.assertEqual(_m0_partition(row), partition_expected)
            self.assertEqual(_m0_exact_log_bin(row), 27)

    def test_09_m0_git_blob_fails_on_tampering_even_if_json_valid(self):
        original = Path('experiments/v028_m0_four_block_candidates_20261110.json')
        if not original.is_file():
            self.skipTest('repository M0 artifact not mounted in isolated test environment')
        raw = original.read_bytes()
        self.assertEqual(verify_m0_frozen_manifest(raw)['decision'],
                         'M0_FROZEN_MANIFEST_STRUCTURAL_PASS_OPTIMIZER_UNVERIFIED')
        altered = raw.replace(b'20261110', b'20261111', 1)
        self.assertEqual(verify_m0_frozen_manifest(altered)['decision'],
                         'M0_FROZEN_MANIFEST_FAIL')
        self.assertFalse(verify_m0_frozen_manifest(altered)['checks']['immutable_manifest_blob'])

    def test_10_prize_report_tamper_is_rejected(self):
        reported = {'observed': {'arms': {}}}
        for arm in ('random', 'coverage', 'repair', 'hybrid'):
            exact = independent_exact_prizes(FROZEN_EXAMPLE)
            exact['research_only_tickets'] = copy.deepcopy(FROZEN_EXAMPLE)
            exact['exact_fair_special_ball_prizes'] = exact.copy()
            reported['observed']['arms'][arm] = exact
        upstream = {'decision':'STRATEGY_COMPLIANCE_PASS','checks':{'x':True}}
        good = verify_p2b_prize_report(reported, upstream)
        self.assertEqual(good['decision'], 'EXACT_PRIZE_COMPLIANCE_PASS', good['failures'])
        altered = copy.deepcopy(reported)
        altered['observed']['arms']['hybrid']['exact_fair_special_ball_prizes']['exact_counts_winning_lines_0_to_4'][1] += 1
        bad = verify_p2b_prize_report(altered, upstream)
        self.assertEqual(bad['decision'], 'EXACT_PRIZE_COMPLIANCE_FAIL')
        self.assertIn('hybrid.winning_line_histogram', bad['failures'])

    def test_11_untrusted_v1_audit_prevents_prize_only_pass(self):
        report = {'observed': {'arms': {}}}
        for arm in ('random', 'coverage', 'repair', 'hybrid'):
            exact = independent_exact_prizes(FROZEN_EXAMPLE)
            report['observed']['arms'][arm] = {
                'research_only_tickets': copy.deepcopy(FROZEN_EXAMPLE),
                'exact_fair_special_ball_prizes': exact,
            }
        v = verify_p2b_prize_report(report, {'decision':'STRATEGY_COMPLIANCE_FAIL','checks':{'source_gate': False}})
        self.assertEqual(v['decision'], 'EXACT_PRIZE_COMPLIANCE_FAIL')
        self.assertIn('upstream_strategy_verification_pass', v['failures'])

    def test_12_dependent_tier_validation_against_independent_histogram(self):
        result = independent_exact_prizes(FROZEN_EXAMPLE)
        weighted = sum(i * n for i, n in enumerate(result['exact_counts_winning_lines_0_to_4']))
        tier_weighted = sum(result['expected_ticket_counts_by_tier'].values())
        self.assertAlmostEqual(weighted / result['total_equally_likely_outcomes'], tier_weighted, 14)


if __name__ == '__main__':
    unittest.main()
