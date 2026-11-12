"""Exact independent brute-force checks of the four-ticket bonus-ball model."""
from itertools import combinations
from math import comb
import unittest

from lotto7x7.portfolio_prize_v030 import exact_prize_distribution


class TestPortfolioPrizeV030(unittest.TestCase):
    def test_small_universe_matches_full_main_and_special_enumeration(self):
        lines = ((1, 2, 3), (1, 4, 5), (2, 6, 7), (3, 8, 9))
        result = exact_prize_distribution(lines, universe=10, draw_size=3)
        brute = [0] * 5
        for main in combinations(range(1, 11), 3):
            for special in set(range(1, 11)) - set(main):
                hit_count = sum(
                    len(set(ticket) & set(main)) >= 3 or
                    (len(set(ticket) & set(main)) == 2 and special in ticket)
                    for ticket in lines
                )
                brute[hit_count] += 1
        self.assertEqual(result["exact_counts_winning_lines_0_to_4"], brute)
        self.assertEqual(result["total_equally_likely_outcomes"], 10 * comb(9, 3))

    def test_prize_coverage_changes_any_win_but_not_expected_count(self):
        disjoint = [tuple(range(i, i + 6)) for i in (1, 7, 13, 19)]
        overlap = [(1, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 7),
                   (1, 2, 3, 4, 5, 8), (1, 2, 3, 4, 5, 9)]
        a, b = [exact_prize_distribution(t) for t in (disjoint, overlap)]
        self.assertGreater(a["probability_any_prize"], b["probability_any_prize"])
        self.assertAlmostEqual(a["expected_number_winning_lines"],
                               b["expected_number_winning_lines"], places=13)
        self.assertAlmostEqual(a["expected_nominal_fixed_lower_tier_payout_ntd"],
                               b["expected_nominal_fixed_lower_tier_payout_ntd"], places=13)
        self.assertAlmostEqual(a["expected_nominal_fixed_lower_tier_payout_ntd"],
                               58.28738021152452, places=10)
        self.assertEqual(a["total_equally_likely_outcomes"], comb(49, 6) * 43)

    def test_illegal_and_duplicate_lines_rejected(self):
        lines = [(1,2,3,4,5,6),(1,2,3,4,5,7),(1,2,3,4,5,8),(1,2,3,4,5,9)]
        for bad in [lines[:3]+[lines[0]], lines[:3]+[(6,5,4,3,2,1)],
                    lines[:3]+[(1,1,3,4,5,6)], lines[:3]+[(0,1,2,3,4,5)]]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                exact_prize_distribution(bad)


if __name__ == "__main__":
    unittest.main()
