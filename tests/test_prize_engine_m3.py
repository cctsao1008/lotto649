"""M3 independent eight-tier and dynamic published-payout reference checks."""
from itertools import combinations
from math import comb
import json
import unittest

from lotto7x7 import prize_engine_m3 as m3
from lotto7x7.official_prize_v026 import PRIZE_ORDER, official_prize_coverage


class TestPrizeEngineM3(unittest.TestCase):
    DRAW = dict(m3.ANCHOR)

    @staticmethod
    def ticket_for(hits, with_bonus):
        main = m3.ANCHOR["main"]
        bonus = m3.ANCHOR["bonus"]
        selected = main[:hits] + ([bonus] if with_bonus else [])
        candidates = [n for n in range(1, 50) if n not in main and n != bonus]
        selected += candidates[:6 - len(selected)]
        return sorted(selected)

    def test_all_eight_official_tiers_independent_constructed_tickets(self):
        expected = {
            (6, False): "first",
            (5, True): "second",
            (5, False): "third",
            (4, True): "fourth",
            (4, False): "fifth",
            (3, True): "sixth",
            (2, True): "seventh",
            (3, False): "regular",
        }
        for (main_hits, bonus_hit), expected_tier in expected.items():
            row = m3.grade_ticket(self.ticket_for(main_hits, bonus_hit), self.DRAW)
            self.assertEqual(expected_tier, row["official_prize_tier"], expected_tier)
            self.assertEqual(main_hits, row["main_hits"])
            self.assertEqual(bonus_hit, row["special_hit"])
            self.assertTrue(row["any_prize"])
            self.assertIsNone(row["actual_settled_payout_twd"])
            self.assertIsNone(row["published_per_winner_reference_twd"])

    def test_nonwinning_tickets_and_bonus_without_main_hits(self):
        for hits, bonus in ((0, False), (0, True), (1, False),
                            (1, True), (2, False)):
            row = m3.grade_ticket(self.ticket_for(hits, bonus), self.DRAW)
            self.assertIsNone(row["official_prize_tier"])
            self.assertFalse(row["any_prize"])
        self.assertEqual("regular", m3.grade_ticket(
            self.ticket_for(3, False), self.DRAW)["official_prize_tier"])
        self.assertEqual("sixth", m3.grade_ticket(
            self.ticket_for(3, True), self.DRAW)["official_prize_tier"])

    def test_actual_published_20261110_table_and_no_zero_jackpot_assumption(self):
        table = m3.official_anchor_table()
        self.assertEqual(set(PRIZE_ORDER), set(table["published_prizes_twd"]))
        self.assertEqual(56494, table["published_prizes_twd"]["third"])
        self.assertEqual(16880, table["published_prizes_twd"]["fourth"])
        first = m3.grade_ticket(self.ticket_for(6, False), self.DRAW,
                                published_table=table)
        self.assertEqual("first", first["official_prize_tier"])
        self.assertIsNone(first["published_per_winner_reference_twd"])
        self.assertIn("HYPOTHETICAL_CHANGE_UNKNOWN", first["payout_status"])
        third = m3.grade_ticket(self.ticket_for(5, False), self.DRAW,
                                published_table=table)
        self.assertEqual(56494, third["published_per_winner_reference_twd"])
        self.assertIsNone(third["actual_settled_payout_twd"])
        none = m3.grade_ticket(self.ticket_for(1, False), self.DRAW,
                               published_table=table)
        self.assertEqual(0, none["published_per_winner_reference_twd"])

    def test_unreviewed_payouts_cannot_be_used_as_verified_reference(self):
        table = m3.official_anchor_table()
        table["source"]["status"] = "DECLARED_PUBLIC_TABLE_REQUIRES_INDEPENDENT_REVIEW"
        table["published_prizes_twd"]["third"] = 100000
        third = m3.grade_ticket(self.ticket_for(5, False), self.DRAW,
                                published_table=table)
        self.assertIsNone(third["published_per_winner_reference_twd"])
        self.assertEqual("UNREVIEWED_SOURCE_AMOUNT_NOT_USABLE",
                         third["payout_status"])

    def test_tampered_claimed_reviewed_table_and_wrong_draw_id_rejected(self):
        table = m3.official_anchor_table()
        table["published_prizes_twd"]["third"] += 1
        with self.assertRaisesRegex(ValueError, "only independently reviewed"):
            m3.validate_published_table(table, draw=self.DRAW)
        table = m3.official_anchor_table()
        table["draw_id"] += 1
        with self.assertRaisesRegex(ValueError, "draw ID"):
            m3.validate_published_table(table, draw=self.DRAW)
        table = m3.official_anchor_table()
        table["published_prizes_twd"]["regular"] = -1
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            m3.validate_published_table(table, draw=self.DRAW)
        table = m3.official_anchor_table()
        del table["published_prizes_twd"]["sixth"]
        with self.assertRaisesRegex(ValueError, "eight tiers"):
            m3.validate_published_table(table, draw=self.DRAW)

    def test_bad_ticket_or_draw_and_bool_rejected(self):
        for bad in ([1, 2, 3, 4, 5, 5], [1, 2, 3, 4, 5, 0],
                    [1, 2, 3, 4, 5, True], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 50]):
            with self.assertRaises(ValueError):
                m3.grade_ticket(bad, self.DRAW)
        with self.assertRaisesRegex(ValueError, "special ball"):
            m3.grade_ticket([1, 2, 3, 4, 5, 6], {**self.DRAW, "bonus": 11})
        with self.assertRaisesRegex(ValueError, "main balls"):
            m3.grade_ticket([1, 2, 3, 4, 5, 6],
                            {**self.DRAW, "main": [11, 11, 23, 32, 40, 41]})

    def test_portfolio_highest_hits_and_mutually_exclusive_tier_counts(self):
        portfolio = [
            self.ticket_for(5, False),
            self.ticket_for(3, True),
            self.ticket_for(2, False),
            self.ticket_for(1, False),
        ]
        result = m3.grade_portfolio(portfolio, self.DRAW,
                                    published_table=m3.official_anchor_table())
        self.assertEqual(4, result["ticket_count"])
        self.assertEqual(5, result["max_main_hits"])
        self.assertEqual(2, result["awarded_ticket_count"])
        self.assertTrue(result["any_award"])
        self.assertEqual(1, result["tier_counts"]["third"])
        self.assertEqual(1, result["tier_counts"]["sixth"])
        self.assertEqual(2, sum(result["tier_counts"].values()))
        self.assertEqual(200, result["nominal_unpurchased_ticket_cost_twd"])
        self.assertIsNone(result["actual_realized_net_profit_twd"])
        self.assertIsNone(result["actual_settled_total_twd"])
        self.assertIsInstance(json.dumps(result, allow_nan=False), str)

    def test_portfolio_duplicate_rejected_and_disjointness_is_not_required(self):
        ticket = self.ticket_for(3, True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            m3.grade_portfolio([ticket, ticket], self.DRAW)
        with self.assertRaisesRegex(ValueError, "1..4"):
            m3.grade_portfolio([], self.DRAW)
        with self.assertRaisesRegex(ValueError, "1..4"):
            m3.grade_portfolio([ticket]*5, self.DRAW)
        overlapping = [self.ticket_for(4, True), self.ticket_for(3, True)]
        graded = m3.grade_portfolio(overlapping, self.DRAW)
        self.assertEqual(2, graded["ticket_count"])
        self.assertEqual(2, graded["awarded_ticket_count"])

    def test_one_ticket_exact_award_probability_matches_independent_combinatorics(self):
        selected = self.ticket_for(3, False)
        theory = official_prize_coverage([selected])
        self.assertEqual(comb(49, 6) * 43, theory["total_outcomes"])
        counts = {}
        for hits in range(7):
            ways_main = comb(6, hits) * comb(43, 6-hits)
            for bonus_hit, remaining_bonus in (
                (True, 6 - hits), (False, 37 + hits)
            ):
                tier = m3.prize_tier(hits, bonus_hit)
                if tier is not None:
                    counts[tier] = counts.get(tier, 0) + ways_main * remaining_bonus
        for tier in PRIZE_ORDER:
            self.assertEqual(counts[tier],
                             theory["any_ticket_by_tier"][tier]["outcomes"])
        self.assertEqual(sum(counts.values()), theory["any_prize_outcomes"])

    def test_p5a_frozen_four_disjoint_arms_have_equal_exact_fair_odds(self):
        result = m3.verify_prize_engine()
        self.assertEqual("PASS_EIGHT_TIERS_ANCHOR_AND_EXACT_FOUR_DISJOINT_PROBABILITY",
                         result["status"])
        self.assertEqual(8, len(result["official_tier_matrix"]))
        self.assertEqual(2, len(result["shadow_arms"]))
        probabilities = [
            x["fair_any_prize_probability"] for x in result["shadow_arms"].values()
        ]
        self.assertAlmostEqual(probabilities[0], probabilities[1], places=14)
        self.assertAlmostEqual(probabilities[0], 0.122731458961909, places=14)
        self.assertEqual(601304088,
                         next(iter(result["shadow_arms"].values()))["total_exact_outcomes"])
        self.assertIn("NO_RESULTS_EVALUATED", result["frozen_shadow_verification"])

    def test_fixed_tiers_are_not_assumed_to_be_paid_when_no_source(self):
        graded = m3.grade_portfolio(
            [self.ticket_for(4, False), self.ticket_for(3, False)],
            self.DRAW)
        for row in graded["tickets"]:
            self.assertIsNone(row["published_per_winner_reference_twd"])
            self.assertIsNone(row["actual_settled_payout_twd"])
        self.assertIsNone(graded["actual_realized_net_profit_twd"])
        self.assertEqual(m3.STANDARD_FIXED_TWD, {
            "fifth": 2000, "sixth": 1000, "seventh": 400, "regular": 400,
        })


if __name__ == "__main__":
    unittest.main()
