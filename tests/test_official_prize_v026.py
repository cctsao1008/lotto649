"""Independent exact probability checks for v0.26 prize tier engine."""
from itertools import combinations
from math import comb

import pytest

from lotto7x7.official_prize_v026 import (
    PRIZE_ORDER, prize_tier, official_prize_coverage,
)
from lotto7x7.deficit_compensation_v025 import REFERENCE
from lotto7x7.coverage_v06 import exact_coverage


def test_all_eight_awards_and_unawarded_cases():
    expected = {
        (6, False): "first", (5, True): "second",
        (5, False): "third", (4, True): "fourth",
        (4, False): "fifth", (3, True): "sixth",
        (2, True): "seventh", (3, False): "regular",
        (2, False): None, (1, True): None, (0, False): None,
    }
    assert [prize_tier(*key) for key in expected] == list(expected.values())
    with pytest.raises(ValueError):
        prize_tier(7, False)


def test_single_ticket_combinatorial_independent_anchor():
    result = official_prize_coverage([REFERENCE[0]])
    assert result["total_outcomes"] == comb(49, 6) * 43
    expected = {name: 0 for name in PRIZE_ORDER}
    for main_hits in range(7):
        main_choices = comb(6, main_hits) * comb(43, 6 - main_hits)
        for special_hit, remaining in (
                (True, 6 - main_hits), (False, 37 + main_hits)):
            tier = prize_tier(main_hits, special_hit)
            if tier is not None:
                expected[tier] += main_choices * remaining
    for tier in PRIZE_ORDER:
        assert result["any_ticket_by_tier"][tier]["outcomes"] == expected[tier]
        assert result["expected_tickets_by_tier"][tier]["ticket_outcomes"] == expected[tier]
    assert result["any_prize_outcomes"] == sum(expected.values())


def brute(bets, universe_size):
    counts = {name: 0 for name in PRIZE_ORDER}
    totals = {name: 0 for name in PRIZE_ORDER}
    by_winners = [0] * (len(bets) + 1)
    for main in combinations(range(1, universe_size + 1), 6):
        main = set(main)
        for special in set(range(1, universe_size + 1)) - main:
            tiers = []
            for bet in bets:
                inside = set(bet)
                tiers.append(prize_tier(len(main & inside), special in inside))
            by_winners[sum(x is not None for x in tiers)] += 1
            for tier in PRIZE_ORDER:
                occurrences = tiers.count(tier)
                if occurrences:
                    counts[tier] += 1
                    totals[tier] += occurrences
    return counts, totals, by_winners


@pytest.mark.parametrize("bets", [
    [(1, 2, 3, 4, 5, 6)],
    [(1, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 7)],
    [(1, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 7),
     (1, 2, 3, 4, 6, 8), (1, 2, 3, 5, 6, 9)],
])
def test_exhaustive_nine_number_universe_matches_compressed_exact(bets):
    compressed = official_prize_coverage(bets, universe_size=9)
    by_tier, totals, counts = brute(bets, 9)
    assert compressed["total_outcomes"] == comb(9, 6) * 3
    assert compressed["awarded_ticket_count_outcomes"] == counts
    for tier in PRIZE_ORDER:
        assert compressed["any_ticket_by_tier"][tier]["outcomes"] == by_tier[tier]
        assert compressed["expected_tickets_by_tier"][tier]["ticket_outcomes"] == totals[tier]


def test_four_disjoint_reference_conservation_and_head_prize():
    out = official_prize_coverage(REFERENCE)
    assert out["total_outcomes"] == comb(49, 6) * 43
    assert out["any_ticket_by_tier"]["first"]["outcomes"] == 4 * 43
    assert out["any_ticket_by_tier"]["first"]["probability"] == pytest.approx(
        4 / comb(49, 6))
    assert sum(out["awarded_ticket_count_outcomes"]) == out["total_outcomes"]
    assert out["any_prize_outcomes"] == (
        out["total_outcomes"] - out["awarded_ticket_count_outcomes"][0])
    # Winning any official tier includes the main-hit >=3 event, and 2+bonus.
    assert out["any_prize_probability"] >= exact_coverage(REFERENCE)["at_least"]["3"]
    assert out["expected_awarded_tickets"] >= out["any_prize_probability"]


def test_legal_portfolios_rejected_correctly():
    for args in (
        [(1, 2, 3, 4, 5, 5)],
        [REFERENCE[0], REFERENCE[0]],
        [(1, 2, 3, 4, 5, 50)],
        list(REFERENCE) + [REFERENCE[0]],
    ):
        with pytest.raises(ValueError):
            official_prize_coverage(args)
    with pytest.raises(ValueError):
        official_prize_coverage(REFERENCE, universe_size=8)
