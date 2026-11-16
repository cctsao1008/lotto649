"""Exact special-ball-aware prize-event distribution for a four-ticket 6/49 set.

The special ball is drawn uniformly from the remaining 43 numbers after the
six main balls. Equivalently, draw the special ball first from 49, then the
six main balls uniformly from the remaining 48. Group identical ticket
membership masks and use integer multivariate hypergeometric DP. This counts
all (main-six-set, special-ball) outcomes without sampling or independence
assumptions between ticket wins.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from math import comb

import numpy as np


PRIZE_TIERS = ("jackpot", "second", "third", "fourth", "fifth",
               "sixth", "seventh", "regular")
NOMINAL_FIXED_TIER_PRIZES = {"fifth": 2000, "sixth": 1000,
                              "seventh": 400, "regular": 400}


def _grade(main_hits: int, special: bool) -> str | None:
    """Return a category; current 6/49 base game has eight prize tiers."""
    if main_hits == 6:
        return "jackpot"
    if main_hits == 5:
        return "second" if special else "third"
    if main_hits == 4:
        return "fourth" if special else "fifth"
    if main_hits == 3:
        return "sixth" if special else "regular"
    if main_hits == 2 and special:
        return "seventh"
    return None


def exact_prize_distribution(tickets: object, *, universe: int = 49,
                             draw_size: int = 6) -> dict:
    """Compute exact per-outcome number of winning lines and expected tiers."""
    if type(universe) is not int or type(draw_size) is not int or not (
            2 <= draw_size < universe <= 49):
        raise ValueError("invalid lottery universe")
    try:
        rows = tuple(tuple(row) for row in tickets)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid ticket collection") from exc
    if len(rows) != 4 or any(len(row) != draw_size or
           any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_))
               or not 1 <= int(v) <= universe for v in row) or
           len(set(row)) != draw_size for row in rows):
        raise ValueError("exactly four legal tickets required")
    if len(set(tuple(sorted(map(int, row))) for row in rows)) != 4:
        raise ValueError("duplicate ticket combination")
    grouped = Counter(
        sum(1 << i for i, row in enumerate(rows) if ball in row)
        for ball in range(1, universe + 1)
    )
    wins_distribution = [0] * 5
    tier_counts = Counter()
    denominator = universe * comb(universe - 1, draw_size)
    for special_mask, special_choices in grouped.items():
        remaining = dict(grouped)
        remaining[special_mask] -= 1
        states = {(0, (0, 0, 0, 0)): 1}
        for mask, group_size in sorted(remaining.items()):
            nxt = defaultdict(int)
            for (main_count, hits), ways in states.items():
                for amount in range(min(group_size, draw_size - main_count) + 1):
                    new_hits = tuple(hits[j] + (amount if mask & (1 << j) else 0)
                                     for j in range(4))
                    nxt[(main_count + amount, new_hits)] += ways * comb(group_size, amount)
            states = nxt
        for (main_count, hits), ways in states.items():
            if main_count != draw_size:
                continue
            weight = special_choices * ways
            if draw_size == 6:
                grades = [_grade(hits[j], bool(special_mask & (1 << j)))
                          for j in range(4)]
                for grade in grades:
                    if grade is not None:
                        tier_counts[grade] += weight
                count_winners = sum(g is not None for g in grades)
            else:
                # Small-universe brute-force testing uses the same base
                # any-prize condition but without actual 6/49 tier names.
                count_winners = sum(hits[j] >= 3 or
                                    (hits[j] == 2 and bool(special_mask & (1 << j)))
                                    for j in range(4))
            wins_distribution[count_winners] += weight
    if sum(wins_distribution) != denominator:
        raise AssertionError("weighted special/main partition is incomplete")
    if draw_size == 6 and tier_counts["jackpot"] != 4 * (universe - 6):
        raise AssertionError("four-ticket jackpot count does not match exact null")
    return {
        "total_equally_likely_outcomes": denominator,
        "exact_counts_winning_lines_0_to_4": wins_distribution,
        "probabilities_winning_lines_0_to_4": [v / denominator for v in wins_distribution],
        "probability_any_prize": 1 - wins_distribution[0] / denominator,
        "expected_number_winning_lines": sum(i * n for i, n in enumerate(wins_distribution)) / denominator,
        "expected_ticket_counts_by_tier": {
            tier: tier_counts[tier] / denominator for tier in PRIZE_TIERS
        } if draw_size == 6 else {},
        "expected_nominal_fixed_lower_tier_payout_ntd": (
            sum(tier_counts[tier] * value for tier, value
                in NOMINAL_FIXED_TIER_PRIZES.items()) / denominator
        ) if draw_size == 6 else None,
        "fixed_tier_payout_assumption_ntd": NOMINAL_FIXED_TIER_PRIZES if draw_size == 6 else None,
        "assumptions": "fair 6 main balls plus 1 special from the remaining 43; regular base prize tiers; no promotions",
        "limits": "not the full monetary EV; jackpot/second/third/fourth prizes vary, taxes and ticket cost excluded",
    }
