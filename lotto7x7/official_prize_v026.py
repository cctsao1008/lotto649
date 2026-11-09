"""Exact Taiwan Lotto 6/49 official-tier portfolio probability engine.

No IID sampling, history, or prize-value assumptions. Each outcome is six
unordered main balls plus one distinct special ball. Group labels by portfolio
membership mask; enumerate possible main-ball counts per group and weight by
combinations, then enumerate the special ball's remaining membership group.
Works with up to four unique bets; supports a smaller universe for brute-force
regression tests. Official game uses numbers 1..49 and six main balls.
"""
from __future__ import annotations

from collections import Counter
from math import comb
from pathlib import Path
import argparse
import json

from .coverage_v06 import normalized_bets

PRIZE_ORDER = ("first", "second", "third", "fourth", "fifth", "sixth",
               "seventh", "regular")
SCHEMA = "lotto7x7.official_prize_exact.v0.26.p0"

def prize_tier(main_hits: int, bonus_hit: bool) -> str | None:
    """Eight mutually exclusive Taiwan Lotto 6/49 official award tiers.

    Special number never contributes to the count of six main-number hits.
    """
    if not 0 <= main_hits <= 6:
        raise ValueError("main_hits must be between 0 and 6")
    if main_hits == 6:
        return "first"
    if main_hits == 5:
        return "second" if bonus_hit else "third"
    if main_hits == 4:
        return "fourth" if bonus_hit else "fifth"
    if main_hits == 3:
        return "sixth" if bonus_hit else "regular"
    if main_hits == 2 and bonus_hit:
        return "seventh"
    return None


def official_prize_coverage(bets, *, universe_size=49):
    """Exact counts for multi-ticket joint prize events.

    The output denominator is C(universe_size,6)*(universe_size-6).
    All official Lotto649 uses universe_size=49. Lower sizes are reserved
    for small-universe exhaustive unit-test comparisons.
    """
    if (not isinstance(universe_size, int) or isinstance(universe_size, bool)
            or not 7 <= universe_size <= 49):
        raise ValueError("universe_size must be 7..49")
    rows = normalized_bets(bets, maximum=4)
    if any(row[-1] > universe_size for row in rows):
        raise ValueError("ticket number exceeds universe size")
    count = len(rows)
    groups = Counter()
    membership = [set(row) for row in rows]
    for num in range(1, universe_size + 1):
        mask = sum(1 << j for j in range(count) if num in membership[j])
        groups[mask] += 1
    group_rows = tuple(sorted(groups.items()))
    denominator = comb(universe_size, 6) * (universe_size - 6)
    any_prize = 0
    tier_any = {tier: 0 for tier in PRIZE_ORDER}
    tier_ticket_total = {tier: 0 for tier in PRIZE_ORDER}
    award_count_histogram = [0] * (count + 1)
    outcomes = 0

    # Recursion depth is bounded by number of occupied membership classes
    # (<= 2^4=16); at most six main balls are distributed among them.
    def traverse(index, left, multiplicity, hits, picked):
        nonlocal outcomes, any_prize
        if index == len(group_rows):
            if left:
                return
            # Exactly one of the remaining unused balls is special.
            for (mask, group_size), chosen in zip(group_rows, picked):
                possible_bonus = group_size - chosen
                if possible_bonus <= 0:
                    continue
                weight = multiplicity * possible_bonus
                outcomes += weight
                tiers = [prize_tier(hits[i], bool(mask & (1 << i)))
                         for i in range(count)]
                awards = sum(tier is not None for tier in tiers)
                award_count_histogram[awards] += weight
                if awards:
                    any_prize += weight
                for tier in PRIZE_ORDER:
                    occurrences = tiers.count(tier)
                    if occurrences:
                        tier_any[tier] += weight
                        tier_ticket_total[tier] += weight * occurrences
            return

        mask, capacity = group_rows[index]
        for k in range(min(left, capacity) + 1):
            next_hits = tuple(hits[i] + (k if mask & (1 << i) else 0)
                              for i in range(count))
            traverse(index + 1, left - k, multiplicity * comb(capacity, k),
                     next_hits, picked + (k,))

    traverse(0, 6, 1, (0,) * count, ())
    if outcomes != denominator or sum(award_count_histogram) != denominator:
        raise AssertionError("official prize universe mass does not close")
    if sum(tier_ticket_total.values()) != sum(i * c for i, c in enumerate(award_count_histogram)):
        raise AssertionError("ticket award expected-count identity failed")
    if sum(tier_any.values()) < any_prize:
        raise AssertionError("tier union monotonicity failed")

    return {
        "schema": SCHEMA,
        "method": "exact_membership_mask_combinatorial_enumeration",
        "tickets": [list(row) for row in rows],
        "ticket_count": count,
        "universe_size": universe_size,
        "total_outcomes": denominator,
        "main_draws": comb(universe_size, 6),
        "special_choices_per_draw": universe_size - 6,
        "any_prize_outcomes": any_prize,
        "any_prize_probability": any_prize / denominator,
        "no_prize_probability": (denominator - any_prize) / denominator,
        "awarded_ticket_count_outcomes": award_count_histogram,
        "awarded_ticket_count_probabilities": [v / denominator for v in award_count_histogram],
        "expected_awarded_tickets": sum(i * v for i, v in enumerate(award_count_histogram)) / denominator,
        "any_ticket_by_tier": {
            tier: {"outcomes": tier_any[tier], "probability": tier_any[tier] / denominator}
            for tier in PRIZE_ORDER
        },
        "expected_tickets_by_tier": {
            tier: {"ticket_outcomes": tier_ticket_total[tier],
                   "expected_count": tier_ticket_total[tier] / denominator}
            for tier in PRIZE_ORDER
        },
        "note": ("Fair-outcome portfolio probabilities, not forecasting. "
                 "Tier union events overlap across different tickets; "
                 "variable prize amounts/returns are not inferred."),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickets", required=True, nargs="+",
                        help="each ticket as six comma-separated numbers, e.g. 4,9,17,24,29,42")
    parser.add_argument("--out", type=Path, default=Path("results/v026_official_prize.json"))
    args = parser.parse_args(argv)
    try:
        rows = [tuple(map(int, item.split(","))) for item in args.tickets]
    except ValueError as exc:
        parser.error("each ticket must contain six comma-separated integers")
    output = official_prize_coverage(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"[P0] official-prize exact outcomes={output['total_outcomes']}")
    print(f"[PORTFOLIO] tickets={output['ticket_count']} "
          f"any_prize={output['any_prize_probability']:.12f} "
          f"expected_awarded_tickets={output['expected_awarded_tickets']:.12f}")
    print(f"[REPORT] {args.out}")


if __name__ == "__main__":
    main()
