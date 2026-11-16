"""Independent small-universe and exact power regressions."""
from itertools import combinations
from math import comb

import pytest

from research.lotto7x7.portfolio_validation_v025 import paired_award_counts, sign_test_power


def test_exact_paired_census_matches_small_brute_force():
    repair = [(1, 2, 3, 4, 5, 6), (2, 3, 4, 5, 6, 7),
              (1, 3, 4, 8, 9, 10), (6, 7, 8, 10, 11, 12)]
    control = [(2, 5, 7, 8, 11, 12), (1, 4, 7, 8, 9, 10),
               (1, 2, 3, 4, 5, 7), (3, 5, 6, 7, 9, 11)]
    labels = set(range(1, 16))
    categories = {(True, False): "repair_only", (False, True): "control_only",
                  (True, True): "both", (False, False): "neither"}
    counts = dict.fromkeys(categories.values(), 0)
    for draw in combinations(sorted(labels), 6):
        main = set(draw)
        for bonus in labels - main:
            wins = tuple(any(
                len(main.intersection(ticket)) >= 3
                or (len(main.intersection(ticket)) == 2 and bonus in ticket)
                for ticket in portfolio
            ) for portfolio in (repair, control))
            counts[categories[wins]] += 1
    assert counts["repair_only"] > 0 and counts["control_only"] > 0
    assert paired_award_counts(repair, control, universe_size=15) == {
        **counts, "total": comb(15, 6) * 9,
    }


def test_sign_test_power_boundary_and_small_horizon():
    assert sign_test_power(4, 1.0, 0.25) == 0.0
    assert sign_test_power(5, 1.0, 0.25) == pytest.approx(0.625**5)
    assert sign_test_power(5, 1.0, 0.0, alpha=1 / 32) == pytest.approx(1 / 32)
    assert sign_test_power(5, 0.5, 0.25) == pytest.approx((0.5 * 0.75)**5)
    with pytest.raises(ValueError, match="incidence"):
        sign_test_power(10, 0.2, 0.21)
