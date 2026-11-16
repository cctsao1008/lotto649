"""Regression tests for the coverage/repair comparison."""
from math import comb

import pytest

from lotto7x7 import portfolio_pareto_v025 as p
from lotto7x7.coverage_v06 import UNIVERSE, joint_coverage_counts, single_bet_anchor


def test_joint_threshold_counts_preserve_mass_and_expectation():
    for portfolio in (
        p.DISJOINT_REPRESENTATIVE,
        ((1, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 7), (1, 8, 9, 10, 11, 12)),
    ):
        result = joint_coverage_counts(portfolio)
        n = len(portfolio)
        for t in (3, 4, 5, 6):
            counts = result["qualifying_ticket_counts"][str(t)]
            assert sum(counts) == UNIVERSE
            assert sum(counts[1:]) == sum(result["max_hits"][t:])
            assert sum(i * v for i, v in enumerate(counts)) == n * sum(single_bet_anchor()[t:])
    disjoint = joint_coverage_counts(p.DISJOINT_REPRESENTATIVE)
    assert disjoint["qualifying_ticket_counts"]["3"][2] == comb(4, 2) * comb(6, 3)**2
    assert disjoint["qualifying_ticket_counts"]["4"][2:] == [0, 0, 0]


def test_pareto_keeps_tradeoffs_and_coalesces_equal_objectives():
    assert p.pareto_frontier({
        (0,): ((8, 4, 2), 0.1, 22, 3),
        (1,): ((9, 4, 2), 0.1, 22, 3),
        (2,): ((9, 4, 2), 0.1, 22, 3),
        (3,): ((9, 4, 2), 0.2, 23, 3),
        (4,): ((9, 4, 2), 0.2, 22, 3),
    }) == [((1,), 2), ((3,), 1)]


def test_swap_search_distinguishes_floors_from_repair():
    def score(indices):
        candidate = max(indices)
        return (candidate, candidate, candidate), float(candidate), 24, 4

    for policy, expected in (("hybrid", 4), ("coverage_only", 5), ("compensation_only", 3)):
        indices, visited = p.swap_search(6, score, policy, (4, 4, 4))
        assert max(indices) == expected
        assert len(set(indices)) == 4
        assert visited <= 1 + p.PASSES * 4 * (6 - 4)


def test_archived_exact_coverage_and_strategy_budget():
    import json
    from pathlib import Path

    path = Path("research/artifacts/v025_exact_covariance/v025_portfolio_pareto_20261116.json")
    result = json.loads(path.read_text(encoding="utf-8"))
    assert set(result["strategies"]) == {
        "uniform", "coverage_v021", "shape_score_v021",
        "compensation_only", "coverage_only", "hybrid",
    }
    assert result["independent_main_ball_audit"]["full_dp_parity_pass"] == [
        "synthetic_disjoint", "uniform",
    ]
    census = result["exact_theoretical_coverage_census"]
    assert census["structures"] == 2957
    assert census["structures_attaining_all_maxima"] == 1
    assert census["maximum_outcomes"] == {"3": 1040096, "4": 55216, "5": 1036}
    cap = result["config"]["portfolio_evaluation_cap_per_strategy"]
    for row in result["strategies"].values():
        assert row["evaluations"] <= cap
        assert len(set(map(tuple, row["tickets"]))) == 4
        assert row["main_ball_at_least_counts"]["6"] == 4
    hybrid = result["strategies"]["hybrid"]
    assert hybrid["meets_hybrid_floor"]
    best = result["pareto"]["best_repair_at_hybrid_floors_among_evaluated"]
    assert best["repair_after"] <= hybrid["repair_after"] + 1e-12
    assert hybrid["official_any_prize_probability"] == pytest.approx(0.1227314589619088)
