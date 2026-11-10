"""Issue #15 P2: exact joint support, integrity gates and certification honesty."""
from __future__ import annotations

from bisect import bisect_right
from itertools import combinations
from math import comb, prod
from random import Random

import pytest

from lotto7x7.coverage_v06 import exact_coverage
from lotto7x7.deficit_compensation_v025 import REFERENCE
from lotto7x7.distribution_repair_portfolio_v027 import validate_four_disjoint
from lotto7x7.log_product_v027 import _grid
from lotto7x7.log_product_dual_portfolio_v027 import (
    SCHEMA, _describe_portfolio, _exact_coverage_parity, _fallback_four,
    _ticket_mask, _witness_for_cells, enumerate_joint_counts,
    frozen_weights, pareto_frontier, score_joint_cell,
)
from lotto7x7.official_prize_v026 import PRIZE_ORDER, official_prize_coverage


def test_small_universe_joint_counts_and_every_mass_exact():
    n, k = 8, 3
    grid = _grid(n, k, "0.5")
    counts, total = enumerate_joint_counts(n=n, k=k,
                                           thresholds=grid["integer_product_thresholds"])
    assert total == comb(n, k) == 56
    assert sum(map(sum, counts)) == 56
    assert len(counts) == k * (2*n-k+1)//2 - k*(k+1)//2 + 1
    # Independent brute assignment in float-log space rather than integer threshold comparisons.
    expected = [[0]*grid["bin_count"] for _ in counts]
    for nums in combinations(range(1, n + 1), k):
        from math import log
        index = bisect_right([float(x) for x in grid["edges"]], log(prod(nums))) - 1
        expected[sum(nums)-6][index] += 1
    assert counts == expected


@pytest.mark.parametrize("args", [
    (1, 1, [1, 2]), (7, 7, [1, 2]), (7, 3, []),
    (7, 3, [10, 10]), (7, 3, [10, 8, 1]),
    (True, 3, [1, 2]), (7, 3.5, [1, 2]),
])
def test_joint_rejects_invalid_universe_or_thresholds(args):
    n, k, thresholds = args
    with pytest.raises(ValueError):
        enumerate_joint_counts(n=n,k=k,thresholds=thresholds)


def test_fixed_weights_and_exact_integer_cell_ranking():
    ws, wl = frozen_weights(1e-6, 2e-6)
    assert ws == 10**18
    assert wl == 5*10**17
    assert score_joint_cell(0, 1, [3, 5], [1, 4], (ws, wl)) == 3*ws + 4*wl
    assert frozen_weights(1e-6, 2e-6) == (ws, wl)
    for a, b in ((0., 1.), (-1., 1.), (1., float("nan")), (1., float("inf"))):
        with pytest.raises(ValueError):
            frozen_weights(a,b)


def test_true_two_feature_pareto_frontier_dominance_and_ties():
    rows = [
        {"sum": 50, "log_bin": 1, "sum_numerator": 4, "log_numerator": 3, "objective_key": 7},
        {"sum": 60, "log_bin": 2, "sum_numerator": 5, "log_numerator": 1, "objective_key": 6},
        {"sum": 70, "log_bin": 3, "sum_numerator": 1, "log_numerator": 5, "objective_key": 6},
        {"sum": 80, "log_bin": 4, "sum_numerator": 2, "log_numerator": 2, "objective_key": 4},
        {"sum": 90, "log_bin": 5, "sum_numerator": 4, "log_numerator": 3, "objective_key": 7},
    ]
    result = pareto_frontier(rows)
    # (4,3) and (4,3) identical objectives: one deterministic feature-cell representative.
    assert {(x["sum_numerator"],x["log_numerator"]) for x in result} == {(4,3),(5,1),(1,5)}
    assert len(result) == 3
    assert result[0]["sum"] == 50
    assert pareto_frontier(rows) == result


def test_construct_four_disjoint_witness_and_exact_certificate_bound(monkeypatch):
    original = [
        (4, 21, 27, 28, 31, 41),
        (5, 7, 9, 38, 46, 47),
        (6, 15, 23, 29, 39, 40),
        (11, 12, 19, 32, 35, 43),
    ]
    assert [sum(t) for t in original] == [152]*4
    assert len(set.union(*(set(x) for x in original))) == 24
    cell = (152, 23)
    pools = {cell: [(_ticket_mask(t),t) for t in original]}
    witness = _witness_for_cells([cell]*4, pools, rng=Random(649), restarts=5)
    assert validate_four_disjoint(witness)
    assert _witness_for_cells([cell]*4,pools,rng=Random(649),restarts=5) == witness

    def fake_ticket(_report, numbers):
        return {"ticket": list(numbers), "sum": sum(numbers),
                "log_product_bin_index": 23, "dual_positive": True,
                "sum_gain": 2e-6, "log_gain": 1e-6}
    import lotto7x7.log_product_dual_portfolio_v027 as module
    monkeypatch.setattr(module,"dual_score_ticket",fake_ticket)
    ref = official_prize_coverage(REFERENCE)
    hits = exact_coverage(REFERENCE)
    nums = [0]*259
    nums[152-21] = 3
    logs = [0]*34
    logs[23] = 7
    weights=(2,3)
    single_key = score_joint_cell(152-21,23,nums,logs,weights)
    proof = _describe_portfolio(witness,name="dual_positive_unrestricted",
        upper=single_key*4, dual_p1={},sum_num=nums,log_num=logs,
        weights=weights,coverage_ref=ref,hits_ref=hits,search={})
    assert proof["status"] == "GLOBAL_OPTIMUM_CERTIFIED"
    assert proof["certified_global_optimum"]
    assert proof["unique_numbers"] == 24
    assert proof["sum_bins"] == [152]*4
    assert proof["coverage"]["exact_official_eight_tier_parity"]
    assert proof["coverage"]["exact_main_hit_coverage_parity"]
    assert proof["coverage"]["any_prize_probability"] == ref["any_prize_probability"]
    for tier in PRIZE_ORDER:
        assert proof["coverage"]["by_tier"][tier]["outcomes"] == ref["any_ticket_by_tier"][tier]["outcomes"]
    fallback = _describe_portfolio(witness,name="dual_positive_unrestricted",
        upper=single_key*4+1, dual_p1={},sum_num=nums,log_num=logs,
        weights=weights,coverage_ref=ref,hits_ref=hits,search={})
    assert fallback["status"] == "HEURISTIC_UNCERTIFIED"
    assert fallback["gap_exact_integer"] == "1"
    with pytest.raises(AssertionError, match="upper bound"):
        _describe_portfolio(witness,name="dual_positive_unrestricted",
            upper=single_key*4-1, dual_p1={},sum_num=nums,log_num=logs,
            weights=weights,coverage_ref=ref,hits_ref=hits,search={})
    none = _describe_portfolio(None,name="dual_positive_unrestricted",
        upper=single_key*4, dual_p1={},sum_num=nums,log_num=logs,
        weights=weights,coverage_ref=ref,hits_ref=hits,search={})
    assert none["status"] == "INFEASIBLE_UNCERTIFIED"
    assert not none["certified_global_optimum"]


def test_no_fallback_without_valid_dual_positive_partition(monkeypatch):
    import lotto7x7.log_product_dual_portfolio_v027 as module
    monkeypatch.setattr(module,"dual_score_ticket",
                        lambda report,ticket: {"dual_positive": False})
    candidate, details = _fallback_four({}, rng=Random(7), trials=12,
        distinct_sums=False, numerator_sum=[0]*259,numerator_log=[0]*34,
        weights=(1,1))
    assert candidate is None
    assert details["dual_positive_24_number_partitions"] == 0
