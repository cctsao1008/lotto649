"""Issue #15 P3d: unchanged legacy P2 search protocol and prior-only gates."""
from __future__ import annotations

from math import comb
from pathlib import Path

import pytest

import lotto7x7.log_product_p3d_legacy_replay_v027 as p3d
from lotto7x7.deficit_compensation_v025 import REFERENCE
from lotto7x7.distribution_repair_portfolio_v027 import validate_four_disjoint
from lotto7x7.official_prize_v026 import official_prize_coverage


EXAMPLE = (
    (1, 6, 20, 39, 41, 45),
    (2, 5, 8, 42, 47, 48),
    (3, 4, 7, 43, 46, 49),
    (14, 28, 31, 32, 36, 37),
)


def test_strict_fixed_indices_and_cutoff_conventions():
    assert p3d.validate_indices([300,1388,2476],2497)==(300,1388,2476)
    assert p3d.TARGET_INDICES==(300,1388,2476)
    assert p3d._date_from_yyyymmdd(20261106)=="2026-11-06"
    assert p3d.FINAL_CUTOFF=="2026-11-06"
    assert len(p3d.FROZEN_HASH)==64
    assert p3d.CONFIG=={
        "seed":649, "pool_per_cell":800, "cell_limit":24,
        "restarts":512, "fallback_trials":15000,
    }
    for bad in ([],[2],[300,300],[300,200],[2497],[True],[2.0],[300,2497]):
        with pytest.raises(ValueError):
            p3d.validate_indices(bad,2497)


def test_random_control_exact_pairwise_disjoint_and_deterministic():
    for t in p3d.TARGET_INDICES:
        rows=p3d._random_control(t)
        assert validate_four_disjoint(rows)==rows
        assert p3d._random_control(t)==rows
        assert len({n for bet in rows for n in bet})==24
        exact=official_prize_coverage(rows)
        baseline=official_prize_coverage(REFERENCE)
        assert exact["any_prize_outcomes"]==baseline["any_prize_outcomes"]
        assert exact["total_outcomes"]==comb(49,6)*43


def _variant(tickets, *, status="HEURISTIC_UNCERTIFIED", certificate=False):
    return {
        "tickets":[list(t) for t in tickets],
        "status":status,
        "certified_global_optimum":certificate,
        "ticket_details":[{"dual_positive":True} for _ in tickets],
        "coverage":{"exact_official_eight_tier_parity":True},
        "sum_gain_total":12e-6,
        "log_gain_total":21e-6,
        "objective_exact_integer":"100",
        "upper_bound_exact_integer":"1000",
    }


def test_audit_uses_same_eight_official_tiers_for_held_out_draw():
    row=p3d._audit_selected(_variant(EXAMPLE),
                            (1,2,3,4,5,6),bonus=7,distinct=False)
    assert not row["selection_failure"]
    assert row["tickets"]==[list(t) for t in EXAMPLE]
    assert row["status"]=="HEURISTIC_UNCERTIFIED"
    assert row["outcome"]["any_award"] in (0,1)
    assert row["outcome"]["max_main_hits"]<=6
    assert len(row["outcome"]["tiers"])==4
    with pytest.raises(AssertionError,match="distinct"):
        p3d._audit_selected(_variant((
            (1,2,3,4,5,6),(7,8,9,10,11,12),
            (13,14,15,16,17,18),(19,20,21,22,23,24))),
            (1,2,3,4,5,6),bonus=25,distinct=True)
    invalid=_variant(EXAMPLE)
    invalid["ticket_details"][0]["dual_positive"]=False
    with pytest.raises(AssertionError,match="dual-positive"):
        p3d._audit_selected(invalid,(1,2,3,4,5,6),bonus=7,distinct=False)


def test_uncertified_search_failure_record_is_explicit():
    result=p3d._audit_selected({
        "tickets":[],"status":"INFEASIBLE_UNCERTIFIED",
        "certified_global_optimum":False,
    }, (1,2,3,4,5,6),bonus=7,distinct=False)
    assert result["selection_failure"] is True
    assert result["outcome"] is None
    with pytest.raises(AssertionError):
        p3d._audit_selected({
            "tickets":[],"status":"GLOBAL_OPTIMUM_CERTIFIED",
            "certified_global_optimum":True,
        },(1,2,3,4,5,6),bonus=7,distinct=False)


def test_cached_comparison_refuses_unmatched_target_and_tampered_result():
    original={
        "primary":{"selection_failure":False,"tickets":[list(t) for t in EXAMPLE],
                   "outcome":p3d.evaluate_draw(EXAMPLE,(1,2,3,4,5,6),7)},
        "distinct_sums":{"selection_failure":True},
    }
    result=p3d.evaluate_draw(EXAMPLE,(1,2,3,4,5,6),7)
    pilot={"historical":{"records":[{
        "target_index_zero_based":300,"training_count":300,
        "target_main":[1,2,3,4,5,6],"target_bonus":7,
        "decisions":{
            "primary":{"tickets":[list(t) for t in EXAMPLE],**result},
            "distinct_sums":{"no_feasible_witness":True},
        },
    }]}}
    comparison=p3d._compare_cached(pilot,300,(1,2,3,4,5,6),7,original)
    assert comparison["primary"]["same_ticket_portfolio"]
    assert comparison["primary"]["original_any_award"]==result["any_award"]
    assert comparison["distinct_sums"]["original_failure"]
    assert p3d._compare_cached(None,300,(1,2,3,4,5,6),7,original) is None
    with pytest.raises(ValueError,match="no record"):
        p3d._compare_cached(pilot,1388,(1,2,3,4,5,6),7,original)
    with pytest.raises(ValueError,match="held-out"):
        p3d._compare_cached(pilot,300,(1,2,3,4,5,6),8,original)
    pilot["historical"]["records"][0]["decisions"]["primary"]["any_award"]=1-result["any_award"]
    with pytest.raises(ValueError,match="audit"):
        p3d._compare_cached(pilot,300,(1,2,3,4,5,6),7,original)


def test_original_import_identity_unchanged_and_missing_source_rejected(tmp_path):
    from lotto7x7.log_product_dual_portfolio_v027 import optimize
    assert p3d.original_p2_optimize is optimize
    with pytest.raises((ValueError,FileNotFoundError)):
        p3d.replay(tmp_path)
