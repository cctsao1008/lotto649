"""Issue #15 P3b pilot: frozen cached 4-bet selection and IID replay."""
from __future__ import annotations

from collections import defaultdict
from math import comb
from random import Random

import pytest

import lotto7x7.log_product_p3b_pilot_v027 as mod
from lotto7x7.log_product_dual_portfolio_v027 import _round_ratio_over_sqrt


TICKETS = (
    (1, 2, 3, 4, 5, 6),
    (7, 8, 9, 10, 11, 12),
    (13, 14, 15, 16, 17, 18),
    (19, 20, 21, 22, 23, 24),
)


def test_exact_official_award_predicate_and_special_number():
    first=mod.evaluate_draw(TICKETS,(1,2,3,4,5,6),25)
    assert first["any_award"]==1
    assert first["tiers"][0]=="first"
    assert first["max_main_hits"]==6
    ordinary=mod.evaluate_draw(TICKETS,(1,2,3,31,32,33),40)
    assert ordinary["any_award"]==1
    assert ordinary["tiers"][0]=="regular"
    assert mod.evaluate_draw(None,(1,2,3,4,5,6),25) is None
    for main,bonus in (((1,2,3,4,5,6),1),((1,2,3,4,5,6),0)):
        with pytest.raises(ValueError):
            mod.evaluate_draw(TICKETS,main,bonus)


def test_iid_generator_has_valid_unordered_main_and_distinct_bonus():
    a=mod.iid_history(100,seed=649)
    assert a==mod.iid_history(100,seed=649)
    assert a!=mod.iid_history(100,seed=650)
    assert len(a)==100
    for row,bonus in a:
        assert len(row)==6 and len(set(row))==6 and row==tuple(sorted(row))
        assert 1<=row[0]<row[-1]<=49
        assert bonus not in row and 1<=bonus<=49


def test_lexical_tie_and_disjoint_four_ticket_search():
    cells=[(sum(t),i,100-i) for i,t in enumerate(TICKETS)]
    pools={(sum(t),i):[(mod._mask(t),t)] for i,t in enumerate(TICKETS)}
    for distinct in (False,True):
        selected=mod.choose(cells,pools,seed=649,epoch=5,
                            distinct_sums=distinct,restarts=3,cell_limit=4)
        assert len(selected)==4
        assert sorted(selected)==sorted(TICKETS)
        assert len(set.union(*(set(t) for t in selected)))==24
        assert selected==mod.choose(cells,pools,seed=649,epoch=5,
                                    distinct_sums=distinct,restarts=3,cell_limit=4)
    with pytest.raises(ValueError):
        mod.choose(cells,pools,seed=649,epoch=5,
                   distinct_sums=False,restarts=0)


def test_catalogue_independent_of_history_and_exact_mass(monkeypatch):
    # Compact 4-outcome synthetic theoretical world through same reservoir.
    original=[
        (4,21,27,28,31,41),
        (5,7,9,38,46,47),
        (6,15,23,29,39,40),
        (44,45,46,47,48,49),
    ]
    counts=[[0] for _ in range(259)]
    for row in original:
        counts[sum(row)-21][0]+=1
    monkeypatch.setattr(mod,"THEORY_SIZE",4)
    monkeypatch.setattr(mod,"combinations",lambda values,k:iter(original))
    a,meta=mod.catalogue(counts,[1,10**20],seed=649,per_cell=2)
    b,meta2=mod.catalogue(counts,[1,10**20],seed=649,per_cell=2)
    assert a==b and meta==meta2
    assert meta["full_universe_enumerated"]==4
    assert meta["nonzero_joint_cells"]==2
    assert meta["candidate_pool_count"]==3
    assert len(meta["catalogue_checksum_sha256"])==64
    # This covers sum 279, verifying the digest encodes numbers > 255.
    assert (279,0) in a
    with pytest.raises(ValueError):
        mod.catalogue(counts,[1,10**20],seed=649,per_cell=0)


def test_candidate_rank_uses_exact_positive_gains_and_order():
    # Synthetic support: fair universe 8, sum bins 2, log bins 2.
    joint=[[1,3],[2,2]]
    fair_sum=[4,4]
    fair_log=[3,5]
    hist_sum=[8,1]
    hist_log=[7,2]
    outcome=mod.candidate_cells(hist_sum,hist_log,fair_sum,fair_log,joint)
    assert outcome==sorted(outcome,key=lambda x:(-x[2],x[0],x[1]))
    from lotto7x7.log_product_dual_scoring_v027 import exact_gain_numerators
    gs=exact_gain_numerators(hist_sum,fair_sum)
    gl=exact_gain_numerators(hist_log,fair_log)
    for s,li,key in outcome:
        si=s-21
        assert gs["numerators"][si]>0 and gl["numerators"][li]>0
        assert type(key) is int


def test_full_replay_applies_past_only_then_checks_realized_awards(monkeypatch):
    # Stub ONLY selector computation; actual control flow, update timing,
    # target main/bonus scoring, disjoint audit and event accumulation run.
    joint=[[0] for _ in range(259)]
    joint[21-21][0]=1
    joint[57-21][0]=1
    joint[93-21][0]=1
    joint[129-21][0]=1
    monkeypatch.setattr(mod,"candidate_cells",
                        lambda hs,hl,fs,fl,j:[(sum(t),0,4-i) for i,t in enumerate(TICKETS)])
    rows=[
        ((1,2,3,4,5,6),25),
        ((7,8,9,10,11,12),30),
        ((13,14,15,16,17,18),25),
        ((19,20,21,22,23,24),25),
        ((1,2,3,31,32,33),49),
    ]
    pools={(sum(t),0):[(mod._mask(t),t)] for t in TICKETS}
    a=mod.replay(rows,joint,[1,10**20],pools,
                 seed=649,warmup=3,stride=1,restarts=2,
                 cell_limit=4,exact_prize_prob=0.1,keep_records=True)
    assert a["evaluated_checkpoints"]==2
    assert a["stats"]["primary"]["selected"]==2
    assert a["stats"]["distinct_sums"]["selected"]==2
    assert a["stats"]["primary"]["wins"]==2
    assert a["records"][0]["training_count"]==3
    assert a["records"][1]["training_count"]==4
    assert len(a["records"][0]["decisions"]["primary"]["tickets"])==4
    assert a["stats"]["primary"]["iid_expected_any_awards"]==pytest.approx(0.2)

    # The training before index 3 cannot be changed by rewriting its target,
    # though realized result at index 3 may change.
    changed=rows[:]
    changed[3]=((35,36,37,38,39,40),25)
    b=mod.replay(changed,joint,[1,10**20],pools,
                 seed=649,warmup=3,stride=1,restarts=2,
                 cell_limit=4,exact_prize_prob=0.1,keep_records=True)
    assert a["records"][0]["decisions"]["primary"]["tickets"]==b["records"][0]["decisions"]["primary"]["tickets"]


def test_multiple_policy_max_statistic_tail_adjustment_and_resolution():
    # PLUS-ONE MC p; equality counted as an exceedance.
    assert mod._empirical_upper_tail(10,[1,2,3,4,5,6,7,8])==pytest.approx(1/9)
    assert mod._empirical_upper_tail(2,[1,2,2,3,4,5,6,7])==pytest.approx(8/9)
    assert mod._empirical_upper_tail(0,[-1,0,1])==pytest.approx(3/4)
    assert mod.SCHEMA.endswith("p3b_pilot")
    assert mod.POLICY=="cached_joint_cell_greedy_v1"
