"""v0.28 M0 new method: exact D4-shape, Sum, Log, Joint and 4-disjoint.

All synthetic universes below are explicitly test-only and must never be
mistaken for 6/49 full theory in a research report.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import comb
import json

import numpy as np
import pytest

import lotto7x7.multiobjective_v028 as m
from lotto7x7.shape_theory_v016 import LEVELS,classification_keys
from lotto7x7.shape_taxonomy_v015 import classify_numbers
from lotto7x7.distribution_repair_portfolio_v027 import validate_four_disjoint
from lotto7x7.official_prize_v026 import official_prize_coverage
from lotto7x7.deficit_compensation_v025 import REFERENCE


def test_exact_joint_and_shape_count_simultaneously_small_bruteforce(monkeypatch):
    # Tiny fake combinatorial stream; no broad declaration about full 6/49.
    original=[
        (1,2,3,4,5,6),(2,7,9,12,17,22),(5,8,11,15,19,21),
        (18,19,21,24,25,26),(41,42,43,44,45,49),
        (3,10,16,20,33,35),(4,6,17,18,29,49),
    ]
    monkeypatch.setattr(m,"TOTAL",len(original))
    monkeypatch.setattr(m,"combinations",lambda *args:iter(original))
    threshold=[1,10**20]
    joint,shapes,ref=m.exact_joint_shape_baselines(threshold,batch_size=3)
    assert ref["enumerated"]==len(original)
    assert len(joint)==259 and len(joint[0])==1
    assert sum(sum(row) for row in joint)==len(original)
    counts=Counter(sum(ticket) for ticket in original)
    for s,number in counts.items():
        assert joint[s-21][0]==number
    enc=classification_keys(classify_numbers(np.asarray(original,dtype=np.uint8)))
    for level in LEVELS:
        expected=Counter(int(x) for x in enc[level])
        assert shapes["counts"][level]=={str(k):v for k,v in sorted(expected.items())}
        assert sum(map(int,shapes["counts"][level].values()))==len(original)
    assert len(ref["joint_sha256"])==64
    assert len(ref["shape_counts_sha256"])==64


def test_shape_counterfactual_gain_matches_explicit_regularized_objective(monkeypatch):
    monkeypatch.setattr(m,"TOTAL",100)
    fair={1:20,2:30,3:50}
    observed={1:2,2:6,3:9}
    n=sum(observed.values())
    alpha=4.0
    scores,meta=m.shape_per_class_gain(observed,fair,n,alpha=alpha)
    def loss(hist,at_n):
        return sum((hist.get(k,0)/at_n-count/100)**2/
                   (count/100+alpha/n) for k,count in fair.items())
    before=loss(observed,n)
    assert meta["before"]==pytest.approx(before,abs=1e-15)
    for key,gain in scores.items():
        increased={**observed,key:observed.get(key,0)+1}
        assert gain==pytest.approx(before-loss(increased,n+1),abs=1e-15)
    assert m.fair_standard_deviation(list(scores.values()),list(fair.values()))["stddev"]>0


def test_taxonomy_levels_are_one_block_and_seeded_pool_is_not_old_p2():
    assert m.BLOCKS==("sum","log","joint","shape")
    assert m.SHAPE_INTERNAL_WEIGHTS=={
        "component":0.25,"partition":0.35,"edge":0.40
    }
    assert sum(m.SHAPE_INTERNAL_WEIGHTS.values())==1
    assert m.KNOWN_TICKETS[0]==(4,9,17,24,29,42)
    assert len(m.KNOWN_TICKETS)==12
    a=m._uniform_sample(20261110,100)
    b=m._uniform_sample(20261110,100)
    assert a==b and len(a)==100
    assert all(tuple(sorted(t))==t and len(set(t))==6 for t in a)
    assert all(1<=n<=49 for t in a for n in t)
    assert len(set(a))==100


def test_four_block_maximin_selection_and_shape_partitions():
    rows=[]
    for i in range(8):
        t=tuple(range(i*6+1,i*6+7))
        # i=7 reaches 48, all legal
        rows.append({
            "ticket":t,
            "mask":m._mask(t),
            "partition_code":i+10,
            "rank":[60-i,50+i,40+2*i,45+(i%3)],
        })
    for preference in ("balanced","shape","joint"):
        selected,objective=m.search_portfolio(rows,seed=649,style=preference,
                                              restarts=8)
        bets=validate_four_disjoint([r["ticket"] for r in selected])
        assert len({x for ticket in bets for x in ticket})==24
        assert len({r["partition_code"] for r in selected})==4
        assert objective["status"]=="HEURISTIC_UNCERTIFIED"
        assert objective["criterion"]==preference
        assert objective["maximin_rank_total"]>0
        assert objective["total_rank_sum"]>0
        cov=official_prize_coverage(bets)
        ref=official_prize_coverage(REFERENCE)
        assert cov["any_prize_outcomes"]==ref["any_prize_outcomes"]
    assert m._compare_key(selected,style="balanced")
    with pytest.raises(ValueError):
        m._compare_key(selected,style="unknown")


def test_hard_disjoint_and_distinct_partition_fail_closed():
    t=(1,2,3,4,5,6)
    source=[
        {"ticket":t,"mask":m._mask(t),"partition_code":1,"rank":[10]*4},
        {"ticket":(7,8,9,10,11,12),"mask":m._mask((7,8,9,10,11,12)),"partition_code":1,"rank":[10]*4},
        {"ticket":(13,14,15,16,17,18),"mask":m._mask((13,14,15,16,17,18)),"partition_code":1,"rank":[10]*4},
        {"ticket":(19,20,21,22,23,24),"mask":m._mask((19,20,21,22,23,24)),"partition_code":1,"rank":[10]*4},
    ]
    with pytest.raises(ValueError,match="INFEASIBLE_UNCERTIFIED"):
        m.search_portfolio(source,restarts=2)
    with pytest.raises(ValueError):
        m.search_portfolio(source[:2],restarts=2)


def test_full_universe_model_checksums_are_fixed_not_retuned_from_candidates():
    assert m.TOTAL==comb(49,6)
    assert m.CUTOFF=="2026-11-06"
    assert m.DRAW_ID==115000103
    assert m.INPUT_SHA=="d865e1d241b607488e58a00b22fb595afd9d8850718808d6e8b7f8"
    assert m.JOINT_SHA=="31136d04347ff7b84fa7b5d17e7d009068b784e46f43f1d49aa058bc7fcd422a"
    assert m.DEFAULT_SEED==20261110
    assert m.SCHEMA=="lotto7x7.multiobjective.v0.28.m0"
    assert len(set(m.KNOWN_TICKETS))==12
