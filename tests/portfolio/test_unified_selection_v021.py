"""v0.21 independent deterministic unit tests and full committed freeze integration."""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np
import pytest

from lotto7x7.unified_selection_v021 import (
    SCHEMA, UNIVERSE, _digest_candidates, _objective_key, build_selection,
    coverage_select, frozen_shape_scores, selection_html, uniform_candidates
)
from lotto7x7.coverage_v06 import exact_coverage, normalized_bets
from lotto7x7.shape_prospective_v019 import check_freeze


def test_pool_legality_and_uniqueness():
    x=uniform_candidates(200,649)
    assert x.shape==(200,6)
    assert x.dtype==np.uint8
    assert np.all(x[:,1:]>x[:,:-1])
    assert x.min()>=1 and x.max()<=49
    assert len({tuple(row) for row in x.tolist()})==200


def test_pool_reproducible():
    a=uniform_candidates(90,64)
    assert np.array_equal(a,uniform_candidates(90,64))
    assert not np.array_equal(a,uniform_candidates(90,65))


@pytest.mark.parametrize('size,seed',[(4,1),(0,1),(5001,1),(False,1),(10,-1),(10,True)])
def test_pool_rejects_bad_args(size,seed):
    with pytest.raises(ValueError):uniform_candidates(size,seed)


def test_digest_deterministic_and_sensitive_to_order():
    a=uniform_candidates(10,649)
    assert _digest_candidates(a)==_digest_candidates(a.copy())
    assert _digest_candidates(a)!=_digest_candidates(a[::-1])


@pytest.mark.parametrize('n',range(1,6))
def test_exact_six_hit_probability_invariant(n):
    a=uniform_candidates(15,101)
    p=exact_coverage([list(map(int,row)) for row in a[:n]])
    assert p['universe']==comb(49,6)
    assert sum(p['max_hits_counts_0_to_6'])==comb(49,6)
    assert p['at_least_counts']['6']==n
    assert p['at_least']['6']==pytest.approx(n/comb(49,6))


def test_fair_single_ticket_matches_hypergeometric():
    a=uniform_candidates(5,78)[0]
    p=exact_coverage([a.tolist()])
    assert p['max_hits_counts_0_to_6']==[comb(6,k)*comb(43,6-k) for k in range(7)]


def test_select_exact_objective_never_below_same_pool_random():
    pool=uniform_candidates(28,649)
    for objective in ('3plus','4plus','5plus'):
        ix,diagnostic=coverage_select(pool,5,objective=objective)
        assert len(ix)==5 and len(set(ix))==5
        assert diagnostic['globally_optimal'] is False
        assert diagnostic['oracle_evaluations']>0
        assert tuple(diagnostic['selected_objective'])>=tuple(diagnostic['baseline_objective'])
        candidate=exact_coverage([pool[i].tolist() for i in ix])
        baseline=exact_coverage([pool[i].tolist() for i in range(5)])
        assert _objective_key(candidate,objective)>=_objective_key(baseline,objective)


def test_count_one_equals_uniform_six_odds():
    pool=uniform_candidates(20,649)
    ix,diag=coverage_select(pool,1,objective='4plus')
    assert ix==[0]
    assert diag['selected_objective']==diag['baseline_objective']


def test_evaluation_budget_enforced():
    pool=uniform_candidates(20,649)
    with pytest.raises(RuntimeError,match='budget'):
        coverage_select(pool,5,objective='3plus',max_evaluations=2)


def test_duplicate_pool_rejected():
    a=uniform_candidates(5,649)
    a[2]=a[0]
    with pytest.raises(ValueError,match='duplicate'):
        coverage_select(a,5,objective='3plus')


def test_objective_validation():
    with pytest.raises(ValueError):_objective_key({'at_least_counts':{'3':1,'4':1,'5':1}},'7plus')


def test_html_contains_3_strategy_grids():
    p={ 'settings':{'pool_size':5,'tickets':1,'objective':'3plus'},
         'strategies':{},'data':{'freeze_sha256':'test'} }
    ticket=[1,2,3,4,5,6]
    coverage=exact_coverage([ticket])
    for name in ('uniform','coverage','shape_score'):
        p['strategies'][name]={'tickets':[ticket],'description':'test','exact_fair_portfolio_coverage':coverage}
    page=selection_html(p)
    assert page.count('role="img"')==3
    assert 'shape_score' in page and 'coverage' in page and 'uniform' in page
    assert '13,983,816' in page


def test_schema_and_universe_constants():
    assert SCHEMA.endswith('v0.21')
    assert UNIVERSE==13983816
