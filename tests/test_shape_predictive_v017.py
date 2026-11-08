"""P2 statistical tests, self-contained legal shape fixtures."""
from math import comb
from pathlib import Path
import json
import numpy as np
import pytest

from lotto7x7.shape_predictive_v017 import (
    UNIVERSE,RARE_CODE_NAME,validate_history,_baseline_tables,_index_tags,
    _prequential_one,_iid_prequential_simulation,_fresh_uniform_tickets,
    _ticket_matches,_ticket_summary,candidate_backtest,analyze,per_draw_rows,
    last_candidate_rows)
from lotto7x7.shape_taxonomy_v015 import classify_numbers,edge_name
from lotto7x7.shape_theory_v016 import RULE_VERSION,classification_keys


def toy_inputs(n=32):
    rare=np.array([1,8,16,22,24,30],dtype=np.uint8)
    isolated=np.array([1,3,5,29,31,33],dtype=np.uint8)
    x=np.stack([rare if (j%9==0) else isolated for j in range(n)])
    q=classification_keys(classify_numbers(np.stack([rare,isolated])))
    assert edge_name(q['edge'][0])==RARE_CODE_NAME
    b=json.loads((Path(__file__).parent/'fixtures'/'p1_exact_universe_v1.json').read_text(encoding='utf-8'))
    return np.arange(1001,1001+n),x,b


def test_history_must_be_strictly_chronological():
    ids,x,_=toy_inputs()
    validate_history(ids,x)
    ids[5],ids[6]=ids[6],ids[5]
    with pytest.raises(ValueError):validate_history(ids,x)


def test_illegal_numbers_and_repetition_fail():
    ids,x,_=toy_inputs();x[0]=[1,1,2,3,4,5]
    with pytest.raises(ValueError):validate_history(ids,x)


def test_exact_prior_requires_full_mass():
    _,_,b=toy_inputs()
    _baseline_tables(b)
    b['counts']['edge'][next(iter(b['counts']['edge']))]-=1
    with pytest.raises(ValueError):_baseline_tables(b)


def test_codes_match_theory_index():
    _,x,b=toy_inputs()
    tab=_baseline_tables(b)
    tags=_index_tags(x,tab)
    assert all(len(tags[k])==len(x) for k in tab)


def test_zero_leakage_first_evaluation():
    idx=np.array([0,0,1,1,1,0,1],dtype=np.int32)
    p=np.array([.5,.5])
    s=_prequential_one(idx,p,3,2.)
    assert np.isnan(s[:3]).all()
    # after three draws counts [2,1], next draw is 1: (1+1)/(3+2)/0.5
    assert np.isclose(s[3],np.log(.8))


def test_changing_future_does_not_change_past_prediction():
    idx=np.array([0,1,1,0,0,1,0,1,1,1])
    a=_prequential_one(idx,[.5,.5],3,3.)
    idx[7:]=0
    b=_prequential_one(idx,[.5,.5],3,3.)
    assert np.array_equal(a[:7],b[:7],equal_nan=True)


def test_online_constant_fair_prior_with_only_prior_weights():
    out=_prequential_one(np.zeros(12,dtype=int),[.9,.1],3,20.)
    assert np.isfinite(out[3:]).all()


def test_iid_refits_and_nested_labels():
    _,_,b=toy_inputs()
    out=_iid_prequential_simulation(_baseline_tables(b),4,12,1.,20,649)
    assert set(out)=={'component','partition','edge'}
    assert all(v.shape==(20,) for v in out.values())
    out2=_iid_prequential_simulation(_baseline_tables(b),4,12,1.,20,649)
    assert all(np.array_equal(out[k],out2[k]) for k in out)


def test_iid_refit_requires_positive_count():
    _,_,b=toy_inputs()
    with pytest.raises(ValueError):_iid_prequential_simulation(_baseline_tables(b),4,12,2.,0,649)


def test_uniform_tickets_legal_unique_reproducible():
    a=_fresh_uniform_tickets(np.random.default_rng(100),120)
    b=_fresh_uniform_tickets(np.random.default_rng(100),120)
    assert np.array_equal(a,b)
    assert len({tuple(r) for r in a})==120
    assert np.all((a>=1)&(a<=49)) and np.all(np.diff(a,axis=1)>0)


def test_ticket_match_exact_case():
    bets=np.array([[[1,2,3,4,5,6],[1,10,11,12,13,14]]])
    x=np.array([[1,2,3,4,5,6]])
    assert _ticket_matches(bets,x).tolist()==[[6,1]]
    summary=_ticket_summary(_ticket_matches(bets,x))
    assert summary['total_match_6']==1 and summary['total_match_at_least_3']==1


def test_ticket_fair_probability_sum():
    assert sum(comb(6,j)*comb(43,6-j) for j in range(7))==UNIVERSE


def test_candidate_pool_before_target_and_same_budget():
    _,x,b=toy_inputs()
    tab=_baseline_tables(b);tags=_index_tags(x,tab)
    r,a=candidate_backtest(x,tags['edge'],tab,min_train=10,tau=20.,seed=12,
       pool_size=12,bets=3,ticket_null_reps=8)
    assert r['draws_evaluated']==len(x)-10
    for s in ('uniform','shape_ranked'):
        assert a['tickets'][s].shape==(len(x)-10,3,6)
        assert a['matches'][s].max()<=6
        assert r['strategies'][s]['total_bets']==3*(len(x)-10)


def test_future_draw_does_not_affect_candidate_picks_at_that_time():
    _,x,b=toy_inputs()
    tab=_baseline_tables(b);tags=_index_tags(x,tab)
    r,a=candidate_backtest(x,tags['edge'],tab,min_train=10,tau=20.,seed=7,pool_size=9,bets=2,ticket_null_reps=3)
    x2=x.copy();x2[10]=x[0]
    tags2=_index_tags(x2,tab)
    r2,a2=candidate_backtest(x2,tags2['edge'],tab,min_train=10,tau=20.,seed=7,pool_size=9,bets=2,ticket_null_reps=3)
    for s in ('uniform','shape_ranked'):
        assert np.array_equal(a['tickets'][s][0],a2['tickets'][s][0])


def test_p2_end_to_end_small_fixture():
    ids,x,b=toy_inputs()
    report,a=analyze(ids,x,b,min_train=10,tau=20.,iid_replicates=16,ticket_null_replicates=8,
          seed=649,pool_size=12,bets=3)
    assert report['data']['evaluated_draws']==len(x)-10
    assert len(list(per_draw_rows(a,10)))==len(x)-10
    assert len(list(last_candidate_rows(a,report)))==6
    assert report['rare_shape_watchlist']['restriction'].startswith('Discovered using')
    for level in ('component','partition','edge'):
        assert np.isnan(a['gain_'+level][:10]).all()
        assert np.isfinite(a['gain_'+level][10:]).all()


def test_reproducible_end_to_end():
    ids,x,b=toy_inputs()
    kw=dict(min_train=10,tau=20.,iid_replicates=8,ticket_null_replicates=3,
           seed=649,pool_size=12,bets=3)
    r,a=analyze(ids,x,b,**kw)
    r2,a2=analyze(ids,x,b,**kw)
    assert r['prequential_log_scores']==r2['prequential_log_scores']
    assert np.array_equal(a['tickets_shape_ranked'],a2['tickets_shape_ranked'])
