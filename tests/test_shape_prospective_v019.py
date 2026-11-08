"""P3 future-only protocol: real P1 artifacts + synthetic post-freeze events."""
from __future__ import annotations

import csv
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np
import pytest

from lotto7x7.shape_prospective_v019 import (
    make_freeze,check_freeze,content_hash,verify_seal,read_updates,
    prospective_evaluate,report_html,RARE_CODE_NAME)

ROOT=Path('/mnt/data')
CAT=ROOT/'v015_v016_p1_shape_catalog.npz'
UNIVERSE=ROOT/'v015_v016_p1_shape_catalog_universe.json'
WATCH=ROOT/'v017_v018_p2_complete_future_watchlist.json'

@pytest.fixture(scope='module')
def frozen():
    f=make_freeze(CAT,UNIVERSE,WATCH,frozen_at=datetime(2026,11,8,9,20,tzinfo=timezone.utc))
    ids,nums,tables,_=check_freeze(f,CAT,UNIVERSE)
    return f,ids,nums,tables


def _csv(path,rows,headers=('date','draw_id','n1','n2','n3','n4','n5','n6','special')):
    with path.open('w',newline='',encoding='utf-8') as out:
        w=csv.DictWriter(out,fieldnames=headers);w.writeheader();w.writerows(rows)
    return str(path)


def _new(day='2026-11-09',number=115000104,nums=(1,8,16,22,24,30)):
    return {'date':day,'draw_id':str(number),**{f'n{i+1}':str(x) for i,x in enumerate(nums)},'special':'49'}


def test_freeze_fixes_real_historical_boundary(frozen):
    f,ids,nums,_=frozen
    assert f['schema'].endswith('v0.19')
    assert len(ids)==2497 and int(ids[-1])==115000103
    assert f['freeze_local_date_taipei']=='2026-11-08'
    assert f['source']['historical_draw_count']==2497
    assert f['tests']['rare_alternative_probability']==10*f['tests']['rare_fair_probability']
    assert f['tests']['mixture_weights']==[.25]*4
    verify_seal(f)


def test_no_new_returns_no_evidence(frozen):
    f,_,_,tables=frozen
    result,rows=prospective_evaluate(f,tables,[],[])
    assert rows==[] and result['status']=='NO_PROSPECTIVE_DRAWS'
    assert result['decision']=='NO_FUTURE_DATA' and result['mixture_e']==1
    assert result['anytime_p_upper_bound']==1
    assert '尚' not in report_html(result) or '新期別' in report_html(result)


def test_real_history_csv_yields_no_future(frozen):
    f,ids,nums,_=frozen
    updates,sources=read_updates([str(ROOT/'v015_v016_p1_shape_catalog_draws.csv')],
                                freeze=f,reference_ids=ids,reference_numbers=nums)
    assert updates==[] and len(sources)==1 and len(sources[0]['sha256'])==64


def test_future_one_event_and_rare_hit(frozen,tmp_path):
    f,ids,nums,tables=frozen
    record=_new(nums=(17,22,23,25,31,33))  # Synthetic case matching a known rare topology.
    csvpath=_csv(tmp_path/'future.csv',[record]); events,src=read_updates([csvpath],freeze=f,reference_ids=ids,reference_numbers=nums)
    r,rows=prospective_evaluate(f,tables,events,src)
    assert len(rows)==1 and r['rare_future_count']==1
    assert rows[0]['edge_class']==RARE_CODE_NAME
    assert rows[0]['rare_log_gain']==pytest.approx(np.log(10))
    assert rows[0]['mixture_e']==pytest.approx(sum(np.exp(v) for v in r['cumulative_log_e'].values())/4)
    assert r['last_new_draw_id']==115000104


def test_own_result_does_not_change_own_forecast(frozen):
    f,_,_,tables=frozen
    # At t=0 all class-model pre-draw probabilities must use identical frozen counts.
    a=[{'draw_id':115000104,'date':'2026-11-09','numbers':[1,8,16,22,24,30]}]
    b=[{'draw_id':115000104,'date':'2026-11-09','numbers':[4,8,12,16,33,49]}]
    ra,rowsa=prospective_evaluate(f,tables,a,[])
    rb,rowsb=prospective_evaluate(f,tables,b,[])
    # Every predicted vector sums to one based on counts; own event is not added to numerator until after scoring.
    level='component'
    ca=f['frozen_rule']['historical_counts'][level]
    for r in (rowsa[0],rowsb[0]):
        code=int(r[level+'_class'][1:]);p=r[level+'_fair_p']
        expected=(ca[str(code)]+384*p)/(2497+384)
        assert r[level+'_q_before_draw']==pytest.approx(expected)
    assert ra['eligible_new_draws']==rb['eligible_new_draws']==1


def test_sequential_second_draw_updates_after_first(frozen):
    f,_,_,tables=frozen
    seq=[{'draw_id':115000104,'date':'2026-11-09','numbers':[1,8,16,22,24,30]},
         {'draw_id':115000105,'date':'2026-11-12','numbers':[1,8,16,22,24,30]}]
    _,rows=prospective_evaluate(f,tables,seq,[])
    for lev in ('component','partition','edge'):
        p=rows[0][lev+'_fair_p']
        before=rows[0][lev+'_q_before_draw']
        after=rows[1][lev+'_q_before_draw']
        expected=(before*(2497+384)+1)/(2498+384)
        assert after==pytest.approx(expected)
        assert after>=before


def test_freeze_checksum_tamper_is_rejected(frozen):
    f,_,_,_=frozen
    bad=json.loads(json.dumps(f));bad['source']['last_draw_id']+=1
    with pytest.raises(ValueError,match='checksum'):verify_seal(bad)


def test_freeze_source_hash_tamper_is_rejected(frozen):
    f,_,_,_=frozen
    bad=json.loads(json.dumps(f));bad['source']['catalog_sha256']='0'*64
    bad['self_checksum_sha256']=content_hash({k:v for k,v in bad.items() if k!='self_checksum_sha256'})
    with pytest.raises(ValueError,match='bytes changed'):check_freeze(bad,CAT,UNIVERSE)


def test_prior_historical_disagreement_rejected(frozen,tmp_path):
    f,ids,nums,_=frozen
    bad=_new(number=93000001,day='2004-01-01',nums=(1,2,3,4,5,6))
    p=_csv(tmp_path/'wrong.csv',[bad])
    with pytest.raises(ValueError,match='differs from frozen'):
        read_updates([p],freeze=f,reference_ids=ids,reference_numbers=nums)

@pytest.mark.parametrize('bad',[
    _new(day='',number=115000104),
    _new(day='2026-11-08',number=115000104),
    _new(day='2025-11-09',number=115000104),
    _new(day='2026-13-09',number=115000104),
    _new(day='2026-11-09',number=115000106),
])
def test_invalid_new_draw_rejected(frozen,tmp_path,bad):
    f,ids,nums,_=frozen
    p=_csv(tmp_path/'bad.csv',[bad])
    with pytest.raises(ValueError):
        read_updates([p],freeze=f,reference_ids=ids,reference_numbers=nums)


def test_duplicate_source_id_rejected(frozen,tmp_path):
    f,ids,nums,_=frozen
    p=_csv(tmp_path/'dup.csv',[_new(),_new()])
    with pytest.raises(ValueError,match='duplicate draw'):
        read_updates([p],freeze=f,reference_ids=ids,reference_numbers=nums)


def test_sorted_new_updates_and_chain_previous(frozen,tmp_path):
    f,ids,nums,tables=frozen
    data=[_new(day='2026-11-12',number=115000105),_new(day='2026-11-09',number=115000104)]
    p=_csv(tmp_path/'future.csv',data)
    events,sources=read_updates([p],freeze=f,reference_ids=ids,reference_numbers=nums)
    assert [z['draw_id'] for z in events]==[115000104,115000105]
    first,_=prospective_evaluate(f,tables,events[:1],sources)
    full,rows=prospective_evaluate(f,tables,events,sources,previous=first)
    assert full['eligible_new_draws']==2
    assert len(rows)==2


def test_previous_changed_events_rejected(frozen):
    f,_,_,tables=frozen
    data=[{'draw_id':115000104,'date':'2026-11-09','numbers':[1,8,16,22,24,30]}]
    prior,_=prospective_evaluate(f,tables,data,[])
    data2=[dict(data[0],numbers=[2,8,16,22,24,30])]
    with pytest.raises(ValueError,match='changed'):
        prospective_evaluate(f,tables,data2,[],previous=prior)


def test_forecast_likelihood_proper_probabilities(frozen):
    f,_,_,tables=frozen
    for lev in ('component','partition','edge'):
        codes=tables[lev]['codes'];p=tables[lev]['p']
        before=np.array([f['frozen_rule']['historical_counts'][lev][str(int(k))] for k in codes])
        q=(before+384*p)/(2497+384)
        assert q.sum()==pytest.approx(1,abs=1e-12)
        assert np.all(q>0)


def test_rare_e_martingale_expected_value_one(frozen):
    f,_,_,_=frozen
    p=f['tests']['rare_fair_probability'];q=f['tests']['rare_alternative_probability']
    assert p*(q/p)+(1-p)*((1-q)/(1-p))==pytest.approx(1)


def test_seal_is_stable_to_dict_key_order(frozen):
    f,_,_,_=frozen
    unsealed={k:v for k,v in f.items() if k!='self_checksum_sha256'}
    assert content_hash(unsealed)==f['self_checksum_sha256']
    assert content_hash(dict(reversed(list(unsealed.items()))))==f['self_checksum_sha256']


def test_previous_modified_score_rejected(frozen):
    f,_,_,tables=frozen
    data=[{'draw_id':115000104,'date':'2026-11-09','numbers':[1,8,16,22,24,30]}]
    prior,_=prospective_evaluate(f,tables,data,[])
    prior['events'][0]['edge_log_gain']+=0.1
    with pytest.raises(ValueError,match='field edge_log_gain'):
        prospective_evaluate(f,tables,data,[],previous=prior)


def test_previous_modified_aggregate_rejected(frozen):
    f,_,_,tables=frozen
    data=[{'draw_id':115000104,'date':'2026-11-09','numbers':[1,8,16,22,24,30]}]
    prior,_=prospective_evaluate(f,tables,data,[])
    prior['mixture_e']+=10
    with pytest.raises(ValueError,match='aggregate'):
        prospective_evaluate(f,tables,data,[],previous=prior)
