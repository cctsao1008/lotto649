"""Portable synthetic contract tests for v0.20, with no real future outcomes."""
from __future__ import annotations

from datetime import datetime, timezone
import json

import numpy as np
import pytest

from lotto7x7.shape_pre_draw_v020 import make_commitment, audit_commitment
from lotto7x7.shape_prospective_v019 import (
    make_freeze, check_freeze, prospective_evaluate, seal_object,
)
from _prospective_fixture_v019 import make_test_inputs


@pytest.fixture(scope='module')
def data(tmp_path_factory):
    x=make_test_inputs(tmp_path_factory.mktemp('v020'))
    cat, uni, watch = x['catalog'], x['universe'], x['watchlist']
    freeze=make_freeze(cat,uni,watch,frozen_at=datetime(2026,11,8,9,23,tzinfo=timezone.utc))
    ids,nums,tables,_=check_freeze(freeze,cat,uni)
    zero,_=prospective_evaluate(freeze,tables,[],[{'file':'annual.csv','sha256':'dummy'}])
    return freeze,tables,zero


def make(data, *, issue=115000104, day='2026-11-10', instant=None):
    f,t,zero=data
    return make_commitment(f,t,zero,issue_id=issue,issue_date=day,
       now=instant or datetime(2026,11,8,9,25,tzinfo=timezone.utc))


def event(issue=115000104,day='2026-11-10',numbers=None):
    return {'draw_id':issue,'date':day,'numbers':numbers or [5,9,26,32,41,48]}


def test_zero_evidence_is_not_prediction_of_numbers(data):
    x=make(data)
    assert x['schema'].endswith('v0.20')
    assert x['target']['draw_id']==115000104
    assert x['previous_events_count']==0
    assert set(x['distributions'])=={'component','partition','edge'}
    assert {'numbers','six_numbers','winning_numbers'}.isdisjoint(x)
    assert x['rare_class']['name']=='P6_O1_D5'


def test_complete_fair_and_causal_probabilities(data):
    x=make(data)
    for lev,expected in [('component',6),('partition',11),('edge',94)]:
        a=x['distributions'][lev]
        assert len(a)==expected
        assert sum(o['p_fair'] for o in a)==pytest.approx(1.,abs=1e-13)
        assert sum(o['q_before_draw'] for o in a)==pytest.approx(1.,abs=1e-13)
        assert all(o['q_before_draw']>0 for o in a)


def test_before_taipei_draw_cutoff_enforced(data):
    with pytest.raises(ValueError,match='too late'):
        make(data,instant=datetime(2026,11,10,12,30,tzinfo=timezone.utc))
    make(data,instant=datetime(2026,11,10,12,29,tzinfo=timezone.utc))


def test_bad_target_calendar_and_issue_rejected(data):
    for issue,day in [(115000104,'2026-11-11'),(115000105,'2026-11-10'),(115000104,'2026-11-06'),(116000001,'2026-11-10')]:
        with pytest.raises(ValueError):make(data,issue=issue,day=day)


def test_missing_timezone_rejected(data):
    with pytest.raises(ValueError,match='timezone-aware'):
        make(data,instant=datetime(2026,11,8,17,0))


def test_pre_draw_report_strictly_rechecked(data):
    f,t,z=data
    tampered=dict(z);tampered['eligible_new_draws']=1
    with pytest.raises(ValueError,match='count'):
        make_commitment(f,t,tampered,issue_id=115000104,issue_date='2026-11-10',now=datetime(2026,11,8,9,30,tzinfo=timezone.utc))
    tampered=dict(z);tampered['freeze_sha256']='wrong'
    with pytest.raises(ValueError,match='different freeze'):
        make_commitment(f,t,tampered,issue_id=115000104,issue_date='2026-11-10',now=datetime(2026,11,8,9,30,tzinfo=timezone.utc))


def test_post_draw_reconciliation_pass(data):
    f,t,_=data
    seal=make(data)
    after,rows=prospective_evaluate(f,t,[event()],[])
    result=audit_commitment(seal,f,t,after)
    assert result['decision']=='AUDIT_MATCH'
    assert result['verified_class_forecast_arithmetic']
    assert not result['source_authenticity_verified']
    assert len(rows)==1


def test_pre_draw_reconciliation_awaits_event(data):
    f,t,z=data
    seal=make(data)
    result=audit_commitment(seal,f,t,z)
    assert result['decision']=='AWAITING_DRAW'


def test_post_draw_reconciliation_rejects_modified_event(data):
    f,t,_=data
    seal=make(data)
    after,_=prospective_evaluate(f,t,[event()],[])
    altered=json.loads(json.dumps(after));altered['events'][0]['component_q_before_draw']*=1.01
    with pytest.raises(ValueError,match='previously scored'):
        audit_commitment(seal,f,t,altered)


def test_commitment_modified_not_authenticated(data):
    f,t,_=data
    seal=make(data)
    after,_=prospective_evaluate(f,t,[event()],[])
    modified=json.loads(json.dumps(seal));modified['target']['draw_id']=115000105
    with pytest.raises(ValueError,match='checksum'):
        audit_commitment(modified,f,t,after)


def test_malicious_resealed_probability_rejected(data):
    f,t,_=data
    seal=make(data)
    after,_=prospective_evaluate(f,t,[event()],[])
    bad=dict(seal);bad['distributions']=json.loads(json.dumps(seal['distributions']))
    bad['distributions']['component'][0]['q_before_draw']+=0.01
    bad=seal_object({k:v for k,v in bad.items() if k!='self_checksum_sha256'})
    with pytest.raises(ValueError,match='not reproducible'):
        audit_commitment(bad,f,t,after)


def test_evidence_after_one_prior_event_uses_updated_counts(data):
    f,t,_=data
    prior,_=prospective_evaluate(f,t,[event()],[])
    a=make_commitment(f,t,prior,issue_id=115000105,issue_date='2026-11-13',now=datetime(2026,11,11,tzinfo=timezone.utc))
    assert a['previous_events_count']==1
    for lev in ('component','partition','edge'):
        assert any(x['q_before_draw']!=y['q_before_draw'] for x,y in zip(a['distributions'][lev],make(data)['distributions'][lev]))
    after,_=prospective_evaluate(f,t,[event(),event(115000105,'2026-11-13',[1,2,3,4,5,6])],[])
    assert audit_commitment(a,f,t,after)['decision']=='AUDIT_MATCH'


def test_reject_wrong_next_draw_with_valid_scored_data(data):
    f,t,_=data
    seal=make(data)
    after,_=prospective_evaluate(f,t,[event(115000105,'2026-11-13')],[])
    with pytest.raises(ValueError,match='next prospective draw'):
        audit_commitment(seal,f,t,after)


def test_previous_prefix_must_not_change(data):
    f,t,_=data
    prior,_=prospective_evaluate(f,t,[event()],[])
    seal=make_commitment(f,t,prior,issue_id=115000105,issue_date='2026-11-13',now=datetime(2026,11,11,tzinfo=timezone.utc))
    updated,_=prospective_evaluate(f,t,[event(115000104,'2026-11-10',[1,2,3,4,5,6]),event(115000105,'2026-11-13')],[])
    with pytest.raises(ValueError,match='earlier prospective events'):
        audit_commitment(seal,f,t,updated)
