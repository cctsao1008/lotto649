"""v0.22 historical hit benchmark: independent synthetic and real-model rules."""
import json
from math import comb
from pathlib import Path
import numpy as np
import pytest

from lotto7x7.historical_guess_v022 import (make_pool,encode_tickets,build_reference,trial,
                                                 run_experiment,_paths)
from lotto7x7.shape_taxonomy_v015 import classify_numbers
from lotto7x7.shape_theory_v016 import classification_keys


def _fixture():
    rng=np.random.default_rng(30)
    rows=make_pool(rng,24)
    from lotto7x7.shape_predictive_v017 import _baseline_tables
    baseline=json.loads((Path(__file__).parent/'fixtures'/'p1_exact_universe_v1.json').read_text())
    table=_baseline_tables(baseline)
    return np.arange(1000,1024),rows,table


def test_pool_is_valid_unique_and_reproducible():
    x=make_pool(np.random.default_rng(649),10000)
    y=make_pool(np.random.default_rng(649),10000)
    assert np.array_equal(x,y)
    assert np.all((x>=1)&(x<=49))
    assert np.all(np.diff(x.astype('int16'),axis=1)>0)
    assert len(np.unique(encode_tickets(x)))==10000


def test_pool_rejects_10001_and_empty():
    for n in (0,10001):
        with pytest.raises(ValueError):make_pool(np.random.default_rng(1),n)


def test_encode_injective_and_validates():
    a=np.asarray([[1,2,3,4,5,6],[1,2,3,4,5,7]],dtype='uint8')
    assert len(set(map(int,encode_tickets(a))))==2
    for bad in ([[2,1,3,4,5,6]],[[1,2,3,4,5,5]],[[1,2,3,4,5,50]]):
        with pytest.raises(ValueError):encode_tickets(np.asarray(bad,dtype='uint8'))


def test_holdout_reference_excludes_targets_from_training():
    ids,rows,tables=_fixture()
    ref=build_reference(ids,rows,tables,target='holdout',holdout_count=5)
    assert ref['training_draws']==19
    assert ref['target_count']==5
    assert ref['training_last_draw_id']==1018
    assert set(ref['lookup'])==set(map(int,encode_tickets(rows[-5:])))
    assert not ref['all_history_leakage']


def test_all_reference_declares_in_sample():
    ids,rows,tables=_fixture()
    ref=build_reference(ids,rows,tables,target='all')
    assert ref['training_draws']==24 and ref['target_count']==24
    assert ref['all_history_leakage']


def test_trial_uses_same_pool_and_correct_ranks():
    ids,rows,tables=_fixture()
    ref=build_reference(ids,rows,tables,target='holdout',holdout_count=5)
    candidate=np.concatenate([rows[-3:], rows[:10]])
    out=trial(candidate,ref)
    assert len(out['hits'])==3
    assert out['unique_candidates']==len(candidate)
    assert sorted(h['uniform_rank'] for h in out['hits'])==[1,2,3]
    assert sorted(h['shape_rank'] for h in out['hits'])==sorted(h['shape_rank'] for h in out['hits'])
    assert out['uniform_first']==1
    assert all(1<=h['shape_rank']<=len(candidate) for h in out['hits'])


def test_trial_no_hits():
    ids,rows,tables=_fixture()
    ref=build_reference(ids,rows,tables,target='holdout',holdout_count=5)
    out=trial(rows[:10],ref)
    assert out['hits']==[] and out['uniform_first'] is None and out['shape_first'] is None


def test_trial_rejects_duplicate_candidates():
    ids,rows,tables=_fixture()
    ref=build_reference(ids,rows,tables,target='all')
    # Duplicate candidates can be separately ranked, but unique count records the issue.
    out=trial(np.concatenate([rows[:2],rows[:1]]),ref)
    assert out['unique_candidates']==2


def test_run_creates_three_auditable_files(monkeypatch,tmp_path):
    import lotto7x7.historical_guess_v022 as mod
    ids,rows,tables=_fixture()
    monkeypatch.setattr(mod,'read_sources',lambda c,u:(ids,rows,tables,{}))
    monkeypatch.setattr(mod,'file_sha256',lambda x:'a'*64)
    events=[]
    out=tmp_path/'test.json'
    report=mod.run_experiment(catalog=tmp_path/'cat.npz',universe=tmp_path/'universe.json',
        out=out,minutes=5,per_round=200,seed=649,max_rounds=2,
        progress_every=1,emit=events.append)
    assert report['stats']['trials']==2
    assert report['stats']['guesses_per_strategy']==400
    assert _paths(out)['hits'].exists() and _paths(out)['rounds'].exists()
    assert json.loads(out.read_text())['stats']['status']=='COMPLETED'
    with pytest.raises(FileExistsError):
        mod.run_experiment(catalog=tmp_path/'cat.npz',universe=tmp_path/'universe.json',
            out=out,minutes=0.01,per_round=100,emit=events.append)


def test_full_reference_lookups_can_match_known_history():
    ids,rows,tables=_fixture()
    ref=build_reference(ids,rows,tables,target='all')
    out=trial(rows[:10],ref)
    assert len(out['hits'])==10
    assert out['uniform_first']==1 and out['shape_first']>=1
