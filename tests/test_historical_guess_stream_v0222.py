"""v0.22.2 fast continuous historic guess engine tests."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import lotto7x7.historical_guess_stream_v0222 as m
from lotto7x7.historical_guess_v022 import encode_tickets


def _sources(monkeypatch, historical=None):
    """Synthetic exact target with a real 6/49 draw and known fixed class table."""
    known = np.asarray(historical if historical is not None else [[1, 2, 3, 4, 5, 6]], dtype=np.uint8)
    base = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.uint8)
    monkeypatch.setattr(m, 'read_sources', lambda *_: (np.arange(500, 500 + len(known)), known, {}, {}))
    monkeypatch.setattr(m, 'file_sha256', lambda _: '1' * 64)
    # One-category shape model; allows shape_score and uniform to match identically.
    from lotto7x7.shape_predictive_v017 import _baseline_tables
    fixture = json.loads((Path(__file__).parent/'fixtures'/'p1_exact_universe_v1.json').read_text())
    edge = _baseline_tables(fixture)['edge']
    ref = {
        'lookup': {int(key): [500 + i] for i, key in enumerate(encode_tickets(known))},
        'target_count': len(known), 'training_draws': len(known),
        'all_history_leakage': True,
        'class_codes': edge['codes'],
        'class_ratio': np.ones(len(edge['codes']), dtype=np.float64),
    }
    monkeypatch.setattr(m, 'build_reference', lambda *args, **kwargs: ref)
    return base


def _run(monkeypatch, tmp_path, *, batch_size=4, max_batches=2, strategy='uniform',
         show_all_hits=False, checkpoint_seconds=10., seed=1, emit=None):
    messages = [] if emit is None else emit
    result = m.run_stream(catalog=tmp_path/'catalog.npz', universe=tmp_path/'univ.json',
                          out=tmp_path/'stream.json',batch_size=batch_size,
                          strategy=strategy,seed=seed,max_batches=max_batches,
                          checkpoint_seconds=checkpoint_seconds,
                          show_all_hits=show_all_hits,emit=messages.append)
    return result, messages


def test_random_batches_contain_legal_sorted_sixes():
    tickets = m.make_fast_batch(np.random.default_rng(649), 8192)
    assert tickets.shape == (8192, 6)
    assert tickets.dtype == np.uint8
    assert np.all((tickets >= 1) & (tickets <= 49))
    assert np.all(np.diff(tickets.astype(np.int16), axis=1) > 0)
    assert np.array_equal(tickets, m.make_fast_batch(np.random.default_rng(649), 8192))


@pytest.mark.parametrize('bad', [0, 65537])
def test_batch_size_validation(bad):
    with pytest.raises(ValueError):m.make_fast_batch(np.random.default_rng(0), bad)


def test_hist_match_vector_is_exact_and_ordered():
    pool = np.asarray([[1,2,3,4,5,6],[1,2,3,4,5,7],[2,3,4,5,6,7],[1,2,3,4,5,6]],dtype=np.uint8)
    target=np.sort(encode_tickets(pool[[0,2]]))
    assert m.historical_matches(pool,target).tolist()==[0,2,3]


def test_uniform_reproducibility_and_score_frozen(monkeypatch):
    pool=m.make_fast_batch(np.random.default_rng(649),100)
    assert np.array_equal(m.ordered_batch(pool,'uniform',{}),pool)
    with pytest.raises(ValueError):m.ordered_batch(pool,'bad',{})


def test_ratio_calculation_and_zero():
    assert m._ratio_fields(0,0)['guesses_per_hit'] is None
    assert m._ratio_fields(1,10000)['hit_rate_percent']==pytest.approx(.01)
    assert m._ratio_fields(2,10000)['guesses_per_hit']==5000
    assert m._ratio_fields(3,10000)['hit_rate_fraction']=='3/10000'


def test_repeated_hits_in_total_but_one_unique_csv(monkeypatch,tmp_path):
    x=_sources(monkeypatch)
    pool=np.tile(x,(4,1))
    monkeypatch.setattr(m,'make_fast_batch',lambda rng,n:pool.copy())
    report,msgs=_run(monkeypatch,tmp_path,batch_size=4,max_batches=2)
    assert report['stats']['guesses_completed']==8
    assert report['stats']['hit_events']==8
    assert report['stats']['unique_historical_six_sets_hit']==1
    assert report['stats']['unique_historical_draw_ids_hit']==1
    assert report['stats']['hits_per_guess']==1.0
    assert len([x for x in msgs if x.startswith('[NEW HIT')])==1
    with (tmp_path/'stream_hits.csv').open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==1
    assert rows[0]['guess_number']=='1'
    assert '8/8' in msgs[-2]
    assert json.loads((tmp_path/'stream.json').read_text())['stats']['hit_rate_fraction']=='8/8'


def test_no_hit_prints_only_two_final_lines(monkeypatch,tmp_path):
    x=_sources(monkeypatch)
    missed=np.asarray([2,3,4,5,6,7],dtype=np.uint8)
    monkeypatch.setattr(m,'make_fast_batch',lambda rng,n:np.tile(missed,(n,1)))
    report,msgs=_run(monkeypatch,tmp_path)
    assert report['stats']['hit_events']==0
    assert report['stats']['hit_rate_fraction']=='0/8'
    assert report['stats']['guesses_per_hit'] is None
    assert len(msgs)==2
    assert msgs[0].startswith('[DONE] COMPLETED')


def test_ctrl_c_stops_unlimited_and_persists(monkeypatch,tmp_path):
    x=_sources(monkeypatch)
    calls=[]
    def next_pool(rng,n):
        calls.append(n)
        if len(calls)==3:raise KeyboardInterrupt()
        return np.tile(x,(n,1))
    monkeypatch.setattr(m,'make_fast_batch',next_pool)
    result,msgs=_run(monkeypatch,tmp_path,batch_size=3,max_batches=0)
    assert calls==[3,3,3]
    assert result['stats']['status']=='INTERRUPTED'
    assert result['stats']['stop_reason']=='ctrl_c'
    assert result['stats']['batches']==2
    assert result['stats']['hit_events']==6
    assert result['stats']['guesses_completed']==6
    assert result['stats']['hit_rate_fraction']=='6/6'
    assert json.loads((tmp_path/'stream.json').read_text())['stats']['status']=='INTERRUPTED'
    assert '[DONE] INTERRUPTED' in msgs[-2]
    assert len(list(csv.DictReader((tmp_path/'stream_rounds.csv').open(newline=''))))==2


def test_partial_batch_dropped_on_ctrl_c(monkeypatch,tmp_path):
    _sources(monkeypatch)
    def interrupted(rng,n):raise KeyboardInterrupt()
    monkeypatch.setattr(m,'make_fast_batch',interrupted)
    res,_=_run(monkeypatch,tmp_path,max_batches=0)
    assert res['stats']['guesses_completed']==0
    assert res['stats']['hit_events']==0
    assert res['stats']['status']=='INTERRUPTED'


def test_os_seed_generated_and_saved(monkeypatch,tmp_path):
    x=_sources(monkeypatch)
    monkeypatch.setattr(m,'secrets',type('Dummy',(),{'randbits':lambda bits:123456789}))
    monkeypatch.setattr(m,'make_fast_batch',lambda rng,n:np.tile(x,(n,1)))
    result,_=_run(monkeypatch,tmp_path,batch_size=2,max_batches=1,seed=None)
    assert result['settings']['seed']==123456789
    assert result['settings']['seed_source']=='OS_CSPRNG'
    assert result['settings']['batch_size']==2


def test_output_nonoverwrite(monkeypatch,tmp_path):
    x=_sources(monkeypatch)
    (tmp_path/'stream.json').write_text('DO NOT OVERWRITE')
    with pytest.raises(FileExistsError):
        _run(monkeypatch,tmp_path)
    assert (tmp_path/'stream.json').read_text()=='DO NOT OVERWRITE'


def test_default_unique_notifications_are_bounded(monkeypatch,tmp_path):
    history=m.make_fast_batch(np.random.default_rng(15),210)
    _sources(monkeypatch,history)
    monkeypatch.setattr(m,'make_fast_batch',lambda rng,n:history[:n].copy())
    res,msgs=_run(monkeypatch,tmp_path,batch_size=210,max_batches=1)
    newlines=[x for x in msgs if x.startswith('[NEW HIT')]
    assert len(newlines)==12  # first 10; #100; #200
    assert res['stats']['hit_events']==210
    assert res['stats']['unique_historical_six_sets_hit']==210
    assert res['stats']['guesses_completed']==210
    assert '[INFO]' in '\n'.join(msgs)
    assert len(list(csv.DictReader((tmp_path/'stream_hits.csv').open(newline=''))))==210


def test_explicit_full_unique_display(monkeypatch,tmp_path):
    history=m.make_fast_batch(np.random.default_rng(23),20)
    _sources(monkeypatch,history)
    monkeypatch.setattr(m,'make_fast_batch',lambda rng,n:history[:n].copy())
    _,msgs=_run(monkeypatch,tmp_path,batch_size=20,max_batches=1,show_all_hits=True)
    assert len([x for x in msgs if x.startswith('[NEW HIT')])==20
    assert not any(x.startswith('[INFO]') for x in msgs)


def test_shape_vs_uniform_complete_batch_same_hit_rate(monkeypatch,tmp_path):
    _sources(monkeypatch)
    p=m.make_fast_batch(np.random.default_rng(17),4000)
    monkeypatch.setattr(m,'make_fast_batch',lambda rng,n:p.copy())
    a,_=_run(monkeypatch,tmp_path,batch_size=4000,max_batches=1,strategy='shape_score')
    b=m.run_stream(catalog=tmp_path/'cat',universe=tmp_path/'univ',
                   out=tmp_path/'other.json',batch_size=4000,max_batches=1,
                   strategy='uniform',seed=3,emit=lambda x:None)
    assert a['stats']['hit_events']==b['stats']['hit_events']
    assert a['stats']['guesses_completed']==b['stats']['guesses_completed']
