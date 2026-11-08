"""v0.22.1 quiet, paced historic guessing regression and correctness tests."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from lotto7x7.historical_guess_v022 import make_pool
from lotto7x7.historical_guess_watch_v0221 import run_paced_experiment


class FakeTime:
    def __init__(self, interrupt_after=None):
        self.value = 100.0
        self.interrupt_after = interrupt_after
        self.calls = 0

    def clock(self):
        return self.value

    def sleep(self, seconds):
        assert seconds >= 0
        self.calls += 1
        if self.interrupt_after is not None and self.calls > self.interrupt_after:
            raise KeyboardInterrupt()
        self.value += seconds


def setup(monkeypatch, *, hits=True):
    import lotto7x7.historical_guess_watch_v0221 as m
    from lotto7x7.shape_predictive_v017 import _baseline_tables
    fixture = json.loads((Path(__file__).parent/'fixtures'/'p1_exact_universe_v1.json').read_text())
    table = _baseline_tables(fixture)
    pool = make_pool(np.random.default_rng(17), 24)
    ids = np.arange(93000001,93000025,dtype=np.int64)
    draws = pool.copy() if hits else make_pool(np.random.default_rng(3), 24)
    monkeypatch.setattr(m, 'read_sources', lambda _a,_b:(ids,draws,table,{}))
    monkeypatch.setattr(m, 'file_sha256', lambda _: 'f'*64)
    monkeypatch.setattr(m, 'make_pool', lambda rng, count: pool[:count].copy())
    return m,pool,draws


def call(monkeypatch,tmp_path,*,count=8,minutes=.02,strategy='shape_score',hits=True,interrupt=None):
    m,_,_=setup(monkeypatch,hits=hits)
    f=FakeTime(interrupt_after=interrupt)
    output=[]
    report=m.run_paced_experiment(catalog=tmp_path/'c.npz',universe=tmp_path/'u.json',
        out=tmp_path/'watch.json',minutes=minutes,max_guesses=count,seed=3,
        strategy=strategy, clock=f.clock,sleep=f.sleep,emit=output.append)
    return report,output,f


def test_max_guesses_applies_to_entire_experiment_not_each_round(monkeypatch,tmp_path):
    report,output,t=call(monkeypatch,tmp_path,count=8,minutes=.02)
    assert report['stats']['guesses_completed']==8
    assert report['stats']['hit_events']==8
    assert report['stats']['status']=='COMPLETED'
    assert report['stats']['elapsed_seconds']==pytest.approx(1.2)
    assert t.value==pytest.approx(101.2)
    assert sum(s.startswith('[HIT') for s in output)==8
    assert not any(s.startswith('[PROGRESS]') for s in output)
    assert output[-1].startswith('[DONE]')


def test_hit_messages_include_event_count_and_cumulative_guess(monkeypatch,tmp_path):
    report,output,_=call(monkeypatch,tmp_path,count=4,strategy='uniform')
    for index,msg in enumerate(output[:-1],1):
        assert msg.startswith(f'[HIT #{index}] guess #{index}/4 ')
    with (tmp_path/'watch_hits.csv').open(encoding='utf-8',newline='') as f:
        records=list(csv.DictReader(f))
    assert len(records)==4
    assert [int(x['hit_number']) for x in records]==[1,2,3,4]
    assert [int(x['guess_number']) for x in records]==[1,2,3,4]
    assert records[0]['historical_draw_ids']=='93000001'
    assert report['stats']['first_hit_guess']==1


def test_no_hit_is_silent_except_final_summary(monkeypatch,tmp_path):
    report,output,_=call(monkeypatch,tmp_path,count=8,hits=False)
    assert report['stats']['hit_events']==0
    assert len(output)==1 and output[0].startswith('[DONE]')
    assert report['stats']['first_hit_guess'] is None
    assert len(list(csv.DictReader((tmp_path/'watch_hits.csv').open())))==0


def test_ctrl_c_preserves_partial_output(monkeypatch,tmp_path):
    report,output,_=call(monkeypatch,tmp_path,count=8,interrupt=2)
    assert report['stats']['status']=='INTERRUPTED'
    assert 1<=report['stats']['guesses_completed']<8
    assert json.loads((tmp_path/'watch.json').read_text())['stats']['status']=='INTERRUPTED'
    assert (tmp_path/'watch_rounds.csv').is_file()
    assert output[-1].startswith('[DONE] INTERRUPTED')


def test_duplicate_output_is_rejected(monkeypatch,tmp_path):
    m,_,_=setup(monkeypatch)
    (tmp_path/'watch.json').write_text('{}')
    with pytest.raises(FileExistsError):
        m.run_paced_experiment(catalog=tmp_path/'c.npz',universe=tmp_path/'u.json',
            out=tmp_path/'watch.json',minutes=.01,max_guesses=3)


@pytest.mark.parametrize('bad', [0,10001])
def test_guess_cap_validation(monkeypatch,tmp_path,bad):
    m,_,_=setup(monkeypatch)
    with pytest.raises(ValueError):
        m.run_paced_experiment(catalog=tmp_path/'c.npz',universe=tmp_path/'u.json',
            out=tmp_path/'watch.json',minutes=.01,max_guesses=bad)


def test_distribution_is_independent_of_strategy(monkeypatch,tmp_path):
    # Both strategies see the identical distinct tickets and find the same
    # exact past six-number sets. Only order/timing can differ.
    a,_,_=setup(monkeypatch)
    f=FakeTime()
    x=a.run_paced_experiment(catalog=tmp_path/'c.npz',universe=tmp_path/'u.json',
        out=tmp_path/'shape.json',minutes=.01,max_guesses=9,strategy='shape_score',
        clock=f.clock,sleep=f.sleep,emit=lambda _:None)
    f2=FakeTime()
    y=a.run_paced_experiment(catalog=tmp_path/'c.npz',universe=tmp_path/'u.json',
        out=tmp_path/'uniform.json',minutes=.01,max_guesses=9,strategy='uniform',
        clock=f2.clock,sleep=f2.sleep,emit=lambda _:None)
    assert x['stats']['hit_events']==y['stats']['hit_events']==9
    with (tmp_path/'shape_hits.csv').open(newline='',encoding='utf-8') as sf, \
         (tmp_path/'uniform_hits.csv').open(newline='',encoding='utf-8') as uf:
        assert {r['numbers'] for r in csv.DictReader(sf)} == {r['numbers'] for r in csv.DictReader(uf)}
