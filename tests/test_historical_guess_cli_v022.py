"""v0.22.1 CLI uses a total cap; old argument still runs safely."""
from __future__ import annotations
import argparse
from pathlib import Path
import pytest
from lotto7x7.historical_guess_cli_v022 import configure_parsers,run


def parser():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='command',required=True)
    configure_parsers(sub)
    return p


def test_defaults():
    a=parser().parse_args(['historical-hit-benchmark'])
    assert a.minutes==20 and a.max_guesses==10000
    assert a.target=='all' and a.strategy=='shape_score' and a.seed==649


def test_legacy_per_round_flag_means_total_now():
    a=parser().parse_args(['historical-hit-benchmark','--per-round','5000',
        '--progress-every','100'])
    assert a.max_guesses==5000 and a.progress_every==100


def test_explicit_new_flag():
    a=parser().parse_args(['historical-hit-benchmark','--max-guesses','900',
        '--strategy','uniform','--out','results/x.json'])
    assert a.max_guesses==900 and a.strategy=='uniform' and a.out==Path('results/x.json')


def test_invalid_mode():
    with pytest.raises(SystemExit):parser().parse_args(['historical-hit-benchmark','--target','future'])


def test_dispatch(monkeypatch,tmp_path):
    from lotto7x7 import historical_guess_cli_v022 as m
    kwargs=[]
    monkeypatch.setattr(m,'run_paced_experiment',lambda **kw:kwargs.append(kw) or {'ok':True})
    a=parser().parse_args(['historical-hit-benchmark','--minutes','.05',
        '--max-guesses','25','--out',str(tmp_path/'watch.json')])
    assert run(a)=={'ok':True}
    assert kwargs[0]['minutes']==.05 and kwargs[0]['max_guesses']==25
    assert kwargs[0]['strategy']=='shape_score'
