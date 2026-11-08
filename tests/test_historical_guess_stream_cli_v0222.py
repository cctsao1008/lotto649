"""v0.22.2 CLI contract: default is unlimited and OS-random seeded."""
from __future__ import annotations

import argparse
from pathlib import Path

from lotto7x7.historical_guess_stream_cli_v0222 import configure_parsers, run


def _parser():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='command',required=True)
    configure_parsers(sub)
    return p


def test_defaults_are_unlimited_no_pacing_and_random_seed():
    args=_parser().parse_args(['historical-hit-stream'])
    assert args.strategy=='shape_score'
    assert args.batch_size==10000
    assert args.max_batches==0
    assert args.seed is None
    assert args.checkpoint_seconds==10.0
    assert args.target=='all'
    assert args.show_all_hits is False


def test_explicit_flags_are_forwarded(monkeypatch,tmp_path):
    import lotto7x7.historical_guess_stream_cli_v0222 as cli
    args=_parser().parse_args(['historical-hit-stream','--strategy','uniform',
         '--target','holdout','--holdout-count','50','--batch-size','65536',
         '--seed','123456','--max-batches','2','--checkpoint-seconds','2',
         '--show-all-hits','--out',str(tmp_path/'test.json')])
    calls=[]
    monkeypatch.setattr(cli,'run_stream',lambda **kwargs:calls.append(kwargs))
    cli.run(args)
    assert calls[0]['seed']==123456
    assert calls[0]['target']=='holdout'
    assert calls[0]['holdout_count']==50
    assert calls[0]['batch_size']==65536
    assert calls[0]['show_all_hits'] is True
    assert calls[0]['max_batches']==2
    assert calls[0]['out']==tmp_path/'test.json'


def test_cli_does_not_offer_minutes_or_guess_cap():
    args=_parser().parse_args(['historical-hit-stream','--batch-size','5000'])
    assert not hasattr(args,'minutes')
    assert not hasattr(args,'max_guesses')
