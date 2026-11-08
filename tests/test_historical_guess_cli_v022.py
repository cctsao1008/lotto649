"""CLI integration for timed historical exact-hit retrieval."""
from __future__ import annotations

import argparse
from pathlib import Path
import pytest

from lotto7x7.historical_guess_cli_v022 import configure_parsers,run


def _parser():
    main=argparse.ArgumentParser()
    sub=main.add_subparsers(dest='command',required=True)
    configure_parsers(sub)
    return main


def test_cli_defaults():
    args=_parser().parse_args(['historical-hit-benchmark'])
    assert args.minutes==20
    assert args.per_round==10000
    assert args.target=='all'
    assert args.seed==649


def test_cli_holdout_custom_output():
    args=_parser().parse_args(['historical-hit-benchmark','--target','holdout',
        '--minutes','1','--holdout-count','500','--out','results/retry.json'])
    assert args.target=='holdout' and args.holdout_count==500
    assert args.out==Path('results/retry.json')


def test_cli_rejects_unknown_mode():
    with pytest.raises(SystemExit):
        _parser().parse_args(['historical-hit-benchmark','--target','predictive'])


def test_run_dispatches(monkeypatch,tmp_path):
    from lotto7x7 import historical_guess_cli_v022 as mod
    got=[]
    monkeypatch.setattr(mod,'run_experiment',lambda **kwargs:got.append(kwargs) or {'ok':True})
    args=_parser().parse_args(['historical-hit-benchmark','--minutes','0.5',
                               '--max-rounds','2','--out',str(tmp_path/'out.json')])
    result=run(args)
    assert result['ok']
    assert got[0]['minutes']==0.5 and got[0]['max_rounds']==2
