"""v0.20 CLI smoke tests with synthetic-only prospective records."""
from __future__ import annotations

import argparse
import json
from datetime import datetime,timezone

import pytest

from lotto7x7.shape_pre_draw_cli_v020 import configure_parsers,run
from lotto7x7.shape_prospective_v019 import check_freeze,make_freeze,prospective_evaluate
from _prospective_fixture_v019 import make_test_inputs


def parse(argv):
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='command',required=True)
    configure_parsers(sub)
    return p.parse_args(argv)


def test_cli_subcommands_registered():
    assert parse(['shape-prospective-seal','--report','x','--issue-id','115000104',
                  '--issue-date','2026-11-10','--out','y.json']).command=='shape-prospective-seal'
    assert parse(['shape-prospective-audit','--seal','s.json','--report','r.json',
                  '--out','audit.json']).command=='shape-prospective-audit'


def test_cli_realistic_initial_and_audit(tmp_path):
    a=make_test_inputs(tmp_path)
    f=make_freeze(a['catalog'],a['universe'],a['watchlist'],
                  frozen_at=datetime(2026,11,8,9,23,tzinfo=timezone.utc))
    freeze=tmp_path/'freeze.json';freeze.write_text(json.dumps(f))
    _,_,tables,_=check_freeze(f,a['catalog'],a['universe'])
    zero,_=prospective_evaluate(f,tables,[],[])
    zero_path=tmp_path/'zero.json';zero_path.write_text(json.dumps(zero))
    common=['--freeze',str(freeze),'--catalog',str(a['catalog']),
            '--universe',str(a['universe'])]
    seal=tmp_path/'seal.json'
    cmd=parse(['shape-prospective-seal',*common,'--report',str(zero_path),
               '--issue-id','115000104','--issue-date','2026-11-10','--out',str(seal)])
    run(cmd)
    assert seal.exists() and seal.with_suffix('.html').exists()
    assert json.loads(seal.read_text())['previous_events_count']==0
    with pytest.raises(FileExistsError):run(cmd)
    audit=tmp_path/'audit.json'
    run(parse(['shape-prospective-audit',*common,'--seal',str(seal),
               '--report',str(zero_path),'--out',str(audit)]))
    assert json.loads(audit.read_text())['decision']=='AWAITING_DRAW'


def test_output_path_conflicts_and_extensions(tmp_path):
    from lotto7x7.shape_pre_draw_cli_v020 import _reserve
    with pytest.raises(ValueError,match='json'):_reserve(tmp_path/'x.txt',[])
    path=tmp_path/'a.json';path.write_text('existing')
    with pytest.raises(FileExistsError):_reserve(path,[])
    with pytest.raises(ValueError,match='conflicts'):_reserve(tmp_path/'new.json',[tmp_path/'new.json'])
