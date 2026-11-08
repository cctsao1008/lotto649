"""v0.21 portable synthetic end-to-end integration; no user workspace paths."""
from __future__ import annotations
import argparse
import json
from datetime import datetime,timezone
from math import comb
import numpy as np
import pytest

from lotto7x7.shape_prospective_v019 import make_freeze
from lotto7x7.unified_selection_v021 import build_selection
from lotto7x7.unified_selection_cli_v021 import configure_parsers,run,output_paths
from _prospective_fixture_v019 import make_test_inputs


@pytest.fixture(scope='module')
def inputs(tmp_path_factory):
    folder=tmp_path_factory.mktemp('lotto021')
    data=make_test_inputs(folder)
    freeze=make_freeze(data['catalog'],data['universe'],data['watchlist'],
                       frozen_at=datetime(2026,11,8,9,20,tzinfo=timezone.utc))
    dest=folder/'freeze.json'
    dest.write_text(json.dumps(freeze,ensure_ascii=False),encoding='utf-8')
    data['freeze']=dest
    return data


def test_synthetic_frozen_end_to_end(inputs):
    r,pool,bets=build_selection(inputs['catalog'],inputs['universe'],inputs['freeze'],
                                 pool_size=22,tickets=4,objective='3plus',seed=123)
    assert len(pool)==22 and len(bets)==12
    assert r['data']['history_count']==2497
    assert r['data']['fair_universe']==comb(49,6)
    assert len(r['strategies'])==3
    for values in r['strategies'].values():
        assert len(values['tickets'])==4
        assert values['exact_fair_portfolio_coverage']['at_least']['6']==pytest.approx(4/comb(49,6))
        assert len({tuple(x) for x in values['tickets']})==4


def test_synthetic_cli_generates_four_outputs_and_protects_rewrite(inputs,tmp_path):
    cmd=['shape-select-unified','--catalog',str(inputs['catalog']),
         '--universe',str(inputs['universe']),'--freeze',str(inputs['freeze']),
         '--pool-size','15','--tickets','3','--objective','4plus','--seed','123',
         '--out',str(tmp_path/'report.json')]
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    configure_parsers(sub)
    args=parser.parse_args(cmd)
    run(args)
    paths=output_paths(args.out)
    assert all(x.exists() for x in paths.values())
    with paths['JSON'].open() as f:report=json.load(f)
    assert report['settings']['objective']=='4plus'
    assert report['strategies']['coverage']['exact_fair_portfolio_coverage']['at_least_counts']['6']==3
    with pytest.raises(FileExistsError):run(args)


def test_edited_freeze_refused(inputs,tmp_path):
    snap=json.loads(inputs['freeze'].read_text())
    snap['source']['last_draw_id']=115000105
    altered=tmp_path/'tamper.json';altered.write_text(json.dumps(snap),encoding='utf-8')
    with pytest.raises(ValueError,match='checksum'):
        build_selection(inputs['catalog'],inputs['universe'],altered,pool_size=10,tickets=1)
