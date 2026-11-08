"""P3 CLI bootstraps a sealed snapshot and does not fabricate future evidence."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest

from lotto7x7.shape_prospective_cli_v019 import configure_parsers,run

from _prospective_fixture_v019 import make_test_inputs


def _args(argv):
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='command',required=True)
    configure_parsers(sub)
    return p.parse_args(argv)


def test_v019_command_registration():
    f=_args(['shape-prospective-freeze'])
    u=_args(['shape-prospective-update'])
    assert f.command=='shape-prospective-freeze'
    assert u.command=='shape-prospective-update'


def test_full_cli_freeze_and_zero_new_sample(tmp_path):
    inputs=make_test_inputs(tmp_path)
    base=str(inputs['catalog'])
    theory=str(inputs['universe'])
    watch=str(inputs['watchlist'])
    freeze=tmp_path/'freeze.json'
    args=_args(['shape-prospective-freeze','--catalog',base,'--universe',theory,'--watchlist',watch,'--out',str(freeze)])
    run(args)
    assert freeze.exists()
    report=tmp_path/'zero.json'
    args=_args(['shape-prospective-update','--freeze',str(freeze),'--catalog',base,'--universe',theory,
                '--csv',str(inputs['historical_csv']),'--out',str(report)])
    run(args)
    obj=json.loads(report.read_text())
    assert obj['status']=='NO_PROSPECTIVE_DRAWS'
    assert obj['mixture_e']==1 and obj['eligible_new_draws']==0
    assert report.with_suffix('.html').exists()
    with (tmp_path/'zero_events.csv').open() as fh:
        assert len(list(csv.reader(fh)))==1
    with pytest.raises(FileExistsError):run(args)


def test_update_empty_csv_option(tmp_path):
    inputs=make_test_inputs(tmp_path)
    base=str(inputs['catalog'])
    theory=str(inputs['universe'])
    watch=str(inputs['watchlist'])
    freeze=tmp_path/'freeze.json'
    run(_args(['shape-prospective-freeze','--catalog',base,'--universe',theory,'--watchlist',watch,'--out',str(freeze)]))
    output=tmp_path/'empty.json'
    run(_args(['shape-prospective-update','--freeze',str(freeze),'--catalog',base,'--universe',theory,'--out',str(output)]))
    data=json.loads(output.read_text())
    assert data['status']=='NO_PROSPECTIVE_DRAWS' and data['source_csv_files']==[]
