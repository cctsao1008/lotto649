"""Portable CLI tests; avoid any local source files under /mnt/data."""
import argparse
import json
from pathlib import Path

import pytest

from lotto7x7.unified_selection_cli_v021 import configure_parsers, output_paths, _reserve
from lotto7x7.unified_selection_v021 import write_csv


def parse(line):
    parent=argparse.ArgumentParser()
    actions=parent.add_subparsers(dest='command',required=True)
    configure_parsers(actions)
    return parent.parse_args(line)


def test_cli_default_routes_and_settings():
    ns=parse(['shape-select-unified'])
    assert ns.pool_size==100 and ns.tickets==5
    assert ns.objective=='3plus' and ns.seed==649
    assert ns.catalog==Path('results/v015_v016_p1_shape_catalog.npz')


def test_cli_custom_objective():
    ns=parse(['shape-select-unified','--tickets','3','--objective','4plus','--seed','987'])
    assert (ns.tickets,ns.objective,ns.seed)==(3,'4plus',987)


def test_cli_invalid_objective():
    with pytest.raises(SystemExit):parse(['shape-select-unified','--objective','6plus'])


def test_no_overwrite(tmp_path):
    destinations=output_paths(tmp_path/'trial.json')
    _reserve(destinations,[])
    destinations['JSON'].write_text('{}')
    with pytest.raises(FileExistsError):_reserve(destinations,[])


def test_invalid_extension():
    with pytest.raises(ValueError):output_paths(Path('invalid.html'))


def test_input_output_collision(tmp_path):
    p=tmp_path/'source.json'
    with pytest.raises(FileExistsError):_reserve({'JSON':p},[p])


def test_csv_roundtrip(tmp_path):
    rows=[{'strategy':'uniform','numbers':'01 02 03 04 05 06'}]
    dest=tmp_path/'bets.csv'
    write_csv(dest,rows)
    assert 'uniform' in dest.read_text(encoding='utf-8')
    with pytest.raises(FileExistsError):write_csv(dest,rows)
