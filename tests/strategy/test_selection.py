"""Regression tests for the public single-strategy selection pipeline."""
from dataclasses import replace
from datetime import date
import json
import os
import subprocess
import sys

import pytest

from lotto649 import SelectionConfig, load_history, select
from lotto649.pipeline import ROOT, _rank
from lotto649.hexagram import cast


def test_band_sorting_cannot_move_candidate_across_statistical_boundary():
    pool = [
        (1, 2, 4, 6, 8, 10),
        (2, 4, 6, 8, 10, 12),
        (4, 6, 8, 10, 12, 14),
        (1, 3, 5, 7, 9, 11),
    ]
    scores = (0.910, 0.906, 0.895, 0.905)
    parts = [(value,) * 3 for value in scores]
    rows = _rank(pool, parts, scores, code=0, epsilon=0.010, count=4)
    # The candidate at 0.895 matches all six lines but stays in band 2.
    assert [row["numbers"] for row in rows] == [pool[1], pool[0], pool[3], pool[2]]
    assert [row["statistical_band"] for row in rows] == [1, 1, 1, 2]


def test_cwa_uses_taipei_noon_and_leap_month():
    calendar_dir = ROOT / "data/astronomy"
    leap = cast(date(2025, 7, 25), calendar_dir)
    assert (leap["lunar_month"], leap["lunar_day"], leap["leap_month"]) == (6, 1, True)
    # A late evening solar-term transition is not active at the noon reference.
    assert cast(date(2023, 1, 5), calendar_dir)["solar_term"] == "winter_solstice"
    assert cast(date(2023, 1, 6), calendar_dir)["solar_term"] == "minor_cold"
    with pytest.raises(ValueError, match="no lunar date"):
        cast(date(2027, 1, 1), calendar_dir)


def test_selection_depends_only_on_prefix_and_config():
    history = load_history()
    config = SelectionConfig.load(ROOT / "config/strategy.json")
    target = date(2026, 11, 10)
    prefix = [row for row in history if row[0] < target]
    report = select(history, target, config)
    assert report == select(prefix, target, config)
    assert report["history"]["last_date"] == prefix[-1][0].isoformat()
    rows = report["candidates"]
    assert len({row["numbers"] for row in rows}) == config.count
    assert all(-1 <= row["statistical_score"] <= 1 for row in rows)
    assert all(set(row["statistical_components"]) == {"sum", "frequency", "shape"} for row in rows)
    with pytest.raises(ValueError, match="strictly increasing"):
        select([prefix[1], prefix[0]], target, config)
    with pytest.raises(ValueError, match="epsilon"):
        replace(config, epsilon=float("nan"))


def test_primary_cli_writes_exclusive_json(tmp_path):
    out = tmp_path / "nested" / "selection.json"
    command = [
        sys.executable, "-m", "lotto649", "select", "--date", "2026-11-17",
        "--out", str(out),
    ]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
    original = out.read_bytes()
    assert len(json.loads(original)["candidates"]) == 4
    again = subprocess.run(command, cwd=ROOT, capture_output=True)
    assert again.returncode == 2
    assert out.read_bytes() == original


def test_public_and_compatibility_entry_points_match_from_another_directory(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    args = ["select", "--date", "2026-11-17"]
    current = subprocess.check_output(
        [sys.executable, "-m", "lotto649", *args], cwd=tmp_path, env=env,
    )
    previous = subprocess.check_output(
        [sys.executable, "-m", "lotto7x7", *args], cwd=tmp_path, env=env,
    )
    assert current == previous
    assert json.loads(current)["history"]["last_date"] < "2026-11-17"
