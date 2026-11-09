"""Shape-deficit v0.25 P2 tests; small synthetic fixed-universe fixture.

The synthetic category distribution is deliberately NOT the real exhaustive
shape universe. Production load_exact_baseline rejects incomplete catalogs.
"""
from __future__ import annotations

from collections import Counter
import json
from math import comb
from pathlib import Path
from random import Random

import numpy as np
import pytest

from lotto7x7.deficit_compensation_v025 import REFERENCE
from lotto7x7.shape_deficit_v025 import (
    _keys, level_summary, load_exact_baseline, select_shape_portfolio,
    shape_score,
)
from lotto7x7.shape_theory_v016 import LEVELS, RULE_VERSION, UNIVERSE


def history(n=48):
    rng = Random(77)
    return [(f"2026-{(i//28)+1:02d}-{(i%28)+1:02d}", i + 1,
             tuple(sorted(rng.sample(range(1, 50), 6)))) for i in range(n)]


def synthetic_baseline(records, seed=649, pool_size=40):
    # All candidate codes expected in a deterministic test's finite pool.
    rng = Random(seed)
    pool, seen = [], set()
    while len(pool) < pool_size:
        t = tuple(sorted(rng.sample(range(1, 50), 6)))
        if t not in seen:
            seen.add(t)
            pool.append(t)
    combined = [r[2] for r in records] + pool + list(REFERENCE)
    codes = _keys(combined)
    weights = {}
    for level in LEVELS:
        unique = sorted(set(map(int, codes[level])))
        # A fabricated complete-mass fixture for testing arithmetic only.
        q, remainder = divmod(UNIVERSE, len(unique))
        weights[level] = {
            str(k): q + int(i < remainder)
            for i, k in enumerate(unique)
        }
    return {"schema": RULE_VERSION, "total": UNIVERSE,
            "complete": True, "counts": weights}


def test_exact_shape_level_totals_and_stability():
    r = history()
    fixture = synthetic_baseline(r)
    levels = level_summary(r, REFERENCE, fixture)
    for level in LEVELS:
        obj = levels[level]
        assert sum(row["historical_observed"] for row in obj["rows"]) == len(r)
        assert sum(row["selected_count"] for row in obj["rows"]) == 4
        assert sum(row["theory_probability"] for row in obj["rows"]) == pytest.approx(1)
        assert sum(row["historical_deficit"] for row in obj["rows"]) == pytest.approx(0)
        assert obj["before"] >= 0
        assert obj["after"] >= 0
        assert obj["improvement"] == pytest.approx(obj["before"] - obj["after"])
    before = level_summary(r, (), fixture)
    assert all(before[level]["before"] == pytest.approx(before[level]["after"])
               for level in LEVELS)


def test_four_ticket_joint_selector_determinism():
    r = history()
    baseline = synthetic_baseline(r)
    kwargs = {"pool_size": 40, "seed": 649, "alpha": 4.0}
    a = select_shape_portfolio(r, baseline, **kwargs)
    b = select_shape_portfolio(r, baseline, **kwargs)
    assert a == b
    assert len(a["tickets"]) == 4
    assert len({tuple(t) for t in a["tickets"]}) == 4
    score = shape_score(level_summary(r, a["tickets"], baseline),
                        a["weights"])
    assert a["objective"] == pytest.approx(score)
    # All 4 tickets are legal combinations from the search pool.
    assert all(len(t) == len(set(t)) == 6 and min(t) >= 1 and max(t) <= 49
               for t in a["tickets"])


def test_baseline_rejects_partial_or_malformed(tmp_path: Path):
    path = tmp_path / "baseline.json"
    fixture = synthetic_baseline(history())
    path.write_text(json.dumps(fixture), encoding="utf8")
    data, digest = load_exact_baseline(path)
    assert data == fixture
    assert len(digest) == 64
    for bad in (
        {**fixture, "complete": False},
        {**fixture, "total": UNIVERSE - 1},
        {**fixture, "schema": "other-taxonomy"},
        {**fixture, "counts": {"component": fixture["counts"]["component"]}},
    ):
        path.write_text(json.dumps(bad), encoding="utf8")
        with pytest.raises(ValueError):
            load_exact_baseline(path)


def test_shape_selection_rejects_bad_inputs():
    records = history()
    baseline = synthetic_baseline(records)
    for kwargs in ({"pool_size": 3}, {"seed": -1},
                   {"alpha": float("nan")}, {"alpha": -1}):
        with pytest.raises(ValueError):
            select_shape_portfolio(records, baseline, **kwargs)
