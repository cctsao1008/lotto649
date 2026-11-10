"""v0.25 number-deficit foundation regression tests.

Run: python -m pytest tests/test_deficit_compensation_v025.py -q
"""
from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

import pytest

from lotto7x7.deficit_compensation_v025 import (
    REFERENCE, LAMBDA_NUMBER, P, P2, UNIVERSE, deficit_select,
    discrepancy, frequency_audit, load_csv_history, validate_ticket,
)


def synthetic(n=80):
    from random import Random
    rng = Random(22)
    return [(f"2020-01-{i % 28 + 1:02d}", i + 1,
             tuple(sorted(rng.sample(range(1, 50), 6)))) for i in range(n)]


def test_exact_number_moments_and_jackpot():
    from math import comb
    assert UNIVERSE == 13_983_816
    assert comb(48, 5) / UNIVERSE == P
    assert comb(47, 4) / UNIVERSE == P2
    assert LAMBDA_NUMBER > 0
    assert sum((P for _ in range(49))) == pytest.approx(6)


def test_reference_and_ticket_validation():
    assert len({v for row in REFERENCE for v in row}) == 24
    assert [validate_ticket(row) for row in REFERENCE] == list(REFERENCE)
    for invalid in ((0, 1, 2, 3, 4, 5), (1, 1, 3, 4, 5, 6),
                    (1, 2, 3), (True, 2, 3, 4, 5, 6)):
        with pytest.raises(ValueError):
            validate_ticket(invalid)


def test_frequency_marginal_and_covariance_score():
    records = synthetic(40)
    data = frequency_audit(records)
    assert sum(item["observed"] for item in data) == 240
    assert sum(item["deficit"] for item in data) == pytest.approx(0, abs=1e-11)
    before = discrepancy(records)
    assert before >= 0
    # Independently recompute the covariance inverse for 49 Bernoulli
    # indicators on the 48-dim sum-zero subspace.
    assert before == pytest.approx(
        sum((x["observed"] / 40 - P) ** 2 for x in data) / LAMBDA_NUMBER
    )


def test_augmentation_score_and_deterministic_selection():
    records = synthetic(100)
    chosen = deficit_select(records, count=4, pool_size=80, seed=649)
    assert len(chosen["tickets"]) == 4
    assert len(set(map(tuple, chosen["tickets"]))) == 4
    assert chosen["after_discrepancy"] == pytest.approx(
        discrepancy(records, chosen["tickets"]))
    assert chosen == deficit_select(records, count=4, pool_size=80, seed=649)
    assert chosen["after_discrepancy"] <= chosen["baseline_discrepancy"] + 1e-12


def test_invalid_selection_inputs():
    records = synthetic(10)
    for kwargs in ({"count": 0}, {"pool_size": 1}, {"seed": -1},
                   {"ridge": -1}):
        with pytest.raises(ValueError):
            deficit_select(records, **kwargs)
    with pytest.raises(ValueError):
        discrepancy(records, [REFERENCE[0], REFERENCE[0]])


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf8") as fp:
        writer = csv.writer(fp)
        writer.writerow(("date", "draw_id", "n1", "n2", "n3",
                         "n4", "n5", "n6", "bonus"))
        writer.writerows(rows)


def test_csv_validation_cutoff_and_hash(tmp_path: Path):
    path = tmp_path / "lotto649_2026.csv"
    rows = [
        ("2026/11/03", "115000102", 2, 23, 24, 25, 29, 39, 31),
        ("2026/11/06", "115000103", 5, 9, 26, 32, 41, 48, 23),
    ]
    _write_csv(path, rows)
    records, h1 = load_csv_history(tmp_path, cutoff="2026-11-03")
    assert len(records) == 1
    assert records[0][1] == 115000102
    assert h1 == load_csv_history(tmp_path, cutoff="2026-11-03")[1]
    with pytest.raises(ValueError):
        _write_csv(path, rows[::-1])
        load_csv_history(tmp_path)
    with pytest.raises(ValueError):
        _write_csv(path, [rows[0], rows[0]])
        load_csv_history(tmp_path)
    with pytest.raises(ValueError):
        _write_csv(path, [(*rows[0][:8], rows[0][2])])
        load_csv_history(tmp_path)
