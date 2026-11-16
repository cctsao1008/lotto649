"""Regression tests for v0.25 multivariate research-only feature audit."""
from __future__ import annotations

from random import Random

import numpy as np
import pytest

from lotto7x7.deficit_compensation_v025 import REFERENCE
from lotto7x7.multivariate_audit_v025 import (
    DIM_NAMES, encode_features, fair_covariance, iid_tickets,
    portfolio_audit, registry, select_multivariate_portfolio,
)
from lotto7x7.shape_theory_v016 import LEVELS, UNIVERSE
from lotto7x7.shape_deficit_v025 import _keys


def _fixture():
    # Synthetic mass-conserving shape catalog only, NEVER a scientific
    # substitute for exact v0.16 universe in production.
    rng = Random(649)
    rows = [tuple(sorted(rng.sample(range(1, 50), 6))) for _ in range(120)]
    rows += list(REFERENCE)
    # Include the exact IID tickets exercised in the covariance unit test.
    rows += [tuple(map(int, row)) for row in iid_tickets(256, 649)]
    class_keys = _keys(rows)
    weights = {}
    for level in LEVELS:
        ids = sorted(set(map(int, class_keys[level])))
        q, remainder = divmod(UNIVERSE, len(ids))
        weights[level] = {str(k): q + (int(i < remainder))
                          for i, k in enumerate(ids)}
    baseline = {"counts": weights}
    records = [(f"2026-01-{i % 28 + 1:02d}", i, t)
               for i, t in enumerate(rows)]
    return baseline, records


def test_registry_exact_means_and_class_mass():
    baseline, _ = _fixture()
    names, mu, catalog = registry(baseline)
    assert len(names) == len(mu)
    assert names[:3] == ("number_01", "number_02", "number_03")
    assert names[49:52] == DIM_NAMES
    np.testing.assert_allclose(mu[:49], 6 / 49)
    np.testing.assert_allclose(mu[49:52], [150, 150 / 49, 250 / 7])
    for level in LEVELS:
        assert len(catalog[level]) >= 1
        assert sum(baseline["counts"][level].values()) == UNIVERSE


def test_feature_matrix_matches_ticket_definitions():
    baseline, _ = _fixture()
    _, _, catalog = registry(baseline)
    matrix = encode_features(REFERENCE, catalog)
    assert matrix.shape[0] == 4
    np.testing.assert_allclose(matrix[:, :49].sum(axis=1), 6)
    np.testing.assert_allclose(matrix[:, 49], [125, 126, 149, 148])
    np.testing.assert_allclose(matrix[:, 50], [3, 2, 1, 4])
    np.testing.assert_allclose(matrix[:, 51], [38, 24, 42, 44])
    assert np.all((matrix[:, :49] == 0) | (matrix[:, :49] == 1))
    offset = 52
    for level in LEVELS:
        width = len(catalog[level])
        np.testing.assert_allclose(matrix[:, offset:offset + width].sum(axis=1), 1)
        offset += width


def test_iid_tickets_are_legal_deterministic():
    a = iid_tickets(256, 22)
    assert np.array_equal(a, iid_tickets(256, 22))
    assert a.shape == (256, 6)
    assert ((1 <= a) & (a <= 49)).all()
    assert (np.diff(a, axis=1) > 0).all()
    with pytest.raises(ValueError):
        iid_tickets(255, 649)


def test_covariance_regularized_and_reference_audit():
    baseline, records = _fixture()
    a = fair_covariance(baseline, samples=256, seed=649, ridge=0.5)
    b = fair_covariance(baseline, samples=256, seed=649, ridge=0.5)
    np.testing.assert_array_equal(a["eigenvalues"], b["eigenvalues"])
    assert a["meta"]["smallest_eigenvalue"] > 0
    assert a["meta"]["dimension"] == len(a["names"])
    p = portfolio_audit(records, a, REFERENCE)
    assert np.isfinite([p["before"], p["after"], p["improvement"]]).all()
    assert p["improvement"] == pytest.approx(p["before"] - p["after"])
    empty = portfolio_audit(records, a, ())
    assert empty["before"] == empty["after"]


def test_bad_covariance_inputs():
    baseline, _ = _fixture()
    for kwargs in ({"samples": 10}, {"seed": -1}, {"ridge": 0},
                   {"ridge": float("nan")}, {"ridge": -0.1}):
        with pytest.raises(ValueError):
            fair_covariance(baseline, **({"samples": 256, **kwargs}))


def test_multivariate_joint_coverage_constrained_optimizer():
    baseline, records = _fixture()
    model = fair_covariance(baseline, samples=256, seed=649, ridge=0.5)
    kwargs = {"pool_size": 35, "seed": 649, "coverage_constraint": "reference-3plus"}
    first = select_multivariate_portfolio(records, model, **kwargs)
    second = select_multivariate_portfolio(records, model, **kwargs)
    assert first == second
    assert first["meets_reference_3plus_floor"]
    assert len(first["tickets"]) == 4
    assert len({number for row in first["tickets"] for number in row}) == 24
    assert first["objective"] == pytest.approx(
        portfolio_audit(records, model, first["tickets"])["after"])
    assert first["exact_main_ball_coverage"]["at_least"]["3"] == pytest.approx(
        0.07437855303588091)


def test_multivariate_joint_invalid_inputs():
    baseline, records = _fixture()
    model = fair_covariance(baseline, samples=256, seed=649, ridge=0.5)
    for kwargs in ({"pool_size": 2}, {"seed": -1},
                   {"coverage_constraint": "unknown"}):
        with pytest.raises(ValueError):
            select_multivariate_portfolio(records, model, **({"pool_size": 4, **kwargs}))
