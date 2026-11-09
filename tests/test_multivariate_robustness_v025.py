"""IID null calibration and robustness audit regression tests."""
from __future__ import annotations

from random import Random
import numpy as np
import pytest

from lotto7x7.deficit_compensation_v025 import REFERENCE
from lotto7x7.multivariate_audit_v025 import iid_tickets
from lotto7x7.multivariate_robustness_v025 import (
    _integer_sequence, _float_sequence, robustness_audit,
)
from lotto7x7.shape_deficit_v025 import _keys
from lotto7x7.shape_theory_v016 import LEVELS, UNIVERSE


def fixture():
    n = 256
    rng = Random(77)
    records = [(None, i + 1, tuple(sorted(rng.sample(range(1, 50), 6))))
               for i in range(n)]
    # Test-only category support includes every possible input used by the
    # deterministic 2-replicate pipeline, not an exhaustive real universe.
    ticket_rows = [r[2] for r in records]
    ticket_rows += list(REFERENCE)
    ticket_rows += [tuple(map(int, x)) for x in iid_tickets(256, 649)]
    for seed in (1650, 1651):
        ticket_rows += [tuple(map(int, x)) for x in iid_tickets(n, seed)]
    pool = Random(649)
    ticket_rows += [tuple(sorted(pool.sample(range(1, 50), 6)))
                    for _ in range(12)]
    keys = _keys(ticket_rows)
    counts = {}
    for level in LEVELS:
        labels = sorted(set(int(v) for v in keys[level]))
        q, remainder = divmod(UNIVERSE, len(labels))
        counts[level] = {
            str(code): q + (1 if i < remainder else 0)
            for i, code in enumerate(labels)
        }
    return records, {"counts": counts}


def test_grid_parsing_rejects_duplicate_and_invalid():
    assert _integer_sequence("256, 512") == (256, 512)
    assert _float_sequence("0.25, 1") == (0.25, 1.0)
    for text in ("", "-1", "1,1", "a"):
        with pytest.raises(ValueError):
            _integer_sequence(text)
    for text in ("", "-0.1", "0", "nan", "0.5,0.5"):
        with pytest.raises(ValueError):
            _float_sequence(text)


def test_null_reoptimizes_and_reports_resolution():
    records, baseline = fixture()
    settings = dict(iid_sizes=(256,), covariance_seeds=(649,),
                    ridges=(0.5,), selector_seed=649, pool_size=12,
                    null_replicates=2, null_seed=1649)
    a = robustness_audit(records, baseline, **settings)
    b = robustness_audit(records, baseline, **settings)
    assert a == b
    assert a["history_draws"] == 256
    assert len(a["scenarios"]) == 1
    scenario = a["scenarios"][0]
    observed = scenario["observed"]
    null = scenario["null_pipeline_matched"]
    assert observed["coverage_floor_pass"]
    assert len(observed["selected_tickets"]) == 4
    assert len({v for row in observed["selected_tickets"] for v in row}) == 24
    assert null["replicates"] == 2
    assert len(null["gains"]) == 2
    assert null["p_resolution"] == pytest.approx(1 / 3)
    assert null["p_empirical_exploratory_upper"] in (1/3, 2/3, 1)
    assert np.isfinite(null["gains"]).all()


def test_invalid_audit_arguments():
    records, baseline = fixture()
    for patch in ({"null_replicates": 0}, {"pool_size": 1},
                  {"covariance_seeds": ()}, {"ridges": (0,)},
                  {"coverage_constraint": "bad"}):
        opts = dict(iid_sizes=(256,), covariance_seeds=(649,),
                    ridges=(0.5,), pool_size=12, null_replicates=2)
        opts.update(patch)
        with pytest.raises(ValueError):
            robustness_audit(records, baseline, **opts)
