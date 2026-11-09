"""Guardrails for an independently auditable S4-reduced structural census."""
from itertools import permutations

import pytest

from lotto7x7.coverage_exhaustive_v026 import (
    canonical_signature, exhaustive_audit, feasible_signatures,
    has_unique_tickets, materialize_tickets,
)
from lotto7x7.coverage_global_research_v026 import canonical_signature as previous_canonical
from lotto7x7.deficit_compensation_v025 import REFERENCE


def test_raw_census_exact_count_independent():
    assert sum(1 for _ in feasible_signatures()) == 52450


def test_valid_labeled_structures_exact_count_independent():
    assert sum(has_unique_tickets(s) for s in feasible_signatures()) == 50985


def test_canonicalization_matches_prior_independent_implementation():
    for idx, sig in enumerate(feasible_signatures()):
        if idx % 117 != 0:
            continue
        assert canonical_signature(sig) == previous_canonical(materialize_tickets(sig))


def test_disjoint_reference_all_permutations_identical():
    sig = previous_canonical(REFERENCE)
    canon = canonical_signature(sig)
    assert canon == sig
    assert len(set(materialize_tickets(sig))) == 4


def test_partial_run_explicitly_cannot_claim_global_proof():
    out = exhaustive_audit(max_classes=12)
    assert not out["completed"]
    assert not out["global_disjoint_optimality_proven"]
    assert out["evaluated_classes"] == 12
    assert out["unique_valid_unlabeled_structures"] is None
    assert out["maximum_improvement_outcomes"] <= 0


def test_invalid_inputs():
    with pytest.raises(ValueError):
        exhaustive_audit(max_classes=0)
    with pytest.raises(ValueError):
        materialize_tickets((0,) * 15)
    with pytest.raises(ValueError):
        canonical_signature((0,) * 14)
