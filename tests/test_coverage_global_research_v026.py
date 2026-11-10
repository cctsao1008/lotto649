"""Research guardrails for exact two-ticket and exploratory four-ticket studies."""
from itertools import permutations

import pytest

from lotto7x7.coverage_global_research_v026 import (
    canonical_signature, random_overlap_portfolio, research,
    two_ticket_overlap_study,
)
from lotto7x7.deficit_compensation_v025 import REFERENCE


def test_two_ticket_exhaustive_overlap_partition_and_monotonicity():
    result = two_ticket_overlap_study()
    assert [r["shared_numbers"] for r in result["cases"]] == list(range(6))
    assert result["best_overlap"] == 0
    assert result["strict_disjoint_optimum"] is True
    assert len(set(r["expected_awarded_tickets"] for r in result["cases"])) == 1
    for row in result["cases"]:
        assert row["any_prize_probability"] + row["both_awarded_probability"] == pytest.approx(
            row["expected_awarded_tickets"])


def test_canonical_signature_invariant_to_ticket_order_and_number_relabeling():
    sig = canonical_signature(REFERENCE)
    assert sum(sig) == 24
    for perm in permutations(REFERENCE):
        assert canonical_signature(perm) == sig
    remap = {i: 50 - i for i in range(1, 50)}
    relabeled = [tuple(remap[n] for n in row) for row in REFERENCE]
    assert canonical_signature(relabeled) == sig


def test_research_smoke_and_consistency():
    report = research(seed=649, samples=8)
    four = report["four_ticket"]
    assert four["distinct_structures_evaluated"] >= 15
    assert four["globally_optimal_proven"] is False
    assert four["expected_awarded_tickets_invariant"] is True
    assert four["best_tested"]["any_prize_probability"] >= (
        four["reference_any_prize_probability"] - 1e-12)
    assert len(four["top_10"]) <= 10


def test_stochastic_generator_is_legal_and_deterministic():
    from random import Random
    left = Random(649)
    right = Random(649)
    for moves in (1, 8, 20):
        a = random_overlap_portfolio(left, mutations=moves)
        b = random_overlap_portfolio(right, mutations=moves)
        assert a == b
        assert len(set(a)) == 4
        assert all(len(row) == len(set(row)) == 6 for row in a)


def test_invalid_search_inputs():
    with pytest.raises(ValueError):
        research(samples=-1)
    with pytest.raises(ValueError):
        research(seed=-1)
