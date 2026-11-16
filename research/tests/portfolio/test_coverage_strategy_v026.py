"""v0.26 P1 profile-search coverage tests with exact prize engine."""
from math import comb

import pytest

from lotto7x7.coverage_strategy_v026 import (
    PROFILES, audit, overlap_summary, portfolio_profile,
)
from lotto7x7.deficit_compensation_v025 import REFERENCE
from lotto7x7.official_prize_v026 import official_prize_coverage


@pytest.mark.parametrize("profile", PROFILES)
def test_all_profiles_legal_unique_seeded(profile):
    rows = portfolio_profile(profile, 649)
    assert rows == portfolio_profile(profile, 649)
    assert len(rows) == 4 and len(set(rows)) == 4
    assert all(len(row) == len(set(row)) == 6 and
               min(row) >= 1 and max(row) <= 49 for row in rows)
    overlap = overlap_summary(rows)
    assert 9 <= overlap["distinct_numbers"] <= 24


def test_structures_are_as_declared():
    for name, distinct in (
        ("disjoint", 24), ("one_pair_1", 23),
        ("one_pair_3", 21), ("one_pair_5", 19),
        ("two_pairs_1", 22), ("two_pairs_2", 20),
        ("two_pairs_3", 18), ("two_pairs_5", 14),
        ("star_1", 21), ("star_2", 18), ("star_3", 15),
        ("star_4", 12), ("star_5", 9),
        ("chain_1", 21), ("chain_2", 18),
    ):
        assert overlap_summary(portfolio_profile(name))["distinct_numbers"] == distinct


def test_disjoint_relabeling_invariance_and_reference():
    baseline = official_prize_coverage(REFERENCE)
    for seed in (1, 649, 2026):
        p = official_prize_coverage(portfolio_profile("disjoint", seed))
        assert p["any_prize_outcomes"] == baseline["any_prize_outcomes"]
        assert p["awarded_ticket_count_outcomes"] == baseline["awarded_ticket_count_outcomes"]
        assert p["any_ticket_by_tier"] == baseline["any_ticket_by_tier"]


def test_one_profile_audit_matches_exact_engine():
    out = audit(profiles=("disjoint",), objective="any-prize", seed=649)
    assert out["best_profile"] == "disjoint"
    assert out["globally_optimal"] is False
    assert out["difference_from_reference"] == pytest.approx(0)
    assert out["reference"]["any_prize_probability"] == pytest.approx(
        0.1227314589619088)
    assert out["evaluated"][0]["total_outcomes"] == comb(49, 6) * 43


def test_multifamily_search_respects_objective_and_expectation():
    subset = ("disjoint", "star_2", "two_pairs_2")
    out = audit(profiles=subset, objective="any-prize")
    assert set(r["profile"] for r in out["evaluated"]) == set(subset)
    assert out["best_objective_value"] == max(
        r["objective_value"] for r in out["evaluated"])
    expected = out["reference"]["expected_awarded_tickets"]
    assert all(r["expected_awarded_tickets"] == pytest.approx(expected)
               for r in out["evaluated"])
    assert all(sum(r["awarded_ticket_count_probabilities"]) == pytest.approx(1)
               for r in out["evaluated"])
    assert len(out["best_tickets"]) == 4


def test_alternative_objective_and_invalid_configs():
    out = audit(profiles=("disjoint", "star_1"),
                objective="two-or-more-awards")
    assert out["best_objective_value"] == max(
        item["two_or_more_awards_probability"] for item in out["evaluated"])
    for invalid in ({"profiles": ()}, {"profiles": ("disjoint", "disjoint")},
                    {"profiles": ("not-a-profile",)}, {"objective": "predicted-profit"},
                    {"seed": -1}):
        with pytest.raises(ValueError):
            audit(**invalid)
