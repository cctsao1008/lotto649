"""v0.26 P1: exact official-prize Coverage Strategy A (structural profiles).

Under uniform 6/49, relabeling numbers leaves all exact prize probabilities
unchanged: only portfolio intersection structure matters. Evaluate a small,
explicit *non-exhaustive* set of representative overlap profiles. Do not
pretend a random-number or in-sample search can improve these probabilities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random

from .deficit_compensation_v025 import REFERENCE
from .official_prize_v026 import PRIZE_ORDER, official_prize_coverage

SCHEMA = "lotto7x7.coverage_strategy_a.v0.26.p1"
PROFILES = (
    "disjoint",
    "one_pair_1", "one_pair_3", "one_pair_5",
    "two_pairs_1", "two_pairs_2", "two_pairs_3", "two_pairs_5",
    "chain_1", "chain_2",
    "star_1", "star_2", "star_3", "star_4", "star_5",
)


def portfolio_profile(profile, seed=649):
    """Generate one legal four-bet realization of a fixed overlap pattern."""
    if profile not in PROFILES:
        raise ValueError(f"unknown overlap profile: {profile}")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = Random(seed)
    labels = list(range(1, 50))
    rng.shuffle(labels)
    rows = [labels[6 * i:6 * (i + 1)] for i in range(4)]
    if profile == "disjoint":
        pass
    elif profile.startswith("one_pair_"):
        k = int(profile.rsplit("_", 1)[1])
        rows[1][:k] = rows[0][:k]
    elif profile.startswith("two_pairs_"):
        k = int(profile.rsplit("_", 1)[1])
        rows[1][:k] = rows[0][:k]
        rows[3][:k] = rows[2][:k]
    elif profile.startswith("star_"):
        k = int(profile.rsplit("_", 1)[1])
        for row in rows[1:]:
            row[:k] = rows[0][:k]
    elif profile.startswith("chain_"):
        k = int(profile.rsplit("_", 1)[1])
        rows[1][:k] = rows[0][:k]
        rows[2][:k] = rows[1][k:2 * k]
        rows[3][:k] = rows[2][k:2 * k]
    result = tuple(tuple(sorted(row)) for row in rows)
    if (len(set(result)) != 4 or
            any(len(row) != 6 or len(set(row)) != 6 for row in result)):
        raise AssertionError("overlap template generated invalid tickets")
    return result


def overlap_summary(tickets):
    sets = [set(row) for row in tickets]
    return {
        "distinct_numbers": len(set.union(*sets)),
        "pairwise_intersections": [
            {"ticket_i": i + 1, "ticket_j": j + 1,
             "shared": len(sets[i] & sets[j])}
            for i in range(4) for j in range(i + 1, 4)
        ],
    }


def _metric_value(result, objective):
    if objective == "any-prize":
        return result["any_prize_probability"]
    if objective == "two-or-more-awards":
        return sum(result["awarded_ticket_count_probabilities"][2:])
    if objective in PRIZE_ORDER:
        return result["any_ticket_by_tier"][objective]["probability"]
    raise ValueError(f"unsupported coverage objective: {objective}")


def audit(*, seed=649, objective="any-prize", profiles=PROFILES):
    if objective not in ("any-prize", "two-or-more-awards", *PRIZE_ORDER):
        raise ValueError("invalid coverage objective")
    profiles = tuple(profiles)
    if not profiles or len(set(profiles)) != len(profiles):
        raise ValueError("profiles must be nonempty and unique")
    if any(profile not in PROFILES for profile in profiles):
        raise ValueError("unknown profile requested")
    original = official_prize_coverage(REFERENCE)
    reference_value = _metric_value(original, objective)
    examined = []
    for profile in profiles:
        rows = portfolio_profile(profile, seed)
        prize_result = official_prize_coverage(rows)
        examined.append({
            "profile": profile, "tickets": [list(t) for t in rows],
            "overlap": overlap_summary(rows),
            "objective_value": _metric_value(prize_result, objective),
            "any_prize_probability": prize_result["any_prize_probability"],
            "two_or_more_awards_probability": sum(
                prize_result["awarded_ticket_count_probabilities"][2:]),
            "expected_awarded_tickets": prize_result["expected_awarded_tickets"],
            "awarded_ticket_count_probabilities":
                prize_result["awarded_ticket_count_probabilities"],
            "by_tier_probability": {
                tier: prize_result["any_ticket_by_tier"][tier]["probability"]
                for tier in PRIZE_ORDER
            },
            "total_outcomes": prize_result["total_outcomes"],
        })
    winner = max(examined, key=lambda item: (
        item["objective_value"], -PROFILES.index(item["profile"])))
    # Linearity: four different fair tickets have identical expected number
    # of winning tickets, regardless of inter-ticket overlap.
    if any(abs(row["expected_awarded_tickets"] -
               original["expected_awarded_tickets"]) > 1e-12 for row in examined):
        raise AssertionError("fair per-ticket expectation depends on portfolio overlap")
    return {
        "schema": SCHEMA,
        "research_only": True,
        "method": "exact finite structural-profile search (not global optimum)",
        "objective": objective,
        "seed": seed,
        "globally_optimal": False,
        "reference": {
            "tickets": [list(t) for t in REFERENCE],
            "objective_value": reference_value,
            "any_prize_probability": original["any_prize_probability"],
            "expected_awarded_tickets": original["expected_awarded_tickets"],
            "awarded_ticket_count_probabilities":
                original["awarded_ticket_count_probabilities"],
        },
        "evaluated": examined,
        "best_profile": winner["profile"],
        "best_tickets": winner["tickets"],
        "best_objective_value": winner["objective_value"],
        "difference_from_reference": winner["objective_value"] - reference_value,
        "note": (
            "All results are exact fair-IID outcome probabilities. "
            "Profiles are an explicitly restricted family; not exhaustive. "
            "Optimizing any prize does not maximize expected monetary return."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=649)
    parser.add_argument("--objective", default="any-prize",
                        choices=("any-prize", "two-or-more-awards", *PRIZE_ORDER))
    parser.add_argument("--profiles", default=",".join(PROFILES),
                        help="comma-separated representative overlap structures")
    parser.add_argument("--out", type=Path,
                        default=Path("results/v026_coverage_strategy_a.json"))
    args = parser.parse_args(argv)
    output = audit(seed=args.seed, objective=args.objective,
                   profiles=tuple(args.profiles.split(",")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"[P1] profiles={len(output['evaluated'])}; objective={args.objective}")
    print(f"[REFERENCE] {output['reference']['objective_value']:.12f}")
    print(f"[BEST] {output['best_profile']} "
          f"objective={output['best_objective_value']:.12f} "
          f"delta={output['difference_from_reference']:+.12f}")
    print(f"[REPORT] {args.out}")


if __name__ == "__main__":
    main()
