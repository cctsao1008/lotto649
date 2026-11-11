"""M3: strict eight-tier Lotto649 result grader, without inventing payouts.

Grade draw/category independently of published per-winner amounts. Shadow
tickets are hypothetical, so real settlement/payout and net profit remain
unknown even if a published prize-table reference is available.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .draw_ingestion_m2 import ANCHOR, ANCHOR_SOURCE, normalize_draw, audit_source
from .official_prize_v026 import PRIZE_ORDER, prize_tier, official_prize_coverage
from .coverage_v06 import normalized_bets
from .geometry_shadow_v029 import verify_shadow, SEAL_PATH

SCHEMA = "lotto649.m3.prize_grading.v1"
LOTTERY_LINE_COST_TWD = 50
TIER_REQUIREMENTS = {
    "first": (6, False),
    "second": (5, True),
    "third": (5, False),
    "fourth": (4, True),
    "fifth": (4, False),
    "sixth": (3, True),
    "seventh": (2, True),
    "regular": (3, False),
}
STANDARD_FIXED_TWD = {"fifth": 2000, "sixth": 1000, "seventh": 400, "regular": 400}
# One independently published example. A non-winning jackpot/second prize is
# listed as 0 in the source's table; this must NOT be treated as what a
# hypothetical winning ticket would receive if it changed the winners count.
ANCHOR_PUBLISHED_PAYOUTS_TWD = {
    "first": 0, "second": 0, "third": 56494, "fourth": 16880,
    "fifth": 2000, "sixth": 1000, "seventh": 400, "regular": 400,
}
ANCHOR_PAYOUT_SOURCE = {
    "publisher": ANCHOR_SOURCE["publisher"],
    "url": ANCHOR_SOURCE["url"],
    "review_date": ANCHOR_SOURCE["review_date"],
    "status": "ONE_PUBLISHED_DRAW_TABLE_MANUALLY_REVIEWED_NOT_LIVE_RETRIEVED",
}


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _parse_bet(ticket: object) -> tuple[int, ...]:
    _require(isinstance(ticket, (list, tuple)) and len(ticket) == 6
             and all(type(n) is int and 1 <= n <= 49 for n in ticket)
             and len(set(ticket)) == 6,
             "ticket must have exactly six distinct integer numbers in 1..49")
    return tuple(sorted(ticket))


def validate_published_table(table: object, *, draw: dict) -> dict:
    """Validate a stated payout table; metadata is NOT source authenticity."""
    _require(isinstance(table, dict) and set(table) ==
             {"draw_id", "published_prizes_twd", "source"},
             "published payouts require complete eight-tier table and source")
    _require(type(table["draw_id"]) is int and table["draw_id"] == draw["draw_id"],
             "payout table draw ID does not match verified draw")
    amounts = table["published_prizes_twd"]
    _require(isinstance(amounts, dict) and set(amounts) == set(PRIZE_ORDER),
             "published payout table must contain exactly eight tiers")
    _require(all(type(value) is int and value >= 0 for value in amounts.values()),
             "published per-winner amounts must be nonnegative integer TWD")
    source = table["source"]
    _require(isinstance(source, dict) and set(source) ==
             {"publisher", "url", "review_date", "status"},
             "payout source provenance must be explicit")
    for k in ("publisher", "url", "review_date", "status"):
        _require(type(source[k]) is str and bool(source[k].strip()),
                 "payout provenance fields cannot be blank")
    _require(source["url"].startswith("https://"),
             "payout source must be HTTPS")
    _require(source["status"] in (
        "ONE_PUBLISHED_DRAW_TABLE_MANUALLY_REVIEWED_NOT_LIVE_RETRIEVED",
        "DECLARED_PUBLIC_TABLE_REQUIRES_INDEPENDENT_REVIEW",
    ), "unsupported payout table source status")
    if source["status"] == "ONE_PUBLISHED_DRAW_TABLE_MANUALLY_REVIEWED_NOT_LIVE_RETRIEVED":
        _require(draw == ANCHOR
                 and source == ANCHOR_PAYOUT_SOURCE
                 and amounts == ANCHOR_PUBLISHED_PAYOUTS_TWD,
                 "only independently reviewed 2026-11-10 fixture may claim this status")
    return amounts.copy()


def official_anchor_table() -> dict:
    return {
        "draw_id": ANCHOR["draw_id"],
        "published_prizes_twd": ANCHOR_PUBLISHED_PAYOUTS_TWD.copy(),
        "source": ANCHOR_PAYOUT_SOURCE.copy(),
    }


def grade_ticket(ticket: object, draw_record: object, *,
                 published_table: dict | None = None) -> dict:
    """Classify one legal hypothetical ticket using 6 main + distinct bonus."""
    selected = _parse_bet(ticket)
    draw = normalize_draw(draw_record)
    main = set(draw["main"])
    hits = len(main.intersection(selected))
    bonus_hit = draw["bonus"] in selected
    tier = prize_tier(hits, bonus_hit)
    _require(prize_tier(*TIER_REQUIREMENTS[tier]) == tier
             if tier is not None else True, "official tier mapping is inconsistent")
    reference_payout = None
    payout_status = "NO_VERIFIED_PUBLISHED_PER_TICKET_PAYOUT"
    if published_table is not None:
        table = validate_published_table(published_table, draw=draw)
        source_status = published_table["source"]["status"]
        if tier is None:
            payout_status = "NO_AWARD"
            reference_payout = 0
        elif table[tier] == 0:
            payout_status = "TIER_HAS_NO_PUBLISHED_WINNER_PAYOUT_HYPOTHETICAL_CHANGE_UNKNOWN"
        elif source_status == "DECLARED_PUBLIC_TABLE_REQUIRES_INDEPENDENT_REVIEW":
            payout_status = "UNREVIEWED_SOURCE_AMOUNT_NOT_USABLE"
        else:
            reference_payout = table[tier]
            payout_status = "PUBLISHED_PER_WINNER_REFERENCE_NOT_COUNTERFACTUAL_SETTLEMENT"
    return {
        "numbers": list(selected),
        "main_hits": hits,
        "special_hit": bonus_hit,
        "official_prize_tier": tier,
        "any_prize": tier is not None,
        "published_per_winner_reference_twd": reference_payout,
        "payout_status": payout_status,
        "actual_settled_payout_twd": None,
        "is_actual_paid_wager": False,
    }


def grade_portfolio(tickets: object, draw_record: object, *,
                    published_table: dict | None = None) -> dict:
    """Read-only grading for 1..4 distinct tickets, with no sale assumption."""
    _require(isinstance(tickets, (list, tuple)) and 1 <= len(tickets) <= 4,
             "portfolio must contain 1..4 ticket combinations")
    entries = [_parse_bet(ticket) for ticket in tickets]
    _require(len(set(entries)) == len(entries),
             "duplicate tickets are not separate portfolio entries")
    draw = normalize_draw(draw_record)
    if published_table is not None:
        validate_published_table(published_table, draw=draw)
    graded = [grade_ticket(ticket, draw, published_table=published_table)
              for ticket in entries]
    prize_tiers = {name: sum(item["official_prize_tier"] == name for item in graded)
                   for name in PRIZE_ORDER}
    fair = official_prize_coverage(entries)
    return {
        "schema": SCHEMA,
        "analysis_scope": "HYPOTHETICAL_UNPAID_PORTFOLIO_NOT_CLAIMABLE_SETTLEMENT",
        "draw": draw,
        "tickets": graded,
        "ticket_count": len(entries),
        "awarded_ticket_count": sum(item["any_prize"] for item in graded),
        "any_award": any(item["any_prize"] for item in graded),
        "max_main_hits": max(item["main_hits"] for item in graded),
        "tier_counts": prize_tiers,
        "nominal_unpurchased_ticket_cost_twd": len(entries) * LOTTERY_LINE_COST_TWD,
        "actual_settled_total_twd": None,
        "actual_realized_net_profit_twd": None,
        "fair_any_prize_probability": fair["any_prize_probability"],
        "fair_exact_outcome_denominator": fair["total_outcomes"],
        "scientific_limitations": [
            "Tier classification uses 6 main balls and the distinct special ball; all eight awards are exclusive.",
            "A hypothetical ticket is not an actual purchased, verified, claimable ticket.",
            "Published pari-mutuel per-winner amounts can change if a hypothetical ticket changes the winner count.",
            "Posted payout amounts can depend on the draw's prize pool and exceptional adjustment rules; no general fixed-return assumption.",
            "No actual winnings, settled revenue, personal net profit, or future predictive advantage is established.",
        ],
    }


def verify_prize_engine() -> dict:
    """Self-contained official-anchor fixture plus exact four-disjoint reference."""
    for tier, (hits, bonus) in TIER_REQUIREMENTS.items():
        _require(prize_tier(hits, bonus) == tier,
                 "official eight-tier predicate contradicts expected matrix")
    _require(prize_tier(1, True) is None and prize_tier(2, False) is None
             and prize_tier(0, False) is None, "non-award predicate mismatch")
    observed = grade_ticket([11, 23, 30, 32, 40, 41], ANCHOR,
                            published_table=official_anchor_table())
    _require(observed["official_prize_tier"] == "first"
             and observed["published_per_winner_reference_twd"] is None,
             "fictional first-prize claimant must not inherit actual zero-winner payout")
    source = verify_shadow()
    manifest = json.loads(SEAL_PATH.read_text(encoding="utf-8"))
    arms = {}
    for name, info in manifest["strategies"].items():
        tickets = info["tickets"]
        _require(len({n for row in tickets for n in row}) == 24,
                 "shadow portfolio must have four disjoint tickets")
        prob = official_prize_coverage(tickets)
        _require(abs(prob["any_prize_probability"] - 0.122731458961909) < 1e-13,
                 "four-disjoint official odds differ from sealed fair constant")
        arms[name] = {
            "tickets": len(tickets),
            "total_exact_outcomes": prob["total_outcomes"],
            "fair_any_prize_probability": prob["any_prize_probability"],
        }
    _require(len(arms) == 2, "missing shadow evaluation arm")
    return {
        "status": "PASS_EIGHT_TIERS_ANCHOR_AND_EXACT_FOUR_DISJOINT_PROBABILITY",
        "official_tier_matrix": {tier: {"main": hits, "special": bonus}
                                 for tier, (hits, bonus) in TIER_REQUIREMENTS.items()},
        "current_published_anchor": {
            "draw": ANCHOR, "per_winner_amounts_twd": ANCHOR_PUBLISHED_PAYOUTS_TWD,
            "source": ANCHOR_PAYOUT_SOURCE,
        },
        "frozen_shadow_verification": source["decision"],
        "shadow_arms": arms,
        "return_limitation": "No actual or hypothetical settled monetary returns are calculated.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-root", type=Path, default=Path("data/csv"))
    parser.add_argument("--bin-root", type=Path, default=Path("data/bin"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    output = args.out.resolve()
    _require(output.is_relative_to(Path("results").resolve()),
             "M3 report path must be under gitignored results/")
    audit = audit_source(args.csv_root, args.bin_root)
    report = {
        "schema": SCHEMA,
        "source_gate": audit,
        "prize_verification": verify_prize_engine(),
        "shadow_current_state": "P5A_NINE_FUTURE_DRAWS_PENDING_NO_PAYOUT_EVALUATION",
        "example_anchor_hypothetical": grade_portfolio([
            [11, 23, 30, 32, 40, 41],
            [11, 23, 28, 32, 40, 41],
            [11, 23, 30, 32, 40, 48],
            [1, 2, 3, 4, 5, 6],
        ], ANCHOR, published_table=official_anchor_table()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii", newline="\n") as fp:
        json.dump(report, fp, indent=2, sort_keys=True, ensure_ascii=True)
        fp.write("\n")
    print("PASS M3: eight prize tiers, special-ball rules, published prize anchors, "
          "and exact four-disjoint fair probability")
    print("PAYOUT: published reference only; no hypothetical settlement or realized net profit")
    print("SHADOW: P5A still pending; no future outcomes evaluated")
    print(f"REPORT: {output}")


if __name__ == "__main__":
    main()
