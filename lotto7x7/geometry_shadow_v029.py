"""Read-only verifier for the pre-draw v0.29 P5A nine-draw shadow seal.

Never append outcomes to, update, overwrite or reselect a frozen shadow
manifest. This verifier is deterministic even after the planned draw window.
"""
from __future__ import annotations

import argparse
from datetime import date
from hashlib import sha1
import json
from math import isclose
from pathlib import Path

SEAL_PATH = Path("prospective/v029_p5a_static9_20261113_seal.json")
ORIGINAL_PATH = Path("prospective/v027_p3f_static10_20261110_seal.json")
SCHEMA = "lotto649.geometry.v0.29.p5a.shadow.static9.v1"
EXPERIMENT_ID = "v029-p5a-geometry-shadow-static9-20261113"
ORIGINAL_BLOB_SHA1 = "b7dec534a44b1ec08d575a4f2e48e5c3580e76d1"
SOURCE_P3B_REPORT_SHA256 = "9866e1c0294f86f1d078952146a5ad59e9f2053c2dc7be3e269de3e9ea480041"
P3B_COMMIT = "b36cf882aae53578a3c2c2c5c6000af08d523313"
P3B_FROZEN_FOUR = (
    (7, 10, 12, 19, 30, 34),
    (11, 26, 27, 31, 40, 46),
    (6, 20, 21, 24, 33, 49),
    (5, 8, 17, 22, 32, 37),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git_blob_sha1(payload: bytes) -> str:
    return sha1(b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload).hexdigest()


def validate_tickets(tickets: object) -> tuple[tuple[int, ...], ...]:
    """Validate four legal six-number tickets with exactly 24 distinct labels."""
    _require(isinstance(tickets, list) and len(tickets) == 4,
             "shadow portfolio requires exactly four tickets")
    output = []
    for row in tickets:
        _require(isinstance(row, list) and len(row) == 6,
                 "each ticket requires six numbers")
        _require(all(type(n) is int and 1 <= n <= 49 for n in row)
                 and row == sorted(set(row)),
                 "ticket numbers must be sorted, unique integers from 1..49")
        output.append(tuple(row))
    _require(len(set(n for row in output for n in row)) == 24,
             "four shadow tickets must have 24 distinct numbers")
    return tuple(output)


def _parse_date(value: object) -> date:
    _require(isinstance(value, str), "draw date must be a string")
    try:
        day = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("invalid ISO draw date") from exc
    _require(day.isoformat() == value, "draw date must be canonical ISO")
    return day


def verify_shadow(seal_path: Path = SEAL_PATH,
                  original_path: Path = ORIGINAL_PATH) -> dict:
    seal_path = Path(seal_path)
    original_path = Path(original_path)
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    original_bytes = original_path.read_bytes().replace(b"\r\n", b"\n")
    _require(_git_blob_sha1(original_bytes) == ORIGINAL_BLOB_SHA1,
             "original P3f control's historic Git blob has changed")
    old = json.loads(original_bytes)
    _require(old["experiment_id"] == "v027-p3f-shadow-static10-20261110",
             "wrong historical control experiment")
    _require(seal.get("schema") == SCHEMA
             and seal.get("experiment_id") == EXPERIMENT_ID
             and seal.get("status") == "SHADOW_PRE_REGISTERED_NO_PURCHASE",
             "unknown P5A schema, experiment or status")
    evidence = seal["source_evidence"]
    p3b = evidence["geometry_optimizer"]
    control = evidence["prior_public_shadow_control"]
    _require(p3b["report_json_sha256"] == SOURCE_P3B_REPORT_SHA256,
             "P3B source artifact JSON SHA256 changed")
    _require(p3b["implementation_commit"] == P3B_COMMIT
             and p3b["github_actions_run_id"] == 42969777398
             and p3b["candidate_pool_sha256"] ==
             "2caabf0223d28b04628ab44d9022d2b5cd6b2b13a8c846677684565d625d49f3",
             "P3B source selection record changed")
    _require(p3b["optimizer_params"] == {
        "seed": 649, "pool_size": 1024, "random_portfolios": 256,
        "iid_replicates": 99, "regularization_ridge": 0.05,
        "coordinate_exchange_max_passes": 4,
    }, "P3B algorithm parameters changed")
    _require(control["git_blob_sha1"] == ORIGINAL_BLOB_SHA1
             and control["original_experiment_id"] == old["experiment_id"]
             and control["original_strategy"] == "history_blind_random_control",
             "P3f control source link changed")
    cutoff = seal["historical_cutoff"]
    _require(cutoff["last_draw_id"] == 115000104
             and cutoff["last_draw_date"] == "2026-11-10"
             and cutoff["draws"] == 2498
             and cutoff["next_draw_not_in_model"] == 115000105
             and cutoff["source_manifest_sha256"] ==
             "003ed03eda889952654b679859f2d465ffb7349c7f005a6df3aa4436be3d54a3",
             "historical input cutoff or manifest SHA changed")
    experiment = seal["experiment"]
    expected_schedule = old["experiment"]["planned_draws"][1:]
    _require(experiment["planned_draws"] == expected_schedule
             and experiment["fixed_draw_count"] == 9
             and experiment["terminal_draw_id"] == 115000113
             and experiment["selection_locked_before_draw_id"] == 115000105,
             "future nine-draw schedule does not match original P3f calendar")
    for i, item in enumerate(expected_schedule):
        day = _parse_date(item["date"])
        _require(day > date(2026, 11, 10)
                 and item["draw_id"] == 115000105 + i
                 and item["draw_id"] // 1000000 == day.year - 1911,
                 "illegal date or future draw ID in frozen schedule")
    _require(experiment["shadow_only"] is True
             and experiment["purchase_authorized"] is False,
             "shadow-only experiment cannot authorize actual wagers")
    primary = experiment["primary_comparison"]
    _require(primary["name"] ==
             "geometry_p3b_counterfactual_vs_history_blind_random_control"
             and primary["direction"] == "geometry greater than random",
             "primary prespecified comparison changed")
    _require("official" in primary["event"]
             and "Paired" in primary["comparison"],
             "prize endpoint or paired comparison changed")
    strategies = seal["strategies"]
    _require(set(strategies) == {
        "geometry_p3b_counterfactual", "history_blind_random_control",
    }, "unexpected shadow strategy count or identities")
    a = validate_tickets(strategies["geometry_p3b_counterfactual"]["tickets"])
    b = validate_tickets(strategies["history_blind_random_control"]["tickets"])
    expected_b = validate_tickets(
        old["strategies"]["history_blind_random_control"]["tickets"])
    _require(a == P3B_FROZEN_FOUR,
             "P3B prospective candidate differs from original CI report")
    _require(b == expected_b,
             "history-blind random control differs from original preseal")
    fair = experiment["fair_iid_context"]
    _require(isclose(fair["marginal_any_prize_probability_for_four_disjoint_tickets"],
                     0.122731458961909, abs_tol=1e-15),
             "fixed four-disjoint fair reference changed")
    _require(isclose(fair["expected_any_prize_draws_per_strategy_over_nine"],
                     9 * 0.122731458961909, abs_tol=1e-13),
             "nine-draw fair expectation inconsistent")
    _require("outcomes" not in seal and "draw_results" not in seal
             and "realized_prizes" not in seal,
             "pre-draw shadow seal cannot contain future outcomes")
    return {
        "experiment_id": seal["experiment_id"],
        "status": seal["status"],
        "source_p3b_report_sha256": SOURCE_P3B_REPORT_SHA256,
        "source_p3f_git_blob_sha1": ORIGINAL_BLOB_SHA1,
        "first_future_draw": expected_schedule[0],
        "last_future_draw": expected_schedule[-1],
        "draw_count": len(expected_schedule),
        "strategies": len(strategies),
        "tickets_per_strategy": 4,
        "no_purchases_authorized": True,
        "decision": "PROSPECTIVE_PILOT_SEALED_NO_RESULTS_EVALUATED",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", type=Path, default=SEAL_PATH)
    parser.add_argument("--original", type=Path, default=ORIGINAL_PATH)
    args = parser.parse_args()
    report = verify_shadow(args.seal, args.original)
    print(json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
