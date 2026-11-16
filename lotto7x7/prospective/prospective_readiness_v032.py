"""v0.32 P3: read-only prospective-study readiness and evidence boundary audit.

Engineering only. A source-archived draw is NEVER a source-authenticated
prospective observation. This module does not calculate prizes, forecasts,
p-values, e-values or wagers. Original prospective evaluators remain authoritative.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from hashlib import sha1, sha256
import json
from pathlib import Path

@dataclass(frozen=True)
class Study:
    key: str
    path: str
    blob: str
    first: int
    first_date: str
    horizon: int
    kind: str

STUDIES = (
    Study("v027_p3f_static10", "prospective/v027_p3f_static10_20261110_seal.json",
          "b7dec534a44b1ec08d575a4f2e48e5c3580e76d1", 115000104, "2026-11-10", 10, "fixed"),
    Study("v029_p5a_static9", "prospective/v029_p5a_static9_20261113_seal.json",
          "546c7352fd0a98a42959b333e51498ef7c56cf9d", 115000105, "2026-11-13", 9, "fixed"),
    Study("v23_m0_static200", "prospective/v23_m0_global_static200_20261113_seal.json",
          "f02a94b37bc2287f89cfbc6dd7f6ab2daa4d16be", 115000106, "2026-11-17", 200, "rolling"),
    Study("v24_p3_iching_static100", "prospective/v24_p3_usno_hexagram_static100_20261114_seal.json",
          "abf4963adf339d11f5bddd75b91ee5409abf82f1", 115000106, "2026-11-17", 100, "rolling"),
)
REPORT_SCHEMA = "lotto649.v032.prospective_readiness.v1"
EXPECTED_SKY_SHA256 = "55bb765698c6a8926626d909290d74013e7038db418a0a1ecad795cfa3179f8c"


def canonical_git_bytes(raw: bytes) -> bytes:
    if b"\r" in raw.replace(b"\r\n", b""):
        raise ValueError("mixed/noncanonical line endings")
    return raw.replace(b"\r\n", b"\n")


def blob_sha1(raw: bytes) -> str:
    b = canonical_git_bytes(raw)
    return sha1(b"blob " + str(len(b)).encode("ascii") + b"\0" + b).hexdigest()


def safe_seal(root: Path, study: Study) -> dict:
    raw = (root / study.path).read_bytes()
    if blob_sha1(raw) != study.blob:
        raise ValueError("immutable seal drift: " + study.key)
    seal = json.loads(raw)
    if not isinstance(seal, dict):
        raise ValueError("seal JSON must be an object")
    return seal


def check_ticket_arms(arms: dict, count: int, require_disjoint: bool = True) -> None:
    if not isinstance(arms, dict) or len(arms) != count:
        raise ValueError("unexpected shadow arm count")
    for name, arm in arms.items():
        tickets = arm.get("tickets") if isinstance(arm, dict) else None
        if not isinstance(name, str) or not name or not isinstance(tickets, list) or len(tickets) != 4:
            raise ValueError("invalid four-ticket arm")
        seen = []
        for ticket in tickets:
            if (not isinstance(ticket, list) or len(ticket) != 6
                    or any(type(x) is not int or not 1 <= x <= 49 for x in ticket)
                    or ticket != sorted(set(ticket))):
                raise ValueError("invalid shadow ticket")
            seen.extend(ticket)
        if require_disjoint and len(set(seen)) != 24:
            raise ValueError("shadow tickets must be disjoint within each arm")


def next_draw_id(current: int) -> int:
    year, sequence = divmod(current, 1000000)
    if year < 93 or sequence <= 0:
        raise ValueError("invalid ROC draw identity")
    return current + 1


def check_fixed_schedule(schedule: object, study: Study) -> list[dict]:
    if not isinstance(schedule, list) or len(schedule) != study.horizon:
        raise ValueError("invalid frozen schedule length: " + study.key)
    last_id, last_date = None, None
    for i, r in enumerate(schedule):
        if not isinstance(r, dict) or set(r) != {"date", "draw_id"}:
            raise ValueError("invalid frozen schedule row")
        n, d = r["draw_id"], r["date"]
        if type(n) is not int or not isinstance(d, str) or date.fromisoformat(d).isoformat() != d:
            raise ValueError("invalid frozen schedule identity")
        if n // 1000000 != date.fromisoformat(d).year - 1911:
            raise ValueError("ROC date/number mismatch")
        if i == 0 and (n, d) != (study.first, study.first_date):
            raise ValueError("frozen schedule first draw drift")
        if last_id is not None and (n != last_id + 1 or d <= last_date):
            raise ValueError("nonconsecutive frozen draw sequence")
        last_id, last_date = n, d
    return schedule


def check_study(root: Path, study: Study) -> dict:
    seal = safe_seal(root, study)
    if study.key == "v027_p3f_static10":
        if seal.get("schema") != "lotto7x7.prospective.v0.27.p3f.static10.v1" or \
                seal.get("status") != "SHADOW_PRE_REGISTERED_NO_PURCHASE":
            raise ValueError("P3f research registration changed")
        experiment = seal["experiment"]
        schedule = check_fixed_schedule(experiment["planned_draws"], study)
        if experiment["fixed_draw_count"] != study.horizon or \
                experiment["terminal_draw_id"] != schedule[-1]["draw_id"]:
            raise ValueError("P3f horizon drift")
        check_ticket_arms(seal["strategies"], 3)
    elif study.key == "v029_p5a_static9":
        if seal.get("schema") != "lotto649.geometry.v0.29.p5a.shadow.static9.v1" or \
                seal.get("status") != "SHADOW_PRE_REGISTERED_NO_PURCHASE":
            raise ValueError("P5A research registration changed")
        experiment = seal["experiment"]
        schedule = check_fixed_schedule(experiment["planned_draws"], study)
        if experiment["fixed_draw_count"] != study.horizon or \
                experiment["terminal_draw_id"] != schedule[-1]["draw_id"] or \
                experiment["purchase_authorized"] is not False:
            raise ValueError("P5A no-wager or horizon drift")
        check_ticket_arms(seal["strategies"], 2)
    elif study.key == "v23_m0_static200":
        p = seal["prospective_protocol"]
        if (seal.get("schema") != "lotto649.v2.3.prospective.fixed_m0_full_universe.v1"
                or seal.get("study_type") != "PROSPECTIVE_STATIC_FROZEN_SHADOW_NO_REAL_WAGERS"
                or p["first_draw_id"] != study.first or p["first_draw_date"] != study.first_date
                or p["horizon_consecutive_official_draws"] != study.horizon
                or p["anytime_valid_monitor"]["review_threshold_e_value"] != "20"):
            raise ValueError("v23 prospective rule drift")
        check_ticket_arms({k: {"tickets":v} for k,v in (
            ("A", seal["selection"]["history_blind_control"]["tickets"]),
            ("B", seal["selection"]["history_optimized_strategy"]["tickets"]))}, 2)
        schedule = []
    elif study.key == "v24_p3_iching_static100":
        first = seal["first_eligible_draw"]
        if (seal.get("schema") != "lotto649.v2.4.p3.future_sky_and_model_seal.v1"
                or seal.get("study_type") != "FUTURE_ONLY_NON_WAGER_PROSPECTIVE_SHADOW"
                or (first["draw_id"], first["date"]) != (study.first, study.first_date)
                or seal["horizon_consecutive_verified_draws"] != study.horizon
                or seal["model"]["review_threshold_e_value"] != 20
                or seal["model"]["never_automatically_conclude_predictive_go"] is not True):
            raise ValueError("v24 P3 prospective rule drift")
        path = seal["sky_table_path"]
        if path != "prospective/v24_p3_sky_calendar_20261114.csv" or \
                seal["sky_table_sha256"] != EXPECTED_SKY_SHA256 or \
                sha256((root / path).read_bytes()).hexdigest() != EXPECTED_SKY_SHA256:
            raise ValueError("P3 pre-draw sky table SHA256 drift")
        schedule = []
    else:
        raise ValueError("unknown study")
    return {"study_id": study.key, "seal_path": study.path, "seal_git_blob_sha1": study.blob,
            "first_eligible_draw": {"draw_id": study.first, "date": study.first_date},
            "horizon": study.horizon, "fixed_schedule": schedule}


def verify_cross_study(meta: list[dict]) -> None:
    byid = {x["study_id"]: x for x in meta}
    a = byid["v027_p3f_static10"]["fixed_schedule"]
    b = byid["v029_p5a_static9"]["fixed_schedule"]
    if b != a[1:]:
        raise ValueError("original P5A is not the 9-draw tail of P3f")
    x, y = byid["v23_m0_static200"], byid["v24_p3_iching_static100"]
    if x["first_eligible_draw"] != y["first_eligible_draw"]:
        raise ValueError("P3 and v23 did not share the frozen first draw")
    if a[2] != x["first_eligible_draw"]:
        raise ValueError("original 10-draw P3f window differs from v23 start")


def archived_prefix(study: dict, draws: list[dict]) -> list[dict]:
    start, horizon = study["first_eligible_draw"]["draw_id"], study["horizon"]
    observed = [x for x in draws if x["draw_id"] >= start]
    if study["fixed_schedule"]:
        window_ids = {x["draw_id"] for x in study["fixed_schedule"]}
        observed = [x for x in observed if x["draw_id"] in window_ids]
        if [(x["draw_id"], x["date"]) for x in observed] != [
                (x["draw_id"], x["date"]) for x in study["fixed_schedule"][:len(observed)]]:
            raise ValueError("archived draws diverge from fixed preseal schedule")
    else:
        observed = observed[:horizon]
        if observed and (observed[0]["draw_id"], observed[0]["date"]) != \
                (start, study["first_eligible_draw"]["date"]):
            raise ValueError("rolling trial skips its first target")
        for a, b in zip(observed, observed[1:]):
            ya, yb = a["draw_id"] // 1000000, b["draw_id"] // 1000000
            if not ((ya == yb and b["draw_id"] == a["draw_id"] + 1)
                    or (yb == ya + 1 and b["draw_id"] == yb * 1000000 + 1)):
                raise ValueError("rolling study has missing archived trial")
            if b["date"] <= a["date"]:
                raise ValueError("nonchronological rolling trial")
    if len(observed) > horizon:
        raise ValueError("archived study exceeds horizon")
    return observed


def audit_readiness(root: Path, *, expected_last_id: int | None = None,
                    expected_draws: int | None = None) -> dict:
    # These existing audited functions are mandatory; never trust only manifest fields.
    from lotto7x7.draw_ingestion_m2 import audit_source, _from_csv_bytes
    from scripts.export_lotto649_csv import generate_yearly_csv
    root = root.resolve()
    studies = [check_study(root, study) for study in STUDIES]
    verify_cross_study(studies)
    archive = audit_source(root / "data/csv", root / "data/bin")
    if expected_last_id is not None and archive["last_draw_id"] != expected_last_id:
        raise ValueError("unexpected archive cutoff (not the intended readiness snapshot)")
    if expected_draws is not None and archive["total_draws"] != expected_draws:
        raise ValueError("unexpected archive size (not the intended readiness snapshot)")
    batches, _ = generate_yearly_csv(root / "data/bin")
    draws = [x for path, raw in sorted(batches.items())
             for x in _from_csv_bytes(path, raw)]
    if len(draws) != archive["total_draws"] or \
            (draws[-1]["draw_id"], draws[-1]["date"]) != (archive["last_draw_id"], archive["last_date"]):
        raise AssertionError("audited archive versus generated draw chronology mismatch")
    state = []
    for item in studies:
        eligible = archived_prefix(item, draws)
        state.append({k:v for k,v in item.items() if k != "fixed_schedule"} |
                     {"source_archived_targets_not_yet_admitted_for_scoring":len(eligible),
                      "remaining_targets_not_in_archive":item["horizon"]-len(eligible),
                      "next_not_archived_target": (
                          item["fixed_schedule"][len(eligible)] if item["fixed_schedule"] and len(eligible) < item["horizon"]
                          else item["first_eligible_draw"] if not eligible and not item["fixed_schedule"]
                          else None),
                      "publisher_evidence_per_target":"NOT_EXAMINED_BY_THIS_PREFLIGHT",
                      "prospective_observations_admitted_by_this_tool":0,
                      "statistical_inference":"NOT_PERFORMED"})
    return {
        "schema": REPORT_SCHEMA,
        "scope":"READINESS_ONLY_NO_OUTCOME_SCORING",
        "source_archive":{
            "gate_status":archive["status"], "total_draws":archive["total_draws"],
            "latest_draw_id":archive["last_draw_id"], "latest_date":archive["last_date"],
            "manifest_sha256":archive["source_fingerprints"]["manifest_sha256"],
            "archive_integrity_is_not_two_publisher_verification":True},
        "studies":state,
        "joint_study_interpretation":{
            "v23_and_iching_share_first_100_future_draws":True,
            "p3f_and_p5a_share_nine_draws":True,
            "v23_and_iching_each_have_e_process_review_threshold_20":True,
            "union_bound_any_one_of_two_valid_e_processes_under_joint_fair_null_at_most":0.10,
            "two_studies_at_20_each_do_not_control_cross_study_familywise_at_0_05":True,
            "p3f_and_p5a_fixed_horizon_tests_not_included_in_this_two_e_process_bound":True,
            "false_review_bound_is_not_proof_of_predictive_effect":True},
        "limitations":[
            "Original M1 experiment ledger remains the authority for paid/private identity; private ticket records are never read.",
            "This readiness check does NOT examine independently archived publisher pages and admits ZERO outcomes to inference.",
            "Source CSV/BIN integrity is not the same as independent two-publisher authentication for each future draw.",
            "Shared lottery draws induce cross-experiment dependence; nominal single-study significance cannot be promoted to familywise proof.",
            "A single-study e-process threshold crossing triggers MANUAL REVIEW ONLY; NEVER automatic PREDICTIVE_GO or ticket action.",
            "Existing original seals, sky table, source data and historical report conclusions are strictly read-only."
        ],
        "predictive_go":False, "wagers_placed":False
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--expected-last-id", type=int)
    p.add_argument("--expected-draws", type=int)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    root, path = args.root.resolve(), args.out.resolve()
    output_root = (root / "results").resolve()
    if not path.is_relative_to(output_root) or path.exists():
        p.error("--out must be a new path inside results/")
    result = audit_readiness(root, expected_last_id=args.expected_last_id,
                             expected_draws=args.expected_draws)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(result, out, ensure_ascii=False, indent=2, sort_keys=True)
        out.write("\n")
    print("PASS: four immutable seals; full archive integrity;", result["source_archive"]["total_draws"],
          "archived draws; NO scoring; NO purchases")
    for row in result["studies"]:
        print(row["study_id"], row["source_archived_targets_not_yet_admitted_for_scoring"],
              "archived only /", row["horizon"], "planned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
