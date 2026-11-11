"""M4: source-gated, read-only evaluation of the frozen P5A 9-draw pilot.

Only future rows explicitly appended to the validated archive AND matching
two locally preserved publisher page snapshots are eligible for scoring.
No betting, source-file mutation, early GO decision or seal modification.
"""
from __future__ import annotations

import argparse
from hashlib import sha1
from math import comb
import json
from pathlib import Path

from .draw_ingestion_m2 import (
    _from_csv_bytes, audit_source, normalize_draw,
)
from .geometry_shadow_v029 import (
    SEAL_PATH, verify_shadow, _git_blob_sha1,
)
from .official_draw_evidence_m2 import verify_local_evidence
from .prize_engine_m3 import grade_portfolio
from scripts.export_lotto649_csv import generate_yearly_csv

SCHEMA = "lotto649.p5a.prospective_paired_evaluation.v1"
FROZEN_SEAL_SHA1 = "546c7352fd0a98a42959b333e51498ef7c56cf9d"
ARMS = ("geometry_p3b_counterfactual", "history_blind_random_control")


def verify_p5a_seal(path: Path = SEAL_PATH) -> dict:
    """Reject even syntactically valid but post-seal modified manifest content."""
    path = Path(path)
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    if _git_blob_sha1(raw) != FROZEN_SEAL_SHA1:
        raise ValueError("P5A original Git blob SHA1 changed after preregistration")
    verify_shadow(path)
    return json.loads(raw)


def exact_paired_one_sided_p(geometry_only: int, random_only: int) -> float:
    """Exact conditional one-sided McNemar sign test, complete trial only."""
    if (type(geometry_only) is not int or type(random_only) is not int
            or min(geometry_only, random_only) < 0):
        raise ValueError("discordant counts must be nonnegative integers")
    d = geometry_only + random_only
    if d > 9:
        raise ValueError("P5A seal has only nine paired trials")
    return sum(comb(d, x) for x in range(geometry_only, d + 1)) / (1 << d)


def evaluate_verified_rows(rows: list[dict], *, seal_path: Path = SEAL_PATH) -> dict:
    """Pure deterministic evaluation for already source-verified draw prefix."""
    seal = verify_p5a_seal(seal_path)
    schedule = seal["experiment"]["planned_draws"]
    if not isinstance(rows, list) or len(rows) > len(schedule):
        raise ValueError("observed draws must be a prefix list of at most nine")
    normalized = [normalize_draw(item) for item in rows]
    if [(item["draw_id"], item["date"]) for item in normalized] != [
        (item["draw_id"], item["date"]) for item in schedule[:len(rows)]
    ]:
        raise ValueError("non-contiguous, duplicate or out-of-schedule future draw prefix")
    if set(seal["strategies"]) != set(ARMS):
        raise ValueError("the sealed two-arm selection was modified")
    by_arm = {}
    for name in ARMS:
        by_arm[name] = {
            "any_award_draws": 0,
            "awarded_ticket_count": 0,
            "max_main_hits_by_draw": [],
            "tier_counts": {
                tier: 0 for tier in ("first", "second", "third", "fourth",
                                    "fifth", "sixth", "seventh", "regular")
            },
        }
    by_draw = []
    geometry_only = random_only = concordant_both = concordant_neither = 0
    for draw in normalized:
        current = {}
        for name in ARMS:
            tickets = seal["strategies"][name]["tickets"]
            evaluation = grade_portfolio(tickets, draw)
            summary = by_arm[name]
            summary["any_award_draws"] += int(evaluation["any_award"])
            summary["awarded_ticket_count"] += evaluation["awarded_ticket_count"]
            summary["max_main_hits_by_draw"].append(evaluation["max_main_hits"])
            for tier, number in evaluation["tier_counts"].items():
                summary["tier_counts"][tier] += number
            current[name] = {
                "any_award": evaluation["any_award"],
                "max_main_hits": evaluation["max_main_hits"],
                "awarded_ticket_count": evaluation["awarded_ticket_count"],
                "tier_counts": {tier: count for tier, count in evaluation["tier_counts"].items()
                                if count},
            }
        a = current[ARMS[0]]["any_award"]
        b = current[ARMS[1]]["any_award"]
        geometry_only += int(a and not b)
        random_only += int(b and not a)
        concordant_both += int(a and b)
        concordant_neither += int(not a and not b)
        by_draw.append({
            "draw_id": draw["draw_id"],
            "date": draw["date"],
            "main": draw["main"],
            "bonus": draw["bonus"],
            "arms": current,
        })
    complete = len(rows) == len(schedule)
    counts = {
        "geometry_only": geometry_only, "control_only": random_only,
        "both": concordant_both, "neither": concordant_neither,
    }
    paired_difference = by_arm[ARMS[0]]["any_award_draws"] - by_arm[ARMS[1]]["any_award_draws"]
    return {
        "schema": SCHEMA,
        "experiment_id": seal["experiment_id"],
        "frozen_manifest_git_blob_sha1": FROZEN_SEAL_SHA1,
        "study_type": "PROSPECTIVE_SHADOW_NO_REAL_WAGERS",
        "status": ("COMPLETE_DESCRIPTIVE_OR_INCONCLUSIVE_NO_PREDICTIVE_GO"
                   if complete else "PENDING_NO_EARLY_SIGNIFICANCE_OR_GO"),
        "planned_draw_count": len(schedule),
        "observed_draw_count": len(rows),
        "pending_draw_count": len(schedule) - len(rows),
        "next_expected_draw": schedule[len(rows)] if not complete else None,
        "prespecified_primary_arm_difference_observed": paired_difference,
        "paired_event_counts_observed": counts,
        "by_arm": by_arm,
        "observed_draws": by_draw,
        "final_one_sided_exact_p": (
            exact_paired_one_sided_p(geometry_only, random_only) if complete else None
        ),
        "winner_declared": False,
        "predictive_go": False,
        "realized_payout_twd": None,
        "realized_profit_twd": None,
        "methodological_warnings": [
            "All nine trials must finish before inferential testing; a partial lead is descriptive only.",
            "Even a completed nine-trial pilot is severely underpowered and cannot establish a lottery edge.",
            "Preregistered geometry and random arms have identical theoretical marginal prize probability.",
            "Archived future draws require external source page verification before score generation.",
            "Eight-tier prize classes are computed independently of published payout and settlement.",
            "No real paid wager or real net profit can be inferred from shadow tickets.",
        ],
    }


def read_archived_verified_prefix(
    csv_root: Path = Path("data/csv"),
    bin_root: Path = Path("data/bin"),
    evidence_root: Path = Path("data/private/lotto649-source"),
) -> tuple[list[dict], dict]:
    """Check source archive and external snapshots before admitting future draws."""
    audit = audit_source(csv_root, bin_root)
    seal = verify_p5a_seal()
    schedule = seal["experiment"]["planned_draws"]
    if audit["total_draws"] < seal["historical_cutoff"]["draws"]:
        raise ValueError("source archive is shorter than frozen P5A training history")
    generated, _ = generate_yearly_csv(Path(bin_root))
    path = Path("lotto649_2026.csv")
    if path not in generated:
        raise ValueError("frozen 2026 annual CSV is unavailable")
    candidates = [
        row for row in _from_csv_bytes(path, generated[path])
        if row["draw_id"] >= schedule[0]["draw_id"]
    ]
    if len(candidates) > len(schedule):
        candidates = [r for r in candidates if r["draw_id"] <= schedule[-1]["draw_id"]]
    if [(r["draw_id"], r["date"]) for r in candidates] != [
        (r["draw_id"], r["date"]) for r in schedule[:len(candidates)]
    ]:
        raise ValueError("archived future draw IDs/dates mismatch frozen P5A schedule")
    approved = []
    evidence = []
    for row in candidates:
        verified = verify_local_evidence(row["draw_id"], Path(evidence_root))
        if row != verified["draw"]:
            raise ValueError("archived draw contradicts two captured publisher pages")
        approved.append(row)
        evidence.append({
            "draw_id": row["draw_id"], "page_sha256": verified["page_sha256"],
            "local_evidence_index_sha256": verified["source_index_sha256"],
        })
    return approved, {
        "source_gate": audit,
        "status": "PASS_ARCHIVE_AND_TWO_INDEPENDENT_PAGE_SNAPSHOTS_FOR_EACH_OBSERVED_DRAW",
        "admitted_draws": len(approved),
        "source_evidence": evidence,
        "caveat": "Local snapshots and HTTPS origin custody still require trustworthy capture practices.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-root", type=Path, default=Path("data/csv"))
    parser.add_argument("--bin-root", type=Path, default=Path("data/bin"))
    parser.add_argument("--evidence-root", type=Path,
                        default=Path("data/private/lotto649-source"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    if not output.is_relative_to(Path("results").resolve()):
        raise ValueError("P5A evaluation must be created under the ignored results directory")
    verified, evidence = read_archived_verified_prefix(
        args.csv_root, args.bin_root, args.evidence_root
    )
    report = evaluate_verified_rows(verified)
    report["source_evidence_gate"] = evidence
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii", newline="\n") as fp:
        json.dump(report, fp, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        fp.write("\n")
    print(f"P5A: {report['status']}; verified {report['observed_draw_count']} / "
          f"{report['planned_draw_count']}; pending {report['pending_draw_count']}")
    print(f"PAIRED DELTA (DESCRIPTIVE): {report['prespecified_primary_arm_difference_observed']}")
    if report["final_one_sided_exact_p"] is not None:
        print(f"FINAL EXACT P (LOW POWER): {report['final_one_sided_exact_p']:.7f}")
    print("NO PREDICTIVE GO / NO REALIZED PROFIT / SEALED TICKETS UNMODIFIED")
    print(f"REPORT: {output}")


if __name__ == "__main__":
    main()
