"""v0.27 P3f: audit a PRE-DRAW SEALED ten-draw static shadow experiment.

This runner READS an immutable preregistration. It never calls an optimizer,
changes tickets, or fills in future draw outcomes. The original P2 predictions
are frozen for all ten future draws; they are NOT an adaptive rolling P2 model.
As yet unavailable results are PENDING, never losses. Exact statistical inference
is reported only when all ten predeclared draws have arrived.

Under a fair IID Lotto 6/49, each valid four-disjoint-ticket strategy has the
same exact conditional chance of at least one official prize, independent of
history. A fixed n=10 sequence has a Binomial(10,p) null. Two named P2
strategies use a conservative two-policy Bonferroni one-sided upper-tail
correction. The random control is a descriptive comparator only.

The public shadow study is entirely separate from the user's already paid
four-line ten-draw ticket. The script never buys or changes any ticket.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from fractions import Fraction
from hashlib import sha1, sha256
import json
from pathlib import Path
from random import Random

from .distribution_repair_portfolio_v027 import validate_four_disjoint
from .distribution_repair_v027 import _prefix_sha256
from .log_product_p3b_pilot_v027 import evaluate_draw
from .log_product_p3c_exact_iid_v027 import exact_binomial_tail, exact_fair_probability
from .temporal_dataset_v024 import load_csv_history

SCHEMA = "lotto7x7.prospective.v0.27.p3f.static10.v1"
REPORT_SCHEMA = "lotto7x7.prospective.v0.27.p3f.static10.report.v1"
SEAL = Path("prospective/v027_p3f_static10_20261110_seal.json")

# Git blob pinned by the PRE-FIRST-DRAW commit ae15748a88e7aad334d68354b04338fa71d6bfdf.
# Do not update this to retroactively accept changed tickets or a changed schedule.
PINNED_GIT_BLOB_SHA1 = "b7dec534a44b1ec08d575a4f2e48e5c3580e76d1"
SEAL_COMMIT = "ae15748a88e7aad334d68354b04338fa71d6bfdf"
EXPECTED_HISTORY_SHA256 = "d865e1d241b607488e58a00b22fb595734f804595afd9d8850718808d6e8b7f8"
EXPECTED_STRATEGIES = (
    "original_p2_primary", "original_p2_distinct_sums",
    "history_blind_random_control",
)
PRIMARY_HYPOTHESES = ("original_p2_primary", "original_p2_distinct_sums")


def _blob_sha1(raw: bytes) -> str:
    return sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()


def load_frozen_seal(path: Path = SEAL) -> tuple[dict, dict]:
    """Reject any edited/replaced preregistration, even if JSON still parses."""
    raw = Path(path).read_bytes()
    fingerprint = _blob_sha1(raw)
    if fingerprint != PINNED_GIT_BLOB_SHA1:
        raise ValueError("PROSPECTIVE SEAL MODIFIED: wrong pinned pre-draw Git blob SHA1")
    manifest = json.loads(raw)
    validate_manifest(manifest)
    return manifest, {
        "first_pre_draw_seal_commit": SEAL_COMMIT,
        "pinned_git_blob_sha1": fingerprint,
        "seal_bytes_sha256": sha256(raw).hexdigest(),
    }


def validate_manifest(manifest: dict) -> None:
    if (not isinstance(manifest, dict) or
            manifest.get("schema") != SCHEMA or
            manifest.get("status") != "SHADOW_PRE_REGISTERED_NO_PURCHASE"):
        raise ValueError("wrong preregistration schema/status")
    historical = manifest["historical_cutoff"]
    if (historical["last_draw_date"] != "2026-11-06" or
            historical["last_draw_id"] != 115000103 or
            historical["draws"] != 2497 or
            historical["next_draw_not_in_model"] != 115000104 or
            historical["input_prefix_sha256"] != EXPECTED_HISTORY_SHA256):
        raise ValueError("model cutoff/source fingerprint mismatch")
    source = manifest["source"]
    if (source["original_p2_module_git_blob_sha1"] !=
            "e190bfbc847658301d69cc9d91a4c26dba27617e" or
            source["selection_parameters"] != {
                "seed":649, "pool_per_cell":800, "cell_limit":24,
                "restarts":512, "fallback_trials":15000,
            }):
        raise ValueError("frozen original P2 provenance/selection budget changed")
    experiment = manifest["experiment"]
    planned = experiment["planned_draws"]
    if (experiment["fixed_draw_count"] != 10 or
            experiment["terminal_draw_id"] != 115000113 or
            not isinstance(planned, list) or len(planned) != 10 or
            [row["draw_id"] for row in planned] != list(range(115000104,115000114)) or
            [row["date"] for row in planned] != [
                "2026-11-10","2026-11-13","2026-11-17",
                "2026-11-20","2026-11-24","2026-11-27",
                "2026-12-01","2026-12-04","2026-12-08",
                "2026-12-11"
            ] or
            experiment["prespecified_primary_hypotheses"] != list(PRIMARY_HYPOTHESES)):
        raise ValueError("precommitted draw schedule or hypotheses changed")
    if set(manifest["strategies"]) != set(EXPECTED_STRATEGIES):
        raise ValueError("three precommitted shadow policies required")
    for name in EXPECTED_STRATEGIES:
        tickets = manifest["strategies"][name]["tickets"]
        canonical = validate_four_disjoint(tickets)
        if [list(x) for x in canonical] != tickets:
            raise ValueError(f"ticket ordering/number duplication in {name}")
        if name == "original_p2_distinct_sums" and len({sum(t) for t in canonical}) != 4:
            raise ValueError("frozen four-distinct-sums policy altered")
    rng = Random(649)
    vals = rng.sample(range(1,50),24)
    random_tickets = [list(x) for x in validate_four_disjoint(
        tuple(sorted(tuple(sorted(vals[i:i+6])) for i in range(0,24,6)))
    )]
    if manifest["strategies"]["history_blind_random_control"]["tickets"] != random_tickets:
        raise ValueError("history-blind random baseline does not reproduce its seed")


def read_verified_history(csv_dir: Path, manifest: dict) -> list[dict]:
    """Verify first 2497 canonical main draws before ingesting post-seal rows.

    The existing loader validates chronology, duplicate numbers, dates,
    valid distinct bonus and CSV structure. No data after the pre-draw
    cutoff feeds back into any static ticket selection.
    """
    dataset = load_csv_history(Path(csv_dir))
    frozen_n = manifest["historical_cutoff"]["draws"]
    if len(dataset.ids) < frozen_n:
        raise ValueError("frozen history incomplete")
    if (int(dataset.ids[frozen_n-1]) != 115000103 or
            int(dataset.dates[frozen_n-1]) != 20261106):
        raise ValueError("historical cutoff marker differs from prospectively sealed input")
    actual_hash = _prefix_sha256(dataset.ids[:frozen_n],
                                 dataset.dates[:frozen_n],
                                 dataset.numbers[:frozen_n])
    if actual_hash != EXPECTED_HISTORY_SHA256:
        raise ValueError("historical prefix SHA256 has changed after the seal")
    # Bonus numbers are not retained in temporal dataset arrays; read
    # only the annual CSV for the 2026 future targets. Full historical
    # CSV already validated by load_csv_history above.
    rows = []
    file = Path(csv_dir) / "lotto649_2026.csv"
    with file.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            draw_id = int(row["draw_id"])
            if 115000104 <= draw_id <= 115000113:
                rows.append({
                    "draw_id":draw_id,
                    "date":datetime.strptime(row["date"].strip(),"%Y/%m/%d").date().isoformat(),
                    "main":sorted(int(row[f"n{i}"]) for i in range(1,7)),
                    "bonus":int(row["bonus"]),
                })
    return rows


def evaluate_frozen_future(manifest: dict, observed_rows: list[dict],
                           provenance: dict | None = None) -> dict:
    """Deterministically grade observed, not invented, post-seal draws."""
    validate_manifest(manifest)
    if not isinstance(observed_rows, list):
        raise ValueError("observed future results must be an explicit list")
    planned = manifest["experiment"]["planned_draws"]
    expected = {row["draw_id"]:row["date"] for row in planned}
    actual = {}
    for row in observed_rows:
        if (not isinstance(row, dict) or
                type(row.get("draw_id")) is not int or
                row["draw_id"] not in expected or
                row["draw_id"] in actual):
            raise ValueError("extra, duplicated, or unplanned future draw ID")
        if row.get("date") != expected[row["draw_id"]]:
            raise ValueError("draw date does not match pre-registered source schedule")
        main, bonus = row.get("main"), row.get("bonus")
        if (not isinstance(main, list) or len(main) != 6 or
                any(type(x) is not int or x < 1 or x > 49 for x in main) or
                main != sorted(main) or len(set(main)) != 6 or
                type(bonus) is not int or not 1 <= bonus <= 49 or bonus in main):
            raise ValueError("malformed future Lotto649 draw/bonus")
        actual[row["draw_id"]] = row
    # Forbid a missing earlier draw with a later one ingested, which could
    # otherwise allow selective reporting of favorable periods.
    arrived_ids = sorted(actual)
    if arrived_ids != [r["draw_id"] for r in planned[:len(arrived_ids)]]:
        raise ValueError("non-contiguous future outcome prefix; cannot skip earlier draws")
    fair = exact_fair_probability()
    records=[]
    stats={name:{"observed":0,"awarded_draws":0,"awarded_tickets":0}
           for name in EXPECTED_STRATEGIES}
    for draw in planned:
        record={
            "draw_id":draw["draw_id"],
            "date":draw["date"],
            "status":"PENDING" if draw["draw_id"] not in actual else "OBSERVED",
        }
        if draw["draw_id"] in actual:
            future=actual[draw["draw_id"]]
            record["actual_main"]=future["main"]
            record["actual_bonus"]=future["bonus"]
            evaluations={}
            for name in EXPECTED_STRATEGIES:
                bets=manifest["strategies"][name]["tickets"]
                info=evaluate_draw(bets,tuple(future["main"]),future["bonus"])
                evaluations[name]=info
                stats[name]["observed"]+=1
                stats[name]["awarded_draws"]+=info["any_award"]
                stats[name]["awarded_tickets"]+=info["awards"]
            record["evaluations"]=evaluations
        records.append(record)
    count=len(actual)
    for stat in stats.values():
        stat["iid_expected_any_prize_draws"]=float(fair*count)
        stat["observed_any_prize_rate"]=stat["awarded_draws"]/count if count else None
    # Do not present inferential p-values until the exact ten precommitted
    # draws have all been reported; this avoids retrospective early stopping.
    final_tests=None
    adjusted=None
    if count == len(planned):
        final_tests={
            name:exact_binomial_tail(count,stats[name]["awarded_draws"],fair)
            for name in EXPECTED_STRATEGIES
        }
        adjusted=min(
            1.,2*min(final_tests[name]["upper_tail_probability"]
                     for name in PRIMARY_HYPOTHESES)
        )
    return {
        "schema":REPORT_SCHEMA,
        "decision":(
            "PROSPECTIVE_STATIC10_COMPLETE; PREDICTIVE_NO_GO_OR_INCONCLUSIVE"
            if count==len(planned) else
            "PENDING_FUTURE_DRAWS; DESCRIPTIVE_ONLY_NO_INFERENCE"
        ),
        "source":manifest["historical_cutoff"],
        "frozen_seal_provenance":provenance,
        "precommitted_policies":list(EXPECTED_STRATEGIES),
        "fixed_draws_total":len(planned),
        "observed_draws":count,
        "pending_draws":len(planned)-count,
        "earliest_missing_draw":next(
            (r["draw_id"] for r in records if r["status"]=="PENDING"),None
        ),
        "fair_IID_probability_exact_rational":[str(fair.numerator),str(fair.denominator)],
        "fair_IID_probability":float(fair),
        "status_per_draw":records,
        "strategy_statistics":stats,
        "final_exact_one_sided_binomial_tests":final_tests,
        "two_P2_policy_bonferroni_upper_tail_p":adjusted,
        "test_scope":[
            "Both P2 four-ticket selections are STATIC fixed 2026-11-06 model results repeated over ten future draw dates; not rolling re-selection.",
            "Missing future outcomes remain PENDING and count neither as losses nor wins.",
            "Only two prespecified P2 variants enter two-policy Bonferroni; the random control is descriptive.",
            "Ten observations have low power and model family was historically explored; no predictive GO based on this experiment alone.",
            "CSV was not independently cross-checked against the lottery operator; obtain independent official verification before relying on graded outcomes.",
            "No user-purchased ticket is stored, altered or purchased in this public shadow test.",
        ],
    }


def markdown_report(report: dict) -> str:
    s=report
    lines=[
        "# v0.27 P3f — Prospectively Sealed Static Ten-Draw Shadow Test","",
        f"- **Pre-draw seal commit:** {SEAL_COMMIT}",
        f"- **Seal Git blob SHA1:** {PINNED_GIT_BLOB_SHA1}",
        f"- **Historical input SHA256:** {EXPECTED_HISTORY_SHA256}",
        f"- **State:** {s['decision']}",
        f"- **Observed:** {s['observed_draws']}/{s['fixed_draws_total']}; pending {s['pending_draws']}",
        f"- **Fair IID P(any award) per draw:** {s['fair_IID_probability']:.12%}",
        "",
        "## Prespecified shadow portfolio results","",
        "| Strategy | Observed draws | Any official award | Fair expected awards |",
        "|---|---:|---:|---:|",
    ]
    for k,row in s["strategy_statistics"].items():
        lines.append(
            f"| {k} | {row['observed']} | {row['awarded_draws']} | "
            f"{row['iid_expected_any_prize_draws']:.6f} |")
    lines.extend(["","## Preregistered ten future draws","",
                  "| Draw | Date | State | Primary | Diverse | Random |",
                  "|---:|---|---|---:|---:|---:|"])
    for item in s["status_per_draw"]:
        ev=item.get("evaluations",{})
        def winner(k):
            return str(ev[k]["any_award"]) if k in ev else "PENDING"
        lines.append(
            f"| {item['draw_id']} | {item['date']} | {item['status']} | "
            f"{winner('original_p2_primary')} | "
            f"{winner('original_p2_distinct_sums')} | "
            f"{winner('history_blind_random_control')} |")
    if s["final_exact_one_sided_binomial_tests"] is not None:
        lines.extend([
            "", "## Final prespecified inference (only after all ten draws)",""
        ])
        for k,row in s["final_exact_one_sided_binomial_tests"].items():
            lines.append(
                f"- **{k}:** exact Binomial(10,p) upper-tail "
                f"p={row['upper_tail_probability']:.9f}")
        lines.append(
            f"- **Two-policy Bonferroni:** p={s['two_P2_policy_bonferroni_upper_tail_p']:.9f}")
    else:
        lines.extend(["","**Statistical test is NOT evaluated before all ten observations.**"])
    lines.extend(["", "## Interpretation",""])
    lines.extend("- "+note for note in s["test_scope"])
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal",type=Path,default=SEAL)
    parser.add_argument("--csv-dir",type=Path,default=Path("data/csv"))
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--md-out",type=Path,required=True)
    args=parser.parse_args(argv)
    if (args.out.resolve()==args.md_out.resolve() or
            args.out.exists() or args.md_out.exists()):
        raise FileExistsError("two distinct unused JSON and Markdown output paths required")
    manifest,provenance=load_frozen_seal(args.seal)
    future=read_verified_history(args.csv_dir,manifest)
    report=evaluate_frozen_future(manifest,future,provenance)
    report["prospective_csv_observations_sha256"]=sha256(
        json.dumps(future,sort_keys=True,separators=(",",":")).encode("ascii")).hexdigest()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.md_out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write("\n")
    args.md_out.write_text(markdown_report(report),encoding="utf-8")
    print(f"[P3F SEAL PASS] commit={SEAL_COMMIT}; blob={PINNED_GIT_BLOB_SHA1}")
    print(f"[P3F STATUS] {report['observed_draws']}/{report['fixed_draws_total']} observed; {report['pending_draws']} pending")
    for name,values in report["strategy_statistics"].items():
        print(f"[{name.upper()}] awarded_draws={values['awarded_draws']}; IID_expected={values['iid_expected_any_prize_draws']:.4f}")
    if report["final_exact_one_sided_binomial_tests"] is None:
        print("[P3F INFERENCE] PENDING; do not treat missing outcomes as losing bets")
    else:
        print(f"[P3F FINAL INFERENCE] two-policy adjusted p={report['two_P2_policy_bonferroni_upper_tail_p']:.9f}")
    print(f"[REPORT] {args.out} / {args.md_out}")
    print("[SCOPE] Static ten-draw shadow only; no paid bets, no rolling re-selection, no predictive GO")


if __name__=="__main__":
    main()
