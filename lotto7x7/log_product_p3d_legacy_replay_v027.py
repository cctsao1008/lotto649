"""v0.27 Issue #15 P3d: strict historical walk-forward of ORIGINAL P2 #18.

This module calls P2 #18's actual optimize() without changing its reservoir,
constructive witness, fallback, certification, seed or search budget. Every
input report is rebuilt from historical draws ending at t-1. The result is
evaluated against target draw t only AFTER both four-ticket portfolios were
committed. It audits formal official award tiers and validates source hashes.

Pilot deliberately fixes three widely spaced target indices [300,1388,2476]
on the sealed #115000103/2026-11-06 dataset. The sample size is too small
to substantiate predictive performance. No IID replay of this expensive
original exact reservoir implementation occurs here; mathematical fair-null
odds were proven in P3c. P3 family-wide calibration remains open.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from random import Random

from .deficit_compensation_v025 import REFERENCE
from .distribution_repair_v027 import baseline, _prefix_sha256
from .distribution_repair_scoring_v027 import score_p0
from .distribution_repair_portfolio_v027 import validate_four_disjoint
from .log_product_v027 import exact_log_theory, historical_baseline
from .log_product_dual_scoring_v027 import dual_score
from .log_product_dual_portfolio_v027 import optimize as original_p2_optimize
from .log_product_joint_p3_v027 import _history_prefix
from .log_product_p3b_pilot_v027 import _bonus_prefix, evaluate_draw
from .log_product_p3c_exact_iid_v027 import exact_binomial_tail, exact_fair_probability
from .official_prize_v026 import official_prize_coverage

SCHEMA = "lotto7x7.log_product.v0.27.p3d_legacy_replay"
FINAL_CUTOFF = "2026-11-06"
FROZEN_HASH = "d865e1d241b607488e58a00b22fb595734f804595afd9d8850718808d6e8b7f8"
FROZEN_ID = 115000103
TARGET_INDICES = (300, 1388, 2476)
CONFIG = {
    "seed": 649,
    "pool_per_cell": 800,
    "cell_limit": 24,
    "restarts": 512,
    "fallback_trials": 15000,
}


def validate_indices(indices, n):
    if (not isinstance(indices, (tuple, list)) or not indices or
            type(n) is not int or
            any(type(v) is not int for v in indices) or
            sorted(set(indices)) != list(indices) or
            any(v < 3 or v >= n for v in indices)):
        raise ValueError("strictly increasing unique earlier-history indices required")
    return tuple(indices)


def _date_from_yyyymmdd(raw):
    return datetime.strptime(str(int(raw)), "%Y%m%d").date().isoformat()


def _random_control(t: int):
    """Same fixed-seed four-disjoint random baseline as cached P3b."""
    rng = Random(CONFIG["seed"]+99991*t+17)
    selected = rng.sample(range(1,50),24)
    return validate_four_disjoint(tuple(sorted(
        tuple(sorted(selected[i*6:i*6+6])) for i in range(4))))


def _audit_selected(variant: dict, main, bonus: int, *, distinct: bool):
    """Score only after original optimize() returned the sealed tickets."""
    tickets = variant["tickets"]
    if not tickets:
        if variant["status"] not in ("INFEASIBLE_UNCERTIFIED",):
            raise AssertionError("missing tickets without explicit search failure")
        return {
            "status": variant["status"],
            "certified_global_optimum": False,
            "tickets": [],
            "selection_failure": True,
            "outcome": None,
        }
    selected=validate_four_disjoint(tickets)
    if distinct and len({sum(t) for t in selected}) !=4:
        raise AssertionError("original four-distinct-sums P2 constraint violated")
    if not all(x["dual_positive"] for x in variant["ticket_details"]):
        raise AssertionError("original P2 selected a ticket without strict dual-positive gains")
    if variant["status"] not in ("HEURISTIC_UNCERTIFIED","GLOBAL_OPTIMUM_CERTIFIED"):
        raise AssertionError("invalid P2 certification state")
    if not variant["coverage"]["exact_official_eight_tier_parity"]:
        raise AssertionError("original P2 exact prize coverage regression")
    fresh=official_prize_coverage(selected)
    if fresh["any_prize_outcomes"] != official_prize_coverage(REFERENCE)["any_prize_outcomes"]:
        raise AssertionError("source code official-prize invariant unexpectedly violated")
    evaluation=evaluate_draw(selected,main,bonus)
    return {
        "status":variant["status"],
        "certified_global_optimum":variant["certified_global_optimum"],
        "tickets":[list(t) for t in selected],
        "selection_failure":False,
        "sum_gain_total":variant["sum_gain_total"],
        "log_gain_total":variant["log_gain_total"],
        "objective_exact_integer":variant["objective_exact_integer"],
        "upper_bound_exact_integer":variant["upper_bound_exact_integer"],
        "outcome":evaluation,
    }


def _compare_cached(pilot:dict|None, t:int, main, bonus, original:dict):
    if pilot is None:
        return None
    records=pilot["historical"]["records"]
    lookup={x["target_index_zero_based"]:x for x in records}
    if t not in lookup:
        raise ValueError(f"cached P3b has no record for predefined checkpoint index {t}")
    row=lookup[t]
    if (row["training_count"]!=t or row["target_main"]!=list(main)
            or row["target_bonus"]!=bonus):
        raise ValueError("cached P3b held-out main/bonus outcome differs from original replay")
    result={}
    for orig_key,pilot_key in (("primary","primary"),
                               ("distinct_sums","distinct_sums")):
        old=original[orig_key]
        cached=row["decisions"][pilot_key]
        if old["selection_failure"]:
            result[orig_key]={
                "same_ticket_portfolio":False,
                "original_failure":True,
                "cached_failure":"tickets" not in cached,
            }
        elif "tickets" in cached:
            cached_tickets=validate_four_disjoint(cached["tickets"])
            reproduced=evaluate_draw(cached_tickets,main,bonus)
            if reproduced != {k:cached[k] for k in ("any_award","awards","tiers","max_main_hits")}:
                raise ValueError("cached P3b prize audit did not reproduce")
            result[orig_key]={
                "same_ticket_portfolio": old["tickets"]==[list(tk) for tk in cached_tickets],
                "original_any_award":old["outcome"]["any_award"],
                "cached_any_award":cached["any_award"],
                "cached_tickets":[list(tk) for tk in cached_tickets],
            }
        else:
            result[orig_key]={"same_ticket_portfolio":False,
                              "cached_failure":True}
    return result


def replay(csv_dir:Path, *, target_indices=TARGET_INDICES,
           cache_pilot:dict|None=None)->dict:
    """Call the unchanged original P2 optimizer on strict prior-history input."""
    csv_dir=Path(csv_dir)
    full=baseline(csv_dir,cutoff=FINAL_CUTOFF)
    source=full["source"]
    if (source["analysis_input_sha256"]!=FROZEN_HASH
            or source["last_draw_id"]!=FROZEN_ID
            or source["historical_draws"]!=2497):
        raise ValueError("immutable original P2 reference history fingerprint mismatch")
    data,n=_history_prefix(csv_dir,source)
    indexes=validate_indices(target_indices,n)
    bonus,bonus_hash=_bonus_prefix(csv_dir,data,n)
    if cache_pilot is not None:
        psrc=cache_pilot.get("source",{})
        if (psrc.get("analysis_input_sha256")!=FROZEN_HASH or
                psrc.get("historical_draws")!=n or
                cache_pilot.get("policy",{}).get("policy")!="cached_joint_cell_greedy_v1"):
            raise ValueError("cached P3b pilot and sealed original reference mismatch")
    theory=exact_log_theory()
    observations=[]
    for t in indexes:
        previous_day=_date_from_yyyymmdd(data.dates[t-1])
        target_day=_date_from_yyyymmdd(data.dates[t])
        target=tuple(int(v) for v in data.numbers[t])
        target_bonus=bonus[t]
        # P0/P1 models accept ONLY a strict t-row cutoff; their hashes
        # must exactly match the independent canonical t-row prefix digest.
        p0=baseline(csv_dir,cutoff=previous_day)
        prior_hash=_prefix_sha256(data.ids[:t],data.dates[:t],data.numbers[:t])
        if (p0["source"]["historical_draws"]!=t or
                p0["source"]["analysis_input_sha256"]!=prior_hash or
                p0["source"]["last_draw_id"]!=int(data.ids[t-1]) or
                p0["source"]["post_cutoff_rows_excluded"]<1):
            raise AssertionError("historical prior-only cutoff/source isolation broken")
        s1=score_p0(p0)
        l0=historical_baseline(csv_dir,cutoff=previous_day,
                               expected_prefix_sha256=prior_hash,theory=theory)
        d1=dual_score(s1,l0)
        # Nothing about the unseen target or its bonus reaches optimize().
        output=original_p2_optimize(s1,l0,d1,**CONFIG)
        if output.get("schema")!="lotto7x7.log_product.v0.27.p2":
            raise ValueError("not an original P2 optimize() output")
        if output.get("source",{}).get("analysis_input_sha256")!=prior_hash:
            raise ValueError("original P2 optimized on the wrong historical prefix")
        portfolios=output.get("portfolios",{})
        original={
            "primary":_audit_selected(portfolios["primary"],target,target_bonus,
                                      distinct=False),
            "distinct_sums":_audit_selected(portfolios["distinct_sum_sensitivity"],
                                            target,target_bonus,distinct=True),
        }
        random_choice=_random_control(t)
        random_evaluation=evaluate_draw(random_choice,target,target_bonus)
        observations.append({
            "target_index_zero_based":t,
            "previous_draw_id":int(data.ids[t-1]),
            "previous_draw_date":previous_day,
            "target_draw_id":int(data.ids[t]),
            "target_draw_date":target_day,
            "training_count":t,
            "source_prefix_sha256":prior_hash,
            "original_P2_schema":output["schema"],
            "original_P2_config":CONFIG,
            "target_main":list(target),
            "target_bonus":target_bonus,
            "original_P2_portfolios":original,
            "random_control":{"tickets":[list(x) for x in random_choice],
                              "outcome":random_evaluation},
            "cached_P3b_comparison":_compare_cached(
                cache_pilot,t,target,target_bonus,original),
        })
    fair=exact_fair_probability()
    statistics={}
    for name in ("primary","distinct_sums","random"):
        summary={"selected":0,"selection_failures":0,"prize_events":0}
        for row in observations:
            entry=(row["random_control"] if name=="random" else
                   row["original_P2_portfolios"][name])
            if entry.get("selection_failure",False):
                summary["selection_failures"]+=1
            else:
                summary["selected"]+=1
                summary["prize_events"]+=entry["outcome"]["any_award"]
        if summary["selected"]+summary["selection_failures"]!=len(indexes):
            raise AssertionError("original P2 observation counts did not close")
        if summary["selection_failures"]==0:
            summary["exact_binomial_test"]=exact_binomial_tail(
                len(indexes),summary["prize_events"],fair)
        else:
            summary["exact_binomial_test"]=None
        statistics[name]=summary
    # Two original P2 variants: only if no outcome-dependent selection
    # failures can the fixed-length binomial Bonferroni gate apply.
    if all(statistics[k]["selection_failures"]==0
           for k in ("primary","distinct_sums")):
        pvals=[
            statistics[k]["exact_binomial_test"]["upper_tail_probability"]
            for k in ("primary","distinct_sums")
        ]
        family_p=min(1.,2*min(pvals))
        decision="DESCRIPTIVE_LEGACY_P2_EXACT_REPLAY_PASS; PREDICTIVE_INCONCLUSIVE"
    else:
        family_p=None
        decision="LEGACY_P2_ADAPTIVE_SELECTION_FAILURE; PREDICTIVE_INCONCLUSIVE"
    return {
        "schema":SCHEMA,
        "stage":"P3D_ORIGINAL_P2_CODE_STRATIFIED_CHRONOLOGICAL_REPLAY",
        "decision":decision,
        "source":source,
        "bonus_prefix_sha256":bonus_hash,
        "actual_original_optimizer":"lotto7x7.log_product_dual_portfolio_v027.optimize",
        "optimizer_parameters":CONFIG,
        "target_indices":list(indexes),
        "checkpoint_count":len(indexes),
        "empirical_results":statistics,
        "two_policy_bonferroni_upper_tail_p":family_p,
        "exact_fair_any_award_probability":float(fair),
        "exact_fair_any_award_rational":[str(fair.numerator),str(fair.denominator)],
        "individual_replays":observations,
        "scope_warnings":[
            "The actual PR #18 P2 optimize() is called unchanged at each checkpoint with its ORIGINAL seed and reservoir policy.",
            "Three prespecified early/mid/late checkpoints are an engineering pilot, not a 35-checkpoint or all-draw backtest.",
            "Original P2 was retrospectively developed using the larger dataset; this historical replay is not a prospectively preregistered test.",
            "P3c fair-IID exact null applies independently of optimizer internals and does not imply real-world predictive advantage.",
            "Full P3 family-wide IID calibration, broader IID replay, and genuinely sealed future outcomes remain open.",
            "No existing ten-draw purchased ticket is altered.",
        ],
    }


def markdown_report(report):
    lines=[
        "# Lotto649 v0.27 Issue #15 — P3d Original P2 Strict Historical Replay","",
        f"- **Frozen input:** {report['source']['historical_draws']} through {report['source']['cutoff']}, #{report['source']['last_draw_id']}",
        f"- **Prefix SHA256:** `{report['source']['analysis_input_sha256']}`",
        f"- **Bonus prefix SHA256:** `{report['bonus_prefix_sha256']}`",
        f"- **Unmodified selector:** `{report['actual_original_optimizer']}`",
        f"- **Exact original config:** `{json.dumps(report['optimizer_parameters'],sort_keys=True)}`",
        f"- **Fixed earlier-history indices:** {', '.join(map(str,report['target_indices']))}",
        "", "| Strategy | Evaluated | Selection failures | Awarded targets | Exact IID E[hits] | Exact single p |",
        "|:---|---:|---:|---:|---:|---:|"
    ]
    for name in ("primary","distinct_sums","random"):
        item=report["empirical_results"][name]
        test=item["exact_binomial_test"]
        lines.append(
            f"| {name} | {item['selected']} | {item['selection_failures']} | "
            f"{item['prize_events']} | "
            f"{format(test['exact_fair_expected_successes'], '.4f') if test else 'N/A'} | "
            f"{format(test['upper_tail_probability'], '.6f') if test else 'N/A'} |"
        )
    lines.extend(["",
        f"- **Two-policy Bonferroni upper-tail p:** "
        f"{report['two_policy_bonferroni_upper_tail_p'] if report['two_policy_bonferroni_upper_tail_p'] is not None else 'INCONCLUSIVE'}",
        "", "## Original P2 choices and held-out results","",
    ])
    for record in report["individual_replays"]:
        lines.extend([
            f"### Next draw #{record['target_draw_id']} on {record['target_draw_date']} (prior N={record['training_count']})","",
            f"- **Historical cutoff:** {record['previous_draw_date']} / #{record['previous_draw_id']}",
            f"- **Training SHA256:** `{record['source_prefix_sha256']}`",
            f"- **Actual next main/bonus:** {' '.join(f'{x:02d}' for x in record['target_main'])} + special {record['target_bonus']:02d}","",
        ])
        for label,entry in record["original_P2_portfolios"].items():
            lines.append(f"**{label}:** {entry['status']}")
            if not entry["selection_failure"]:
                for ticket in entry["tickets"]:
                    lines.append("- "+" ".join(f"{n:02d}" for n in ticket))
                lines.append(f"- **Award event:** {entry['outcome']['any_award']}; tiers {entry['outcome']['tiers']}")
                if record["cached_P3b_comparison"] is not None:
                    lines.append(f"- **Identical cached-P3b portfolio?** {record['cached_P3b_comparison'][label]['same_ticket_portfolio']}")
            lines.append("")
    lines.extend([
        "## Limitations","",
        "This is the literal unchanged original P2 optimizer at three frozen prior-only historical prefixes. "
        "It is a small retrospective integration audit, not evidence of predictive superiority. "
        "Exact IID symmetry still gives the same fair odds to every 24-disjoint portfolio.",
        "Full all-checkpoint legacy replay, broader model-family corrections and sealed future evaluation remain open.","",
        "## Reproduction","", "```powershell",
        "& .\\.venv\\Scripts\\python.exe -m lotto7x7.log_product_p3d_legacy_replay_v027 "
        "--csv-dir data/csv --out results/v027_p3d_legacy.json --md-out results/v027_p3d_legacy.md",
        "```",""
    ])
    return "\n".join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir",type=Path,default=Path("data/csv"))
    parser.add_argument("--out",type=Path,default=Path("results/v027_p3d_legacy.json"))
    parser.add_argument("--md-out",type=Path,default=None)
    parser.add_argument("--cached-p3b",type=Path,default=None)
    args=parser.parse_args(argv)
    out,md=args.out,args.md_out or args.out.with_suffix(".md")
    if out.resolve()==md.resolve() or out.exists() or md.exists():
        raise FileExistsError("two distinct unused report paths required")
    raw=args.cached_p3b.read_bytes() if args.cached_p3b else None
    cached=json.loads(raw) if raw else None
    result=replay(args.csv_dir,cache_pilot=cached)
    if raw:
        result["cached_p3b_report_sha256"]=sha256(raw).hexdigest()
    out.parent.mkdir(parents=True,exist_ok=True)
    md.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x",encoding="utf-8") as fp:
        json.dump(result,fp,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)
        fp.write("\n")
    md.write_text(markdown_report(result),encoding="utf-8")
    print(f"[P3D LEGACY PASS] original P2 optimize() executed unchanged at {result['checkpoint_count']} sealed checkpoints")
    for row in result["individual_replays"]:
        print(f"[CHECKPOINT] t={row['target_index_zero_based']} next=#{row['target_draw_id']} target_date={row['target_draw_date']}")
        for label,detail in row["original_P2_portfolios"].items():
            outcome=detail["outcome"]
            print(f"  {label}: {detail['status']}, awarded={outcome['any_award'] if outcome else 'SKIPPED'}")
    for name,data in result["empirical_results"].items():
        print(f"[{name.upper()}] wins={data['prize_events']}, selected={data['selected']}, failures={data['selection_failures']}")
    print(f"[FAMILYWISE] exact 2-policy Bonferroni p={result['two_policy_bonferroni_upper_tail_p']}")
    print(f"[REPORT] {out} / {md}")
    print("[SCOPE] Original #18 reservoir used, small retrospective 3-target audit only; not predictive GO")


if __name__=="__main__":
    main()
