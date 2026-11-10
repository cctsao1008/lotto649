"""v0.27 Issue #15 P3a: exact joint S x LogProduct baseline and prequential IID event check.

Scope: Exploratory, leakage-safe chronological diagnostic of the PRESPECIFIED
dual-positive single-draw event. All theory bins and distance metrics come
from P0/P1 and are not tuned on later draws. Under a fair IID next draw the
conditional event probability is the EXACT mass of the currently positive
joint cells (not the product of marginal probabilities).

This is NOT a full recalibration of the four-ticket selection/optimizer:
that expensive P3b task remains explicitly incomplete. Do not infer betting
advantage or future-draw self-correction.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from datetime import date
from hashlib import sha256
from itertools import combinations
import json
from math import comb, exp, fsum, log, prod, sqrt
from pathlib import Path

from .distribution_repair_v027 import (
    LOW, TOTAL, _prefix_sha256, squared_l2_probability,
)
from .log_product_dual_scoring_v027 import exact_gain_numerators
from .log_product_dual_portfolio_v027 import _check_inputs, enumerate_joint_counts

SCHEMA = "lotto7x7.log_product.v0.27.p3a"


def joint_exact_reference(sum_p1: dict, log_p0: dict, dual_p1: dict) -> dict:
    """Exact fair 259×34 reference and integrity of both marginals."""
    sum_rows, log_theory, sum_gains, log_gains = _check_inputs(sum_p1,log_p0,dual_p1)
    grid = log_theory["grid"]
    joint, m = enumerate_joint_counts(thresholds=grid["integer_product_thresholds"])
    if m != comb(49,6):
        raise AssertionError("wrong theoretical joint mass")
    sum_counts = [sum(row) for row in joint]
    log_counts = [sum(row[i] for row in joint) for i in range(grid["bin_count"])]
    if sum_counts != [row["theory_combinations"] for row in sum_rows]:
        raise AssertionError("joint theoretical sum marginals differ from frozen theory")
    if log_counts != log_theory["counts"]:
        raise AssertionError("joint theoretical log marginals differ from frozen theory")
    fingerprint = sha256(json.dumps(joint,separators=(",",":")).encode("ascii")).hexdigest()
    return {
        "joint": joint, "joint_sha256": fingerprint,
        "nonzero_cells": sum(v>0 for row in joint for v in row),
        "sum_counts": sum_counts, "log_counts": log_counts,
        "theory": log_theory, "sum_gain_numerators": sum_gains["numerators"],
        "log_gain_numerators": log_gains["numerators"],
    }


def exact_event_mass(sum_positive: list[bool], log_positive: list[bool],
                     joint: list[list[int]]) -> dict:
    """Exact conditional IID mass for the intersected feature-sign event."""
    if (len(joint)!=len(sum_positive) or not joint or
            any(len(row)!=len(log_positive) for row in joint)
            or not log_positive or
            any(type(x) is not bool for x in sum_positive+log_positive)
            or any(type(v) is not int or v<0 for row in joint for v in row)):
        raise ValueError("expected nonnegative integer joint table and Boolean sign masks")
    marginal_sum = [sum(row) for row in joint]
    marginal_log = [sum(row[i] for row in joint) for i in range(len(log_positive))]
    total = sum(marginal_sum)
    if total<=0:
        raise ValueError("empty theoretical joint mass")
    n_sum = sum(v for i,v in enumerate(marginal_sum) if sum_positive[i])
    n_log = sum(v for i,v in enumerate(marginal_log) if log_positive[i])
    n_both = sum(v for i,row in enumerate(joint) if sum_positive[i]
                 for j,v in enumerate(row) if log_positive[j])
    return {
        "sum_positive_mass": n_sum,
        "log_positive_mass": n_log,
        "dual_positive_mass": n_both,
        "sum_positive_probability": n_sum/total,
        "log_positive_probability": n_log/total,
        "dual_positive_probability": n_both/total,
        "independence_product_not_valid": (n_sum/total)*(n_log/total),
        "dependence_difference_from_independence": n_both/total-(n_sum/total)*(n_log/total),
    }


def _theory_pearson_sum_log() -> dict:
    """Finite-population covariance for sums of labels vs sums of log labels."""
    from .log_product_v027 import theoretical_log_moments
    k,n=6,49
    xbar=25
    lbar=fsum(log(v) for v in range(1,n+1))/n
    pop_cov=fsum((v-xbar)*(log(v)-lbar) for v in range(1,n+1))/n
    cov=k*(n-k)/(n-1)*pop_cov
    var_s=1075.
    var_l=theoretical_log_moments()["variance"]
    corr=cov/sqrt(var_s*var_l)
    return {
        "sum_log_covariance": cov,
        "sum_log_pearson_correlation": corr,
        "method": "independent finite-population-without-replacement covariance identities using unbinned ln(integer product)",
        "note": "Correlation uses unbinned L, unlike the binned joint histogram.",
    }


def _history_prefix(csv_dir: Path, source: dict):
    from .temporal_dataset_v024 import load_csv_history
    data=load_csv_history(Path(csv_dir))
    n=source["historical_draws"]
    if type(n) is not int or n < 3 or len(data.ids)<n or data.dates is None:
        raise ValueError("historical prefix missing or too short")
    cutoff=int(date.fromisoformat(source["cutoff"]).strftime("%Y%m%d"))
    if int(data.dates[n-1])!=cutoff or int(data.ids[n-1])!=source["last_draw_id"]:
        raise ValueError("historical cutoff does not equal frozen P0-P2 prefix")
    digest=_prefix_sha256(data.ids[:n],data.dates[:n],data.numbers[:n])
    if digest!=source["analysis_input_sha256"]:
        raise ValueError("historical prefix SHA256 differs from frozen research")
    return data,n


def _observed_features(data, n: int, thresholds):
    features=[]
    for ticket in data.numbers[:n]:
        seq=tuple(int(v) for v in ticket)
        idx=bisect_right(thresholds,prod(seq))-1
        if not 0<=idx<len(thresholds)-1:
            raise ValueError("historical product outside theory grid")
        features.append((sum(seq)-LOW,idx,log(prod(seq))))
    return features


def prequential_test(features: list[tuple[int,int,float]],joint,
                     *, warmup: int=300) -> dict:
    """Chronological conditional fair-IID test for a FIXED single-draw event.

    At each t, evaluate the realized draw *after* creating both histograms
    from strictly earlier observed tickets, then update counts. The fair
    conditional null has q_t = exact currently-qualifying joint mass/M.

    The output is descriptive; reported z-score is a martingale normal
    approximation, while Hoeffding-Azuma supplies a conservative
    one-sided bound for the explicitly fixed sequential event.
    It does not cover portfolio selection, model-search or retrospective
    choice of event/metric.
    """
    n=len(features)
    ns, nl=len(joint),len(joint[0])
    if (type(warmup) is not int or warmup < 3 or warmup >= n
            or any(not 0<=si<ns or not 0<=li<nl for si,li,_ in features)):
        raise ValueError("prequential features/warmup invalid")
    fair_sum=[sum(row) for row in joint]
    fair_log=[sum(row[j] for row in joint) for j in range(nl)]
    total=sum(fair_sum)
    if total < 1:
        raise ValueError("empty joint theory")
    hist_sum=[0]*ns
    hist_log=[0]*nl
    for si,li,_ in features[:warmup]:
        hist_sum[si]+=1
        hist_log[li]+=1
    observations=[]
    for t in range(warmup,n):
        gain_s=exact_gain_numerators(hist_sum,fair_sum)
        gain_l=exact_gain_numerators(hist_log,fair_log)
        mask_s=[v>0 for v in gain_s["numerators"]]
        mask_l=[v>0 for v in gain_l["numerators"]]
        mass=exact_event_mass(mask_s,mask_l,joint)
        si,li,_=features[t]
        observed=int(mask_s[si] and mask_l[li])
        observations.append({
            "next_index_zero_based":t,
            "prior_history_draws":t,
            "realized_dual_positive":observed,
            "conditional_fair_probability":mass["dual_positive_probability"],
            "conditional_fair_mass":mass["dual_positive_mass"],
            "fair_probability_independence_product_wrong":mass["independence_product_not_valid"],
            "realized_sum_positive":bool(mask_s[si]),
            "realized_log_positive":bool(mask_l[li]),
        })
        hist_sum[si]+=1
        hist_log[li]+=1
    q=[row["conditional_fair_probability"] for row in observations]
    count=sum(row["realized_dual_positive"] for row in observations)
    expected=fsum(q)
    variance=fsum(p*(1-p) for p in q)
    diff=count-expected
    z=diff/sqrt(variance) if variance>0 else None
    length=len(observations)
    # Conditional Bernoulli with adaptive probabilities is a bounded
    # martingale difference; Hoeffding's lemma gives a conservative
    # pre-specified one-sided upper tail <= exp(-2*max(diff,0)^2 / T).
    upper_bound=exp(-2*max(diff,0)**2/length)
    lower_bound=exp(-2*max(-diff,0)**2/length)
    return {
        "definition": "fixed dual-positive single-next-draw repair event evaluated against exact fair conditional probabilities",
        "warmup":warmup, "evaluated_draws":length,
        "realized_positive_count":count,
        "expected_positive_count":expected,
        "observed_positive_rate":count/length,
        "IID_expected_average_positive_rate":expected/length,
        "observed_minus_conditional_IID_expected":diff,
        "predictable_variance":variance,
        "martingale_z_approximation":z,
        "hoeffding_azuma_upper_tail_bound":upper_bound,
        "hoeffding_azuma_lower_tail_bound":lower_bound,
        "interpretation": "exploratory walk-forward test, NOT selection-pipeline IID replication or prospective proof",
        "records":observations,
    }


def analyze(sum_p1:dict,log_p0:dict,dual_p1:dict,csv_dir:Path,
            *,warmup:int=300)->dict:
    reference=joint_exact_reference(sum_p1,log_p0,dual_p1)
    data,n=_history_prefix(csv_dir,dual_p1["source"])
    grid=reference["theory"]["grid"]
    features=_observed_features(data,n,grid["integer_product_thresholds"])
    joint=reference["joint"]
    ns,nl=len(joint),len(joint[0])
    empirical=[[0]*nl for _ in range(ns)]
    for si,li,_ in features:
        empirical[si][li]+=1
    flat_h=[v for row in empirical for v in row]
    flat_f=[v for row in joint for v in row]
    joint_loss=squared_l2_probability(flat_h,flat_f)
    jgain=exact_gain_numerators(flat_h,flat_f)
    signsS=[v>0 for v in reference["sum_gain_numerators"]]
    signsL=[v>0 for v in reference["log_gain_numerators"]]
    exact_dual=exact_event_mass(signsS,signsL,joint)
    if exact_dual["dual_positive_mass"]!=dual_p1["qualifying"]["dual_positive_ticket_count"]:
        raise AssertionError("P3 frozen dual-positive IID mass differs from P1")
    jointpositive=0
    jointnegative=0
    jointzero=0
    for i,row in enumerate(joint):
        for j,value in enumerate(row):
            if value>0 and signsS[i] and signsL[j]:
                v=jgain["numerators"][i*nl+j]
                if v>0:jointpositive+=value
                elif v<0:jointnegative+=value
                else:jointzero+=value
    if jointpositive+jointnegative+jointzero!=exact_dual["dual_positive_mass"]:
        raise AssertionError("P3 joint repair partitions do not close")
    theoretical_pair=_theory_pearson_sum_log()
    svalues=[si+LOW for si,_,_ in features]
    lvalues=[v for _,_,v in features]
    ms=fsum(svalues)/n
    ml=fsum(lvalues)/n
    vs=fsum((v-ms)**2 for v in svalues)/n
    vl=fsum((v-ml)**2 for v in lvalues)/n
    cv=fsum((s-ms)*(l-ml) for s,l in zip(svalues,lvalues))/n
    empirical_corr=cv/sqrt(vs*vl) if vs>0 and vl>0 else None
    cells=[]
    for i in range(ns):
        for j in range(nl):
            p=joint[i][j]
            if p==0 and empirical[i][j]!=0:
                raise AssertionError("observed historical outcome in theory-impossible cell")
            if p:
                e=n*p/TOTAL
                cells.append({
                    "sum":i+LOW,"log_bin":j,
                    "exact_combinations":p,"observed_count":empirical[i][j],
                    "expected_count":e,
                    "residual_count":empirical[i][j]-e,
                    "joint_gain_numerator":str(jgain["numerators"][i*nl+j]),
                    "joint_gain":jgain["numerators"][i*nl+j]/jgain["denominator"],
                    "sum_positive":signsS[i],
                    "log_positive":signsL[j],
                    "dual_positive":signsS[i] and signsL[j],
                    "joint_positive":jgain["numerators"][i*nl+j]>0,
                })
    high=sorted(cells,key=lambda x:(-x["residual_count"],x["sum"],x["log_bin"]))[:15]
    low=sorted(cells,key=lambda x:(x["residual_count"],x["sum"],x["log_bin"]))[:15]
    preq=prequential_test(features,joint,warmup=warmup)
    return {
        "schema":SCHEMA,
        "stage":"P3A_JOINT_EXACT_THEORY_AND_PREQUENTIAL_DUAL_EVENT_AUDIT",
        "decision":"JOINT_DESCRIPTIVE_PASS; PREDICTIVE_NO_GO_OR_INCONCLUSIVE_PENDING_FULL_PIPELINE_IID",
        "source":dual_p1["source"],
        "theory_checksums":{
            "sum":sum_p1["theory"]["checksum_sha256"],
            "log":reference["theory"]["checksum_sha256"],
            "joint_counts":reference["joint_sha256"],
        },
        "joint":{
            "support_sums":[21,279], "log_bins":nl,
            "universe":TOTAL,"nonzero_joint_cells":reference["nonzero_cells"],
            "exact_theory_counts":joint,
            "historical_counts":empirical,
            "squared_l2_probability_v1":joint_loss,
            "zero_theory_cell_policy":"no divide by individual bin probability; count must be zero",
            "largest_surplus_cells":high,
            "largest_deficit_cells":low,
        },
        "correlation":{
            "theory_unbinned":theoretical_pair,
            "empirical_unbinned":{
                "mean_sum":ms,"mean_log":ml,
                "covariance":cv,"pearson_correlation":empirical_corr,
            },
        },
        "repairs":{
            "marginal_dual_positive_combinations":exact_dual["dual_positive_mass"],
            "conditional_IID_dual_positive_probability":exact_dual["dual_positive_probability"],
            "incorrect_independence_assumption_probability":exact_dual["independence_product_not_valid"],
            "dual_and_joint_positive_combinations":jointpositive,
            "dual_positive_but_joint_negative_combinations":jointnegative,
            "dual_positive_with_joint_zero_gain_combinations":jointzero,
            "joint_gain_metric":"squared_l2_probability_v1 applied to full 259x34 joint probabilities (exploratory separate metric, not used to select P2 bets)",
        },
        "prequential":preq,
        "warnings":[
            "The exact 2D joint histogram and conditional IID mass do not establish prediction.",
            "Historical walk-forward single-event test uses a frozen metric; it does not simulate the four-ticket optimizer.",
            "Selection/search over model families and portfolios still requires separate full-pipeline IID calibration.",
            "This is retrospective exploratory analysis; PREDICTIVE_GO is not asserted.",
        ],
    }


def markdown_report(report:dict)->str:
    s=report["source"];r=report["repairs"];p=report["prequential"]
    c=report["correlation"]
    lines=[
        "# v0.27 Issue #15 P3a — Exact Joint and Prequential IID Diagnostic","",
        f"- **Cutoff:** {s['cutoff']} / #{s['last_draw_id']} ({s['historical_draws']} draws)",
        f"- **Prefix SHA256:** `{s['analysis_input_sha256']}`",
        f"- **Joint theory SHA256:** `{report['theory_checksums']['joint_counts']}`",
        f"- **Nonzero fair joint cells:** {report['joint']['nonzero_joint_cells']}",
        f"- **Joint squared L2:** {report['joint']['squared_l2_probability_v1']:.12g}",
        f"- **Theory Pearson corr(S,L):** {c['theory_unbinned']['sum_log_pearson_correlation']:.8f}",
        f"- **Historical Pearson corr(S,L):** {c['empirical_unbinned']['pearson_correlation']:.8f}",
        "",
        "## Exact conditional IID overlap of repairs","",
        f"- **Dual-positive combinations:** {r['marginal_dual_positive_combinations']:,}",
        f"- **Dual-positive with joint improvement:** {r['dual_and_joint_positive_combinations']:,}",
        f"- **Dual-positive but joint worsening:** {r['dual_positive_but_joint_negative_combinations']:,}",
        f"- **Conditional IID dual-positive mass:** {r['conditional_IID_dual_positive_probability']:.9%}",
        f"- **Incorrect independence product:** {r['incorrect_independence_assumption_probability']:.9%}",
        "",
        "## Chronological single-draw dual-positive event test","",
        f"- **Warmup:** {p['warmup']} draws; evaluated={p['evaluated_draws']}",
        f"- **Observed dual-positive next-draw events:** {p['realized_positive_count']}",
        f"- **IID conditional expected events:** {p['expected_positive_count']:.6f}",
        f"- **Observed rate:** {p['observed_positive_rate']:.6%}",
        f"- **IID expected rate:** {p['IID_expected_average_positive_rate']:.6%}",
        f"- **Martingale difference:** {p['observed_minus_conditional_IID_expected']:+.6f}",
        f"- **Exploratory martingale z:** {p['martingale_z_approximation']:+.5f}",
        f"- **Conservative Hoeffding-Azuma one-sided upper tail bound:** {p['hoeffding_azuma_upper_tail_bound']:.8g}",
        "",
        "**Interpretation:** conditional IID q_t is exactly recomputed using only prior draws; no future data enters a test step. However retrospective choices and the four-ticket search are NOT covered by this one-event test.","",
        "## Largest theoretical joint-cell deficits","",
        "| Sum | Log bin | Observed | IID expected | Observed−Expected |",
        "|---:|---:|---:|---:|---:|",
    ]
    for x in report["joint"]["largest_deficit_cells"]:
        lines.append(f"| {x['sum']} | {x['log_bin']} | {x['observed_count']} | {x['expected_count']:.3f} | {x['residual_count']:+.3f} |")
    lines.extend([
        "", "## Research decision", "",
        "**JOINT_DESCRIPTIVE_PASS** if this exact joint baseline and walk-forward audit run. "
        "**PREDICTIVE_GO is NOT authorized** without sealed future data, replicated out-of-sample evidence, and full four-ticket selection-pipeline IID calibration.",
        "", "## Reproduce", "", "```powershell",
        "& .\\.venv\\Scripts\\python.exe -m lotto7x7.log_product_joint_p3_v027 "
        "--sum-p1 results/v027_p1_repair.json "
        "--log-p0 results/v027_log_product_p0.json "
        "--dual-p1 results/v027_dual_p1.json "
        "--csv-dir data/csv --cutoff 2026-11-06 --warmup 300 "
        "--out results/v027_joint_p3a.json --md-out results/v027_joint_p3a.md",
        "```",""
    ])
    return "\n".join(lines)


def main(argv=None)->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sum-p1",type=Path,required=True)
    parser.add_argument("--log-p0",type=Path,required=True)
    parser.add_argument("--dual-p1",type=Path,required=True)
    parser.add_argument("--csv-dir",type=Path,default=Path("data/csv"))
    parser.add_argument("--cutoff",default="2026-11-06")
    parser.add_argument("--warmup",type=int,default=300)
    parser.add_argument("--out",type=Path,default=Path("results/v027_joint_p3a.json"))
    parser.add_argument("--md-out",type=Path,default=None)
    args=parser.parse_args(argv)
    out,md=args.out,args.md_out or args.out.with_suffix(".md")
    if out.resolve()==md.resolve() or out.exists() or md.exists():
        raise FileExistsError("separate unused P3 JSON/Markdown output paths required")
    raw=[p.read_bytes() for p in (args.sum_p1,args.log_p0,args.dual_p1)]
    p1,l0,d1=[json.loads(x) for x in raw]
    if args.cutoff!=d1["source"]["cutoff"]:
        raise ValueError("CLI cutoff must exactly match frozen dual P1 cutoff")
    result=analyze(p1,l0,d1,args.csv_dir,warmup=args.warmup)
    result["input_report_sha256"]={k:sha256(v).hexdigest() for k,v in zip(
        ("sum_p1","log_p0","dual_p1"),raw)}
    out.parent.mkdir(parents=True,exist_ok=True)
    md.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)
        stream.write("\n")
    md.write_text(markdown_report(result),encoding="utf-8")
    p=result["prequential"];r=result["repairs"]
    print(f"[JOINT-P3A PASS] nonzero exact cells={result['joint']['nonzero_joint_cells']}, SHA256={result['theory_checksums']['joint_counts']}")
    print(f"[JOINT] L2={result['joint']['squared_l2_probability_v1']:.12g}; theory Pearson(S,L)={result['correlation']['theory_unbinned']['sum_log_pearson_correlation']:.8f}")
    print(f"[REPAIR] both-positive fair mass={r['marginal_dual_positive_combinations']:,}, joint-positive={r['dual_and_joint_positive_combinations']:,}, joint-negative={r['dual_positive_but_joint_negative_combinations']:,}")
    print(f"[PREQUENTIAL] {p['evaluated_draws']} draws; actual dual-positive={p['realized_positive_count']}; IID expected={p['expected_positive_count']:.6f}; z={p['martingale_z_approximation']:.4f}")
    print(f"[REPORT] {out} / {md}")
    print("[SCOPE] P3a exploratory fixed-event diagnostic only; complete portfolio-pipeline IID/P3b remains pending")


if __name__=="__main__":
    main()
