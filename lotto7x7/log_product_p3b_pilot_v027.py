"""v0.27 Issue #15 P3b PILOT: IID replay of a frozen fast four-ticket policy.

This is a *new explicitly versioned cached-catalogue implementation* of the
same P2 mathematical objective and hard constraints, not byte-identical to
P2's history-dependent reservoir construction. Both real and IID histories
use the identical fixed candidate catalogue, solver, checkpoint schedule,
and two strategies. No ahead-of-time outcomes are available to the solver.

A P3b pilot is NOT a full certification of legacy P2 #18, nor prospective
evidence. We calibrate the maximum of the two strategy excess-win scores
under whole-history IID simulation (not two unadjusted p values).
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
from hashlib import sha256
from itertools import combinations
import csv
import json
from math import comb, prod, sqrt
from pathlib import Path
from random import Random

from .deficit_compensation_v025 import REFERENCE
from .distribution_repair_portfolio_v027 import validate_four_disjoint
from .log_product_joint_p3_v027 import joint_exact_reference, _history_prefix
from .log_product_dual_portfolio_v027 import frozen_weights
from .log_product_dual_scoring_v027 import exact_gain_numerators
from .official_prize_v026 import official_prize_coverage, prize_tier

SCHEMA = "lotto7x7.log_product.v0.27.p3b_pilot"
POLICY = "cached_joint_cell_greedy_v1"
THEORY_SIZE = comb(49, 6)
LOW_SUM = 21


def _mask(ticket):
    return sum(1 << (value - 1) for value in ticket)


def catalogue(joint, thresholds, *, seed=649, per_cell=32):
    """One history-independent reservoir for EVERY nonzero exact joint cell.

    Each cell has a separate RNG seeded solely by frozen (seed, sum, log-bin)
    so the candidate pool does not depend on any chosen history or date.
    It is cached across both real and simulated history replays. This differs
    from the original P2 selected-cell reservoir, and is disclosed as such.
    """
    if (type(seed) is not int or seed<0 or type(per_cell) is not int
            or not 1<=per_cell<=256 or len(joint)!=259 or
            any(len(row)!=len(thresholds)-1 for row in joint)):
        raise ValueError("invalid frozen candidate catalogue specification")
    pools={}
    observed=defaultdict(int)
    rngs={}
    for i,row in enumerate(joint):
        for j,mass in enumerate(row):
            if mass:
                cell=(i+LOW_SUM,j)
                pools[cell]=[]
                rngs[cell]=Random(seed ^ ((i+LOW_SUM)<<12) ^ (j<<2))
    for ticket in combinations(range(1,50),6):
        cell=(sum(ticket),bisect_right(thresholds,prod(ticket))-1)
        if cell not in pools:
            raise AssertionError("legal ticket mapped outside exact joint support")
        observed[cell]+=1
        arr=pools[cell]
        entry=(_mask(ticket),ticket)
        if len(arr)<per_cell:
            arr.append(entry)
        else:
            pick=rngs[cell].randrange(observed[cell])
            if pick<per_cell:
                arr[pick]=entry
    if (sum(observed.values())!=THEORY_SIZE or
            any(observed[(i+LOW_SUM,j)]!=mass
                for i,row in enumerate(joint) for j,mass in enumerate(row) if mass)):
        raise AssertionError("catalogue enumeration not identical to exact fair joint support")
    fingerprint=sha256()
    for cell in sorted(pools):
        for _,ticket in pools[cell]:
            fingerprint.update(bytes(ticket))
        fingerprint.update(bytes((cell[0],cell[1])))
    return pools,{
        "policy":POLICY,"seed":seed,"per_cell":per_cell,
        "candidate_pool_count":sum(map(len,pools.values())),
        "nonzero_joint_cells":len(pools),
        "full_universe_enumerated":THEORY_SIZE,
        "catalogue_checksum_sha256":fingerprint.hexdigest(),
    }


def choose(cells,pools,*,seed:int,epoch:int,distinct_sums:bool,
           restarts:int=12,cell_limit:int=24):
    """Bounded deterministic four-ticket search over top currently viable cells.

    This is the precisely frozen POLICY used for BOTH real and IID timelines.
    Returns None if no witness is found. It never fabricates or relaxes bets.
    """
    if (type(restarts) is not int or restarts<1 or
            type(cell_limit) is not int or cell_limit<4):
        raise ValueError("invalid search budget")
    candidates=cells[:cell_limit]
    best,best_score=None,None
    rng=Random(seed + 131071*epoch + (7 if distinct_sums else 0))
    for _ in range(restarts):
        used=0; used_sums=set(); selected=[]; total=0
        for step in range(4):
            match=None
            for cell in candidates:
                s,li,key=cell
                if distinct_sums and s in used_sums:
                    continue
                rows=pools.get((s,li),[])
                if not rows:
                    continue
                start=rng.randrange(len(rows))
                for shift in range(len(rows)):
                    mask,ticket=rows[(start+shift)%len(rows)]
                    if mask & used:
                        continue
                    match=(mask,ticket,s,key)
                    break
                if match:
                    break
            if match is None:
                break
            mask,ticket,s,key=match
            used|=mask; selected.append(ticket); used_sums.add(s); total+=key
        if len(selected)==4:
            normalized=validate_four_disjoint(tuple(sorted(selected)))
            if (best_score is None or total>best_score
                    or (total==best_score and normalized<best)):
                best,best_score=normalized,total
    return best


def candidate_cells(hist_sum,hist_log,fair_sum,fair_log,joint):
    """Exact hard positive signs and the original P2 integer-SD objective."""
    gs=exact_gain_numerators(hist_sum,fair_sum)
    gl=exact_gain_numerators(hist_log,fair_log)
    weights=frozen_weights(gs["numerators"],fair_sum,
                           gl["numerators"],fair_log,gs["denominator"])
    out=[]
    for i,row in enumerate(joint):
        num_s=gs["numerators"][i]
        if num_s<=0:
            continue
        for j,mass in enumerate(row):
            num_l=gl["numerators"][j]
            if mass and num_l>0:
                key=weights[0]*num_s+weights[1]*num_l
                out.append((i+LOW_SUM,j,key))
    return sorted(out,key=lambda item:(-item[2],item[0],item[1]))


def evaluate_draw(portfolio,draw,bonus):
    """Exactly the official eight-tier award predicate; never assume payout."""
    if portfolio is None:
        return None
    if (type(bonus) is not int or bonus in draw or not 1<=bonus<=49 or
            len(draw)!=6 or len(set(draw))!=6):
        raise ValueError("invalid target Lotto649 draw/bonus")
    main=set(draw)
    tiers=[]
    for ticket in portfolio:
        overlap=len(main.intersection(ticket))
        tier=prize_tier(overlap,bonus in ticket)
        tiers.append(tier)
    return {"any_award":int(any(t is not None for t in tiers)),
            "awards":sum(t is not None for t in tiers),
            "tiers":tiers,"max_main_hits":max(len(main.intersection(t)) for t in portfolio)}


def _bonus_prefix(csv_dir, data, n):
    """Bonus is validated but not retained by temporal dataset. Read it with
    exact original chronology and fingerprint its prefix separately.
    """
    records=[]
    for path in sorted(Path(csv_dir).glob("lotto649_????.csv")):
        with path.open(encoding="utf-8-sig",newline="") as stream:
            for row in csv.DictReader(stream):
                records.append((int(row["draw_id"]),int(row["bonus"])))
    if len(records)!=len(data.ids):
        raise ValueError("bonus CSV count differs from validated main-draw dataset")
    answer=[]
    digest=sha256()
    for i in range(n):
        draw_id,bonus=records[i]
        balls={int(x) for x in data.numbers[i]}
        if (draw_id!=int(data.ids[i]) or type(bonus) is not int
                or bonus in balls or not 1<=bonus<=49):
            raise ValueError("bonus row / main numbers chronology mismatch")
        answer.append(bonus)
        digest.update((json.dumps([draw_id,bonus],separators=(",",":"))+"\n").encode("ascii"))
    return answer,digest.hexdigest()


def iid_history(n,seed):
    """Each unordered main 6/49 and distinct bonus are exactly fair."""
    rng=Random(seed)
    rows=[]
    for _ in range(n):
        seven=rng.sample(range(1,50),7)
        rows.append((tuple(sorted(seven[:6])),seven[6]))
    return rows


def replay(rows,joint,thresholds,pools,*,seed:int,warmup:int,stride:int,
           restarts:int,cell_limit:int,exact_prize_prob:float,
           keep_records:bool=False):
    """Same leakage-safe portfolio pipeline on real and synthetic histories.

    At t: update histograms through t-1, select four main tickets, THEN
    inspect the unseen row[t] main draw plus bonus. All model parameters and
    the (sum,log) pool are pre-frozen.
    """
    if (type(warmup) is not int or type(stride) is not int or warmup<3
            or stride<1 or warmup>=len(rows)):
        raise ValueError("invalid prequential schedule")
    fair_sum=[sum(row) for row in joint]
    fair_log=[sum(row[j] for row in joint) for j in range(len(joint[0]))]
    features=[]
    for main,bonus in rows:
        s=sum(main)-LOW_SUM
        li=bisect_right(thresholds,prod(main))-1
        if not (0<=s<len(joint) and 0<=li<len(joint[0])):
            raise ValueError("target main balls outside fixed theoretical bins")
        features.append((s,li,main,bonus))
    hsum=[0]*len(joint); hlog=[0]*len(joint[0])
    pointer=0
    records=[]
    stats={key:{"selected":0,"wins":0,"total_awards":0,
                "sum_max_main_hits":0,"failures":0}
           for key in ("primary","distinct_sums","random")}
    for t in range(warmup,len(rows),stride):
        while pointer<t:
            si,li,_,_=features[pointer]
            hsum[si]+=1;hlog[li]+=1;pointer+=1
        cells=candidate_cells(hsum,hlog,fair_sum,fair_log,joint)
        primary=choose(cells,pools,seed=seed,epoch=t,distinct_sums=False,
                       restarts=restarts,cell_limit=cell_limit)
        diverse=choose(cells,pools,seed=seed,epoch=t,distinct_sums=True,
                       restarts=restarts,cell_limit=cell_limit)
        baseline_rng=Random(seed+99991*t+17)
        values=baseline_rng.sample(range(1,50),24)
        random_four=validate_four_disjoint(tuple(sorted(
            tuple(sorted(values[6*i:6*(i+1)])) for i in range(4))))
        si,li,main,bonus=features[t]
        details={}
        for name,tickets in (("primary",primary),
                             ("distinct_sums",diverse),("random",random_four)):
            if tickets is not None:
                validate_four_disjoint(tickets)
                a=evaluate_draw(tickets,main,bonus)
                stats[name]["selected"]+=1
                stats[name]["wins"]+=a["any_award"]
                stats[name]["total_awards"]+=a["awards"]
                stats[name]["sum_max_main_hits"]+=a["max_main_hits"]
                details[name]={"tickets":[list(ticket) for ticket in tickets],
                               **a} if keep_records else {"any_award":a["any_award"]}
            else:
                stats[name]["failures"]+=1
                details[name]={"no_feasible_witness":True}
        if keep_records:
            records.append({
                "target_index_zero_based":t,
                "training_count":t,
                "target_main":list(main),
                "target_bonus":bonus,
                "candidate_positive_joint_cells":len(cells),
                "decisions":details,
            })
    evaluated=len(range(warmup,len(rows),stride))
    for name,item in stats.items():
        if item["selected"]+item["failures"]!=evaluated:
            raise AssertionError("selection/skip count mismatch")
        item["iid_expected_any_awards"]=item["selected"]*exact_prize_prob
        item["excess_any_awards"]=item["wins"]-item["iid_expected_any_awards"]
        item["realized_any_award_rate"]=item["wins"]/item["selected"] if item["selected"] else None
    return {"evaluated_checkpoints":evaluated,"stats":stats,"records":records}


def _empirical_upper_tail(observed,values):
    # Correct plus-one Monte Carlo p for a fixed statistic and full replay.
    return (1+sum(v>=observed for v in values))/(1+len(values))


def analyze(sum_p1,log_p0,dual_p1,csv_dir:Path,*,seed:int=649,
            warmup:int=300,stride:int=64,replicates:int=32,
            per_cell:int=32,restarts:int=12,cell_limit:int=24):
    """Pilot two-policy IID pipeline calibration with explicit power caveat."""
    if any(type(v) is not int or v<minimum for v,minimum in (
        (seed,0),(warmup,3),(stride,1),(replicates,8),
        (per_cell,1),(restarts,1),(cell_limit,4))):
        raise ValueError("invalid prespecified replay budgets")
    from .log_product_joint_p3_v027 import _observed_features
    reference=joint_exact_reference(sum_p1,log_p0,dual_p1)
    joint=reference["joint"]
    data,n=_history_prefix(csv_dir,dual_p1["source"])
    if warmup>=n:
        raise ValueError("warmup must precede actual historical cutoff")
    thresholds=reference["theory"]["grid"]["integer_product_thresholds"]
    bonuses,bonus_hash=_bonus_prefix(csv_dir,data,n)
    history=[(tuple(int(x) for x in data.numbers[i]),bonuses[i]) for i in range(n)]
    pools,info=catalogue(joint,thresholds,seed=seed,per_cell=per_cell)
    price=official_prize_coverage(REFERENCE)["any_prize_probability"]
    real=replay(history,joint,thresholds,pools,
                seed=seed,warmup=warmup,stride=stride,
                restarts=restarts,cell_limit=cell_limit,
                exact_prize_prob=price,keep_records=True)
    real_stat=real["stats"]
    # Test BOTH primary and diversity without unadjusted post-hoc policy pick.
    def compound(s):
        return max(s["primary"]["excess_any_awards"],
                   s["distinct_sums"]["excess_any_awards"])
    actual_max=compound(real_stat)
    replicate_summaries=[]
    for index in range(replicates):
        synthetic=iid_history(n,seed=seed+104729*(index+1))
        item=replay(synthetic,joint,thresholds,pools,
                    seed=seed,warmup=warmup,stride=stride,
                    restarts=restarts,cell_limit=cell_limit,
                    exact_prize_prob=price,keep_records=False)
        replicate_summaries.append({
            "replicate":index,
            "primary_excess":item["stats"]["primary"]["excess_any_awards"],
            "diverse_excess":item["stats"]["distinct_sums"]["excess_any_awards"],
            "max_excess":compound(item["stats"]),
            "primary_selections":item["stats"]["primary"]["selected"],
            "diverse_selections":item["stats"]["distinct_sums"]["selected"],
            "random_excess":item["stats"]["random"]["excess_any_awards"],
        })
    values=[r["max_excess"] for r in replicate_summaries]
    p_adjusted=_empirical_upper_tail(actual_max,values)
    p_primary=_empirical_upper_tail(real_stat["primary"]["excess_any_awards"],
                                    [v["primary_excess"] for v in replicate_summaries])
    p_diverse=_empirical_upper_tail(real_stat["distinct_sums"]["excess_any_awards"],
                                    [v["diverse_excess"] for v in replicate_summaries])
    return {
        "schema":SCHEMA,
        "stage":"P3B_PILOT_FROZEN_CACHED_POLICY_FULL_HISTORY_IID_REPLAY",
        "decision":"EXPLORATORY_PILOT; PREDICTIVE_NO_GO_OR_INCONCLUSIVE",
        "source":dual_p1["source"],
        "bonus_prefix_sha256":bonus_hash,
        "theory_checksums":{
            "sum":sum_p1["theory"]["checksum_sha256"],
            "log":reference["theory"]["checksum_sha256"],
            "joint":reference["joint_sha256"],
        },
        "policy":{
            **info,"selection":"deterministic bounded multicell greedy with exact rational dual-positive filter and integer IID-SD weighted objective",
            "strategies":["primary","distinct_sums"],
            "both_24_distinct_ball_requirement":True,
            "explicit_difference_from_P2_PR18":"P3b uses a history-independent per-cell cached ticket reservoir; P2 #18 samples a history-selected-cell reservoir. Therefore this pilot calibrates its own frozen policy, not exact P2 #18 outputs.",
            "no_candidate_tuning_after_target":True,
            "search_not_globally_certified":True,
            "one_step_ahead_no_future_input":True,
            "seed":seed,"warmup":warmup,"stride":stride,
            "restarts":restarts,"cell_limit":cell_limit,
            "iid_replicates":replicates,
            "schedule":"every stride-th historical target after warmup, fixed before seeing outcomes",
        },
        "fair_reference":{
            "official_eight_tier_any_prize_probability":price,
            "rule":"24 mutually disjoint ticket numbers implies identical exact official fair odds for all selected portfolios",
            "iid_target_generation":"uniform 7 distinct labels; first 6 unordered main balls, seventh distinct special",
        },
        "historical":real,
        "iid_null":{
            "replay_count":replicates,
            "statistic":"maximum of primary and distinct_sum (wins - conditional fair expectation) across fixed checkpoints",
            "observed_max_excess":actual_max,
            "IID_max_excess_mean":sum(values)/len(values),
            "IID_max_excess_standard_deviation":sqrt(sum((v-sum(values)/len(values))**2 for v in values)/len(values)),
            "max_statistic_multiple_policy_adjusted_upper_tail_p":p_adjusted,
            "primary_unadjusted_upper_tail_p":p_primary,
            "diverse_unadjusted_upper_tail_p":p_diverse,
            "random_baseline_observed_excess":real_stat["random"]["excess_any_awards"],
            "replicates":replicate_summaries,
            "pvalue_resolution":1/(replicates+1),
            "warning":"retrospective test, low Monte Carlo resolution and subset of targets; only the two explicitly frozen policies are adjusted. Model-family search and exact legacy P2 #18 are NOT fully calibrated.",
        },
        "warnings":[
            "All selected P3b portfolios have the same EXACT fair odds as any four disjoint tickets under IID.",
            "A good historical hit rate cannot be interpreted as evidence of self-correction without calibrated replication.",
            "Prequential checkpoints use strictly earlier main and bonus outcomes, but this is a retrospective research exercise.",
            "The cached portfolio search is a separately versioned policy, not the byte-identical P2 #18 optimizer.",
            "No PREDICTIVE_GO; future sealed trials, broader strategy comparison and higher-power Monte Carlo are needed.",
        ],
    }


def markdown_report(report):
    a=report["historical"];stats=a["stats"];null=report["iid_null"];p=report["policy"]
    out=[
        "# v0.27 Issue #15 — P3b Pilot: Four-Ticket IID Replay","",
        f"- **Frozen cutoff:** {report['source']['cutoff']}, #{report['source']['last_draw_id']}, N={report['source']['historical_draws']}",
        f"- **History SHA256:** `{report['source']['analysis_input_sha256']}`",
        f"- **Bonus prefix SHA256:** `{report['bonus_prefix_sha256']}`",
        f"- **Joint theory SHA256:** `{report['theory_checksums']['joint']}`",
        f"- **Policy:** `{p['policy']}`; catalogue SHA256 `{p['catalogue_checksum_sha256']}`",
        f"- **Cache:** {p['candidate_pool_count']} candidates across {p['nonzero_joint_cells']} cells (all 13,983,816 theoretical outcomes enumerated once)",
        f"- **Checkpoints:** {a['evaluated_checkpoints']}, warmup {p['warmup']}, stride {p['stride']}",
        f"- **IID full-history replicates:** {p['iid_replicates']}; p resolution {null['pvalue_resolution']:.5f}",
        f"- **Exact fair P(any official award):** {report['fair_reference']['official_eight_tier_any_prize_probability']:.9%}",
        "", "## Prequential real selected portfolios vs fair baseline","",
        "| Strategy | Valid selections | Any prizes | Fair expected prizes | Excess |",
        "|---|---:|---:|---:|---:|"
    ]
    for key in ("primary","distinct_sums","random"):
        item=stats[key]
        out.append(f"| {key} | {item['selected']} | {item['wins']} | {item['iid_expected_any_awards']:.4f} | {item['excess_any_awards']:+.4f} |")
    out.extend([
        "", "## Replay entire frozen pipeline under fair IID","",
        f"- **Observed max(primary, diverse) excess:** {null['observed_max_excess']:+.4f}",
        f"- **IID mean max-excess:** {null['IID_max_excess_mean']:+.4f}",
        f"- **Paired-policy max statistic Monte Carlo p (upper tail):** {null['max_statistic_multiple_policy_adjusted_upper_tail_p']:.6f}",
        f"- **Primary unadjusted MC p:** {null['primary_unadjusted_upper_tail_p']:.6f}",
        f"- **Diverse unadjusted MC p:** {null['diverse_unadjusted_upper_tail_p']:.6f}",
        "", "## Mandatory scope qualifications","",
        "- P3b pilot calibrates the exact documented cached-reservoir policy, NOT the original history-dependent reservoir implementation in P2 PR #18.",
        "- The fixed cached pool, all candidate scoring, dual-positive filtering, 24-ball disjoint selection and both policies are applied identically to real and synthetic histories.",
        "- Failure to find four positive/disjoint tickets is counted, not silently relaxed or removed.",
        "- This is a low-powered retrospective pilot. Model-family data snooping and original P2 are NOT covered; do not infer predictive skill.",
        "", "## Reproduce","", "```powershell",
        "& .\\.venv\\Scripts\\python.exe -m lotto7x7.log_product_p3b_pilot_v027 "
        "--sum-p1 results/v027_p1_repair.json --log-p0 results/v027_log_product_p0.json "
        "--dual-p1 results/v027_dual_p1.json --csv-dir data/csv --cutoff 2026-11-06 "
        "--warmup 300 --stride 64 --replicates 32 --seed 649 "
        "--out results/v027_p3b_pilot.json --md-out results/v027_p3b_pilot.md",
        "```",""
    ])
    return "\n".join(out)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sum-p1",type=Path,required=True)
    parser.add_argument("--log-p0",type=Path,required=True)
    parser.add_argument("--dual-p1",type=Path,required=True)
    parser.add_argument("--csv-dir",type=Path,default=Path("data/csv"))
    parser.add_argument("--cutoff",default="2026-11-06")
    parser.add_argument("--warmup",type=int,default=300)
    parser.add_argument("--stride",type=int,default=64)
    parser.add_argument("--replicates",type=int,default=32)
    parser.add_argument("--per-cell",type=int,default=32)
    parser.add_argument("--restarts",type=int,default=12)
    parser.add_argument("--cell-limit",type=int,default=24)
    parser.add_argument("--seed",type=int,default=649)
    parser.add_argument("--out",type=Path,default=Path("results/v027_p3b_pilot.json"))
    parser.add_argument("--md-out",type=Path,default=None)
    args=parser.parse_args(argv)
    out,md=args.out,args.md_out or args.out.with_suffix(".md")
    if out.resolve()==md.resolve() or out.exists() or md.exists():
        raise FileExistsError("two separate unused output paths required")
    raw=[p.read_bytes() for p in (args.sum_p1,args.log_p0,args.dual_p1)]
    s,l,d=[json.loads(x) for x in raw]
    if args.cutoff!=d["source"]["cutoff"]:
        raise ValueError("P3b cutoff must equal frozen P0/P1 source")
    report=analyze(s,l,d,args.csv_dir,seed=args.seed,
                   warmup=args.warmup,stride=args.stride,
                   replicates=args.replicates,per_cell=args.per_cell,
                   restarts=args.restarts,cell_limit=args.cell_limit)
    report["input_report_sha256"]={
        key:sha256(x).hexdigest() for key,x in zip(("sum_p1","log_p0","dual_p1"),raw)}
    out.parent.mkdir(parents=True,exist_ok=True)
    md.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)
        stream.write("\n")
    md.write_text(markdown_report(report),encoding="utf-8")
    h=report["historical"];n=report["iid_null"]
    print(f"[P3B PILOT PASS] checkpoints={h['evaluated_checkpoints']}; full-history IID replicates={n['replay_count']} (cached policy)")
    for k in ("primary","distinct_sums","random"):
        x=h["stats"][k]
        print(f"[{k.upper()}] selected={x['selected']} wins={x['wins']} IID_expected={x['iid_expected_any_awards']:.4f} excess={x['excess_any_awards']:+.4f}")
    print(f"[IID MAX-TEST] observed_max_excess={n['observed_max_excess']:+.4f}; adjusted_MC_p={n['max_statistic_multiple_policy_adjusted_upper_tail_p']:.6f}")
    print(f"[CATALOGUE] sha256={report['policy']['catalogue_checksum_sha256']}")
    print(f"[REPORT] {out} / {md}")
    print("[SCOPE] Pilot cached policy NOT byte-identical to P2 #18; no PREDICTIVE_GO")


if __name__=="__main__":
    main()
