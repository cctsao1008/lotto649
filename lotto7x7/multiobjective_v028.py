"""v0.28 M0 exploratory four-ticket multiobjective research.

Four genuinely different OBJECTIVE BLOCKS, not simply a rerun of P2:
 (1) exact Sum single-append L2 repair,
 (2) exact Log Product single-append L2 repair,
 (3) exact JOINT (Sum,Log bin) single-append L2 repair,
 (4) predeclared single Shape Taxonomy block, using exact L1/L2/L3 fair
     combinatorial classes and regularized counterfactual class repair.

The 4th block combines nested L1/L2/L3 scores ONLY WITHIN shape (0.25,
0.35, 0.40). Scores of the four blocks are not added as independent
likelihoods: use percentile normalization followed by maximin/Pareto-aware
portfolio optimization with constraints. Correlation remains explicit.

All 13,983,816 legal tickets are enumerated ONCE to construct an exact fair
joint histogram and exact L1/L2/L3 shape baselines simultaneously. For
candidate search, use a deterministic predeclared finite uniform sample;
label its result HEURISTIC_UNCERTIFIED, never claim a global optimum.

Source cutoff is immutable 2026-11-06/#115000103/2497 draws. Exclude every
individual ticket in existing purchased and original P2 portfolios to
ensure the new experimental choice is different, without implying greater
true probability. The existing P3f pre-draw static seal is untouched.

NO predictive GO or sales claim. All features were explored historically,
and output is descriptive multiobjective optimization only.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
from hashlib import sha256
from itertools import combinations, islice
import json
from math import comb, fsum, prod, sqrt
from pathlib import Path
from random import Random

import numpy as np

from .distribution_repair_portfolio_v027 import validate_four_disjoint, verify_p1
from .distribution_repair_v027 import LOW, TOTAL, exact_theory
from .log_product_dual_portfolio_v027 import _check_inputs
from .log_product_dual_scoring_v027 import exact_gain_numerators
from .log_product_joint_p3_v027 import _history_prefix
from .shape_taxonomy_v015 import classify_numbers
from .shape_theory_v016 import classification_keys, RULE_VERSION, LEVELS, category_name
from .official_prize_v026 import official_prize_coverage
from .deficit_compensation_v025 import REFERENCE

SCHEMA="lotto7x7.multiobjective.v0.28.m0"
CUTOFF="2026-11-06"
DRAW_ID=115000103
HISTORY_N=2497
INPUT_SHA="d865e1d241b607488e58a00b22fb595734f804595afd9d8850718808d6e8b7f8"
JOINT_SHA="31136d04347ff7b84fa7b5d17e7d009068b784e46f43f1d49aa058bc7fcd422a"
DEFAULT_SEED=20261110
SHAPE_ALPHA=4.0
SHAPE_INTERNAL_WEIGHTS={"component":0.25,"partition":0.35,"edge":0.40}
KNOWN_TICKETS=(
    (4,9,17,24,29,42), (10,13,14,27,28,34),
    (6,12,19,20,44,48), (3,7,18,33,40,47),
    (1,6,20,39,41,45), (2,5,8,42,47,48),
    (3,4,7,43,46,49), (14,28,31,32,36,37),
    (1,5,23,35,42,46), (3,8,11,20,34,40),
    (13,22,26,38,39,48), (19,25,30,31,36,37),
)
BLOCKS=("sum","log","joint","shape")


def _shape_codes(matrix):
    classes=classification_keys(classify_numbers(np.asarray(matrix,dtype=np.uint8)))
    return {k:[int(x) for x in v] for k,v in classes.items()}


def exact_joint_shape_baselines(thresholds, *, batch_size=120000):
    """One complete bounded-memory enumeration for JOINT + L1/L2/L3 shape.

    This independently reproduces both earlier theoretical specifications,
    rather than treating correlated marginal features as independent fair laws.
    """
    if (not isinstance(thresholds,list) or len(thresholds)<2 or
            thresholds!=sorted(thresholds) or len(set(thresholds))!=len(thresholds)
            or batch_size<1):
        raise ValueError("invalid frozen integer Log Product threshold grid")
    joint=np.zeros((259,len(thresholds)-1),dtype=np.int64)
    counts={level:Counter() for level in LEVELS}
    iteration=combinations(range(1,50),6)
    observed=0
    while True:
        group=list(islice(iteration,batch_size))
        if not group:
            break
        arr=np.asarray(group,dtype=np.uint8)
        si=arr.sum(axis=1,dtype=np.int64)-LOW
        product=arr.astype(np.int64).prod(axis=1)
        li=np.searchsorted(thresholds,product,side="right")-1
        if np.any(si<0) or np.any(si>=259) or np.any(li<0) or np.any(li>=joint.shape[1]):
            raise AssertionError("theoretical ticket maps outside fixed bins")
        np.add.at(joint,(si,li),1)
        keys=classification_keys(classify_numbers(arr))
        for level in LEVELS:
            names,frequency=np.unique(keys[level],return_counts=True)
            counts[level].update({int(x):int(v) for x,v in zip(names,frequency)})
        observed+=len(group)
    if observed!=TOTAL or int(joint.sum())!=TOTAL:
        raise AssertionError("did not enumerate all 13,983,816 legal tickets exactly once")
    if any(sum(counts[level].values())!=TOTAL for level in LEVELS):
        raise AssertionError("exact shape taxonomy levels did not sum to universe")
    joint_lists=joint.tolist()
    fingerprint=sha256(json.dumps(joint_lists,separators=(",",":")).encode("ascii")).hexdigest()
    fair_shapes={
        level:{str(k):int(v) for k,v in sorted(counts[level].items())}
        for level in LEVELS
    }
    baseline={
        "schema":RULE_VERSION,
        "rule_hash_sha256":sha256(RULE_VERSION.encode("ascii")).hexdigest(),
        "total":observed,
        "complete":True,
        "counts":fair_shapes,
    }
    shape_sha=sha256(json.dumps(baseline,sort_keys=True,
                                separators=(",",":")).encode("ascii")).hexdigest()
    return joint_lists,baseline,{
        "enumerated":observed,"joint_sha256":fingerprint,
        "shape_rule":RULE_VERSION,"shape_counts_sha256":shape_sha,
        "joint_nonzero_cells":int(np.count_nonzero(joint)),
        "shape_category_counts":{level:len(counts[level]) for level in LEVELS},
    }


def shape_per_class_gain(observed:dict[int,int],fair:dict[int,int],n:int,
                         *,alpha:float=SHAPE_ALPHA):
    """v0.25 fixed-alpha regularized class-L2, one hypothetical appended draw.

    Unlike the flat Sum/Log/Joint scores, shape L1/L2/L3 use v0.25's
    (p+alpha/N) denominator. Freeze it at historical N before AND after.
    """
    if n<1 or alpha<0 or not fair or sum(fair.values())!=TOTAL:
        raise ValueError("shape baseline or regularizer invalid")
    if set(observed)-set(fair):
        raise ValueError("unknown observed shape class")
    p={key:value/TOTAL for key,value in fair.items()}
    den={key:v+alpha/n for key,v in p.items()}
    before=fsum((observed.get(key,0)/n-p[key])**2/den[key] for key in p)
    after_base=fsum((observed.get(key,0)/(n+1)-p[key])**2/den[key] for key in p)
    gains={}
    for key,prob in p.items():
        seen=observed.get(key,0)
        old=(seen/(n+1)-prob)**2
        new=((seen+1)/(n+1)-prob)**2
        gains[key]=before-after_base-(new-old)/den[key]
    return gains,{"before":before,"alpha":alpha,"denominator_rule":"p + alpha/N frozen at historical N"}


def fair_standard_deviation(scores,frequencies):
    if len(scores)!=len(frequencies) or not scores:
        raise ValueError("invalid fair score distribution")
    m=sum(frequencies)
    if m<=0:
        raise ValueError("zero fair theory mass")
    mean=fsum(float(x)*int(v) for x,v in zip(scores,frequencies))/m
    variance=fsum(int(v)*(float(x)-mean)**2 for x,v in zip(scores,frequencies))/m
    if variance<=0:
        raise ValueError("zero IID score dispersion")
    return {"mean":mean,"stddev":sqrt(variance)}


def _mask(ticket):
    return sum(1<<(v-1) for v in ticket)


def _compare_key(portfolio,*,style="balanced"):
    total=[sum(c["rank"][i] for c in portfolio) for i in range(4)]
    minimum=min(total)
    if style=="balanced":
        quality=(minimum,sum(total),total[3],total[2])
    elif style=="shape":
        quality=(total[3],minimum,sum(total),total[2])
    elif style=="joint":
        quality=(total[2],minimum,sum(total),total[3])
    else:
        raise ValueError("unknown optimizer preference")
    return quality


def _legal_add(selected,candidate,*,omit_index=None):
    occupied=0
    used_parts=set()
    for idx,item in enumerate(selected):
        if idx==omit_index:
            continue
        occupied|=item["mask"]
        used_parts.add(item["partition_code"])
    return not (candidate["mask"]&occupied) and candidate["partition_code"] not in used_parts


def search_portfolio(pool,*,seed=DEFAULT_SEED,style="balanced",restarts=40):
    """Deterministic finite-pool heuristic, maximin over FOUR block ranks.

    L1/L2/L3 are one shape block. Different shape partitions across four
    selected bets are a mandatory design diversity constraint.
    """
    if restarts<1 or len(pool)<4:
        raise ValueError("insufficient candidate/search budget")
    order=sorted(pool,key=lambda row:(
        -min(row["rank"]),-sum(row["rank"]),row["ticket"]))
    rng=Random(seed+(0 if style=="balanced" else 1 if style=="shape" else 2))
    top=order[:min(200,len(order))]
    best=None; best_key=None
    for i in range(restarts):
        first=top[i%len(top)] if i< len(top) else rng.choice(top)
        selected=[first]
        for step in range(1,4):
            options=(c for c in pool if _legal_add(selected,c))
            best_add=None;best_add_key=None
            for item in options:
                prospective=selected+[item]
                quality=_compare_key(prospective,style=style)
                if best_add_key is None or quality>best_add_key or (
                    quality==best_add_key and item["ticket"]<best_add["ticket"]):
                    best_add,best_add_key=item,quality
            if best_add is None:
                break
            selected.append(best_add)
        if len(selected)!=4:
            continue
        # Local 1-for-1 upgrades preserve 24 disjoint balls and four
        # distinct partition codes; cannot certify global optimum.
        for _ in range(2):
            changed=False
            for slot in range(4):
                current=_compare_key(selected,style=style)
                replacement=None; replacement_key=current
                for item in pool:
                    if not _legal_add(selected,item,omit_index=slot):
                        continue
                    if item["ticket"] in {c["ticket"] for j,c in enumerate(selected) if j!=slot}:
                        continue
                    proposal=selected.copy()
                    proposal[slot]=item
                    score=_compare_key(proposal,style=style)
                    if score>replacement_key or (score==replacement_key and
                            replacement is not None and item["ticket"]<replacement["ticket"]):
                        if score>replacement_key:
                            replacement, replacement_key=item,score
                        elif replacement is not None:
                            replacement=item
                if replacement is not None:
                    selected[slot]=replacement
                    changed=True
            if not changed:
                break
        key=_compare_key(selected,style=style)
        normalized=tuple(sorted(c["ticket"] for c in selected))
        if best_key is None or key>best_key or (key==best_key and normalized<
                tuple(sorted(c["ticket"] for c in best))):
            best=selected.copy();best_key=key
    if best is None:
        raise ValueError("INFEASIBLE_UNCERTIFIED: no four-disjoint partition-diverse portfolio found in finite pool")
    return sorted(best,key=lambda c:c["ticket"]),{"criterion":style,
            "rank_vector_sum":[sum(c["rank"][i] for c in best) for i in range(4)],
            "maximin_rank_total":min(sum(c["rank"][i] for c in best) for i in range(4)),
            "total_rank_sum":sum(sum(c["rank"]) for c in best),
            "status":"HEURISTIC_UNCERTIFIED","restarts":restarts}


def _uniform_sample(seed, count):
    if type(seed) is not int or type(count) is not int or count<4 or seed<0:
        raise ValueError("seed/count invalid")
    rng=Random(seed)
    chosen=set()
    while len(chosen)<count:
        ticket=tuple(sorted(rng.sample(range(1,50),6)))
        chosen.add(ticket)
    return sorted(chosen)


def optimize(sum_p1:dict,log_p0:dict,dual_p1:dict,csv_dir:Path,
             *,seed=DEFAULT_SEED,sample_count=80000,pool_cap=2000,
             restarts=40,alpha=SHAPE_ALPHA,batch_size=120000):
    sum_rows,log_theory,sg,lg=_check_inputs(sum_p1,log_p0,dual_p1)
    source=dual_p1["source"]
    if (source["cutoff"]!=CUTOFF or source["last_draw_id"]!=DRAW_ID or
            source["historical_draws"]!=HISTORY_N or
            source["analysis_input_sha256"]!=INPUT_SHA):
        raise ValueError("v0.28 new-method experiment must use exact frozen 2026-11-06 source")
    if (type(sample_count) is not int or sample_count<1000 or
            type(pool_cap) is not int or pool_cap<80 or
            type(restarts) is not int or restarts<1 or alpha != SHAPE_ALPHA):
        raise ValueError("invalid declared finite candidate/shape hyperparameters")
    data,n=_history_prefix(csv_dir,source)
    thresholds=log_theory["grid"]["integer_product_thresholds"]
    joint,shape,reference=exact_joint_shape_baselines(thresholds,batch_size=batch_size)
    if reference["joint_sha256"]!=JOINT_SHA:
        raise AssertionError("joint theory SHA disagrees with original P3a")
    if [sum(row) for row in joint]!=exact_theory()["counts"]:
        raise AssertionError("joint Sum marginal differs from exact fair sum counts")
    if [sum(row[i] for row in joint) for i in range(len(joint[0]))]!=log_theory["counts"]:
        raise AssertionError("joint Log marginal differs from exact fair product histogram")
    historical=np.asarray(data.numbers[:n],dtype=np.uint8)
    hsum=np.bincount(historical.sum(axis=1).astype(np.int64)-LOW,minlength=259)
    hproduct=np.prod(historical.astype(np.int64),axis=1)
    hlog=np.searchsorted(thresholds,hproduct,side="right")-1
    hlog_hist=np.bincount(hlog,minlength=len(joint[0]))
    if ([int(x) for x in hsum]!=[x["observed_count"] for x in sum_rows] or
            [int(x) for x in hlog_hist]!=log_p0["observed"]["counts"]):
        raise AssertionError("frozen observed marginal counts differ from history")
    h_joint=np.zeros((259,len(joint[0])),dtype=np.int64)
    np.add.at(h_joint,(historical.sum(axis=1).astype(np.int64)-LOW,hlog),1)
    jgain=exact_gain_numerators([int(x) for x in h_joint.flat],
                                 [int(x) for row in joint for x in row])
    shape_history=_shape_codes(historical)
    shape_gains={};shape_stats={}; shape_means={}
    for level in LEVELS:
        fair={int(k):int(v) for k,v in shape["counts"][level].items()}
        observed=Counter(shape_history[level])
        gains,metadata=shape_per_class_gain(observed,fair,n,alpha=alpha)
        weights=fair
        iid=fair_standard_deviation(list(gains.values()),[weights[k] for k in gains])
        shape_gains[level]=gains;shape_means[level]=iid
        shape_stats[level]={**metadata,**iid,"categories":len(fair),
                            "positive_classes":sum(g>0 for g in gains.values())}
    # Block SDs are computed against EXACT IID fair theory (no independence
    # approximation between Sum and Log). All 4 blocks are separately ranked.
    sum_fair=exact_theory()["counts"]
    log_fair=log_theory["counts"]
    total=comb(49,6)
    sigma_sum=fair_standard_deviation([x/sg["denominator"] for x in sg["numerators"]],sum_fair)
    sigma_log=fair_standard_deviation([x/lg["denominator"] for x in lg["numerators"]],log_fair)
    sigma_joint=fair_standard_deviation([x/jgain["denominator"] for x in jgain["numerators"]],
                                         [x for row in joint for x in row])
    raw=_uniform_sample(seed,sample_count)
    known={tuple(row) for row in KNOWN_TICKETS}
    eligible_raw=[t for t in raw if t not in known]
    classes=_shape_codes(eligible_raw)
    qualified=[]
    rejection=Counter()
    for i,ticket in enumerate(eligible_raw):
        si=sum(ticket)-LOW
        li=bisect_left(thresholds,prod(ticket)+1)-1
        # thresholds are sorted exact integers and bins are left-closed;
        # bisect_left(product+1)-1 equals bisect_right(product)-1.
        ji=si*len(joint[0])+li
        if sg["numerators"][si]<=0 or lg["numerators"][li]<=0:
            rejection["marginal_nonpositive"]+=1
            continue
        if jgain["numerators"][ji]<=0:
            rejection["joint_nonpositive"]+=1
            continue
        level_scores={level:shape_gains[level][classes[level][i]]/
                      shape_means[level]["stddev"] for level in LEVELS}
        shape_block=sum(SHAPE_INTERNAL_WEIGHTS[level]*level_scores[level] for level in LEVELS)
        if shape_block<=0:
            rejection["shape_block_nonpositive"]+=1
            continue
        values=[
            sg["numerators"][si]/sg["denominator"]/sigma_sum["stddev"],
            lg["numerators"][li]/lg["denominator"]/sigma_log["stddev"],
            jgain["numerators"][ji]/jgain["denominator"]/sigma_joint["stddev"],
            shape_block
        ]
        qualified.append({
            "ticket":ticket,"mask":_mask(ticket),"sum":si+LOW,
            "log_bin":li,"joint_cell":[si+LOW,li],
            "partition_code":classes["partition"][i],
            "component_code":classes["component"][i],
            "edge_code":classes["edge"][i],
            "shape_categories":{
                level:category_name(level,classes[level][i]) for level in LEVELS
            },
            "block_standardized_gain":values,
            "shape_level_standardized_gain":level_scores,
        })
    if len(qualified)<100:
        raise ValueError("INFEASIBLE_UNCERTIFIED: too few joint+shape positive candidates")
    ranks=[sorted(v["block_standardized_gain"][i] for v in qualified) for i in range(4)]
    for candidate in qualified:
        candidate["rank"]=[
            bisect_left(ranks[i],candidate["block_standardized_gain"][i])
            for i in range(4)
        ]
    pool_candidates={}
    # A transparent sampling budget: 1,200 highest balanced feasible
    # candidates plus 200 best in EACH axis (may overlap).
    for item in sorted(qualified,key=lambda c:(-min(c["rank"]),-sum(c["rank"]),c["ticket"]))[:max(80,pool_cap-800)]:
        pool_candidates[item["ticket"]]=item
    per_axis=max(50,pool_cap//10)
    for axis in range(4):
        for item in sorted(qualified,key=lambda c:(-c["rank"][axis],-min(c["rank"]),c["ticket"]))[:per_axis]:
            pool_candidates[item["ticket"]]=item
    remaining=max(0,pool_cap-len(pool_candidates))
    if remaining:
        # Seeded exploration ensures rare Pareto compromises are considered.
        options=[item for item in qualified if item["ticket"] not in pool_candidates]
        sampler=Random(seed+119)
        for item in sampler.sample(options,min(remaining,len(options))):
            pool_candidates[item["ticket"]]=item
    pool=list(pool_candidates.values())
    primary,pr=search_portfolio(pool,seed=seed,style="balanced",restarts=restarts)
    shape_first,sr=search_portfolio(pool,seed=seed,style="shape",restarts=max(8,restarts//2))
    joint_first,jr=search_portfolio(pool,seed=seed,style="joint",restarts=max(8,restarts//2))
    fair=official_prize_coverage(REFERENCE)
    def describe(selected,meta):
        lines=[list(c["ticket"]) for c in selected]
        portfolio=validate_four_disjoint(lines)
        if len({c["partition_code"] for c in selected})!=4:
            raise AssertionError("four exact Shape partition taxonomy classes not distinct")
        if any(tuple(x) in known for x in lines):
            raise AssertionError("old P2/purchased individual bet was accidentally reused")
        legal=official_prize_coverage(portfolio)
        if (legal["any_prize_outcomes"]!=fair["any_prize_outcomes"] or
                legal["total_outcomes"]!=fair["total_outcomes"]):
            raise AssertionError("four-disjoint eight-tier fair coverage mismatch")
        return {
            **meta,"tickets":lines,
            "sum_values":[c["sum"] for c in selected],
            "shape_partitions":[c["shape_categories"]["partition"] for c in selected],
            "individual":[{
                "ticket":list(c["ticket"]),"sum":c["sum"],"log_bin":c["log_bin"],
                "joint_cell":c["joint_cell"],
                "shape_taxonomy":c["shape_categories"],
                "block_gain_in_exact_IID_std_units":dict(zip(BLOCKS,c["block_standardized_gain"])),
                "block_percentile_ranks_in_frozen_sample":dict(zip(BLOCKS,c["rank"])),
                "shape_internal_L1_L2_L3_scores":c["shape_level_standardized_gain"],
            } for c in selected],
            "exact_official_any_prize_probability":legal["any_prize_probability"],
            "old_ticket_overlap_same_line_count":0,
            "distinct_number_count":len({v for c in selected for v in c["ticket"]}),
            "shape_partition_count":len({c["partition_code"] for c in selected}),
        }
    checksum=sha256()
    for ticket in raw:
        checksum.update(bytes(ticket))
    return {
        "schema":SCHEMA,
        "stage":"M0_EXPLORATORY_SHAPE_SUM_LOG_JOINT_PARETO_MAXIMIN",
        "decision":"DESCRIPTIVE_MULTI_OBJECTIVE_HEURISTIC; PREDICTIVE_NO_GO_OR_INCONCLUSIVE",
        "source":source,
        "theory":{
            **reference,"sum_theory_sha256":sum_p1["theory"]["checksum_sha256"],
            "log_theory_sha256":log_theory["checksum_sha256"],
        },
        "shape_block":{
            "nested_hierarchy_levels":list(LEVELS),
            "internal_fixed_weights":SHAPE_INTERNAL_WEIGHTS,
            "alpha":alpha,"shape_level_stats":shape_stats,
            "shape_score_rule":"single-draw frozen-alpha regularized shape loss reduction divided by exact IID per-level SD; nested levels treated as ONE objective block",
        },
        "sum_log_joint_iid_standards":{
            "sum":sigma_sum,"log":sigma_log,"joint":sigma_joint,
        },
        "search":{
            "seed":seed,"iid_uniform_candidate_sample_size":sample_count,
            "complete_universe_for_fair_theory":total,
            "sample_sha256":checksum.hexdigest(),
            "feasible_four_disjoint_requirement":True,
            "four_distinct_shape_partitions_required":True,
            "all_single_draw_sum_log_joint_gains_strictly_positive":True,
            "shape_block_gain_strictly_positive":True,
            "excluded_individual_historical_bets":len(known),
            "candidate_eligible_after_old_ticket_exclusion":len(eligible_raw),
            "candidate_passing_four_positive_objective_blocks":len(qualified),
            "candidate_rejection_reasons":dict(rejection),
            "bounded_search_pool_size":len(pool),
            "search_restarts":restarts,
            "search_optimality":"HEURISTIC_UNCERTIFIED",
            "rank_rule":"empirical percentile ranks from fixed seeded candidate sample, not marginal likelihoods",
            "objective_rule":"lexicographic maximin of FOUR portfolio percentile-rank sums; total rank sum as secondary criterion",
        },
        "primary":describe(primary,pr),
        "sensitivity":{
            "shape_first":describe(shape_first,sr),
            "joint_first":describe(joint_first,jr),
        },
        "limitations":[
            "This is a NEW exploratory four-block selection algorithm, not a P2 restatement and not a predictive GO.",
            "S and Log Product strongly correlate; joint is derived from both; scores are NOT independent statistical evidence.",
            "L1/L2/L3 taxonomy features are nested and counted as a SINGLE shape block.",
            "Exact full-universe theory; finite sampled candidate search and local swaps are HEURISTIC_UNCERTIFIED.",
            "High-dimensional joint positive repair may reflect sparse-bin noise; results require IID and prospective evaluation.",
            "Prior purchased tickets, original P2 and sealed P3f research portfolios remain unchanged.",
        ],
    }


def markdown_report(r):
    lines=[
        "# v0.28 M0 — Shape + Sum + LogProduct + Joint Multiobjective Selection","",
        f"- **Cutoff:** {r['source']['cutoff']} / #{r['source']['last_draw_id']}",
        f"- **Input SHA256:** "+r["source"]["analysis_input_sha256"],
        f"- **Joint fair SHA256:** "+r["theory"]["joint_sha256"],
        f"- **Shape exact theory SHA256:** "+r["theory"]["shape_counts_sha256"],
        f"- **Total fair support:** {r['theory']['enumerated']:,}",
        f"- **Candidate fair sample:** {r['search']['iid_uniform_candidate_sample_size']:,}; eligible {r['search']['candidate_passing_four_positive_objective_blocks']:,}",
        f"- **Search pool:** {r['search']['bounded_search_pool_size']}; restarts {r['search']['search_restarts']}",
        "",
        "## Selected NEW four-ticket experiment (balanced maximin of four blocks)","",
        "| Line | Six numbers | Sum | Shape partition |",
        "|:---|:---|---:|:---|",
    ]
    for name,item,total,shape in zip("ABCD",r["primary"]["tickets"],r["primary"]["sum_values"],r["primary"]["shape_partitions"]):
        lines.append(f"| {name} | {' '.join(f'{x:02d}' for x in item)} | {total} | {shape} |")
    lines.extend(["", "## Four separate counterfactual repair blocks","",
                  "| Line | Sum G/sigma | Log G/sigma | Joint G/sigma | Shape block |",
                  "|:---|---:|---:|---:|---:|"])
    for n,item in zip("ABCD",r["primary"]["individual"]):
        g=item["block_gain_in_exact_IID_std_units"]
        lines.append(f"| {n} | {g['sum']:.6f} | {g['log']:.6f} | {g['joint']:.6f} | {g['shape']:.6f} |")
    lines.extend(["",f"- **Result:** {r['decision']}",
        f"- **Search status:** {r['primary']['status']}",
        f"- **Fair IID any eight-tier award probability:** {r['primary']['exact_official_any_prize_probability']:.12%}",
        "- The objective is a balanced maximin over percentile-ranked FOUR correlated blocks, not a sum of independent probabilities.",
        "- Strictly positive Sum, Log, Joint and composite Shape single-draw repair is a design constraint, not proof of better draw odds.",
        "",
        "## Sensitivity (different objective priority, same fair baseline)",""
    ])
    for key,result in r["sensitivity"].items():
        lines.append(f"- **{key}:** "+ " | ".join(" ".join(f"{v:02d}" for v in t) for t in result["tickets"]))
    lines.extend(["","## Research caveats",""])
    lines.extend("- "+t for t in r["limitations"])
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sum-p1",type=Path,required=True)
    parser.add_argument("--log-p0",type=Path,required=True)
    parser.add_argument("--dual-p1",type=Path,required=True)
    parser.add_argument("--csv-dir",type=Path,default=Path("data/csv"))
    parser.add_argument("--seed",type=int,default=DEFAULT_SEED)
    parser.add_argument("--sample-count",type=int,default=80000)
    parser.add_argument("--pool-cap",type=int,default=2000)
    parser.add_argument("--restarts",type=int,default=40)
    parser.add_argument("--batch-size",type=int,default=120000)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--md-out",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.out.resolve()==args.md_out.resolve() or args.out.exists() or args.md_out.exists():
        raise FileExistsError("two separate unused output report filenames required")
    s,l,d=(json.loads(p.read_bytes()) for p in (args.sum_p1,args.log_p0,args.dual_p1))
    report=optimize(s,l,d,args.csv_dir,
                    seed=args.seed,sample_count=args.sample_count,
                    pool_cap=args.pool_cap,restarts=args.restarts,
                    batch_size=args.batch_size)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.md_out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf-8") as fh:
        json.dump(report,fh,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
        fh.write("\n")
    args.md_out.write_text(markdown_report(report),encoding="utf-8")
    print("[V028 MULTIOBJECTIVE PASS] independent exact Shape Theory + Sum/Log/Joint diagnostics")
    print(f"[THEORY] {report['theory']['enumerated']:,} exact 6/49; joint sha256={report['theory']['joint_sha256']}")
    print(f"[POOL] {report['search']['candidate_passing_four_positive_objective_blocks']:,} all-four-positive; search={report['search']['bounded_search_pool_size']}")
    for name,item in zip("ABCD",report["primary"]["tickets"]):
        print(f"[NEW {name}] "+" ".join(f"{v:02d}" for v in item))
    print(f"[PORTFOLIO] maximin={report['primary']['maximin_rank_total']}; 24 distinct balls; {report['primary']['status']}")
    print(f"[SENSITIVITY] shape-first diff={report['sensitivity']['shape_first']['tickets']!=report['primary']['tickets']}; joint-first diff={report['sensitivity']['joint_first']['tickets']!=report['primary']['tickets']}")
    print(f"[REPORT] {args.out} / {args.md_out}")
    print("[SCOPE] novel 4-block sampled Pareto/maximin descriptive research; NO predictive GO, old seals and paid tickets unchanged")


if __name__=="__main__":
    main()
