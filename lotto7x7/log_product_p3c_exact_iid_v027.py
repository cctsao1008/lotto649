"""v0.27 Issue #15 P3c: EXACT fair IID test of four-ticket official prize events.

Mathematical theorem (not Monte Carlo):
For any history-measurable selector that commits FOUR PAIRWISE-DISJOINT
six-number tickets before an IID fair Lotto649 draw, the conditional
probability of ANY official prize is the same rational constant p=A/D.
Reason: every four-disjoint portfolio is a relabeling of any other, and
the fair main+bonus draw law is invariant to relabeling.

For a deterministic, fixed-in-advance list of checkpoints, if the
selector successfully commits exactly four disjoint tickets at EVERY
checkpoint, the 0/1 indicators are iid Bernoulli(p) despite arbitrary
adaptivity to previous draws. Therefore the count is EXACTLY Binomial(n,p).

Skip/failure or outcome-driven stopping invalidates naive Binomial(n,p)
conditioning on a random number of selections; such inputs are rejected.
Bonferroni controls the two named strategy tests under arbitrary
within-checkpoint dependence. It does NOT correct choices made across
other retrospectively explored model families or past research runs.

This calibration applies equally to the mathematical original P2 policy,
the cached P3b pilot and any other history-adaptive four-disjoint policy
under a FAIR IID NULL. It is not an empirical backtest of the original
P2 code on the actual historical sequence.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from hashlib import sha256
import json
from math import comb
from pathlib import Path

from .deficit_compensation_v025 import REFERENCE
from .distribution_repair_portfolio_v027 import validate_four_disjoint
from .log_product_p3b_pilot_v027 import (
    SCHEMA as P3B_SCHEMA,
    evaluate_draw,
)
from .official_prize_v026 import official_prize_coverage

SCHEMA = "lotto7x7.log_product.v0.27.p3c_exact_iid"


def exact_fair_probability() -> Fraction:
    """Full eight-tier ANY-award event for a canonical four-disjoint set."""
    baseline = official_prize_coverage(REFERENCE)
    if (baseline["ticket_count"] != 4 or
            baseline["total_outcomes"] != comb(49, 6)*43):
        raise AssertionError("reference must be four legal 6/49 tickets")
    validate_four_disjoint(baseline["tickets"])
    numerator = baseline["any_prize_outcomes"]
    denominator = baseline["total_outcomes"]
    if not 0 < numerator < denominator:
        raise AssertionError("invalid official prize rational probability")
    return Fraction(numerator, denominator)


def exact_binomial_tail(n: int, wins: int, fair_p: Fraction) -> dict:
    """Exact rational one-sided tail probabilities, all integer arithmetic."""
    if (type(n) is not int or type(wins) is not int or
            n < 1 or not 0 <= wins <= n or
            not isinstance(fair_p, Fraction) or not 0 < fair_p < 1):
        raise ValueError("requires n>0, wins in [0,n], and a fair exact probability")
    q = 1 - fair_p
    mass = [
        comb(n,k)*fair_p**k*q**(n-k)
        for k in range(n+1)
    ]
    assert sum(mass, Fraction(0, 1)) == 1
    upper = sum(mass[wins:], Fraction(0,1))
    lower = sum(mass[:wins+1], Fraction(0,1))
    return {
        "trials": n,
        "successes": wins,
        "exact_fair_expected_successes": float(n*fair_p),
        "upper_tail_probability": float(upper),
        "lower_tail_probability": float(lower),
        "upper_tail_exact_numerator": str(upper.numerator),
        "upper_tail_exact_denominator": str(upper.denominator),
        "null": "EXACT binomial under fixed checkpoints and four disjoint tickets chosen before each fair IID outcome",
    }


def _validate_and_score_records(pilot: dict, fair_p: Fraction):
    if not isinstance(pilot, dict) or pilot.get("schema") != P3B_SCHEMA:
        raise ValueError("requires full frozen P3b cached-policy report")
    if pilot.get("decision") != "EXPLORATORY_PILOT; PREDICTIVE_NO_GO_OR_INCONCLUSIVE":
        raise ValueError("P3b research decision changed")
    if pilot.get("policy",{}).get("policy") != "cached_joint_cell_greedy_v1":
        raise ValueError("unknown retrospective P3b policy")
    entries = pilot.get("historical",{}).get("records",[])
    n = pilot.get("historical",{}).get("evaluated_checkpoints")
    if type(n) is not int or n < 1 or len(entries) != n:
        raise ValueError("missing complete predetermined P3b checkpoint records")
    schedule = pilot["policy"]
    first = schedule.get("warmup")
    stride = schedule.get("stride")
    if (type(first) is not int or type(stride) is not int or
            first < 3 or stride < 1 or
            n != len(range(first,pilot["source"]["historical_draws"],stride))):
        raise ValueError("checkpoint schedule is malformed")
    source=pilot["source"]
    if (source.get("cutoff")!="2026-11-06" or
            source.get("last_draw_id") != 115000103 or
            source.get("historical_draws") != 2497 or
            source.get("analysis_input_sha256") !=
            "d865e1d241b607488e58a00b22fb595734f804595afd9d8850718808d6e8b7f8"):
        raise ValueError("requires frozen #115000103 and its historical prefix fingerprint")
    for field in ("catalogue_checksum_sha256",):
        if len(schedule.get(field,""))!=64:
            raise ValueError("cached-policy candidate catalogue hash missing")
    if pilot.get("theory_checksums",{}).get("joint") != (
            "31136d04347ff7b84fa7b5d17e7d009068b784e46f43f1d49aa058bc7fcd422a"):
        raise ValueError("frozen exact joint-theory fingerprint mismatch")
    fair_reference=pilot.get("fair_reference",{}).get("official_eight_tier_any_prize_probability")
    if not isinstance(fair_reference,(int,float)) or abs(fair_reference-float(fair_p))>1e-14:
        raise ValueError("P3b official fair reference probability mismatch")
    results={key:{"wins":0,"selected":0,"failures":0,"total_awards":0}
             for key in ("primary","distinct_sums","random")}
    for index,item in enumerate(entries):
        t=first+stride*index
        if (item.get("target_index_zero_based")!=t or
                item.get("training_count")!=t):
            raise ValueError("not a fixed chronological strictly prior-data checkpoint")
        main=item.get("target_main")
        bonus=item.get("target_bonus")
        if (not isinstance(main,list) or len(main)!=6 or
                any(type(v) is not int for v in main) or
                main!=sorted(main) or
                len(set(main))!=6 or
                type(bonus) is not int or bonus in main or
                not 1<=bonus<=49):
            raise ValueError("invalid held-out official outcome")
        decisions=item.get("decisions",{})
        for strategy in results:
            decision=decisions.get(strategy,{})
            if "tickets" not in decision:
                # Do not silently condition away adaptive failures.
                if decision.get("no_feasible_witness") is not True:
                    raise ValueError("missing portfolio and missing explicit failure indicator")
                results[strategy]["failures"]+=1
                continue
            tickets=validate_four_disjoint(decision["tickets"])
            if strategy=="distinct_sums" and len({sum(t) for t in tickets}) !=4:
                raise ValueError("distinct sum variant broke extra constraint")
            evaluated=evaluate_draw(tickets,main,bonus)
            if (evaluated["any_award"]!=decision.get("any_award") or
                    evaluated["awards"]!=decision.get("awards") or
                    evaluated["tiers"]!=decision.get("tiers") or
                    evaluated["max_main_hits"]!=decision.get("max_main_hits")):
                raise ValueError("cached policy prize outcome record has been altered")
            results[strategy]["selected"]+=1
            results[strategy]["wins"]+=evaluated["any_award"]
            results[strategy]["total_awards"]+=evaluated["awards"]
    for name,stat in results.items():
        expected=pilot["historical"]["stats"][name]
        for old,new in (
            ("wins","wins"),("selected","selected"),
            ("failures","failures"),("total_awards","total_awards")):
            if stat[new]!=expected[old]:
                raise ValueError(f"reported P3b {name} {old} mismatch")
    return n, results


def audit_pilot(pilot: dict) -> dict:
    """Exact null law for two frozen strategies and audit of every checkpoint."""
    p=exact_fair_probability()
    checkpoints, counts=_validate_and_score_records(pilot,p)
    # Fixed-length binomial law must not drop adaptive failures.
    if any(item["failures"]!=0 or item["selected"]!=checkpoints for item in counts.values()):
        return {
            "schema":SCHEMA,
            "decision":"INCONCLUSIVE_ADAPTIVE_SELECTION_FAILURES",
            "reason":"Exact fixed-length Binomial(n,p) does not apply after dropping adaptive strategy failures; no p-value reported.",
            "source":pilot["source"],
            "fair_probability_exact":[str(p.numerator),str(p.denominator)],
            "counts":counts,
        }
    tests={strategy:exact_binomial_tail(checkpoints,counts[strategy]["wins"],p)
           for strategy in counts}
    nominated=("primary","distinct_sums")
    pmin=min(tests[s]["upper_tail_probability"] for s in nominated)
    adjusted=min(1.,len(nominated)*pmin)
    max_observed=max(counts[s]["wins"] for s in nominated)
    return {
        "schema":SCHEMA,
        "stage":"P3C_EXACT_IID_NULL_ANY_OFFICIAL_PRIZE_FOR_ANY_ADAPTIVE_FOUR_DISJOINT_SELECTOR",
        "decision":"EXACT_IID_NULL_CALIBRATED_FOR_FOUR_DISJOINT_POLICY; PREDICTIVE_NO_GO_OR_INCONCLUSIVE",
        "source":pilot["source"],
        "bonus_prefix_sha256":pilot["bonus_prefix_sha256"],
        "catalogue_sha256":pilot["policy"]["catalogue_checksum_sha256"],
        "conditional_theorem":{
            "claim":"If a policy chooses 4 pairwise-disjoint 6/49 tickets using only preceding outcomes, P(any official award at next draw | history) is identical for ALL such policies.",
            "proof":"Permutation symmetry: any 24-disjoint-number four-ticket portfolio maps bijectively to any other under a relabeling of 1..49; the fair draw including distinct bonus is invariant under such relabelings.",
            "iid_bernoulli_sequence_rule":"For a fixed set of checkpoint times and no adaptive selection failures, conditional Bernoulli success probability is constant p, therefore hit indicators are iid Bernoulli(p) and total successes Binomial(n,p).",
            "null_applies_to":"Original PR #18 P2 algorithm, cached P3b policy, random disjoint portfolio and any other preceding-history-measurable valid four-disjoint selection under FAIR IID.",
            "not_an_empirical_claim":"This is a conditional fair-null theorem, NOT a claim that actual lottery history obeys IID or that all historical strategies were tested.",
            "number_of_predeclared_policies_adjusted":2,
            "other_model_family_search_unadjusted":True,
        },
        "fair_reference":{
            "any_official_award_exact_rational":[str(p.numerator),str(p.denominator)],
            "any_official_award_probability":float(p),
            "number_of_main_outcomes":comb(49,6),
            "distinct_bonus_choices":43,
            "total_outcomes":comb(49,6)*43,
        },
        "historical":{
            "fixed_checkpoints":checkpoints,
            "observed_policy_counts":counts,
            "exact_binomial_tests":tests,
            "two_policy_max_wins":max_observed,
        },
        "two_policy_familywise_upper_tail":{
            "method":"Bonferroni / union bound on two exact one-sided Binomial(n,p) tests; valid for arbitrary within-checkpoint policy dependence",
            "adjusted_p_upper_bound":adjusted,
            "minimum_individual_one_sided_p":pmin,
            "hypotheses":list(nominated),
        },
        "original_cached_pilot_reference":{
            "pilot_MC_p":pilot["iid_null"]["max_statistic_multiple_policy_adjusted_upper_tail_p"],
            "pilot_replicates":pilot["iid_null"]["replay_count"],
            "warning":"MC and exact Bonferroni p-values need not match; one is calibrated max-excess simulation, other a conservative valid union-bound on exact marginal binomial tests.",
        },
        "scope":[
            "This audit independently rescored every cached P3b held-out draw, but it did NOT run original PR #18 P2 on the actual timeline.",
            "Exact FAIR-IID chance of a prize is invariant to how valid four-disjoint tickets are chosen, including original P2.",
            "Retrospective design, limited 35 fixed checkpoints and many previously explored model families; no PREDICTIVE_GO.",
            "The exact theorem eliminates Monte Carlo error in the marginal null, but not empirical uncertainty or multiple-model exploration.",
        ],
    }


def markdown_report(report:dict)->str:
    title="# v0.27 Issue #15 — P3c Exact Four-Disjoint Fair IID Prize Null"
    if report.get("decision")=="INCONCLUSIVE_ADAPTIVE_SELECTION_FAILURES":
        return title+"\n\n"+report["reason"]+"\n"
    r=report["fair_reference"]
    h=report["historical"]
    lines=[title,"",f"- **Cutoff:** {report['source']['cutoff']} / #{report['source']['last_draw_id']}",
           f"- **Exact fair P(any prize):** {r['any_official_award_exact_rational'][0]}/{r['any_official_award_exact_rational'][1]} = {r['any_official_award_probability']:.12%}",
           f"- **Fixed checkpoints:** {h['fixed_checkpoints']}",
           "- **Theorem:** Any pre-draw history-adaptive choice of 4 disjoint six-number tickets has the SAME exact official fair prize odds.",
           "", "| Policy | Awards | Fair expected | Exact one-sided Binomial upper-tail p |",
           "|:---|---:|---:|---:|"]
    for name in ("primary","distinct_sums","random"):
        t=h["exact_binomial_tests"][name]
        lines.append(f"| {name} | {t['successes']} | {t['exact_fair_expected_successes']:.4f} | {t['upper_tail_probability']:.8f} |")
    a=report["two_policy_familywise_upper_tail"]
    lines.extend([
        "",f"- **Two-policy Bonferroni-adjusted upper-tail p:** {a['adjusted_p_upper_bound']:.8f}",
        f"- **Prior 24-replay max-excess MC p:** {report['original_cached_pilot_reference']['pilot_MC_p']:.8f}",
        "", "The two p-values use different valid procedures; do not interpret their small numerical difference as proof of selection effects.",
        "", "## Exact theorem versus empirical validation", "",
        "The per-next-draw conditional fair prize probability applies to the ORIGINAL P2 #18 algorithm as well as the P3b cached policy. It does not reproduce the original P2's historical choices, and cannot by itself test whether actual draws violate fair IID assumptions.",
        "The exact Binomial(n,p) test additionally requires a fixed checkpoint schedule and no adaptive selection failures; this report verifies all 35 checkpoints were covered.",
        "Historical pilot tests remain retrospective and do not correct all model families explored before P3b.",
        "", "## Reproduce","", "```powershell",
        "& .\\.venv\\Scripts\\python.exe -m lotto7x7.log_product_p3c_exact_iid_v027 --p3b results/v027_p3b_pilot.json --out results/v027_p3c_exact_iid.json --md-out results/v027_p3c_exact_iid.md",
        "```",""
    ])
    return "\n".join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p3b",type=Path,required=True)
    parser.add_argument("--out",type=Path,default=Path("results/v027_p3c_exact_iid.json"))
    parser.add_argument("--md-out",type=Path,default=None)
    args=parser.parse_args(argv)
    out,md=args.out,args.md_out or args.out.with_suffix(".md")
    if out.resolve()==md.resolve() or out.exists() or md.exists():
        raise FileExistsError("two unused distinct JSON/Markdown report paths required")
    raw=args.p3b.read_bytes()
    result=audit_pilot(json.loads(raw))
    result["input_p3b_report_sha256"]=sha256(raw).hexdigest()
    out.parent.mkdir(parents=True,exist_ok=True)
    md.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)
        stream.write("\n")
    md.write_text(markdown_report(result),encoding="utf-8")
    print(f"[P3C EXACT] {result['decision']}")
    if "historical" in result:
        n=result["historical"]["fixed_checkpoints"]
        for name in ("primary","distinct_sums","random"):
            test=result["historical"]["exact_binomial_tests"][name]
            print(f"[{name.upper()}] {test['successes']}/{n}; exact upper-tail p={test['upper_tail_probability']:.8f}")
        print(f"[EXACT FAMILYWISE] two-policy Bonferroni p<={result['two_policy_familywise_upper_tail']['adjusted_p_upper_bound']:.8f}")
    print(f"[REPORT] {out} / {md}")
    print("[SCOPE] Exact fair-null mathematical invariant, not retrospective replay of legacy P2 #18 or proof of predictive skill")


if __name__=="__main__":
    main()
