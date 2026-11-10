"""v0.27 P2: four disjoint tickets maximizing SUM of individual P1 repair gains.

The globally valid upper bound for four tickets is 4 * max_s G(s).
If a disjoint four-ticket witness attains it, global optimality is PROVED.
For the optional "four different sums" comparison the bound is the sum
of the four highest *distinct* sum-bin gains. An attained bound likewise
proves optimality subject to that extra diversity constraint.

If no bound-attaining witness is found within a bounded deterministic
search, output is explicitly HEURISTIC_UNCERTIFIED, never "globally optimal".
Four tickets are four ALTERNATIVE one-draw outcomes, not four appended draws.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
from random import Random

from .coverage_v06 import exact_coverage
from .deficit_compensation_v025 import REFERENCE, validate_ticket
from .distribution_repair_v027 import LOW, exact_theory
from .distribution_repair_scoring_v027 import (
    SCHEMA as P1_SCHEMA, conditional_repair, score_ticket,
)
from .official_prize_v026 import PRIZE_ORDER, official_prize_coverage

SCHEMA = "lotto7x7.distribution_repair.v0.27.p2"
PORTFOLIO_SIZE = 4
ALL_NUMBERS = 49
RANKING_RULE = "maximize sum of exact integer rank keys (N+1)*theory_count - M*history_count"


def validate_four_disjoint(tickets) -> tuple[tuple[int, ...], ...]:
    """Reject overlapping, malformed, duplicated or unsorted four-line bets."""
    if not isinstance(tickets, (tuple, list)) or len(tickets) != PORTFOLIO_SIZE:
        raise ValueError("exactly four tickets required")
    validated = []
    used = set()
    for row in tickets:
        if not isinstance(row, (tuple, list)):
            raise ValueError("each ticket must be a sequence of six numbers")
        normalized = validate_ticket(row)
        if tuple(row) != normalized:
            raise ValueError("each ticket must already be sorted")
        if used.intersection(normalized):
            raise ValueError("four tickets must be pairwise disjoint")
        used.update(normalized)
        validated.append(normalized)
    if len(used) != 24:
        raise AssertionError("invalid four-ticket union size")
    return tuple(validated)


def verify_p1(p1: dict) -> list[dict]:
    """Recompute full 259-bin conditional law from archived P1 integer counts.

    This is an integrity gate against stale/malformed P1 inputs. It is not a
    cryptographic authentication of the original source; provenance is P0.
    """
    if not isinstance(p1, dict) or p1.get("schema") != P1_SCHEMA:
        raise ValueError("requires a complete v0.27 P1 score report")
    rows = p1.get("sums")
    source = p1.get("source", {})
    if not isinstance(rows, list) or len(rows) != 259:
        raise ValueError("P1 requires all 259 sum bins")
    theory = exact_theory()
    if p1.get("theory", {}).get("checksum_sha256") != theory["checksum_sha256"]:
        raise ValueError("P1 exact theory checksum mismatch")
    if p1.get("metric", {}).get("name") != "squared_l2_probability_v1":
        raise ValueError("P1 primary metric changed")
    n = source.get("historical_draws")
    if type(n) is not int or n < 1 or not isinstance(source.get("cutoff"), str):
        raise ValueError("P1 source metadata/cutoff missing")
    if not isinstance(source.get("analysis_input_sha256"), str) or len(source["analysis_input_sha256"]) != 64:
        raise ValueError("P1 historical prefix SHA-256 missing")
    if any(type(row.get("sum")) is not int or row["sum"] != LOW + i
           or type(row.get("observed_count")) is not int or row["observed_count"] < 0
           for i, row in enumerate(rows)):
        raise ValueError("P1 sum ordering/counts invalid")
    counts = [row["observed_count"] for row in rows]
    if sum(counts) != n:
        raise ValueError("P1 observed counts do not sum to historical N")
    recalculated = conditional_repair(counts, theory["counts"])
    if not isclose(recalculated["loss_before"], p1.get("baseline_loss", float("nan")),
                   rel_tol=1e-12, abs_tol=1e-17):
        raise ValueError("P1 baseline loss inconsistent with counts")
    for i, (stored, checked) in enumerate(zip(rows, recalculated["rows_by_index"])):
        if stored.get("rank_integer") != checked["rank_integer"] or stored.get("theory_combinations") != theory["counts"][i]:
            raise ValueError("P1 exact sum-bin rank or multiplicity altered")
        for field in ("gain", "loss_after", "expected_count"):
            if not isinstance(stored.get(field), (float, int)) or not isclose(
                stored[field], checked[field], rel_tol=1e-10, abs_tol=2e-15
            ):
                raise ValueError(f"P1 sum {LOW+i} {field} inconsistent")
    return rows


def _try_target_sums(targets: tuple[int, ...], *, seed: int,
                     restarts: int, trials: int):
    """Find a disjoint feasible witness for given four ordered total sums.

    Bounded randomized constructive feasibility search, NOT a proof of
    non-existence on failure. The optimum proof comes only from attained
    independent upper bounds, never from this search's exhaustiveness.
    """
    rng = Random(seed)
    attempted = 0
    for retry in range(restarts):
        used = set()
        chosen = []
        for target in targets:
            remaining = [v for v in range(1, 50) if v not in used]
            found = None
            for _ in range(trials):
                attempted += 1
                first_five = rng.sample(remaining, 5)
                last = target - sum(first_five)
                if 1 <= last <= 49 and last not in used and last not in first_five:
                    found = tuple(sorted((*first_five, last)))
                    break
            if found is None:
                break
            chosen.append(found)
            used.update(found)
        if len(chosen) == 4:
            return validate_four_disjoint(chosen), {"restarts_used": retry + 1, "proposals": attempted}
    return None, {"restarts_used": restarts, "proposals": attempted}


def _sample_fallback(rows: list[dict], *, distinct_sums: bool,
                     seed: int, samples: int):
    """Guaranteed-feasible bounded history-scored baseline, not exact search."""
    rng = Random(seed)
    best, best_key = None, None
    valid = 0
    for _ in range(samples):
        values = rng.sample(range(1, 50), 24)
        tickets = tuple(tuple(sorted(values[j * 6:(j + 1) * 6])) for j in range(4))
        sums = tuple(sum(t) for t in tickets)
        if distinct_sums and len(set(sums)) != 4:
            continue
        valid += 1
        key = sum(rows[s - LOW]["rank_integer"] for s in sums)
        canon = tuple(sorted(tickets))
        if best_key is None or key > best_key or (key == best_key and canon < best):
            best, best_key = canon, key
    if best is None:
        raise ValueError("no feasible portfolio in fallback sample (increase samples)")
    return validate_four_disjoint(best), {"random_portfolios": samples, "feasible_random_portfolios": valid}


def _coverage_audit(tickets, reference_official: dict, reference_main: dict):
    official = official_prize_coverage(tickets)
    main = exact_coverage(tickets)
    official_fields = (
        "any_prize_outcomes", "awarded_ticket_count_outcomes",
    )
    for field in official_fields:
        if official[field] != reference_official[field]:
            raise AssertionError(f"v0.26 exact official coverage parity failed: {field}")
    for tier in PRIZE_ORDER:
        for field in ("outcomes",):
            if official["any_ticket_by_tier"][tier][field] != reference_official["any_ticket_by_tier"][tier][field]:
                raise AssertionError(f"v0.26 official tier parity failed: {tier}")
        if official["expected_tickets_by_tier"][tier]["ticket_outcomes"] != reference_official["expected_tickets_by_tier"][tier]["ticket_outcomes"]:
            raise AssertionError(f"official expected award count mismatch: {tier}")
    if main["max_hits_counts_0_to_6"] != reference_main["max_hits_counts_0_to_6"]:
        raise AssertionError("v0.26 exact six-main-ball coverage parity failed")
    return {
        "exact_official_eight_tier_parity": True,
        "exact_main_ball_coverage_parity": True,
        "official_probability": official["any_prize_probability"],
        "official_award_outcomes": official["any_prize_outcomes"],
        "official_total_outcomes": official["total_outcomes"],
        "awarded_ticket_count_outcomes": official["awarded_ticket_count_outcomes"],
        "main_max_hits_counts_0_to_6": main["max_hits_counts_0_to_6"],
        "per_tier_any_award_probability": {
            t: official["any_ticket_by_tier"][t]["probability"] for t in PRIZE_ORDER
        },
        "reference": "v0.26 proven pairwise-disjoint global maximum; same eight-tier profile",
    }


def _build_result(p1: dict, rows: list[dict], *, different_sums: bool,
                  seed: int, restarts: int, trials: int, samples: int,
                  official_ref: dict, main_ref: dict):
    ranked = sorted(rows, key=lambda r: (-r["rank_integer"], r["sum"]))
    target = tuple(r["sum"] for r in ranked[:4]) if different_sums else (ranked[0]["sum"],) * 4
    upper_key = sum(rows[s - LOW]["rank_integer"] for s in target)
    upper_float = sum(rows[s - LOW]["gain"] for s in target)

    witness, search = _try_target_sums(target, seed=seed, restarts=restarts, trials=trials)
    if witness is None:
        witness, search_more = _sample_fallback(rows, distinct_sums=different_sums, seed=seed, samples=samples)
        search.update(search_more)
    ordered_tickets = tuple(sorted(witness))
    validate_four_disjoint(ordered_tickets)
    sums = [sum(t) for t in ordered_tickets]
    if different_sums and len(set(sums)) != 4:
        raise AssertionError("diversity constraint violated")
    actual_key = sum(rows[s - LOW]["rank_integer"] for s in sums)
    if actual_key > upper_key:
        raise AssertionError("invalid portfolio upper bound")
    globally_optimal = actual_key == upper_key
    scores = [score_ticket(p1, t) for t in ordered_tickets]
    objective = sum(r["gain"] for r in scores)
    reference_parity = _coverage_audit(ordered_tickets, official_ref, main_ref)
    return {
        "portfolio_name": "distinct_sum_diversity" if different_sums else "unconstrained_sum_optimum",
        "constraint": "four distinct ticket sums" if different_sums else "four tickets may have equal sums",
        "tickets": [list(t) for t in ordered_tickets],
        "sum_bins": sums,
        "unique_ticket_sums": len(set(sums)),
        "unique_numbers": len({v for t in ordered_tickets for v in t}),
        "per_ticket": scores,
        "objective": {
            "definition": "sum of four separately evaluated G_N(C_j), NOT loss after appending four draws",
            "gain_sum": objective,
            "gain_sum_micro_l2": objective * 1e6,
            "exact_integer_objective": actual_key,
            "exact_integer_upper_bound": upper_key,
            "upper_bound_gain_sum": upper_float,
            "gap_in_exact_integer_units": upper_key - actual_key,
        },
        "certificate": {
            "globally_optimal_proven": globally_optimal,
            "status": "GLOBAL_OPTIMUM_CERTIFIED" if globally_optimal else "HEURISTIC_UNCERTIFIED",
            "theorem": (
                "Each legal ticket's gain is bounded by max_s G(s); four disjoint tickets "
                "attain 4*max_s G(s). Integer-rank equality proves global optimality."
                if not different_sums else
                "With four distinct sums, the four highest distinct sum gains bound "
                "every feasible portfolio. A disjoint witness attains that exact bound."
            ) if globally_optimal else
            "The target upper bound was not attained by bounded feasible search. No global-optimality claim.",
            "target_upper_bound_sums": list(target),
            "algorithm": "deterministic seeded bounded sum-target witness; random baseline if infeasible within budget",
            "search": search,
            "seed": seed,
            "restarts_budget": restarts,
            "proposals_per_ticket_budget": trials,
            "fallback_samples": samples,
            "tied_portfolios": "one deterministic witness; no claim of lexicographically smallest global optimizer",
        },
        "coverage": reference_parity,
    }


def optimize(p1: dict, *, seed: int = 649, restarts: int = 64,
             trials: int = 1000, samples: int = 2000) -> dict:
    """Primary P2 and four-distinct-sums sensitivity portfolio, both audited."""
    if any(type(v) is not int or v < minimum for v, minimum in (
        (seed, 0), (restarts, 0), (trials, 1), (samples, 1)
    )):
        raise ValueError("invalid seed/search budgets")
    rows = verify_p1(p1)
    official_ref = official_prize_coverage(REFERENCE)
    main_ref = exact_coverage(REFERENCE)
    kwargs = dict(seed=seed, restarts=restarts, trials=trials, samples=samples,
                  official_ref=official_ref, main_ref=main_ref)
    primary = _build_result(p1, rows, different_sums=False, **kwargs)
    diverse = _build_result(p1, rows, different_sums=True, **kwargs)
    delta = primary["objective"]["gain_sum"] - diverse["objective"]["gain_sum"]
    return {
        "schema": SCHEMA,
        "stage": "P2_FOUR_TICKET_OPTIMIZATION",
        "decision": "REPAIR_DESCRIPTIVE_PASS; PREDICTIVE_ADVANTAGE_NOT_EVALUATED",
        "source": p1["source"],
        "metric": p1["metric"],
        "theory": p1["theory"],
        "p1_iid_reference": {
            "expected_single_draw_gain": p1["iid_null"]["expected_gain"],
            "stddev_single_draw_gain": p1["iid_null"]["stddev_gain"],
            "warning": "Single-draw IID null; four-ticket scores are NOT four independent real draws.",
        },
        "optimization": {
            "count": 4,
            "numbers_per_ticket": 6,
            "distinct_numbers_required": 24,
            "primary_objective": RANKING_RULE,
            "objective_is_four_separate_counterfactual_gains": True,
            "no_joint_append_to_history": True,
            "primary": primary,
            "distinct_sums_sensitivity": diverse,
            "diversity_tradeoff": {
                "gain_cost_of_distinct_sums": delta,
                "gain_cost_micro_l2": delta * 1e6,
                "coverage_probability_change": diverse["coverage"]["official_probability"] -
                                               primary["coverage"]["official_probability"],
                "interpretation": "More varied ticket sums are an OPTIONAL constraint, not an improvement in fair draw odds.",
            },
        },
        "note": "All four-ticket portfolios have exact fair IID Coverage parity because they use 24 different numbers. Historical repair is not predictive performance.",
    }


def markdown_report(report: dict) -> str:
    """Reproducible P2 certificate, exact coverage and diversity comparison."""
    top = report["optimization"]
    s = report["source"]
    lines = [
        "# Lotto649 v0.27 — P2 Four-Ticket Repair Optimization", "",
        f"- **Frozen cutoff:** {s['cutoff']} / #{s['last_draw_id']}, N={s['historical_draws']}",
        f"- **Historical prefix SHA256:** `{s['analysis_input_sha256']}`",
        f"- **Theory SHA256:** `{report['theory']['checksum_sha256']}`",
        "- **Objective:** sum of four SEPARATE single-outcome G values; no four-draw append to history.",
        "- **Coverage:** exact official eight-tier probabilities, 24 numbers mutually distinct.",
        "- **Decision:** descriptive optimization, not improved next-draw odds.", "",
    ]
    for key, title in (("primary", "Primary: unrestricted sums"),
                       ("distinct_sums_sensitivity", "Sensitivity: four different sums")):
        selected = top[key]
        lines.extend([
            f"## {title}", "",
            f"- **Certificate:** {selected['certificate']['status']}",
            f"- **Global upper integer bound:** {selected['objective']['exact_integer_upper_bound']}",
            f"- **Attained integer objective:** {selected['objective']['exact_integer_objective']}",
            f"- **Portfolio ΣG:** {selected['objective']['gain_sum_micro_l2']:+.9f} × 10⁻⁶",
            f"- **Exact official P(any award):** {selected['coverage']['official_probability']:.15g}",
            f"- **Unique numbers/sums:** {selected['unique_numbers']}/{selected['unique_ticket_sums']}",
            "", "| Ticket | Numbers | Sum | G (×10⁶) | Rank |",
            "|:---:|:---|---:|---:|---:|",
        ])
        for idx, ticket in enumerate(selected["per_ticket"]):
            lines.append(f"| {chr(65 + idx)} | {' '.join(f'{v:02d}' for v in ticket['ticket'])} | "
                         f"{ticket['sum']} | {ticket['gain_micro_l2']:+.6f} | {ticket['rank']} |")
        lines.extend(["", selected["certificate"]["theorem"], ""])
    delta = top["diversity_tradeoff"]
    lines.extend(["## Diversity trade-off", "",
                  f"- **ΣG cost of different sums:** {delta['gain_cost_micro_l2']:+.9f} × 10⁻⁶.",
                  "- **Official prize Coverage change:** exactly zero.",
                  "- The same-sum optimum does not imply that one sum is likelier to appear next.",
                  "- The all-distinct alternative does not imply a forecasting advantage.",
                  "- Portfolio score must never be interpreted as an actual four-draw posterior loss.", "",
                  "## Reproduce", "", "```powershell",
                  "& .\\.venv\\Scripts\\python.exe -m lotto7x7.distribution_repair_portfolio_v027 --p1 results/v027_p1_repair.json --out results/v027_p2_portfolio.json --md-out results/v027_p2_portfolio.md",
                  "```", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1", type=Path, required=True, help="frozen P1 JSON report")
    parser.add_argument("--out", type=Path, default=Path("results/v027_p2_portfolio.json"))
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=649)
    parser.add_argument("--restarts", type=int, default=64)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=2000)
    args = parser.parse_args(argv)
    json_path = args.out
    md_path = args.md_out or args.out.with_suffix(".md")
    if json_path.resolve() == md_path.resolve() or json_path.exists() or md_path.exists():
        raise FileExistsError("distinct unused JSON/Markdown paths required; no overwrite")
    raw = args.p1.read_bytes()
    p1 = json.loads(raw)
    report = optimize(p1, seed=args.seed, restarts=args.restarts,
                      trials=args.trials, samples=args.samples)
    report["input_p1_file_sha256"] = sha256(raw).hexdigest()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    for name, row in (("PRIMARY", report["optimization"]["primary"]),
                      ("DIVERSE", report["optimization"]["distinct_sums_sensitivity"])):
        print(f"[P2 {name}] {row['certificate']['status']} sum_bins={row['sum_bins']} "
              f"ΣG={row['objective']['gain_sum_micro_l2']:+.8f} x10^-6")
        for j, ticket in enumerate(row["tickets"]):
            print(f"  {chr(65+j)}: {' '.join(f'{v:02d}' for v in ticket)}")
    print(f"[COVERAGE] P(any official award)={report['optimization']['primary']['coverage']['official_probability']:.15g}; exact v0.26 eight-tier parity PASS")
    print(f"[REPORT] {json_path} / {md_path}")
    print("[SCOPE] Four alternative hypothetical next outcomes, not four appended draws; no predictive claim")


if __name__ == "__main__":
    main()
