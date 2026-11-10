"""v0.27 Issue #15 P2: four disjoint dual-positive Sum+Log Product tickets.

Freeze the objective BEFORE searching:
  weight_S = round_half_even(10**12 / IID_SD(G_sum))
  weight_L = round_half_even(10**12 / IID_SD(G_log))
  score(C) = weight_S * exact_gain_numerator_S(sum(C))
           + weight_L * exact_gain_numerator_L(log_bin(prod(C)))

Both gains have the SAME positive exact rational denominator because both
theories have M=C(49,6) equally likely outcomes and the same N historical
draws. This supplies an exact integer-valued objective and tie-break,
approximately equal to the 50/50 sum of gains in conditional IID SD units.

A universal *portfolio* bound is four times the largest attainable SINGLE
dual-positive cell score. If four disjoint legal tickets attain that bound,
global optimality is mathematically certified without exhaustive portfolio
search. A bounded deterministic witness search IS NOT a proof of infeasibility.
Otherwise results are explicitly HEURISTIC_UNCERTIFIED.

Sum and Log Product are correlated. Weighted ranking is descriptive, NOT a
joint tail p-value, and none of four tickets are appended to draw history.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
from hashlib import sha256
from itertools import combinations
import json
from math import comb, isqrt, prod
from pathlib import Path
from random import Random

from .coverage_v06 import exact_coverage
from .deficit_compensation_v025 import REFERENCE
from .distribution_repair_portfolio_v027 import optimize as sum_only_optimize, validate_four_disjoint, verify_p1
from .distribution_repair_scoring_v027 import score_ticket as sum_score_ticket
from .log_product_dual_scoring_v027 import (
    SCHEMA as DUAL_P1_SCHEMA,
    exact_gain_numerators,
    score_ticket as dual_score_ticket,
    verify_log_p0,
)
from .official_prize_v026 import PRIZE_ORDER, official_prize_coverage

SCHEMA = "lotto7x7.log_product.v0.27.p2"
N = 49
K = 6
TOTAL = comb(N, K)
SUM_OFFSET = 21
WEIGHT_SCALE = 10 ** 12


def _exact_variance_numerator(numerators: list[int], multiplicities: list[int]) -> tuple[int, int]:
    """Var(integer gain numerator) = A / M**2 under exact conditional IID.

    Theory-multiplicities are the *exact* fair combination counts, not
    uniform weighting over histogram bins. No floating point occurs.
    """
    if (not numerators or len(numerators) != len(multiplicities)
            or any(type(x) is not int for x in numerators)
            or any(type(x) is not int or x < 0 for x in multiplicities)):
        raise ValueError("exact IID variance requires matching integer vectors")
    m = sum(multiplicities)
    if m < 1:
        raise ValueError("empty IID theoretical universe")
    s1 = sum(a * c for a, c in zip(numerators, multiplicities))
    s2 = sum(a * a * c for a, c in zip(numerators, multiplicities))
    a = m * s2 - s1 * s1
    if a <= 0:
        raise ValueError("zero conditional IID gain variance")
    return a, m


def _round_ratio_over_sqrt(k: int, a: int) -> int:
    """Return round-half-even(k/sqrt(a)) using *only exact integers*.

    Let floor(k/sqrt(a)) = floor(sqrt(k*k // a)) = q.
    Compare 4*k*k with a*(2*q+1)^2 to decide the half-unit.
    No float, Decimal precision or Python-platform dependence.
    """
    if type(k) is not int or type(a) is not int or k < 0 or a <= 0:
        raise ValueError("k must be nonnegative and a must be positive integers")
    q = isqrt((k*k) // a)
    lhs, rhs = 4*k*k, a*(2*q+1)**2
    return q + (lhs > rhs or (lhs == rhs and (q % 2 == 1)))


def frozen_weights(sum_num: list[int], sum_counts: list[int],
                   log_num: list[int], log_counts: list[int],
                   gain_denominator: int) -> tuple[int, int]:
    """Two reproducible reciprocal conditional-IID-SD weights.

    Each G_bin = integer_gain_numerator / gain_denominator.
    Exact weighted IID Var(G) = A / (M^2 * gain_denominator^2).
    Thus W=round_half_even(10^12 * M * gain_denominator/sqrt(A))
    can be calculated with pure integer comparisons.
    """
    if type(gain_denominator) is not int or gain_denominator <= 0:
        raise ValueError("gain_denominator must be positive integer")
    a_sum, m_sum = _exact_variance_numerator(sum_num, sum_counts)
    a_log, m_log = _exact_variance_numerator(log_num, log_counts)
    if m_sum != m_log:
        raise ValueError("Sum and Log feature IID universes must match")
    k = WEIGHT_SCALE * m_sum * gain_denominator
    w_sum, w_log = _round_ratio_over_sqrt(k, a_sum), _round_ratio_over_sqrt(k, a_log)
    if not (w_sum > 0 and w_log > 0):
        raise AssertionError("positive standardized weights required")
    return w_sum, w_log

def score_joint_cell(sum_idx: int, log_idx: int,
                     sum_num: list[int], log_num: list[int],
                     weights: tuple[int, int]) -> int:
    return weights[0] * sum_num[sum_idx] + weights[1] * log_num[log_idx]


def enumerate_joint_counts(*, n: int = N, k: int = K,
                           thresholds: list[int]) -> tuple[list[list[int]], int]:
    """Count every fair ticket's joint (sum, exact integer product bin).

    Separate helper permits small-universe exhaustive cross-check. This is
    candidate SUPPORT verification, not P3 chronological IID calibration.
    """
    if (type(n) is not int or type(k) is not int
            or not 2 <= k < n <= 49 or not thresholds
            or sorted(set(thresholds)) != thresholds):
        raise ValueError("invalid universe or log thresholds")
    support_min = k * (k + 1) // 2
    support_max = k * (2 * n - k + 1) // 2
    counts = [[0] * (len(thresholds) - 1) for _ in range(support_max - support_min + 1)]
    universe = 0
    for ticket in combinations(range(1, n + 1), k):
        index = bisect_right(thresholds, prod(ticket)) - 1
        if not 0 <= index < len(thresholds) - 1:
            raise AssertionError("theoretical log product bin missing")
        counts[sum(ticket) - support_min][index] += 1
        universe += 1
    if universe != comb(n, k) or sum(sum(r) for r in counts) != universe:
        raise AssertionError("joint exact enumeration failed normalization")
    return counts, universe


def pareto_frontier(cells: list[dict]) -> list[dict]:
    """Feature-pair non-dominance, including exact ties deterministically.

    Cells are hist-class outcomes; no false claim of ticket independence.
    """
    ordered = sorted(cells, key=lambda c: (-c["sum_numerator"], -c["log_numerator"], c["sum"], c["log_bin"]))
    highest_log = None
    frontier = []
    for cell in ordered:
        if highest_log is None or cell["log_numerator"] > highest_log:
            frontier.append(cell)
            highest_log = cell["log_numerator"]
    return sorted(frontier, key=lambda c: (-c["objective_key"], c["sum"], c["log_bin"]))


def _ticket_mask(ticket) -> int:
    return sum(1 << (n - 1) for n in ticket)


def _witness_for_cells(cells, pools, *, rng: Random, restarts: int):
    """Construct a disjoint witness; a failed search is NOT infeasibility proof."""
    if any(not pools.get(cell) for cell in cells):
        return None
    for _ in range(restarts):
        used, selected = 0, []
        for cell in cells:
            sample = pools[cell]
            start = rng.randrange(len(sample))
            match = None
            for shift in range(len(sample)):
                mask, ticket = sample[(start + shift) % len(sample)]
                if not (used & mask):
                    match = (mask, ticket)
                    break
            if match is None:
                break
            used |= match[0]
            selected.append(match[1])
        if len(selected) == 4:
            return validate_four_disjoint(tuple(sorted(selected)))
    return None



def _exact_portfolio_key(tickets, thresholds, sum_num, log_num, weights):
    """Shared integer objective for independent comparisons and baselines."""
    return sum(score_joint_cell(
        sum(t) - SUM_OFFSET,
        bisect_right(thresholds, prod(t)) - 1,
        sum_num, log_num, weights) for t in tickets)


def _best_greedy_from_pools(cells: list[dict], pools: dict, *,
                            rng: Random, restarts: int,
                            sum_num: list[int], log_num: list[int],
                            weights: tuple[int, int], thresholds: list[int]):
    """Bounded deterministic greedy best-of-restarts over high-ranked cells.

    Scores every feasible ticket EXACTLY by its feature-cell integer key.
    This search is neither exhaustive nor a certification by itself.
    """
    best, best_key, valid_trials = None, None, 0
    for _ in range(restarts):
        used, selected = 0, []
        for step in range(4):
            match = None
            for cell in cells:
                key = (cell["sum"], cell["log_bin"])
                arr = pools.get(key)
                if not arr:
                    continue
                start = rng.randrange(len(arr))
                for offset in range(len(arr)):
                    mask, ticket = arr[(start + offset) % len(arr)]
                    if not (mask & used):
                        match = (mask, ticket)
                        break
                if match is not None:
                    break
            if match is None:
                break
            used |= match[0]
            selected.append(match[1])
        if len(selected) != 4:
            continue
        valid_trials += 1
        selected = validate_four_disjoint(tuple(sorted(selected)))
        value = _exact_portfolio_key(
            selected, thresholds,
            sum_num, log_num, weights)
        if best_key is None or value > best_key or (value == best_key and selected < best):
            best, best_key = selected, value
    return best, {"greedy_restarts": restarts,
                  "greedy_feasible_restarts": valid_trials}

def _reservoir_pools(cells, *, per_cell: int, seed: int, thresholds: list[int]) -> tuple[dict, dict]:
    """Uniform bounded reservoir per prespecified (sum,log) cell, two-pass.

    First pass builds the exact joint support. Second pass stores a bounded
    seeded sample of legal tickets in the chosen high-score cells.
    """
    wanted = set(cells)
    rng = Random(seed)
    pools: dict[tuple[int, int], list] = {c: [] for c in wanted}
    counts = dict.fromkeys(wanted, 0)
    sums = {s for s, _ in wanted}
    for ticket in combinations(range(1, 50), 6):
        s = sum(ticket)
        if s not in sums:
            continue
        li = bisect_right(thresholds, prod(ticket)) - 1
        cell = (s, li)
        if cell not in wanted:
            continue
        counts[cell] += 1
        arr = pools[cell]
        item = (_ticket_mask(ticket), ticket)
        if len(arr) < per_cell:
            arr.append(item)
        else:
            j = rng.randrange(counts[cell])
            if j < per_cell:
                arr[j] = item
    return pools, counts


def _fallback_four(dual_report: dict, *, rng: Random, trials: int, distinct_sums: bool,
                   numerator_sum: list[int], numerator_log: list[int], weights: tuple[int, int]):
    """Sample feasible disjoint 24-label partitions, retaining best objective.

    NEVER a global certificate. If no valid sample, report uncertified.
    """
    best, best_key = None, None
    accepted = 0
    for _ in range(trials):
        chosen = rng.sample(range(1, 50), 24)
        tickets = tuple(tuple(sorted(chosen[6*i:6*i+6])) for i in range(4))
        sums = tuple(sum(t) for t in tickets)
        if distinct_sums and len(set(sums)) < 4:
            continue
        score = [dual_score_ticket(dual_report, t) for t in tickets]
        if not all(x["dual_positive"] for x in score):
            continue
        accepted += 1
        objective = sum(score_joint_cell(
            sum(t)-SUM_OFFSET, x["log_product_bin_index"],
            numerator_sum, numerator_log, weights) for t, x in zip(tickets, score))
        canon = validate_four_disjoint(tuple(sorted(tickets)))
        if best_key is None or objective > best_key or (objective == best_key and canon < best):
            best_key, best = objective, canon
    return best, {"trials": trials, "dual_positive_24_number_partitions": accepted}


def _exact_coverage_parity(tickets, ref_official, ref_hits) -> dict:
    official = official_prize_coverage(tickets)
    hits = exact_coverage(tickets)
    if official["any_prize_outcomes"] != ref_official["any_prize_outcomes"]:
        raise AssertionError("official at-least-one-prize coverage parity violated")
    if official["awarded_ticket_count_outcomes"] != ref_official["awarded_ticket_count_outcomes"]:
        raise AssertionError("official multiplicity histogram parity violated")
    if hits["max_hits_counts_0_to_6"] != ref_hits["max_hits_counts_0_to_6"]:
        raise AssertionError("main hit coverage parity violated")
    for tier in PRIZE_ORDER:
        if official["any_ticket_by_tier"][tier]["outcomes"] != ref_official["any_ticket_by_tier"][tier]["outcomes"]:
            raise AssertionError(f"official eight-tier parity broken for {tier}")
        if official["expected_tickets_by_tier"][tier]["ticket_outcomes"] != ref_official["expected_tickets_by_tier"][tier]["ticket_outcomes"]:
            raise AssertionError(f"expected awards parity broken for {tier}")
    return {
        "exact_official_eight_tier_parity": True,
        "exact_main_hit_coverage_parity": True,
        "any_prize_probability": official["any_prize_probability"],
        "any_prize_outcomes": official["any_prize_outcomes"],
        "total_official_outcomes": official["total_outcomes"],
        "by_tier": {tier: official["any_ticket_by_tier"][tier] for tier in PRIZE_ORDER},
        "awarded_ticket_count_outcomes": official["awarded_ticket_count_outcomes"],
        "max_hit_counts": hits["max_hits_counts_0_to_6"],
    }


def _describe_portfolio(tickets, *, name: str, upper: int,
                        dual_p1: dict, sum_num: list[int], log_num: list[int],
                        weights: tuple[int, int], coverage_ref, hits_ref,
                        search: dict):
    if tickets is None:
        return {"name": name, "status": "INFEASIBLE_UNCERTIFIED",
                "certified_global_optimum": False, "tickets": [],
                "upper_bound_exact_integer": str(upper), "search": search,
                "note": "Bounded search found no valid four-disjoint witness; not a proof of infeasibility."}
    tickets = validate_four_disjoint(tickets)
    detail = [dual_score_ticket(dual_p1, t) for t in tickets]
    if not all(t["dual_positive"] for t in detail):
        raise AssertionError("candidate violates hard dual-positive conditions")
    objective = sum(score_joint_cell(t["sum"]-SUM_OFFSET, t["log_product_bin_index"],
                      sum_num, log_num, weights) for t in detail)
    if objective > upper:
        raise AssertionError("mathematical upper bound violated")
    proof = objective == upper
    return {
        "name": name,
        "status": "GLOBAL_OPTIMUM_CERTIFIED" if proof else "HEURISTIC_UNCERTIFIED",
        "certified_global_optimum": proof,
        "tickets": [list(t) for t in tickets],
        "unique_numbers": len({x for t in tickets for x in t}),
        "unique_sum_bins": len(set(sum(t) for t in tickets)),
        "unique_log_bins": len(set(t["log_product_bin_index"] for t in detail)),
        "sum_bins": [t["sum"] for t in detail],
        "log_bins": [t["log_product_bin_index"] for t in detail],
        "ticket_details": detail,
        "objective_exact_integer": str(objective),
        "upper_bound_exact_integer": str(upper),
        "gap_exact_integer": str(upper - objective),
        "sum_gain_total": sum(t["sum_gain"] for t in detail),
        "log_gain_total": sum(t["log_gain"] for t in detail),
        "iid_standardized_score_approx": 0.5*sum(
            t["sum_gain"] * weights[0]/WEIGHT_SCALE +
            t["log_gain"] * weights[1]/WEIGHT_SCALE for t in detail),
        "coverage": _exact_coverage_parity(tickets, coverage_ref, hits_ref),
        "certificate": (
            "Each legal dual-positive ticket has integer objective <= the best "
            "attainable joint feature-cell score. Four pairwise-disjoint "
            "tickets reach four times that score; hence global optimum."
            if proof and name == "dual_positive_unrestricted" else
            "For each of four distinct ticket sums, its best attainable "
            "dual-positive log bin supplies a valid per-sum upper bound. "
            "The four best distinct sums upper-bound every admissible "
            "portfolio, and this witness reaches that exact bound."
            if proof else
            "No matching exact upper-bound witness was found. This is "
            "a bounded seeded search, not exhaustive portfolio optimization."
        ),
        "search": search,
    }


def _check_inputs(sum_p1, log_p0, dual_p1):
    sum_rows = verify_p1(sum_p1)
    theory, log_observed = verify_log_p0(log_p0, sum_p1)
    if dual_p1.get("schema") != DUAL_P1_SCHEMA:
        raise ValueError("expected dual-P1 JSON")
    for field in ("cutoff", "historical_draws", "last_draw_id", "analysis_input_sha256"):
        if dual_p1.get("source", {}).get(field) != sum_p1["source"][field]:
            raise ValueError("dual P1/Sum P1 source mismatch")
    if dual_p1.get("log_theory", {}).get("checksum_sha256") != theory["checksum_sha256"]:
        raise ValueError("dual P1 theory mismatch")
    if dual_p1.get("metric") != "squared_l2_probability_v1":
        raise ValueError("dual P1 frozen metric mismatch")
    sum_nums = exact_gain_numerators(
        [r["observed_count"] for r in sum_rows],
        [r["theory_combinations"] for r in sum_p1["sums"]])
    log_nums = exact_gain_numerators(log_observed, theory["counts"])
    if sum_nums["denominator"] != log_nums["denominator"]:
        raise AssertionError("features must have same uniform universe and historical count")
    if (len(dual_p1.get("sum_bins", [])) != 259 or
            len(dual_p1.get("log_bins", [])) != len(theory["counts"])):
        raise ValueError("dual P1 missing full per-bin score table")
    if (any(r["exact_gain_numerator"] != v for r, v in zip(dual_p1["sum_bins"],sum_nums["numerators"])) or
            any(r["exact_gain_numerator"] != v for r,v in zip(dual_p1["log_bins"], log_nums["numerators"]))):
        raise ValueError("dual P1 exact gain numerators are inconsistent")
    return sum_rows, theory, sum_nums, log_nums


def optimize(sum_p1: dict, log_p0: dict, dual_p1: dict, *, seed: int = 649,
             pool_per_cell: int = 800, cell_limit: int = 24,
             restarts: int = 512, fallback_trials: int = 15000) -> dict:
    """Optimize both dual-positive portfolio variants, with honest certificates."""
    options = (("seed",seed,0),("pool_per_cell",pool_per_cell,1),
               ("cell_limit",cell_limit,4),("restarts",restarts,0),
               ("fallback_trials",fallback_trials,0))
    if any(type(value) is not int or value < lower for _,value,lower in options):
        raise ValueError("invalid seed or search budget")
    _, log_theory, s_gain, l_gain = _check_inputs(sum_p1, log_p0, dual_p1)
    from .distribution_repair_v027 import exact_theory
    exact_sum_counts = exact_theory()["counts"]
    weights = frozen_weights(
        s_gain["numerators"], exact_sum_counts,
        l_gain["numerators"], log_theory["counts"],
        s_gain["denominator"])
    grid = log_theory["grid"]
    joint, total = enumerate_joint_counts(thresholds=grid["integer_product_thresholds"])
    if total != TOTAL:
        raise AssertionError("wrong joint mass")
    expected_s = exact_sum_counts
    if [sum(row) for row in joint] != expected_s:
        raise AssertionError("joint row marginals != exact sum theory")
    if [sum(row[i] for row in joint) for i in range(grid["bin_count"])] != log_theory["counts"]:
        raise AssertionError("joint column marginals != exact Log Product theory")
    qualifies = 0
    cells = []
    for i, row in enumerate(joint):
        sn = s_gain["numerators"][i]
        if sn <= 0:
            continue
        for j, count in enumerate(row):
            ln = l_gain["numerators"][j]
            if count and ln > 0:
                qualifies += count
                cells.append({
                    "sum": i + SUM_OFFSET, "log_bin": j, "exact_joint_combinations": count,
                    "sum_numerator": sn, "log_numerator": ln,
                    "objective_key": score_joint_cell(i,j,s_gain["numerators"],
                                                       l_gain["numerators"],weights),
                })
    if qualifies != dual_p1["qualifying"]["dual_positive_ticket_count"]:
        raise ValueError("joint exact feasible count disagrees with frozen dual P1")
    cells.sort(key=lambda r: (-r["objective_key"],r["sum"],r["log_bin"]))
    if not cells:
        return {
            "schema": SCHEMA, "source": dual_p1["source"],
            "decision": "INFEASIBLE_NO_DUAL_POSITIVE_TICKETS",
            "qualifying": 0,
            "note": "No dual-positive feasible ticket exists; no portfolio proposed.",
        }
    top_cell = (cells[0]["sum"],cells[0]["log_bin"])
    upper = cells[0]["objective_key"] * 4
    by_sum = {}
    for cell in cells:
        by_sum.setdefault(cell["sum"],cell)
    diverse_best = sorted(by_sum.values(), key=lambda r: (-r["objective_key"],r["sum"]))[:4]
    if len(diverse_best) < 4:
        diverse_cells, diverse_upper = [], None
    else:
        diverse_cells = [(c["sum"],c["log_bin"]) for c in diverse_best]
        diverse_upper = sum(c["objective_key"] for c in diverse_best)
    chosen_cells = {(c["sum"],c["log_bin"]) for c in cells[:cell_limit]}
    chosen_cells.update([top_cell,*diverse_cells])
    pools, inspected_counts = _reservoir_pools(
        chosen_cells, per_cell=pool_per_cell, seed=seed,
        thresholds=grid["integer_product_thresholds"])
    if any(inspected_counts[key] != joint[key[0]-SUM_OFFSET][key[1]] for key in chosen_cells):
        raise AssertionError("reservoir candidate pool differs from exact theoretical joint support")
    rng = Random(seed)
    primary_witness = _witness_for_cells([top_cell]*4, pools, rng=rng, restarts=restarts)
    primary_search = {"attempt": "4 independent tickets all in best exact joint bin",
                      "target_cell": list(top_cell), "restarts": restarts,
                      "pool_per_cell": pool_per_cell, "candidate_cell_limit": cell_limit,
                      "pool_actual_size": len(pools[top_cell])}
    # A restricted four-in-the-same-cell witness may be impossible because
    # reaching a low Log Product often consumes scarce small labels. Search
    # across high-scoring attainable cells as a separate, honest heuristic.
    pool_candidate, pool_diagnostics = _best_greedy_from_pools(
        cells[:cell_limit], pools, rng=rng, restarts=restarts,
        sum_num=s_gain["numerators"], log_num=l_gain["numerators"],
        weights=weights, thresholds=grid["integer_product_thresholds"])
    primary_search["high_score_pool_search"] = pool_diagnostics
    if pool_candidate is not None and (
        primary_witness is None or
        _exact_portfolio_key(pool_candidate, grid["integer_product_thresholds"],
                             s_gain["numerators"], l_gain["numerators"], weights)
        > _exact_portfolio_key(primary_witness, grid["integer_product_thresholds"],
                               s_gain["numerators"], l_gain["numerators"], weights)
    ):
        primary_witness = pool_candidate
        primary_search["best_witness_method"] = "greedy_multicell_reservoir"
    if primary_witness is None:
        primary_witness, fallback = _fallback_four(
            dual_p1, rng=rng, trials=fallback_trials, distinct_sums=False,
            numerator_sum=s_gain["numerators"], numerator_log=l_gain["numerators"], weights=weights)
        primary_search["uncertified_fallback"] = fallback
    diverse_witness = (
        _witness_for_cells(diverse_cells, pools, rng=rng, restarts=restarts)
        if diverse_cells else None
    )
    diverse_search = {"attempt": "4 distinct top per-sum exact joint feature cells",
                      "target_cells": [list(x) for x in diverse_cells],
                      "restarts": restarts, "pool_per_cell": pool_per_cell}
    if diverse_witness is None and diverse_upper is not None:
        diverse_witness, fallback = _fallback_four(
            dual_p1, rng=rng, trials=fallback_trials, distinct_sums=True,
            numerator_sum=s_gain["numerators"], numerator_log=l_gain["numerators"], weights=weights)
        diverse_search["uncertified_fallback"] = fallback
    # The unrestricted feasible set contains the four-distinct-sum subset:
    # it cannot legitimately return a lower objective than a known feasible
    # diversity witness. Reuse the stronger witness when necessary.
    if diverse_witness is not None and (
        primary_witness is None or
        _exact_portfolio_key(diverse_witness, grid["integer_product_thresholds"],
                             s_gain["numerators"], l_gain["numerators"], weights)
        > _exact_portfolio_key(primary_witness, grid["integer_product_thresholds"],
                               s_gain["numerators"], l_gain["numerators"], weights)
    ):
        primary_witness = diverse_witness
        primary_search["best_witness_method"] = "best_known_diverse_feasible_witness"
        primary_search["dominance_guard"] = "unrestricted objective never worse than diversity witness"
    official_ref = official_prize_coverage(REFERENCE)
    hits_ref = exact_coverage(REFERENCE)
    primary = _describe_portfolio(
        primary_witness, name="dual_positive_unrestricted", upper=upper,
        dual_p1=dual_p1, sum_num=s_gain["numerators"], log_num=l_gain["numerators"],
        weights=weights, coverage_ref=official_ref, hits_ref=hits_ref, search=primary_search)
    diverse = _describe_portfolio(
        diverse_witness, name="four_distinct_sum_sensitivity",
        upper=diverse_upper if diverse_upper is not None else 0,
        dual_p1=dual_p1, sum_num=s_gain["numerators"], log_num=l_gain["numerators"],
        weights=weights, coverage_ref=official_ref, hits_ref=hits_ref, search=diverse_search)
    sum_only = sum_only_optimize(sum_p1, seed=seed)
    sum_base = sum_only["optimization"]["primary"]
    base_eval = [dual_score_ticket(dual_p1, t) for t in sum_base["tickets"]]
    base_key = sum(score_joint_cell(t["sum"]-SUM_OFFSET,t["log_product_bin_index"],
                   s_gain["numerators"],l_gain["numerators"],weights) for t in base_eval)
    frontier = pareto_frontier(cells)
    result = {
        "schema": SCHEMA,
        "stage": "DUAL_P2_FOUR_DISJOINT_SUM_LOG_REPAIR",
        "decision": "DESCRIPTIVE_DUAL_REPAIR_P2; PREDICTIVE_ADVANTAGE_NOT_EVALUATED",
        "source": dual_p1["source"],
        "theory_checksums": {
            "sum": sum_p1["theory"]["checksum_sha256"],
            "log": log_theory["checksum_sha256"],
        },
        "objective": {
            "description": "max Σ[w_sum*exact_numerator(G_sum)+w_log*exact_numerator(G_log)]",
            "weights_definition": "exact integer half-even rounding of 10^12 / conditional IID SD(G_feature); exact weighted squared-moment calculation, no float",
            "weight_scale": WEIGHT_SCALE,
            "w_sum": str(weights[0]), "w_log": str(weights[1]),
            "common_exact_gain_denominator": str(s_gain["denominator"]),
            "fixed_50_50_IID_SD_standardized_approx": True,
            "correlation_warning": "Sum and Log Product are highly correlated; weights do not imply independent evidence or tail p-values.",
            "constraint": "G_sum>0 AND G_log>0 exactly for EACH of four disjoint 6-number tickets",
            "four_separate_counterfactual_scores": True,
            "no_joint_append_to_draw_history": True,
        },
        "theoretical_joint_support": {
            "full_universe": TOTAL, "nonzero_joint_cells": sum(v>0 for row in joint for v in row),
            "dual_positive_cells": len(cells),
            "dual_positive_ticket_count": qualifies,
            "dual_positive_fair_probability": qualifies/TOTAL,
            "best_attainable_single_cell": {
                "sum": top_cell[0], "log_bin": top_cell[1],
                "joint_combinations": joint[top_cell[0]-SUM_OFFSET][top_cell[1]],
                "objective_exact_integer": str(cells[0]["objective_key"]),
            },
            "pareto_frontier_cells": [
                {"sum": c["sum"], "log_bin": c["log_bin"],
                 "sum_gain": (c["sum_numerator"]/s_gain["denominator"]),
                 "log_gain": (c["log_numerator"]/l_gain["denominator"]),
                 "joint_combinations": c["exact_joint_combinations"]}
                for c in frontier
            ],
            "joint_counts_sha256": sha256(json.dumps(
                joint, separators=(",", ":")).encode("ascii")).hexdigest(),
            "note": "The joint fair-theory counts are a cross-check of P1 two-marginal feasibility, not a joint predictive P3 test.",
        },
        "search_budget": {
            "method": "exact 6/49 feature joint enumeration + fixed top-cell reservoir + seeded disjoint constructive search; fallback random 24-ball partitions",
            "seed": seed, "pool_per_cell": pool_per_cell,
            "top_cell_limit": cell_limit, "restarts": restarts,
            "fallback_24_ball_trials": fallback_trials,
            "global_certificate_only_if_bound_attained": True,
            "exact_combinatorial_portfolio_exhaustion": False,
            "tie_rule": "integer cell objective descending; then ascending sum and log bin; seeded feasible witness, not lexicographically minimal optimizer",
        },
        "portfolios": {"primary": primary, "distinct_sum_sensitivity": diverse},
        "sum_only_p2_baseline_unchanged": {
            "tickets": sum_base["tickets"],
            "sum_bins": sum_base["sum_bins"],
            "sum_gain_total": sum_base["objective"]["gain_sum"],
            "log_gain_total": sum(t["log_gain"] for t in base_eval),
            "dual_positive_ticket_count": sum(t["dual_positive"] for t in base_eval),
            "dual_score_exact_integer": str(base_key),
            "coverage_probability": sum_base["coverage"]["official_probability"],
            "source": "original frozen v0.27 Sum-only P2, recomputed without modification",
        },
        "notes": [
            "A descriptive repair filter does not change probability of any specific next draw.",
            "Four tickets are four alternate single-draw counterfactuals; none are appended together.",
            "IID SD standardization does not remove Sum/Log dependence; calibrated predictive P3 is pending.",
            "Previously purchased ten-draw tickets and original P0/P1/P2 reports are unchanged.",
        ],
    }
    return result


def markdown_report(report: dict) -> str:
    src = report["source"]
    baseline = report.get("sum_only_p2_baseline_unchanged", {})
    rows = [
        "# v0.27 Issue #15 — Dual-Distribution P2 Four-Disjoint Portfolio", "",
        f"- **Cutoff:** {src['cutoff']}, #{src['last_draw_id']}, N={src['historical_draws']}",
        f"- **Historical prefix SHA256:** `{src['analysis_input_sha256']}`",
        "- **Hard constraint:** every ticket reduces BOTH Sum and Log Product L2 losses.",
        "- **Objective:** 50/50 conditional-IID-SD-standardized gains, weights quantized to exact integer rank keys.",
        "- **Certified** only if four-disjoint witness attains valid independent bound.",
        "- **Predictive status:** not evaluated. Weighted gains are correlated descriptors, not lottery odds.",
        "", "## Feature-joint theoretical support", "",
    ]
    j = report["theoretical_joint_support"]
    rows.extend([
        f"- **Exact fair combinations:** {j['full_universe']:,}",
        f"- **Joint nonzero bins:** {j['nonzero_joint_cells']:,}",
        f"- **Dual-positive cells:** {j['dual_positive_cells']:,}",
        f"- **Dual-positive tickets:** {j['dual_positive_ticket_count']:,}, "
        f"fair mass {j['dual_positive_fair_probability']:.6%}",
        f"- **Best attainable single cell:** Sum={j['best_attainable_single_cell']['sum']}, "
        f"LogBin={j['best_attainable_single_cell']['log_bin']}",
        "",
    ])
    for key, title in (("primary","Primary (dual-positive, sums unrestricted)"),
                       ("distinct_sum_sensitivity","Sensitivity (four different sums)")):
        item = report["portfolios"][key]
        rows.extend([f"## {title}", "", f"- **Certificate:** {item['status']}",
                     f"- **Upper integer key:** `{item['upper_bound_exact_integer']}`",
                     f"- **Search:** `{json.dumps(item['search'],sort_keys=True)}`", ""])
        if item["tickets"]:
            rows.extend([
                f"- **Attained integer key:** `{item['objective_exact_integer']}`",
                f"- **Total Sum gain:** {item['sum_gain_total']*1e6:+.8f} × 10⁻⁶",
                f"- **Total Log gain:** {item['log_gain_total']*1e6:+.8f} × 10⁻⁶",
                f"- **IID normalized objective:** {item['iid_standardized_score_approx']:+.8f}",
                f"- **Exact P(any official prize):** {item['coverage']['any_prize_probability']:.12%}",
                f"- **Unique numbers:** {item['unique_numbers']}; "
                f"unique sum bins: {item['unique_sum_bins']}; unique log bins: {item['unique_log_bins']}",
                "", "| Bet | Six main numbers | Sum | Log bin | Sum G × 10⁶ | Log G × 10⁶ |",
                "|:--:|:---|---:|---:|---:|---:|",
            ])
            for i, ticket in enumerate(item["ticket_details"]):
                rows.append(
                    f"| {chr(65+i)} | {' '.join(f'{v:02d}' for v in ticket['ticket'])} | "
                    f"{ticket['sum']} | {ticket['log_product_bin_index']} | "
                    f"{ticket['sum_gain_micro_l2']:+.6f} | {ticket['log_gain_micro_l2']:+.6f} |")
            rows.extend(["", item["certificate"], ""])
        else:
            rows.extend(["No feasible four-ticket witness found within the stated bounded search. "
                         "This is NOT a proof of impossibility.", ""])
    rows.extend([
        "## Preserved Sum-only P2 reference", "",
        f"- **Sum G:** {baseline.get('sum_gain_total',0)*1e6:+.8f} × 10⁻⁶",
        f"- **Log G:** {baseline.get('log_gain_total',0)*1e6:+.8f} × 10⁻⁶",
        f"- **Dual-positive bets:** {baseline.get('dual_positive_ticket_count','N/A')}/4",
        f"- **Official Coverage P(any prize):** {baseline.get('coverage_probability',0):.12%}",
        "", "## Pareto frontier of realizable feature bins", "",
        "| Sum | Log bin | Sum gain ×10⁶ | Log gain ×10⁶ | Theoretical tickets |",
        "|---:|---:|---:|---:|---:|",
    ])
    for c in j["pareto_frontier_cells"][:40]:
        rows.append(
            f"| {c['sum']} | {c['log_bin']} | {c['sum_gain']*1e6:+.6f} | "
            f"{c['log_gain']*1e6:+.6f} | {c['joint_combinations']} |")
    rows.extend([
        "", "## Reproduce", "", "```powershell",
        "& .\\.venv\\Scripts\\python.exe -m lotto7x7.log_product_dual_portfolio_v027 "
        "--sum-p1 results/v027_p1_repair.json "
        "--log-p0 results/v027_log_product_p0.json "
        "--dual-p1 results/v027_dual_p1.json "
        "--out results/v027_dual_p2.json --md-out results/v027_dual_p2.md",
        "```", "",
        "**Scope:** not a prospective bet recommendation. P3 chronological and complete IID "
        "pipeline calibration are pending; no self-correction forecasting assertion.", "",
    ])
    return "\n".join(rows)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sum-p1", type=Path, required=True)
    parser.add_argument("--log-p0", type=Path, required=True)
    parser.add_argument("--dual-p1", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/v027_dual_p2.json"))
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=649)
    parser.add_argument("--pool-per-cell", type=int, default=800)
    parser.add_argument("--cell-limit", type=int, default=24)
    parser.add_argument("--restarts", type=int, default=512)
    parser.add_argument("--fallback-trials", type=int, default=15000)
    args = parser.parse_args(argv)
    out, md = args.out, args.md_out or args.out.with_suffix(".md")
    if out.resolve() == md.resolve() or out.exists() or md.exists():
        raise FileExistsError("two distinct unused output paths required; never overwrite research reports")
    raw = [p.read_bytes() for p in (args.sum_p1, args.log_p0, args.dual_p1)]
    result = optimize(*[json.loads(d) for d in raw], seed=args.seed,
                      pool_per_cell=args.pool_per_cell, cell_limit=args.cell_limit,
                      restarts=args.restarts, fallback_trials=args.fallback_trials)
    result["input_report_SHA256"] = {
        "sum_p1": sha256(raw[0]).hexdigest(), "log_p0": sha256(raw[1]).hexdigest(),
        "dual_p1": sha256(raw[2]).hexdigest(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    md.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        fp.write("\n")
    md.write_text(markdown_report(result), encoding="utf-8")
    j = result["theoretical_joint_support"]
    print(f"[DUAL-P2 SUPPORT] exact {j['full_universe']:,} combinations; "
          f"{j['dual_positive_ticket_count']:,} dual-positive; "
          f"best_cell={j['best_attainable_single_cell']['sum']}/{j['best_attainable_single_cell']['log_bin']}")
    for label, item in (("PRIMARY", result["portfolios"]["primary"]),
                        ("DIVERSE", result["portfolios"]["distinct_sum_sensitivity"])):
        print(f"[DUAL-P2 {label}] {item['status']}; bound={item['upper_bound_exact_integer']}")
        if item["tickets"]:
            print(f"[DUAL-P2 {label}] sum_G={item['sum_gain_total']*1e6:+.8f}e-6 "
                  f"log_G={item['log_gain_total']*1e6:+.8f}e-6 "
                  f"official_any={item['coverage']['any_prize_probability']:.12%}")
            for i, t in enumerate(item["tickets"]):
                print(f"  {chr(65+i)}: {' '.join(f'{n:02d}' for n in t)}")
    print(f"[SUM-ONLY BASE] dual-positive="
          f"{result['sum_only_p2_baseline_unchanged']['dual_positive_ticket_count']}/4")
    print(f"[REPORT] {out} / {md}")
    print("[SCOPE] Separate counterfactuals; dual-marginal descriptive only; P3 predictive untested")


if __name__ == "__main__":
    main()
