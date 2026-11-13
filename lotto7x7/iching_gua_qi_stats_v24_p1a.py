"""v2.4 P1A: USNO Gua Qi association audit against exact fair 6/49 baselines.

Exploratory retrospective association only; absolutely no betting forecasts.
Inputs are immutable P1 files. Output is write-once and deterministic.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date, datetime, timedelta
from hashlib import sha256
import json
from math import comb, sqrt
from pathlib import Path
import random

SCHEMA = "lotto649.iching_guaqi_association.v2.4.p1a"
SOURCE_SHA256 = "53f35eb6fb0f6f5d6bc36e4b6a78aa63586263c924ba65940cc86c588c4fb1f1"
RULE = "usno_quarter_anchor_centered_12state_half_up_v1"
CLASSIFIER = "shape8-graph-partition-cardinal-diagonal-v1:L1"
SEED = 649
N_REPS = 3999
UNIVERSE = comb(49, 6)
COMPONENT_UNIVERSE_COUNTS = (58620, 669462, 2775454, 5093678, 4166304, 1220298)
GUA_NAMES = ("Fu", "Lin", "Tai", "Da_Zhuang", "Guai", "Qian",
             "Gou", "Dun", "Pi", "Guan", "Bo", "Kun")
GUA_CHINESE = ("復", "臨", "泰", "大壯", "夬", "乾",
               "姤", "遯", "否", "觀", "剝", "坤")
GUA_CODES = (1, 3, 7, 15, 31, 63, 62, 60, 56, 48, 32, 0)
ENDPOINTS = ("sum", "odd", "shape_component_mean", "shape_L1_full_distribution")
INPUT_COUNT = 2498
USABLE_COUNT = 2476


def sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def fair_moments() -> dict:
    """Exactly sampling six distinct labels from population {1,...,49}."""
    if sum(COMPONENT_UNIVERSE_COUNTS) != UNIVERSE:
        raise ValueError("exact L1 shape component counts do not sum to C(49,6)")
    sum_mean = 6 * 25
    sum_variance = 6 * (49 - 6) / (49 - 1) * ((49 * 49 - 1) / 12)
    odd_mean = 6 * 25 / 49
    odd_variance = 6 * (25 / 49) * (24 / 49) * 43 / 48
    p = [x / UNIVERSE for x in COMPONENT_UNIVERSE_COUNTS]
    shape_mean = sum((k + 1) * pk for k, pk in enumerate(p))
    shape_variance = sum(((k + 1) - shape_mean) ** 2 * pk for k, pk in enumerate(p))
    return {
        "sum": {"mean": sum_mean, "variance": sum_variance},
        "odd": {"mean": odd_mean, "variance": odd_variance},
        "shape_component_mean": {"mean": shape_mean, "variance": shape_variance},
        "shape_L1_full_distribution": {
            "counts_by_number_of_components_1_to_6": list(COMPONENT_UNIVERSE_COUNTS),
            "probabilities": p,
        },
    }


def component_count(nums: tuple[int, ...]) -> int:
    """Independent BFS reference for existing v0.15 8-neighbor 7x7 L1 classifier."""
    if (len(nums) != 6 or any(type(n) is not int or not 1 <= n <= 49 for n in nums)
            or tuple(sorted(set(nums))) != nums):
        raise ValueError("expected six unique ascending numbers in 1..49")
    coords = [divmod(n - 1, 7) for n in nums]
    unseen = set(range(6))
    comps = 0
    while unseen:
        comps += 1
        stack = [unseen.pop()]
        while stack:
            i = stack.pop()
            ri, ci = coords[i]
            for j in tuple(unseen):
                rj, cj = coords[j]
                if abs(ri - rj) <= 1 and abs(ci - cj) <= 1:
                    unseen.remove(j)
                    stack.append(j)
    return comps


def read_frozen_source(path: Path) -> tuple[list[dict], dict]:
    if sha(path) != SOURCE_SHA256:
        raise ValueError("v2.4 P1 original SHA256 mismatch; refusing data analysis")
    with path.open(encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames is None or len(set(rd.fieldnames)) != len(rd.fieldnames):
            raise ValueError("bad P1 header")
        required = {"date", "draw_id", "reference_at_taipei", "seasonal_state_12",
                    "seasonal_hexagram_code", "solar_missing", "moon_missing",
                    "astronomy_known_without_outcome", "mapping_version",
                    "same_draw_hexagram_code_OUTCOME_ONLY", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"}
        if not required <= set(rd.fieldnames):
            raise ValueError("P1 schema missing required fields")
        raw = list(rd)
    if len(raw) != INPUT_COUNT or len({r["draw_id"] for r in raw}) != INPUT_COUNT:
        raise ValueError("P1 frozen draw count / ID uniqueness mismatch")
    if ((raw[0]["date"], raw[0]["draw_id"]) != ("2004-01-05", "93000001") or
            (raw[-1]["date"], raw[-1]["draw_id"]) != ("2026-11-10", "115000104")):
        raise ValueError("P1 original date anchor mismatch")
    cleaned = []
    missing_by_year: Counter = Counter()
    prevdate = None
    for r in raw:
        d = date.fromisoformat(r["date"])
        if prevdate is not None and d <= prevdate:
            raise ValueError("nonmonotone frozen draws")
        prevdate = d
        reference = datetime.fromisoformat(r["reference_at_taipei"])
        if reference.date() != d or reference.utcoffset() != timedelta(hours=8) or reference.hour != 12 or reference.minute != 0:
            raise ValueError("unexpected UTC+08:00 noon reference")
        if r["mapping_version"] != RULE or r["astronomy_known_without_outcome"] != "true":
            raise ValueError("unknown P1 mapping or leaked source")
        nums = tuple(int(r[f"n{i}"]) for i in range(1, 7))
        if len(nums) != 6 or tuple(sorted(set(nums))) != nums or nums[0] < 1 or nums[-1] > 49:
            raise ValueError("illegal main numbers")
        bonus = int(r["bonus"])
        if bonus < 1 or bonus > 49 or bonus in nums:
            raise ValueError("illegal bonus")
        observed_hex = sum((n % 2) << i for i, n in enumerate(nums))
        if int(r["same_draw_hexagram_code_OUTCOME_ONLY"]) != observed_hex:
            raise ValueError("outcome hexagram not consistent with sorted main balls")
        if r["solar_missing"] == "true":
            if r["seasonal_state_12"] or r["seasonal_hexagram_code"]:
                raise ValueError("missing seasonal state not blank")
            missing_by_year[d.year] += 1
            continue
        if r["solar_missing"] != "false":
            raise ValueError("unrecognized solar_missing flag")
        state = int(r["seasonal_state_12"])
        if not 0 <= state < 12 or int(r["seasonal_hexagram_code"]) != GUA_CODES[state]:
            raise ValueError("guaxi state or code mismatch")
        cleaned.append({"date": d.isoformat(), "draw_id": r["draw_id"], "year": d.year,
                        "weekday": d.isoweekday(), "state": state,
                        "sum": sum(nums), "odd": sum(n & 1 for n in nums),
                        "shape": component_count(nums)})
    if len(cleaned) != USABLE_COUNT or sum(missing_by_year.values()) != 22 or set(missing_by_year) != {2004}:
        raise ValueError("unexpected source mapping coverage or missing boundary")
    return cleaned, {"unmapped_by_year": {str(k): int(v) for k, v in sorted(missing_by_year.items())},
                     "source_sha256": SOURCE_SHA256, "source_rows": INPUT_COUNT,
                     "mapped_rows": len(cleaned)}


def sufficient_statistics(rows: list[dict]) -> tuple[list[int], list[list[int]], list[list[float]], list[float], list[int]]:
    counts = [0] * 12
    categorical = [[0] * 6 for _ in range(12)]
    moments = [[0.0] * 3 for _ in range(12)]
    grand = [0.0] * 3
    global_categories = [0] * 6
    for r in rows:
        state = r["state"]
        counts[state] += 1
        values = (r["sum"], r["odd"], r["shape"])
        for k in range(3):
            moments[state][k] += values[k]
            grand[k] += values[k]
        category = r["shape"] - 1
        categorical[state][category] += 1
        global_categories[category] += 1
    if not all(counts) or sum(counts) != len(rows):
        raise ValueError("expected at least one observation per seasonal state")
    return counts, categorical, moments, grand, global_categories


def four_statistics(rows: list[dict], exact: dict) -> tuple[list[float], dict]:
    n = len(rows)
    counts, cats, moments, grand, global_cats = sufficient_statistics(rows)
    stats = []
    eta = {}
    for k, name in enumerate(ENDPOINTS[:3]):
        mu = grand[k] / n
        between = sum((moments[s][k] - counts[s] * mu) ** 2 / counts[s] for s in range(12))
        # The variance scale is theoretically exact, and the permutation
        # distribution conditions on the observed response values.
        stats.append(between / exact[name]["variance"])
        total = sum((r[name if k != 2 else "shape"] - mu) ** 2 for r in rows)
        eta[name] = (between / total) if total else 0.0
    chisq = 0.0
    for s, nr in enumerate(counts):
        for cat in range(6):
            expectation = nr * global_cats[cat] / n
            if expectation:
                chisq += (cats[s][cat] - expectation) ** 2 / expectation
    stats.append(chisq)
    cramer_v = sqrt(chisq / (n * min(12 - 1, 6 - 1)))
    return stats, {"eta_squared": eta, "shape_L1_cramers_v": cramer_v,
                   "total_count_by_state": counts,
                   "shape_L1_observed_counts_by_state": cats}


def make_null(rows: list[dict], exact: dict, *, reps: int, seed: int) -> dict:
    """Stratified label permutation, preserving yearly weekday schedules.

    The fixed astronomy state belongs to each date; within a year/weekday
    block, randomize which observed historical draw is assigned to each date.
    This conditions on empirical feature distributions, including all
    correlations between sum, parity and L1 shape labels.
    """
    if reps < 99:
        raise ValueError("at least 99 permutations required")
    rng = random.Random(seed)
    blocks: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        blocks[(r["year"], r["weekday"])].append(i)
    obs, effect = four_statistics(rows, exact)
    hits = [0] * 4
    null_max = [0.0] * 4
    null_sums = [0.0] * 4
    # Data are not re-trained; the exact fair baseline is invariant. The
    # null targets an association of outcomes with known-before-draw states.
    perm_rows = [dict(r) for r in rows]
    for _ in range(reps):
        for block in blocks.values():
            perm = block[:]
            rng.shuffle(perm)
            for original_i, sampled_i in zip(block, perm):
                src = rows[sampled_i]
                dst = perm_rows[original_i]
                dst["sum"] = src["sum"]
                dst["odd"] = src["odd"]
                dst["shape"] = src["shape"]
        sim, _ = four_statistics(perm_rows, exact)
        for k in range(4):
            hits[k] += (sim[k] >= obs[k] - 1e-12)
            null_max[k] = max(null_max[k], sim[k])
            null_sums[k] += sim[k]
    p_raw = [(1 + h) / (1 + reps) for h in hits]
    p_adj = [min(1.0, 4 * p) for p in p_raw]
    return {
        "strata": "year_x_weekday_iso",
        "n_strata": len(blocks),
        "stratum_size_min": min(map(len, blocks.values())),
        "stratum_size_max": max(map(len, blocks.values())),
        "replicates": reps, "seed": seed,
        "endpoints": {name: {"observed_statistic": obs[i], "null_statistic_mean": null_sums[i] / reps,
                              "null_statistic_max": null_max[i], "permutation_upper_tail_hits": hits[i],
                              "permutation_p": p_raw[i], "bonferroni_four_families_p": p_adj[i]}
                      for i, name in enumerate(ENDPOINTS)},
        "familywise_control": "Bonferroni across four predeclared global endpoints; conditional permutations account for 12 seasonal states jointly",
        "effect_sizes": effect,
        "interpretation": "exploratory retrospective association conditional on year and weekday; not predictive evidence",
    }


def state_rows(rows: list[dict], exact: dict) -> list[dict]:
    n = len(rows)
    counts, cats, sums, _, _ = sufficient_statistics(rows)
    out = []
    p = exact["shape_L1_full_distribution"]["probabilities"]
    for s in range(12):
        nr = counts[s]
        row = {"state_12": s, "gua": GUA_NAMES[s], "chinese": GUA_CHINESE[s],
               "hexagram_code": GUA_CODES[s], "draws": nr}
        for k, key in enumerate(ENDPOINTS[:3]):
            mu = exact[key]["mean"]
            stdev_mean = sqrt(exact[key]["variance"] / nr)
            row[f"{key}_mean"] = round(sums[s][k] / nr, 6)
            row[f"{key}_fair_mean"] = round(mu, 6)
            row[f"{key}_descriptive_z"] = round((sums[s][k] / nr - mu) / stdev_mean, 6)
        for k in range(6):
            row[f"shape_L1_C{k+1}_observed"] = cats[s][k]
            row[f"shape_L1_C{k+1}_expected_fair"] = round(nr * p[k], 6)
        out.append(row)
    assert sum(row["draws"] for row in out) == n
    return out


def audit(rows: list[dict], provenance: dict, *, reps: int = N_REPS, seed: int = SEED) -> tuple[dict, list[dict]]:
    exact = fair_moments()
    inference = make_null(rows, exact, reps=reps, seed=seed)
    summary = state_rows(rows, exact)
    winners = []
    for endpoint in ENDPOINTS:
        e = inference["endpoints"][endpoint]
        if e["bonferroni_four_families_p"] < 0.05:
            winners.append(endpoint)
    doc = {
        "schema": SCHEMA,
        "input_frozen": provenance,
        "mapping_version": RULE,
        "shape_taxonomy": CLASSIFIER,
        "reference_clock": "Taipei date at 12:00 (not independently verified drawing time)",
        "analysis_draws": len(rows),
        "state_counts": {f"{i}_{GUA_NAMES[i]}": r["draws"] for i, r in enumerate(summary)},
        "fair_6of49": {"universe": UNIVERSE, "moments": exact,
                       "sum_theory": "E[S]=150; Var(S)=1075, six without replacement from 1..49",
                       "odd_theory": "K~Hypergeometric(N=49,K=25,n=6)",
                       "shape_theory": "existing v0.16 exhaustive enumeration, C(49,6)=13,983,816"},
        "tests": inference,
        "bonferroni_0_05_significant_endpoints": winners,
        "decision": "EXPLORATORY_ASSOCIATION_SCREEN" + ("_FOLLOWUP_REQUIRED" if winners else "_NO_EVIDENCE"),
        "limitations": [
            "Only 2,476 of 2,498 historical draws have P1 seasonal states; 22 early-2004 inputs lack the preceding 2003 solstice.",
            "Fixed USNO seasonal states are pre-draw/date-known; outcomes are used only in retrospective statistical tests.",
            "Exact fair theoretical moments and L1 counts are baselines, not evidence of a physical astronomy effect.",
            "Conditional permutation within year x weekday is exploratory; it does not correct for every time-varying technical change.",
            "P1 mapping and feature families were discussed after seeing older history; never claim pre-registered out-of-sample confirmation.",
            "Twelve per-state descriptive z scores are NOT separate inferential discoveries; four family-level endpoints receive Bonferroni adjustment.",
            "Even a statistically significant retrospective association would require later prospective validation to support prediction.",
            "No new or modified tickets, lottery source files, P1 source CSV, or prospective seals.",
        ],
    }
    return doc, summary


def write_new(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="") as f:
        f.write(text)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="One-time P1A 12 Gua Qi x historical fair Lotto649 exploratory association test")
    p.add_argument("--source", type=Path, default=Path("data/research/v24_iching/v24_p1_gua_qi_draw_context_2498.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("data/research/v24_iching"))
    p.add_argument("--reps", type=int, default=N_REPS)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)
    if args.reps < 99 or args.reps > 100_000:
        p.error("permutation count must be between 99 and 100000")
    rows, provenance = read_frozen_source(args.source)
    report, states = audit(rows, provenance, reps=args.reps, seed=args.seed)
    prefix = f"v24_p1a_gua_qi_association_{INPUT_COUNT}"
    report_file = args.out_dir / f"{prefix}.json"
    states_file = args.out_dir / f"{prefix}_states.csv"
    if report_file.exists() or states_file.exists():
        raise FileExistsError("immutable output already exists: use an empty out-dir; no overwrite")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    headings = list(states[0])
    import io
    fp = io.StringIO(newline="")
    writer = csv.DictWriter(fp, fieldnames=headings, lineterminator="\n")
    writer.writeheader()
    writer.writerows(states)
    states_bytes = fp.getvalue().encode("utf-8")
    report["output_states_csv_sha256"] = sha256(states_bytes).hexdigest()
    write_new(states_file, states_bytes.decode("utf-8"))
    write_new(report_file, json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"P1A: {len(rows)} mapped draws, 12 Gua Qi, {args.reps} stratified permutations")
    for name in ENDPOINTS:
        e = report["tests"]["endpoints"][name]
        print(f"  {name}: stat={e['observed_statistic']:.4f}, p={e['permutation_p']:.4f}, p_fwer={e['bonferroni_four_families_p']:.4f}")
    print("DECISION:", report["decision"])
    print("REPORT:", report_file)
    print("STATES:", states_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())