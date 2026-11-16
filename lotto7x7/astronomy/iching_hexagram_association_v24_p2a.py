"""v2.4 P2A: exploratory astronomical 64-hexagram x lottery-outcome association.

This script analyzes a frozen, full-history USNO-derived dataset. Six global
endpoints are calibrated by the identical within-year-and-weekday permutation
protocol, followed by Bonferroni FWER correction. No predictions or bets.
Standard-library only. Inputs are read-only; outputs use exclusive creation.
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

SCHEMA = "lotto649.iching_hexagram_association.v2.4.p2a"
MAPPING_RULE = "usno_winter_anchor_solar8_newmoon_time8_gray_cyclic_v1"
SOURCE_SHA256 = "a46f08269fc8021369f1a045116051e4f9591ba83801d5cdbe8c1a08873edf39"
USNO_EVENT_SHA256 = "f084a9248725cecbef257d23963b2660f2152e2bfce2f12054fd89337e33f6df"
P1_SHA256 = "53f35eb6fb0f6f5d6bc36e4b6a78aa63586263c924ba65940cc86c588c4fb1f1"
EXPECTED_RAW_ROWS = 2498
EXPECTED_MAPPED_ROWS = 2476
UNIVERSE = comb(49, 6)
N_REPS = 3999
SEED = 240649
ENDPOINTS = (
    "outcome_hexagram_64x64",
    "hamming_distance_mean_two_sided",
    "sum_by_sky_64",
    "odd_count_by_sky_64",
    "shape_L1_mean_by_sky_64",
    "shape_L1_64x6",
)
# Counts from the previously exactly enumerated 13,983,816 legal combinations.
L1_COUNTS = (58620, 669462, 2775454, 5093678, 4166304, 1220298)
REPORT_NAME = "v24_p2a_hexagram_association_2498.json"
STATES_NAME = "v24_p2a_hexagram_association_2498_states.csv"
STATE_COLUMNS = (
    "astronomical_hexagram_code", "draw_count", "observed_mean_sum",
    "observed_mean_odd_count", "observed_mean_shape_L1_components",
    "observed_mean_hamming_distance", "observed_most_common_outcome_hexagram_code",
    "source_provenance", "report_scope",
)


def sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def read_csv_rows(contents: bytes) -> list[dict[str, str]]:
    source = contents.decode("utf-8-sig")
    rd = csv.DictReader(source.splitlines())
    if rd.fieldnames is None or len(rd.fieldnames) != len(set(rd.fieldnames)):
        raise ValueError("duplicate or absent CSV headers")
    rows = list(rd)
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError("malformed CSV rows")
    return rows


def gray3(s: int) -> int:
    if not 0 <= s < 8:
        raise ValueError("sky sector must be 0..7")
    return s ^ (s >> 1)


def outcome_code(nums: tuple[int, ...]) -> int:
    if (len(nums) != 6 or any(type(v) is not int or v < 1 or v > 49 for v in nums)
            or tuple(sorted(set(nums))) != nums):
        raise ValueError("six ascending unique integers in 1..49 required")
    return sum((num & 1) << pos for pos, num in enumerate(nums))


def shape_l1(nums: tuple[int, ...]) -> int:
    """Independent 8-neighbor graph BFS on the original sequential 7x7 board."""
    if len(nums) != 6 or len(set(nums)) != 6 or any(not 1 <= x <= 49 for x in nums):
        raise ValueError("invalid six-number shape")
    coords = [divmod(num - 1, 7) for num in nums]
    unseen = set(range(6))
    components = 0
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            u = stack.pop()
            for v in tuple(unseen):
                if max(abs(coords[u][0] - coords[v][0]), abs(coords[u][1] - coords[v][1])) <= 1:
                    unseen.remove(v)
                    stack.append(v)
    return components


def exact_parity_masses() -> list[int]:
    """Exact ascending-six-number odd/even sequence counts, not uniform 1/64."""
    dp = [[0] * 64 for _ in range(7)]
    dp[0][0] = 1
    for n in range(1, 50):
        for k in range(min(6, n), 0, -1):
            shift = (n & 1) << (k - 1)
            for code in range(1 << (k - 1)):
                dp[k][code | shift] += dp[k - 1][code]
    masses = dp[6]
    if sum(masses) != UNIVERSE or any(v <= 0 for v in masses):
        raise AssertionError("broken exact parity DP")
    return masses


def fair_reference() -> dict:
    if sum(L1_COUNTS) != UNIVERSE:
        raise AssertionError("invalid exact L1 combinatorial counts")
    parity = exact_parity_masses()
    odd_mean = 6 * 25 / 49
    odd_var = 6 * (25 / 49) * (24 / 49) * 43 / 48
    l1_mean = sum((i + 1) * n for i, n in enumerate(L1_COUNTS)) / UNIVERSE
    l1_var = sum(((i + 1) - l1_mean) ** 2 * n for i, n in enumerate(L1_COUNTS)) / UNIVERSE
    return {
        "legal_universe": UNIVERSE,
        "sum": {"mean": 150, "variance": 1075.0},
        "odd_count": {"mean": odd_mean, "variance": odd_var},
        "shape_L1_components": {"mean": l1_mean, "variance": l1_var,
                                "exact_universe_counts_components_1_to_6": list(L1_COUNTS)},
        "outcome_parity_hexagram_exact_masses_by_code_0_to_63": parity,
        "parity_mass_min": min(parity), "parity_mass_max": max(parity),
        "parity_mass_not_uniform": len(set(parity)) > 1,
    }


def load_frozen(source: Path, audit: Path) -> tuple[list[dict], dict]:
    raw = source.read_bytes()
    if sha256_bytes(raw) != SOURCE_SHA256:
        raise ValueError("frozen P2 source SHA-256 mismatch")
    meta = json.loads(audit.read_text(encoding="utf-8"))
    if (meta.get("output_csv_sha256") != SOURCE_SHA256 or
            meta.get("frozen_draws") != EXPECTED_RAW_ROWS or
            meta.get("pair_mapped") != EXPECTED_MAPPED_ROWS or
            meta.get("pair_missing") != EXPECTED_RAW_ROWS - EXPECTED_MAPPED_ROWS or
            meta.get("mapping_version") != MAPPING_RULE or
            meta.get("source_sha256", {}).get("usno_event_csv_sha256") != USNO_EVENT_SHA256 or
            meta.get("source_sha256", {}).get("p1_context_sha256") != P1_SHA256):
        raise ValueError("frozen P2 audit provenance / mapping inconsistency")
    rows = read_csv_rows(raw)
    if len(rows) != EXPECTED_RAW_ROWS:
        raise ValueError("frozen P2 row count mismatch")
    required = {
        "date", "draw_id", "reference_at_taipei", "astronomical_hexagram_code",
        "solar_sector_8", "lunar_sector_8", "solar_gray_code", "lunar_gray_code",
        "astronomical_lower_trigram_code", "astronomical_upper_trigram_code",
        "astronomical_hexagram_binary_bottom_to_top", "astronomy_pair_mapped",
        "astronomy_known_without_outcome", "solar_missing", "lunar_missing",
        "n1", "n2", "n3", "n4", "n5", "n6", "bonus",
        "same_draw_hexagram_code_OUTCOME_ONLY", "hamming_distance_OUTCOME_ONLY",
        "mapping_version", "reference_clock_assumption", "source_provenance",
    }
    if not required <= set(rows[0]):
        raise ValueError("frozen P2 input missing schema columns")
    frozen_ids = set()
    mapped: list[dict] = []
    previous_date: date | None = None
    year_coverage = defaultdict(lambda: {"input_rows": 0, "mapped": 0, "unmapped": 0})
    for idx, row in enumerate(rows):
        day = date.fromisoformat(row["date"])
        draw_id = row["draw_id"]
        if previous_date is not None and day <= previous_date:
            raise ValueError("non-monotone draw dates")
        previous_date = day
        if draw_id in frozen_ids or not draw_id.isascii() or not draw_id.isdecimal():
            raise ValueError("duplicate or malformed draw ID")
        frozen_ids.add(draw_id)
        if row["mapping_version"] != MAPPING_RULE:
            raise ValueError("P2 mapping rule mismatch")
        ref = datetime.fromisoformat(row["reference_at_taipei"])
        if (ref.tzinfo is None or ref.utcoffset() != timedelta(hours=8)
                or ref.date() != day or (ref.hour, ref.minute, ref.second) != (12, 0, 0)):
            raise ValueError("timestamp not 12:00 UTC+8 on the draw date")
        if row["astronomy_known_without_outcome"] != "true":
            raise ValueError("leakage-safe provenance flag absent")
        nums = tuple(int(row[f"n{k}"]) for k in range(1, 7))
        code_out = outcome_code(nums)
        bonus = int(row["bonus"])
        if bonus not in range(1, 50) or bonus in nums:
            raise ValueError("invalid special number")
        if row["same_draw_hexagram_code_OUTCOME_ONLY"] != str(code_out):
            raise ValueError("post-draw label is inconsistent with six main numbers")
        mapped_flag = row["astronomy_pair_mapped"]
        if mapped_flag not in ("true", "false"):
            raise ValueError("malformed P2 mapping flag")
        cov = year_coverage[str(day.year)]
        cov["input_rows"] += 1
        if mapped_flag == "false":
            cov["unmapped"] += 1
            if (row["astronomical_hexagram_code"] != "" or
                    row["hamming_distance_OUTCOME_ONLY"] != ""):
                raise ValueError("unmapped P2 record must have blank astronomy code/hamming")
            continue
        cov["mapped"] += 1
        if row["solar_missing"] != "false" or row["lunar_missing"] != "false":
            raise ValueError("mapped P2 record claims missing component")
        solar, lunar = int(row["solar_sector_8"]), int(row["lunar_sector_8"])
        solar_gray, lunar_gray = gray3(solar), gray3(lunar)
        code_sky = (solar_gray << 3) | lunar_gray
        if (int(row["astronomical_hexagram_code"]) != code_sky or
                int(row["solar_gray_code"]) != solar_gray or
                int(row["lunar_gray_code"]) != lunar_gray or
                int(row["astronomical_lower_trigram_code"]) != lunar_gray or
                int(row["astronomical_upper_trigram_code"]) != solar_gray or
                row["astronomical_hexagram_binary_bottom_to_top"] !=
                "".join(str((code_sky >> b) & 1) for b in range(6)) or
                int(row["hamming_distance_OUTCOME_ONLY"]) != (code_sky ^ code_out).bit_count()):
            raise ValueError("P2 six-bit / Gray code mismatch")
        mapped.append({
            "date": str(day), "year": day.year, "weekday": day.weekday(),
            "draw_id": draw_id, "sky": code_sky, "outcome": code_out,
            "sum": sum(nums), "odd": sum(n & 1 for n in nums),
            "shape": shape_l1(nums),
            "hamming": (code_sky ^ code_out).bit_count(),
        })
    if (len(frozen_ids) != EXPECTED_RAW_ROWS or len(mapped) != EXPECTED_MAPPED_ROWS or
            (rows[0]["date"], rows[0]["draw_id"]) != ("2004-01-05", "93000001") or
            (rows[-1]["date"], rows[-1]["draw_id"]) != ("2026-11-10", "115000104")):
        raise ValueError("P2 input archive anchors / mapping coverage mismatch")
    if len({r["sky"] for r in mapped}) != 64:
        raise ValueError("P2 astronomical coding does not cover all 64 states")
    return mapped, {
        "frozen_source_csv_sha256": sha256_bytes(raw),
        "frozen_audit_sha256": sha256_bytes(audit.read_bytes()),
        "mapped": len(mapped), "missing": EXPECTED_RAW_ROWS - len(mapped),
        "coverage_by_year": dict(sorted(year_coverage.items())),
        "mapping_version": MAPPING_RULE,
    }


def build_null_design(rows: list[dict]) -> dict:
    """Create frozen conditional assignment groups and expectations."""
    n = len(rows)
    if not n or any(not 0 <= r["sky"] < 64 or not 0 <= r["outcome"] < 64
                    or not 1 <= r["shape"] <= 6 for r in rows):
        raise ValueError("invalid mapped rows")
    strata: dict[tuple[int, int], list[int]] = defaultdict(list)
    sky_counts = [0] * 64
    hamming_expected_sum = 0.0
    for idx, r in enumerate(rows):
        strata[(r["year"], r["weekday"])].append(idx)
        sky_counts[r["sky"]] += 1
    expected_hex = [0.0] * (64 * 64)
    expected_shape = [0.0] * (64 * 6)
    expected_totals = {key: [0.0] * 64 for key in ("sum", "odd", "shape")}
    for inds in strata.values():
        size = len(inds)
        a = Counter(rows[i]["sky"] for i in inds)
        b = Counter(rows[i]["outcome"] for i in inds)
        comps = Counter(rows[i]["shape"] for i in inds)
        means = {key: sum(rows[i][key] for i in inds) / size for key in expected_totals}
        for sky, cnt_s in a.items():
            weight = cnt_s / size
            for outcome, cnt_o in b.items():
                expected_hex[64 * sky + outcome] += weight * cnt_o
                hamming_expected_sum += weight * cnt_o * (sky ^ outcome).bit_count()
            for shape, cnt_t in comps.items():
                expected_shape[6 * sky + (shape - 1)] += weight * cnt_t
            for key in expected_totals:
                expected_totals[key][sky] += cnt_s * means[key]
    if (abs(sum(expected_hex) - n) > 1e-8 or
            abs(sum(expected_shape) - n) > 1e-8 or
            any(sky_counts[i] == 0 for i in range(64))):
        raise AssertionError("malformed stratum expectation or missing sky class")
    l1_mu = sum((i + 1) * count for i, count in enumerate(L1_COUNTS)) / UNIVERSE
    l1_var = sum(((i + 1) - l1_mu)**2 * count for i, count in enumerate(L1_COUNTS)) / UNIVERSE
    return {
        "n": n,
        "shape_var": l1_var,
        "strata": tuple(tuple(v) for v in strata.values()),
        "strata_year_weekday": len(strata),
        "sky_counts": sky_counts,
        "expected_hex": expected_hex,
        "expected_shape": expected_shape,
        "expected_totals": expected_totals,
        "hamming_expected_sum": hamming_expected_sum,
    }


def independent_permutation(n: int, strata: tuple[tuple[int, ...], ...], rng: random.Random) -> list[int]:
    """Sky indices remain fixed; entire lottery outcome tuples move together."""
    order = list(range(n))
    for group in strata:
        shuffled = list(group)
        rng.shuffle(shuffled)
        for target, source in zip(group, shuffled):
            order[target] = source
    return order


def _pearson_empirical(obs: dict[int, int], expected: list[float], n: int) -> float:
    """Sum (O-E)^2/E = sum_nonzero O^2/E - n, avoiding sparse empty scans."""
    score = -float(n)
    for code, value in obs.items():
        exp = expected[code]
        if exp <= 0:
            raise AssertionError("observed categorical cell impossible under conditional null")
        score += (value * value) / exp
    return max(score, 0.0)  # guard only ~1e-12 floating-point cancellation


def stats_for_assignment(rows: list[dict], design: dict, assignment: list[int]) -> dict[str, float]:
    n = design["n"]
    if len(assignment) != n:
        raise ValueError("wrong assignment length")
    hex_cells: Counter = Counter()
    shape_cells: Counter = Counter()
    totals = {key: [0.0] * 64 for key in ("sum", "odd", "shape")}
    hamming_total = 0
    for target, source in enumerate(assignment):
        sky = rows[target]["sky"]
        v = rows[source]
        hex_cells[64 * sky + v["outcome"]] += 1
        shape_cells[6 * sky + v["shape"] - 1] += 1
        for key in totals:
            totals[key][sky] += v[key]
        hamming_total += (sky ^ v["outcome"]).bit_count()
    # The null distribution (not an asymptotic chi-square) calibrates the scores.
    theory_var = {"sum": 1075.0, "odd": 6 * 25 / 49 * 24 / 49 * 43 / 48,
                  "shape": design["shape_var"]}
    between = {}
    for feature, target_name in (("sum", "sum_by_sky_64"),
                                 ("odd", "odd_count_by_sky_64"),
                                 ("shape", "shape_L1_mean_by_sky_64")):
        expected = design["expected_totals"][feature]
        between[target_name] = sum(
            (totals[feature][sky] - expected[sky]) ** 2 / (design["sky_counts"][sky] * theory_var[feature])
            for sky in range(64)
        )
    return {
        "outcome_hexagram_64x64": _pearson_empirical(hex_cells, design["expected_hex"], n),
        "hamming_distance_mean_two_sided": abs(hamming_total - design["hamming_expected_sum"]) / n,
        **between,
        "shape_L1_64x6": _pearson_empirical(shape_cells, design["expected_shape"], n),
    }


def calibrate(rows: list[dict], *, reps: int = N_REPS, seed: int = SEED) -> tuple[dict, dict]:
    if not isinstance(reps, int) or reps < 99:
        raise ValueError("at least 99 null replicates needed")
    design = build_null_design(rows)
    observed = stats_for_assignment(rows, design, list(range(len(rows))))
    exceed = Counter()
    null_sum = {key: 0.0 for key in ENDPOINTS}
    null_sum2 = {key: 0.0 for key in ENDPOINTS}
    rng = random.Random(seed)
    for _ in range(reps):
        assignment = independent_permutation(len(rows), design["strata"], rng)
        value = stats_for_assignment(rows, design, assignment)
        for name in ENDPOINTS:
            x = value[name]
            null_sum[name] += x
            null_sum2[name] += x * x
            if x >= observed[name] - 1e-10:
                exceed[name] += 1
    result = {}
    for name in ENDPOINTS:
        mean = null_sum[name] / reps
        variance = max(0., null_sum2[name] / reps - mean * mean)
        raw_p = (1 + exceed[name]) / (reps + 1)
        result[name] = {
            "statistic": observed[name],
            "null_mean": mean,
            "null_sd": sqrt(variance),
            "null_exceedances": exceed[name],
            "permutation_p_plus_one": raw_p,
            "bonferroni_six_global_p": min(1., len(ENDPOINTS) * raw_p),
            "significant_at_fwer_0_05": len(ENDPOINTS) * raw_p < .05,
            "statistic_definition": {
                "outcome_hexagram_64x64": "sparse conditional Pearson 64x64; permutation not asymptotic chi-square",
                "hamming_distance_mean_two_sided": "abs(observed mean XOR distance - stratum-conditional null mean)",
                "sum_by_sky_64": "64-state squared centered group sum / (group_n * exact fair variance)",
                "odd_count_by_sky_64": "64-state squared centered group odd count / (group_n * exact fair variance)",
                "shape_L1_mean_by_sky_64": "64-state squared centered graph-component sum / (group_n * exact fair variance)",
                "shape_L1_64x6": "sparse conditional Pearson 64x6; permutation not asymptotic chi-square",
            }[name],
        }
    return result, design


def describe_states(rows: list[dict], reference: dict) -> tuple[list[dict], dict]:
    by_sky = defaultdict(list)
    for row in rows:
        by_sky[row["sky"]].append(row)
    masses = reference["outcome_parity_hexagram_exact_masses_by_code_0_to_63"]
    fair_hamming = sum(
        sum((sky ^ code).bit_count() * masses[code] for code in range(64)) / UNIVERSE
        for sky in (r["sky"] for r in rows)
    ) / len(rows)
    states = []
    for sky in range(64):
        subset = by_sky[sky]
        outcomes = Counter(r["outcome"] for r in subset)
        states.append({
            "astronomical_hexagram_code": sky,
            "draw_count": len(subset),
            "observed_mean_sum": round(sum(r["sum"] for r in subset) / len(subset), 6),
            "observed_mean_odd_count": round(sum(r["odd"] for r in subset) / len(subset), 6),
            "observed_mean_shape_L1_components": round(sum(r["shape"] for r in subset) / len(subset), 6),
            "observed_mean_hamming_distance": round(sum(r["hamming"] for r in subset) / len(subset), 6),
            "observed_most_common_outcome_hexagram_code": min(
                (i for i, value in outcomes.items() if value == max(outcomes.values()))
            ),
            "source_provenance": "FROZEN_USNO_P2_64_GRAY_TIME_SECTOR",
            "report_scope": "RETROSPECTIVE_DESCRIPTIVE_NOT_PREDICTIVE",
        })
    diagnostics = {
        "state_count": len(states),
        "sky_state_min_draws": min(s["draw_count"] for s in states),
        "sky_state_max_draws": max(s["draw_count"] for s in states),
        "observed_overall_mean_hamming": sum(r["hamming"] for r in rows) / len(rows),
        "fair_iid_exact_parity_mass_weighted_hamming_expected": fair_hamming,
        "observed_overall_mean_sum": sum(r["sum"] for r in rows) / len(rows),
        "observed_overall_mean_odd_count": sum(r["odd"] for r in rows) / len(rows),
        "observed_overall_mean_shape_L1_components": sum(r["shape"] for r in rows) / len(rows),
    }
    return states, diagnostics


def write_exclusive_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(raw)


def build_report(source: Path, audit: Path, *, reps: int, seed: int) -> tuple[dict, list[dict]]:
    rows, provenance = load_frozen(source, audit)
    theory = fair_reference()
    results, design = calibrate(rows, reps=reps, seed=seed)
    states, summary = describe_states(rows, theory)
    decision = "EXPLORATORY_ASSOCIATION_NO_EVIDENCE" if not any(
        r["significant_at_fwer_0_05"] for r in results.values()
    ) else "EXPLORATORY_ASSOCIATION_SIGNAL_REQUIRES_INDEPENDENT_CONFIRMATION"
    report = {
        "schema": SCHEMA,
        "decision": decision,
        "no_causal_or_predictive_claim": True,
        "method": "conditional_permutation_fixed_year_and_weekday_6_endpoint_Bonferroni",
        "null": "IID fair Lotto649 outcomes independent of prespecified astronomical sky states; conditional on observed outcome multisets within year×weekday strata",
        "endpoint_preregistration_note": "P2A is an exploratory successor chosen after inspecting P1A; six tests fixed before inspecting P2A outcomes, not an independent holdout",
        "source": provenance,
        "analysis_draws": len(rows),
        "frozen_draws": EXPECTED_RAW_ROWS,
        "mapped_sky_hexagram_states": 64,
        "calendar_strata": design["strata_year_weekday"],
        "simulation_replicates": reps,
        "seed": seed,
        "p_value_resolution": 1 / (reps + 1),
        "alpha_familywise": .05,
        "endpoints": results,
        "fair_6of49_theory": theory,
        "observed_summary": summary,
        "bonferroni_significant_endpoints": [k for k, v in results.items() if v["significant_at_fwer_0_05"]],
        "states_csv": STATES_NAME,
        "limitations": [
            "Six global tests are correlated; Bonferroni remains valid but conservative.",
            "64x64 tables are sparse, so Pearson statistics are permutation-calibrated; power may be low.",
            "Permutation within calendar year and draw weekday conditions on observed outcomes; it does not test whether overall lottery distributions match theoretical IID by itself.",
            "Year×weekday stratification reduces calendar/regime confounding but cannot rule out every time-dependent nuisance; analyses are retrospective and exploratory.",
            "USNO solar/lunar phase bins are experimental time partitions, not authenticated traditional Chinese six-line divination or actual lunar phase angles.",
            "The historical draw time is not verified; astronomical state is sampled at 12:00 Asia/Taipei.",
            "The 22 missing sky states in early 2004 are excluded without extrapolating 2003 events.",
            "Same-draw parity code, hamming, sum, odd count and shape are outcome-only analysis targets, not future-known predictors.",
            "P1A exploratory screen already inspected overlapping historical observations; a significant P2A result would require future-only independent confirmation.",
            "No individual state-level p-values, forecast, candidate tickets or buying decisions are produced.",
        ],
    }
    return report, states


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exploratory P2A 64-state astronomical hexagram association, not lottery prediction")
    parser.add_argument("--p2-source", type=Path, default=Path("data/research/v24_iching_p2/v24_p2_solar_lunar_hexagram_draw_context_2498.csv"))
    parser.add_argument("--p2-audit", type=Path, default=Path("data/research/v24_iching_p2/v24_p2_solar_lunar_hexagram_audit_2498.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/research/v24_iching_p2a"))
    parser.add_argument("--reps", type=int, default=N_REPS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    if args.reps < 99:
        parser.error("--reps must be at least 99")
    dst_json = args.out_dir / REPORT_NAME
    dst_csv = args.out_dir / STATES_NAME
    if dst_json.exists() or dst_csv.exists():
        parser.error("output already exists; frozen research data cannot be overwritten")
    report, states = build_report(args.p2_source, args.p2_audit, reps=args.reps, seed=args.seed)
    from io import StringIO
    buf = StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=STATE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(states)
    csv_raw = buf.getvalue().encode("utf-8")
    report["states_csv_sha256"] = sha256_bytes(csv_raw)
    report_raw = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    # Exclusive create: archived P2 inputs are never written.
    write_exclusive_bytes(dst_csv, csv_raw)
    write_exclusive_bytes(dst_json, report_raw)
    print(f"P2A: {report['analysis_draws']} mapped / {report['frozen_draws']} archived draws; {report['calendar_strata']} strata")
    for name, outcome in report["endpoints"].items():
        print(f"  {name}: statistic={outcome['statistic']:.5f} p={outcome['permutation_p_plus_one']:.5f} bonf={outcome['bonferroni_six_global_p']:.5f}")
    print(f"DECISION: {report['decision']}")
    print(f"REPORT: {dst_json}")
    print(f"STATES: {dst_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
