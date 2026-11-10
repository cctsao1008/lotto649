"""v0.27 Issue #15 P1: exact-sign Sum + Log Product counterfactual repair.

A ticket passes iff BOTH L2 losses decrease if that ONE ticket is
hypothetically appended as the next draw, separately for each feature.
Gain signs are exact rational arithmetic on integer histogram counts;
floating point is ONLY a readable representation of the magnitude.

Enumerates all legal 6/49 tickets only to count exact fair IID mass of the
dual-positive set. This does NOT make those tickets any more likely to win.
No four-ticket optimizer and no four-draw history append occur in P1.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from hashlib import sha256
from itertools import combinations
import json
from math import comb, isclose, prod
from pathlib import Path

from .deficit_compensation_v025 import validate_ticket
from .distribution_repair_v027 import LOW, METRIC, TOTAL
from .distribution_repair_portfolio_v027 import verify_p1
from .distribution_repair_scoring_v027 import conditional_repair
from .log_product_v027 import (
    SCHEMA as LOG_P0_SCHEMA,
    exact_log_theory,
    product_bin,
)

SCHEMA = "lotto7x7.log_product.v0.27.p1"


def exact_gain_numerators(observed: list[int], theoretical: list[int]) -> dict:
    """All possible append-one squared-L2 gains as exact rational numbers.

    Let M=sum(theory counts), N=sum(hist counts), a_i=M*n_i-N*c_i:
    D_N = sum(a_i^2)/(M^2*N^2).
    Define b_i=M*n_i-(N+1)*c_i. D_(N+1,j)
       = [sum(b_i^2) + 2*M*b_j + M^2]/[M^2*(N+1)^2].
    Subtract to obtain G_j with ONE positive common denominator.
    Thus G_j > 0 iff integer numerator_j > 0. There is no arbitrary
    epsilon threshold or vulnerable floating-point sign test.
    """
    if (not isinstance(observed, (tuple, list))
            or not isinstance(theoretical, (tuple, list))
            or len(observed) != len(theoretical) or not observed
            or any(type(x) is not int or x < 0 for x in (*observed, *theoretical))):
        raise ValueError("two same-length nonnegative integer histograms required")
    n, m = sum(observed), sum(theoretical)
    if n < 1 or m < 1:
        raise ValueError("historical and fair theory histograms must be nonempty")
    old_squares = sum((m * h - n * t) ** 2 for h, t in zip(observed, theoretical))
    new_terms = [m * h - (n + 1) * t for h, t in zip(observed, theoretical)]
    new_squares = sum(x * x for x in new_terms)
    numerators = [
        old_squares * (n + 1)**2 - n*n*(new_squares + 2*m*x + m*m)
        for x in new_terms
    ]
    common_denominator = m*m*n*n*(n+1)**2
    return {
        "numerators": numerators,
        "denominator": common_denominator,
        "positive_indices": [i for i, v in enumerate(numerators) if v > 0],
        "zero_indices": [i for i, v in enumerate(numerators) if v == 0],
        "sign_rule": "integer numerator > 0; exact rational arithmetic, zero tolerance",
        "gain_definition": "D_N - D_(N+1, appended single outcome)",
    }


def verify_log_p0(log_p0: dict, sum_p1: dict) -> tuple[dict, list[int]]:
    """Check prefix, frozen metric and EXACT theoretical binning/counts."""
    if not isinstance(log_p0, dict) or log_p0.get("schema") != LOG_P0_SCHEMA:
        raise ValueError("expected complete frozen Log Product P0 report")
    if log_p0.get("metric", {}).get("name") != METRIC:
        raise ValueError("Log P0 primary metric mismatch")
    source, sum_src = log_p0.get("source", {}), sum_p1.get("source", {})
    for field in ("cutoff", "last_draw_id", "historical_draws", "analysis_input_sha256"):
        if source.get(field) != sum_src.get(field):
            raise ValueError(f"Sum P1 and Log P0 source mismatch: {field}")
    if source.get("historical_draws", 0) < 1:
        raise ValueError("missing historical Log P0 count")
    log_theory = log_p0.get("theory", {})
    width = log_theory.get("grid", {}).get("width")
    expected = exact_log_theory(width=width)
    if (log_theory.get("checksum_sha256") != expected["checksum_sha256"]
            or log_theory.get("counts") != expected["counts"]
            or log_theory.get("grid") != expected["grid"]):
        raise ValueError("Log Product exact theoretical histogram or bin boundaries altered")
    counts = log_p0.get("observed", {}).get("counts")
    if (not isinstance(counts, list) or len(counts) != len(expected["counts"])
            or any(type(x) is not int or x < 0 for x in counts)
            or sum(counts) != source["historical_draws"]):
        raise ValueError("Log P0 historical counts invalid")
    from .distribution_repair_v027 import squared_l2_probability
    expected_loss = squared_l2_probability(counts, expected["counts"])
    if not isclose(expected_loss,
                   log_p0["observed"].get("squared_l2_probability", float("nan")),
                   rel_tol=1e-12, abs_tol=1e-16):
        raise ValueError("Log P0 historical distance mismatched")
    return expected, counts


def score_ticket(report: dict, ticket) -> dict:
    if report.get("schema") != SCHEMA or len(report.get("sum_bins", [])) != 259:
        raise ValueError("expected complete v0.27 dual-score report")
    if len(report.get("log_bins", [])) != report.get("log_theory", {}).get("grid", {}).get("bin_count"):
        raise ValueError("Log scoring bins missing")
    numbers = validate_ticket(ticket)
    s = sum(numbers)
    product = prod(numbers)
    li = product_bin(product, report["log_theory"]["grid"])
    a = report["sum_bins"][s - LOW]
    b = report["log_bins"][li]
    return {
        "ticket": list(numbers),
        "sum": s,
        "integer_product": product,
        "log_product_bin_index": li,
        "log_product_interval": [b["log_lower"], b["log_upper_exclusive"]],
        "sum": s,
        "sum_gain": a["gain"],
        "sum_gain_micro_l2": a["gain"] * 1e6,
        "sum_loss_before": a["loss_before"],
        "sum_loss_after": a["loss_after"],
        "sum_rank": a["rank"],
        "sum_observed_count": a["observed_count"],
        "sum_expected_count": a["expected_count"],
        "sum_exact_positive": a["exact_gain_numerator"] > 0,
        "log_gain": b["gain"],
        "log_gain_micro_l2": b["gain"] * 1e6,
        "log_loss_before": b["loss_before"],
        "log_loss_after": b["loss_after"],
        "log_rank": b["rank"],
        "log_observed_count": b["observed_count"],
        "log_expected_count": b["expected_count"],
        "log_exact_positive": b["exact_gain_numerator"] > 0,
        "dual_positive": a["exact_gain_numerator"] > 0 and b["exact_gain_numerator"] > 0,
        "interpretation": "Hypothetical single next draw; this is NOT a forecast of win probability.",
    }


def dual_score(sum_p1: dict, log_p0: dict, *, example_limit: int = 8) -> dict:
    if type(example_limit) is not int or not 1 <= example_limit <= 100:
        raise ValueError("example_limit must be 1..100")
    sum_rows = verify_p1(sum_p1)
    theory, observed = verify_log_p0(log_p0, sum_p1)
    from .distribution_repair_v027 import exact_theory
    sum_counts = [row["observed_count"] for row in sum_rows]
    sum_theory = exact_theory()["counts"]
    sum_gains = exact_gain_numerators(sum_counts, sum_theory)
    log_gains = exact_gain_numerators(observed, theory["counts"])
    log_scored = conditional_repair(observed, theory["counts"])
    if len(sum_gains["positive_indices"]) == 0 or len(log_gains["positive_indices"]) == 0:
        # Enumerate as usual for an exact overall count; no silent relaxation.
        pass
    srows = []
    for i, original in enumerate(sum_rows):
        row = {k: original[k] for k in ("sum", "observed_count", "expected_count",
                                       "loss_before", "loss_after", "gain", "rank")}
        row["exact_gain_numerator"] = sum_gains["numerators"][i]
        row["exact_positive"] = row["exact_gain_numerator"] > 0
        if not isclose(row["gain"], row["exact_gain_numerator"] / sum_gains["denominator"],
                       rel_tol=1e-9, abs_tol=3e-16):
            raise AssertionError("sum P1 float gain disagrees with exact rational")
        srows.append(row)
    lrows = []
    for i, item in enumerate(log_scored["rows_by_index"]):
        row = {k: item[k] for k in ("observed_count", "expected_count",
                                    "loss_before", "loss_after", "gain", "rank",
                                    "theory_combinations", "theory_probability",
                                    "iid_conditional", "iid_z_score")}
        row.update({
            "bin_index": i,
            "log_lower": theory["grid"]["edges"][i],
            "log_upper_exclusive": theory["grid"]["edges"][i + 1],
            "exact_gain_numerator": log_gains["numerators"][i],
            "exact_positive": log_gains["numerators"][i] > 0,
        })
        if not isclose(row["gain"], row["exact_gain_numerator"] / log_gains["denominator"],
                       rel_tol=1e-9, abs_tol=3e-16):
            raise AssertionError("log float gain disagrees with exact rational")
        lrows.append(row)
    positive_sum = set(sum_gains["positive_indices"])
    positive_log = set(log_gains["positive_indices"])
    grid = theory["grid"]
    thresholds = grid["integer_product_thresholds"]
    qualifying = 0
    per_sum = [0] * 259
    per_log = [0] * grid["bin_count"]
    examples = []
    # This is exact fair IID combinatorial mass. Do NOT interpret as the
    # probability that future draws will correct history more often.
    for ticket in combinations(range(1, 50), 6):
        si = sum(ticket) - LOW
        if si not in positive_sum:
            continue
        product = prod(ticket)
        li = bisect_right(thresholds, product) - 1
        if li in positive_log:
            qualifying += 1
            per_sum[si] += 1
            per_log[li] += 1
            if len(examples) < example_limit:
                examples.append(list(ticket))
    if sum(per_sum) != qualifying or sum(per_log) != qualifying:
        raise AssertionError("exact dual-positive fair IID count did not close")
    result = {
        "schema": SCHEMA,
        "stage": "DUAL_P1_EXACT_SIGN_SUM_LOG_PRODUCT_COUNTERFACTUAL_SCORING",
        "decision": "DUAL_CONSTRAINT_DESCRIPTIVE_EVALUATED; PREDICTIVE_ADVANTAGE_NOT_EVALUATED",
        "source": sum_p1["source"],
        "sum_theory": sum_p1["theory"],
        "log_theory": {
            "checksum_sha256": theory["checksum_sha256"],
            "grid": grid,
            "universe": theory["universe"],
        },
        "metric": METRIC,
        "sum_bins": srows,
        "log_bins": lrows,
        "sum_baseline_loss": sum_p1["baseline_loss"],
        "log_baseline_loss": log_scored["loss_before"],
        "exact_rational": {
            "sum_common_denominator": sum_gains["denominator"],
            "log_common_denominator": log_gains["denominator"],
            "strict_threshold": "numerator > 0; no epsilon",
        },
        "conditional_iid": {
            "sum_expected_gain": sum_p1["iid_null"]["expected_gain"],
            "sum_stddev_gain": sum_p1["iid_null"]["stddev_gain"],
            "log_expected_gain": log_scored["iid_null"]["expected_gain"],
            "log_stddev_gain": log_scored["iid_null"]["stddev_gain"],
            "dual_positive_fair_mass": qualifying,
            "dual_positive_fair_probability": qualifying / TOTAL,
            "mass_method": "exhaustive 13,983,816 legal tickets, exact integer products and pre-fixed log bins",
            "caution": "The two features are correlated. Their marginal IID probabilities must NOT be multiplied as if independent.",
        },
        "qualifying": {
            "rule": "G_sum(C) > 0 AND G_log(C) > 0; independently exact integer-numerator signs",
            "positive_sum_bin_count": len(positive_sum),
            "positive_log_bin_count": len(positive_log),
            "dual_positive_ticket_count": qualifying,
            "feasibility": "NONEMPTY" if qualifying else "INFEASIBLE_NO_DUAL_POSITIVE_TICKET",
            "per_sum_counts": per_sum,
            "per_log_bin_counts": per_log,
            "lex_first_examples": examples,
            "portfolio_note": "No four-disjoint portfolio certified in P1; P2 must handle four-ticket feasibility and optimize it.",
        },
        "ticket_examples": [],
        "interpretation": "Repair of two historical marginal distributions is a deterministic counterfactual property, not increased lottery odds.",
    }
    for numbers in examples:
        if not score_ticket(result, numbers)["dual_positive"]:
            raise AssertionError("dual feasible example failed exact score check")
    return result


def markdown_report(report: dict) -> str:
    src = report["source"]
    q = report["qualifying"]
    iid = report["conditional_iid"]
    lines = [
        "# Lotto649 v0.27 Issue #15 — P1 Exact Dual-Positive Repair", "",
        f"- **Cutoff:** {src['cutoff']} / #{src['last_draw_id']}; {src['historical_draws']} draws.",
        f"- **Prefix SHA256:** `{src['analysis_input_sha256']}`",
        f"- **Log theory SHA256:** `{report['log_theory']['checksum_sha256']}`",
        f"- **Metric:** `{METRIC}` for EACH distribution, not a raw-distance weighted sum.",
        "- **Strict condition:** both exact rational gain numerators must be >0.",
        f"- **Positive sum bins:** {q['positive_sum_bin_count']} / 259.",
        f"- **Positive log-product bins:** {q['positive_log_bin_count']} / {report['log_theory']['grid']['bin_count']}.",
        f"- **Legal combinations satisfying BOTH:** {q['dual_positive_ticket_count']:,} / 13,983,816.",
        f"- **Exact fair IID dual-positive mass:** {iid['dual_positive_fair_probability']:.9%}.",
        f"- **Conditional IID E[G_sum],SD:** {iid['sum_expected_gain']:.9g}, {iid['sum_stddev_gain']:.9g}.",
        f"- **Conditional IID E[G_log],SD:** {iid['log_expected_gain']:.9g}, {iid['log_stddev_gain']:.9g}.",
        f"- **Status:** {q['feasibility']}. No four-disjoint portfolio has been selected or certified.", "",
        "## Log Product bin-by-bin repair scores", "",
        "| Bin [lower,upper) | Observed | IID expected | Gain ×10⁶ | Exact positive? |",
        "|:---|---:|---:|---:|:---:|",
    ]
    for b in report["log_bins"]:
        lines.append(f"| [{b['log_lower']},{b['log_upper_exclusive']}) | {b['observed_count']} | "
                     f"{b['expected_count']:.3f} | {b['gain']*1e6:+.6f} | "
                     f"{'YES' if b['exact_positive'] else 'NO'} |")
    lines.extend(["", "## Lexicographically first qualifying examples (NOT recommended bets)", "",
                  "| Example | Ticket | Sum G ×10⁶ | Log G ×10⁶ |",
                  "|---:|:---|---:|---:|"])
    for i, row in enumerate(q["lex_first_examples"], start=1):
        ticket = score_ticket(report, row)
        lines.append(f"| {i} | {' '.join(f'{v:02d}' for v in row)} | "
                     f"{ticket['sum_gain_micro_l2']:+.6f} | {ticket['log_gain_micro_l2']:+.6f} |")
    lines.extend(["", "## Caveats", "",
                  "- These are hypothetical single-draw changes; four tickets are NOT appended simultaneously.",
                  "- The two features are correlated. Their IID probabilities are not independent.",
                  "- An exact fair fraction of dual-positive combinations is not a forecast that these combinations will occur more frequently.",
                  "- Four-ticket feasibility, eight-tier coverage parity and optimum certification belong to P2.",
                  "- P3 must evaluate the full prespecified pipeline under chronological out-of-sample and IID control.",
                  "", "## Reproduce", "", "```powershell",
                  "& .\\.venv\\Scripts\\python.exe -m lotto7x7.log_product_dual_scoring_v027 "
                  "--sum-p1 results/v027_p1_repair.json "
                  "--log-p0 results/v027_log_product_p0.json "
                  "--out results/v027_dual_p1.json --md-out results/v027_dual_p1.md",
                  "```", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sum-p1", type=Path, required=True)
    parser.add_argument("--log-p0", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/v027_dual_p1.json"))
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument("--ticket", action="append", default=[],
                        help="optional six comma-separated main numbers; score without selecting bets")
    args = parser.parse_args(argv)
    json_path = args.out
    md_path = args.md_out or json_path.with_suffix(".md")
    if json_path.resolve() == md_path.resolve() or json_path.exists() or md_path.exists():
        raise FileExistsError("two distinct unused report paths required")
    sum_raw = args.sum_p1.read_bytes()
    log_raw = args.log_p0.read_bytes()
    report = dual_score(json.loads(sum_raw), json.loads(log_raw))
    report["input_sum_p1_sha256"] = sha256(sum_raw).hexdigest()
    report["input_log_p0_sha256"] = sha256(log_raw).hexdigest()
    for text in args.ticket:
        try:
            values = [int(v.strip()) for v in text.split(",")]
        except ValueError as exc:
            raise ValueError("tickets must be six comma-separated integers") from exc
        report["ticket_examples"].append(score_ticket(report, values))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    q = report["qualifying"]
    print(f"[DUAL-P1 PASS] positive_sum_bins={q['positive_sum_bin_count']}, positive_log_bins={q['positive_log_bin_count']}")
    print(f"[IID EXACT] dual-positive legal tickets={q['dual_positive_ticket_count']:,}/{TOTAL:,} ({report['conditional_iid']['dual_positive_fair_probability']:.9%})")
    print(f"[LOSS] sum={report['sum_baseline_loss']:.12g}; log={report['log_baseline_loss']:.12g}")
    for row in report["ticket_examples"]:
        print(f"[TICKET] {' '.join(f'{x:02d}' for x in row['ticket'])} "
              f"sum_G={row['sum_gain_micro_l2']:+.6f}e-6 "
              f"log_G={row['log_gain_micro_l2']:+.6f}e-6 dual_positive={row['dual_positive']}")
    print(f"[REPORT] {json_path} / {md_path}")
    print("[SCOPE] No four-disjoint portfolio/P2 or predictive claim; two correlated features")


if __name__ == "__main__":
    main()
