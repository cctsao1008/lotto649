"""v0.27 P1: exact conditional-IID counterfactual sum repair scoring.

Each legal six-number ticket shares its score with every ticket of equal sum.
The score evaluates what would happen IF that draw were appended; it is not a
conditional probability forecast. Exactly 13,983,816 possible outcomes are
represented by the 259 exact total-sum multiplicities, without Monte Carlo.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isclose, sqrt
from pathlib import Path

from .distribution_repair_v027 import (
    LOW, METRIC, SCHEMA as P0_SCHEMA, exact_theory, squared_l2_probability,
)
from .deficit_compensation_v025 import validate_ticket

SCHEMA = "lotto7x7.distribution_repair.v0.27.p1"


def first_ticket_with_sum(target: int) -> tuple[int, ...]:
    """Lexicographically first legal 6-of-49 combination having this sum.

    This is an illustrative deterministic representative, NOT a preference
    between combinations with the same sum or a coverage-optimized selection.
    """
    if not isinstance(target, int) or isinstance(target, bool) or not 21 <= target <= 279:
        raise ValueError("target sum must be an integer from 21 to 279")
    selected: list[int] = []
    start = 1
    remaining_sum = target
    for left in range(6, 0, -1):
        for value in range(start, 51 - left):
            rest = left - 1
            minimum = rest * (2 * value + rest + 1) // 2
            maximum = rest * (99 - rest) // 2
            if minimum <= remaining_sum - value <= maximum:
                selected.append(value)
                start = value + 1
                remaining_sum -= value
                break
        else:
            raise AssertionError("unreachable sum according to exact 6-of-49 theory")
    result = validate_ticket(selected)
    if sum(result) != target:
        raise AssertionError("constructed representative has incorrect sum")
    return result


def conditional_repair(observed: list[int], exact_counts: list[int]) -> dict:
    """Compute G for every sum and its conditional IID null law, exactly weighted.

    Frozen metric: squared L2 of probability histograms, as in P0.
    For a bin j and N observations:
       after_j = B + (2*n_j + 1)/(N+1)^2 - 2*p_j/(N+1)
       B = sum_i (n_i/(N+1) - p_i)^2.
    Thus ranking is EXACTLY the integer ordering of
       advantage_j = (N+1)*exact_count_j - M*n_j,
    where M = sum(exact_counts); floating-point rounding never breaks ties.
    The null probabilities equal exact_count_j/M, NOT 1/#sum-bins.
    """
    a = list(observed)
    b = list(exact_counts)
    if not a or len(a) != len(b) or any(type(x) is not int or x < 0 for x in a + b):
        raise ValueError("same-size nonnegative integer count vectors are required")
    n, total = sum(a), sum(b)
    if n == 0 or total == 0:
        raise ValueError("empty historical or theoretical histogram")
    before = squared_l2_probability(a, b)
    next_n = n + 1
    base_after = sum((x / next_n - c / total) ** 2 for x, c in zip(a, b))
    rows = []
    for index, (count, exact_count) in enumerate(zip(a, b)):
        p = exact_count / total
        after = base_after + (2 * count + 1) / next_n**2 - 2 * p / next_n
        gain = before - after
        rows.append({
            "index": index,
            "observed_count": count,
            "expected_count": n * p,
            "residual_count": count - n * p,
            "theory_combinations": exact_count,
            "theory_probability": p,
            "rank_integer": next_n * exact_count - total * count,
            "loss_before": before,
            "loss_after": after,
            "gain": gain,
            "gain_micro_l2": gain * 1_000_000,
        })
    # Exact integer grouping prevents spurious rank splits at machine precision.
    group_counts: dict[int, int] = {}
    group_losses: dict[int, float] = {}
    for item in rows:
        key = item["rank_integer"]
        group_counts[key] = group_counts.get(key, 0) + item["theory_combinations"]
        group_losses.setdefault(key, item["gain"])
    below = 0
    group_tail = {}
    for key in sorted(group_counts):
        equal = group_counts[key]
        higher = total - below - equal
        group_tail[key] = {
            "probability_less": below / total,
            "probability_equal": equal / total,
            "probability_greater": higher / total,
            "probability_greater_equal": (higher + equal) / total,
            "midrank_percentile": (below + 0.5 * equal) / total,
        }
        below += equal
    if below != total:
        raise AssertionError("conditional IID law did not normalize")
    order = sorted(rows, key=lambda item: (-item["rank_integer"], item["index"]))
    current_rank, last_key = 0, None
    for i, item in enumerate(order, start=1):
        if item["rank_integer"] != last_key:
            current_rank = i
            last_key = item["rank_integer"]
        item["rank"] = current_rank
        item["iid_conditional"] = group_tail[last_key].copy()

    mean = sum(item["gain"] * item["theory_combinations"] for item in rows) / total
    variance = sum(item["theory_combinations"] * (item["gain"] - mean)**2 for item in rows) / total
    # Independent analytical identity, not another enumeration:
    # E[G|H_N] = (1-(N/(N+1))^2)*D_N - (1-sum(p_i^2))/(N+1)^2.
    theory_sq = sum((c / total)**2 for c in b)
    expected_closed = (1 - (n / next_n)**2) * before - (1 - theory_sq) / next_n**2
    if not isclose(mean, expected_closed, rel_tol=2e-9, abs_tol=5e-17):
        raise AssertionError("IID expectation does not match independent L2 identity")
    sd = sqrt(max(variance, 0.0))
    for item in rows:
        item["iid_z_score"] = (item["gain"] - mean) / sd if sd > 0 else None
    null = {
        "definition": "conditional fair IID next draw; exact 6-of-49 sum multiplicities",
        "expected_gain": mean,
        "expected_gain_closed_form": expected_closed,
        "variance_gain": variance,
        "stddev_gain": sd,
        "probability_positive_gain": sum(r["theory_combinations"] for r in rows if r["gain"] > 0) / total,
        "probability_negative_gain": sum(r["theory_combinations"] for r in rows if r["gain"] < 0) / total,
        "law_by_exact_rank_integer": [
            {"rank_integer": k, "probability": group_counts[k] / total,
             "gain": group_losses[k], **group_tail[k]}
            for k in sorted(group_counts, reverse=True)
        ],
        "caution": "Conditional tail probabilities/ranks are descriptive, not validated predictive p-values.",
    }
    return {"historical_draws": n, "loss_before": before,
            "rows_by_index": rows, "ordered_indices": [r["index"] for r in order],
            "iid_null": null}


def score_p0(p0: dict, *, top_k: int = 10) -> dict:
    """Validate an existing frozen P0 report, then score each hypothetical sum."""
    if p0.get("schema") != P0_SCHEMA or p0.get("metric", {}).get("name") != METRIC:
        raise ValueError("requires validated v0.27 P0 with the frozen squared-L2 metric")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 259:
        raise ValueError("top_k must be in 1..259")
    theory = exact_theory()
    if p0.get("theory", {}).get("counts") != theory["counts"] or p0["theory"].get("checksum_sha256") != theory["checksum_sha256"]:
        raise ValueError("P0 exact sum theory is missing, stale or modified")
    observed = p0.get("observed", {}).get("counts")
    if not isinstance(observed, list) or len(observed) != len(theory["counts"]):
        raise ValueError("P0 observed count histogram missing")
    source = p0.get("source", {})
    n = source.get("historical_draws")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1 or sum(observed) != n:
        raise ValueError("P0 historical count does not match source metadata")
    if not isinstance(source.get("cutoff"), str) or source.get("last_draw_date") != source["cutoff"]:
        raise ValueError("P0 explicit data cutoff missing")
    fingerprint = source.get("analysis_input_sha256")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
        raise ValueError("P0 historical prefix SHA256 missing")
    scored = conditional_repair(observed, theory["counts"])
    before = p0["observed"]["squared_l2_probability"]
    if not isclose(scored["loss_before"], before, rel_tol=1e-12, abs_tol=1e-17):
        raise ValueError("P0 loss is incompatible with P0 counts")
    for index, entry in enumerate(scored["rows_by_index"]):
        entry["sum"] = LOW + index
    ranked = sorted(scored["rows_by_index"], key=lambda item: (-item["rank_integer"], item["sum"]))
    high = []
    low = []
    for group, subset in (("top", ranked[:top_k]), ("bottom", ranked[-top_k:][::-1])):
        target = high if group == "top" else low
        for row in subset:
            target.append({"sum": row["sum"], "rank": row["rank"], "gain": row["gain"],
                           "gain_micro_l2": row["gain_micro_l2"],
                           "loss_after": row["loss_after"], "observed_count": row["observed_count"],
                           "expected_count": row["expected_count"],
                           "rank_integer": row["rank_integer"],
                           "representative_ticket": list(first_ticket_with_sum(row["sum"]))})
    return {
        "schema": SCHEMA,
        "stage": "P1_COUNTERFACTUAL_SUM_REPAIR_SCORING",
        "decision": "REPAIR_DESCRIPTIVE_PASS; PREDICTIVE_ADVANTAGE_NOT_EVALUATED",
        "objective": "G_N(C)=L_N-L_{N+1}(sum(C)); no probability-forecast claim",
        "metric": p0["metric"],
        "source": {k: source.get(k) for k in (
            "cutoff", "first_draw_id", "last_draw_id", "historical_draws",
            "analysis_input_sha256", "analysis_input_sha256_protocol")},
        "theory": {"universe": theory["universe"], "support_min": LOW,
                   "support_max": LOW + len(theory["counts"]) - 1,
                   "checksum_sha256": theory["checksum_sha256"]},
        "baseline_loss": scored["loss_before"],
        "iid_null": scored["iid_null"],
        "sums": scored["rows_by_index"],
        "ranking": {"rule": "descending exact integer (N+1)*theory_count - total_universe*observed_count; then ascending sum",
                    "all_same_sum_tickets_tied": True,
                    "rank_method": "competition ranking (1,1,3); tie probabilities weighted by exact ticket counts",
                    "top": high, "bottom": low},
        "interpretation": "Ranks hypothetical single next draws. No four tickets are jointly appended. Repair is not predictive probability.",
    }


def score_ticket(report: dict, ticket) -> dict:
    """Look up a validated six-number ticket using its exact sum-only tie."""
    if report.get("schema") != SCHEMA or len(report.get("sums", [])) != 259:
        raise ValueError("expected complete v0.27 P1 sum score report")
    numbers = validate_ticket(ticket)
    row = report["sums"][sum(numbers) - LOW]
    return {"ticket": list(numbers), "sum": sum(numbers), "rank": row["rank"],
            "gain": row["gain"], "gain_micro_l2": row["gain_micro_l2"],
            "loss_before": row["loss_before"], "loss_after": row["loss_after"],
            "observed_count": row["observed_count"],
            "expected_count": row["expected_count"],
            "theory_probability": row["theory_probability"],
            "iid_conditional": row["iid_conditional"],
            "iid_z_score": row["iid_z_score"],
            "note": "All legal tickets with this sum have identical P1 repair scores."}


def markdown_report(report: dict) -> str:
    """Human-readable exact IID reference, candidate ranking and rationale."""
    s, null, rank = report["source"], report["iid_null"], report["ranking"]
    lines = [
        "# Lotto649 v0.27 — P1 Counterfactual Repair Ranking", "",
        f"- **Frozen cutoff:** {s['cutoff']} / #{s['last_draw_id']}; N={s['historical_draws']}",
        f"- **Historical prefix SHA-256:** `{s['analysis_input_sha256']}`",
        f"- **Theory SHA-256:** `{report['theory']['checksum_sha256']}`",
        f"- **P0 loss:** {report['baseline_loss']:.15g}",
        f"- **Metric:** `{report['metric']['name']}`",
        f"- **Conditional IID E[G]:** {null['expected_gain']:.15g}",
        f"- **Conditional IID SD[G]:** {null['stddev_gain']:.15g}",
        f"- **IID P(G > 0):** {null['probability_positive_gain']:.6%}",
        "", "**Scope:** each row is ONE possible next draw. All six-number combinations with the same sum are tied. The displayed representative is lexicographically first, not a suggested bet.", "",
        "## Highest hypothetical repair scores", "",
        "| Rank | Sum | Observed | IID expected | G (×10⁶) | Loss if appended | Illustrative combination |",
        "|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for row in rank["top"]:
        nums = " ".join(f"{x:02d}" for x in row["representative_ticket"])
        lines.append(f"| {row['rank']} | {row['sum']} | {row['observed_count']} | {row['expected_count']:.3f} | {row['gain_micro_l2']:+.6f} | {row['loss_after']:.12g} | {nums} |")
    lines.extend(["", "## Lowest hypothetical repair scores", "",
                  "| Rank | Sum | Observed | IID expected | G (×10⁶) | Loss if appended | Illustrative combination |",
                  "|---:|---:|---:|---:|---:|---:|:---|"])
    for row in rank["bottom"]:
        nums = " ".join(f"{x:02d}" for x in row["representative_ticket"])
        lines.append(f"| {row['rank']} | {row['sum']} | {row['observed_count']} | {row['expected_count']:.3f} | {row['gain_micro_l2']:+.6f} | {row['loss_after']:.12g} | {nums} |")
    lines.extend(["", "## Interpretation and validation", "",
                  "- G > 0 mathematically lowers the frozen P0 discrepancy if that sum is drawn next; G < 0 increases it.",
                  "- IID next-draw probability for each sum equals its exact combination count / 13,983,816.",
                  "- The IID null mean, SD, and tie-inclusive tail probabilities are conditional on the historical prefix.",
                  "- Ranks use integer arithmetic; false floating-point tie-breaking is avoided.",
                  "- Neither a positive G nor a high percentile proves next-draw self-correction or improved ticket odds.",
                  "- No P2 four-ticket optimization or P3 historical predictive test is performed.",
                  "", "## Reproduce", "", "```powershell",
                  "& .\\.venv\\Scripts\\python.exe -m lotto7x7.distribution_repair_scoring_v027 --p0 results/v027_p0_baseline.json --out results/v027_p1_repair.json --md-out results/v027_p1_repair.md",
                  "```", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0", required=True, type=Path, help="frozen P0 JSON report")
    parser.add_argument("--out", type=Path, default=Path("results/v027_p1_repair.json"))
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ticket", action="append", default=[], metavar="01,02,03,04,05,06",
                        help="optional legal 6-number example; score only (no portfolio selection)")
    args = parser.parse_args(argv)
    json_path = args.out
    md_path = args.md_out or json_path.with_suffix(".md")
    if json_path.resolve() == md_path.resolve() or json_path.exists() or md_path.exists():
        raise FileExistsError("P1 output paths must be distinct and unused")
    raw = args.p0.read_bytes()
    p0 = json.loads(raw)
    output = score_p0(p0, top_k=args.top_k)
    output["input_p0_file_sha256"] = sha256(raw).hexdigest()
    output["ticket_examples"] = []
    for text in args.ticket:
        try:
            values = [int(x.strip()) for x in text.split(",")]
        except ValueError as exc:
            raise ValueError("ticket must contain six comma-separated integers") from exc
        output["ticket_examples"].append(score_ticket(output, values))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("x", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    md_path.write_text(markdown_report(output), encoding="utf-8")
    print(f"[P1 PASS] 259 sum bins; exact IID 6/49 weighting; cutoff={output['source']['cutoff']}")
    print(f"[BASE] {output['baseline_loss']:.12g}")
    print(f"[IID] E[G]={output['iid_null']['expected_gain']:.12g}; SD[G]={output['iid_null']['stddev_gain']:.12g}")
    print("[TOP] " + ", ".join(f"{r['sum']} ({r['gain_micro_l2']:+.6f} x10^-6)" for r in output["ranking"]["top"]))
    print(f"[REPORT] {json_path} / {md_path}")
    print("[SCOPE] Single hypothetical outcome; no prediction or P2 portfolio search")


if __name__ == "__main__":
    main()
