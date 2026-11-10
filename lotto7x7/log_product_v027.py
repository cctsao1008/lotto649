"""v0.27 Issue #15 P0: exact-binned fair 6/49 Log Product theory and history.

L(C)=ln(prod(C)) for six WITHOUT-REPLACEMENT main balls, 1..49.
Each product remains an exact Python integer. Fixed log-space edges are
translated into integer-product cutoffs using 80-digit Decimal exp/ceil;
bin assignments compare INTEGERS ONLY. All C(49,6)=13,983,816 outcomes
are enumerated. Histogram counts are exact; log moments are real-valued
finite-population formulas, *not* claimed exact rational numbers.

Scope: descriptive theory/historical baseline. No candidate selection,
four-ticket changes or prediction claims. Does not overwrite P0-P2.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from datetime import date
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from hashlib import sha256
from itertools import combinations
import json
from math import comb, fsum, log, prod, sqrt, ceil
from pathlib import Path

from .distribution_repair_v027 import (
    METRIC, TOTAL, baseline as sum_baseline, squared_l2_probability,
)

SCHEMA = "lotto7x7.log_product.v0.27.p0"
DEFAULT_WIDTH = "0.5"
DEFAULT_CUTOFF = "2026-11-06"
EXPECTED_FROZEN_PREFIX = "d865e1d241b607488e58a00b22fb595734f804595afd9d8850718808d6e8b7f8"


def _grid(n: int, k: int, width: str) -> dict:
    """Log-bin half-open edges with integer ceil(exp(edge)) comparisons.

    For integer product P and Decimal-computed edge e, P is assigned to the
    interval [e_i, e_(i+1)) iff ceil(exp(e_i)) <= P < ceil(exp(e_(i+1))).
    For the default half-integer edges none of exp(edge) is an integer, so
    the integer inequalities implement the intended exact real log binning.
    Edges depend ONLY on theoretical minimum/maximum feasible products.
    """
    if type(n) is not int or type(k) is not int or not (4 <= n <= 49 and 2 <= k < n):
        raise ValueError("requires integer 4 <= n <= 49 and 2 <= k < n")
    if not isinstance(width, str):
        raise ValueError("bin width must be a canonical positive decimal string")
    try:
        w = Decimal(width)
    except Exception as exc:
        raise ValueError("invalid decimal bin width") from exc
    if not w.is_finite() or not (Decimal("0.1") <= w <= Decimal("2.0")):
        raise ValueError("log bin width must be in [0.1,2.0]")
    if str(w) != width:
        raise ValueError("bin width must have canonical Decimal string formatting")
    min_product = prod(range(1, k + 1))
    max_product = prod(range(n - k + 1, n + 1))
    with localcontext() as context:
        context.prec = 80
        low_idx = int((Decimal(min_product).ln() / w).to_integral_value(rounding=ROUND_FLOOR))
        high_idx = int((Decimal(max_product).ln() / w).to_integral_value(rounding=ROUND_CEILING))
        if high_idx <= low_idx:
            raise ValueError("empty theoretical log-product bin grid")
        edges_decimal = [w * (low_idx + i) for i in range(high_idx - low_idx + 1)]
        thresholds = [
            int(edge.exp().to_integral_value(rounding=ROUND_CEILING))
            for edge in edges_decimal
        ]
        # Keep a generous margin to integer cutoffs, preventing numerical
        # ambiguity even if another Decimal implementation is used.
        distances = [
            min(abs(edge.exp() - Decimal(upper - 1)),
                abs(Decimal(upper) - edge.exp()))
            for edge, upper in zip(edges_decimal, thresholds)
        ]
        if any(distance < Decimal("1e-30") for distance in distances):
            raise ArithmeticError("a log edge is dangerously close to an integer product")
    if thresholds != sorted(set(thresholds)):
        raise ValueError("bin width produces non-increasing integer thresholds")
    if not thresholds[0] <= min_product or not max_product < thresholds[-1]:
        raise AssertionError("fixed log grid does not contain theoretical support")
    edges = [format(x, "f") for x in edges_decimal]
    return {
        "n": n, "k": k, "width": width, "edge_start": edges[0],
        "edge_end": edges[-1], "edges": edges,
        "integer_product_thresholds": thresholds,
        "min_product": min_product, "max_product": max_product,
        "bin_count": len(thresholds) - 1,
        "edge_rule": "[lower,upper) in log units; integer product P >= ceil(exp(lower)) and P < ceil(exp(upper))",
        "edge_precision": "Decimal 80 digits; only integer product comparisons after edge construction",
        "selection_rule": "round theoretical extremal log products outward to the fixed half-unit grid; no observed history",
    }


def product_bin(product: int, grid: dict) -> int:
    if type(product) is not int or product < grid["min_product"] or product > grid["max_product"]:
        raise ValueError("product is outside theoretical eligible support")
    cuts = grid["integer_product_thresholds"]
    index = bisect_right(cuts, product) - 1
    if not 0 <= index < grid["bin_count"]:
        raise AssertionError("product fails bin partition")
    return index


def theoretical_log_moments(n: int = 49, k: int = 6) -> dict:
    """Analytical first 3 central moments for a simple random k-subset.

    Let y_i = log(i)-mean(log 1..n), with sum y_i=0. Distinct-label
    inclusion probabilities p1,p2,p3 give
      E[(sum_selected y)^3] = (p1 - 3*p2 + 2*p3) * sum_i y_i^3
    using symmetric-polynomial identities. Var uses finite-population
    correction. This is INDEPENDENT of binning and histogram enumeration.
    """
    if type(n) is not int or type(k) is not int or not (4 <= n <= 49 and 2 <= k < n):
        raise ValueError("requires integer 4 <= n <= 49 and 2 <= k < n")
    values = [log(x) for x in range(1, n + 1)]
    center = fsum(values) / n
    y = [v - center for v in values]
    central2 = fsum(x * x for x in y) / n
    central3 = fsum(x * x * x for x in y)
    mean = k * center
    variance = k * (n - k) / (n - 1) * central2
    p1 = k / n
    p2 = k * (k - 1) / (n * (n - 1))
    p3 = k * (k - 1) * (k - 2) / (n * (n - 1) * (n - 2))
    moment3 = (p1 - 3 * p2 + 2 * p3) * central3
    return {
        "mean": mean,
        "variance": variance,
        "stddev": sqrt(variance),
        "third_central_moment": moment3,
        "skewness": moment3 / variance ** 1.5,
        "method": "exact finite-population inclusion-probability identities; real-number log evaluations",
    }


def exact_log_theory(n: int = 49, k: int = 6, *, width: str = DEFAULT_WIDTH) -> dict:
    """Full exact *integer-count* histogram of log products; no sampling."""
    grid = _grid(n, k, width)
    counts = [0] * grid["bin_count"]
    for ticket in combinations(range(1, n + 1), k):
        counts[product_bin(prod(ticket), grid)] += 1
    total = comb(n, k)
    if sum(counts) != total:
        raise AssertionError("full product enumeration did not normalize")
    if counts[product_bin(grid["min_product"], grid)] < 1 or counts[product_bin(grid["max_product"], grid)] < 1:
        raise AssertionError("missing extremal combinations")
    moments = theoretical_log_moments(n, k)
    if n == 49 and k == 6 and total != TOTAL:
        raise AssertionError("6/49 total combinations mismatch")
    # Canonical checksum excludes approximate log moments and source/history;
    # it fingerprints frozen exact integer binning and full counts.
    sealed = {
        "n": n, "k": k, "width": width,
        "edges": grid["edges"],
        "integer_product_thresholds": grid["integer_product_thresholds"],
        "counts": counts,
    }
    canonical = json.dumps(sealed, separators=(",", ":"), sort_keys=True).encode("ascii")
    quantiles = {}
    for q in ("0.01", "0.05", "0.25", "0.5", "0.75", "0.95", "0.99"):
        target_rank = ceil(Decimal(q) * total)
        so_far = 0
        for index, count in enumerate(counts):
            so_far += count
            if so_far >= target_rank:
                quantiles[q] = {"rank_at_or_above": target_rank,
                                "bin_index": index,
                                "log_lower": grid["edges"][index],
                                "log_upper_exclusive": grid["edges"][index + 1]}
                break
        else:
            raise AssertionError("theoretical quantile not found")
    peak = max(counts)
    local_peaks = sum(
        counts[i] > counts[i-1] and counts[i] > counts[i+1]
        for i in range(1, len(counts) - 1)
    )
    return {
        "schema": "lotto7x7.log_product.exact_binned_theory.v1",
        "universe": total,
        "grid": grid,
        "counts": counts,
        "probabilities": [c / total for c in counts],
        "moments": moments,
        "support_log_min": log(grid["min_product"]),
        "support_log_max": log(grid["max_product"]),
        "mode_bin_indices": [i for i, c in enumerate(counts) if c == peak],
        "strict_interior_local_peaks": local_peaks,
        "quantile_brackets": quantiles,
        "checksum_sha256": sha256(canonical).hexdigest(),
        "checksum_protocol": "SHA256 ASCII compact sorted-key JSON {n,k,width,edges,integer_product_thresholds,counts}",
        "method": "complete enumeration of every k-subset; exact product and exact bin integer counts",
    }


def historical_baseline(csv_dir: Path, *, cutoff: str = DEFAULT_CUTOFF,
                        expected_prefix_sha256: str | None = None,
                        theory: dict | None = None) -> dict:
    """Same chronological frozen six-number prefix as v0.27 sum P0."""
    sum_report = sum_baseline(Path(csv_dir), cutoff=cutoff)
    source = sum_report["source"]
    if expected_prefix_sha256 is not None and source["analysis_input_sha256"] != expected_prefix_sha256:
        raise ValueError("frozen historical prefix SHA256 mismatch")
    from .temporal_dataset_v024 import load_csv_history
    data = load_csv_history(Path(csv_dir))
    n = source["historical_draws"]
    if (int(data.ids[n - 1]) != source["last_draw_id"]
            or int(data.dates[n - 1]) != int(date.fromisoformat(cutoff).strftime("%Y%m%d"))):
        raise AssertionError("historical Log Product rows disagree with frozen sum P0")
    theory = exact_log_theory() if theory is None else theory
    if theory["grid"]["n"] != 49 or theory["grid"]["k"] != 6 or theory["universe"] != TOTAL:
        raise ValueError("historical 6/49 requires a full 49 choose 6 Log Product theory")
    if sum(theory["counts"]) != TOTAL:
        raise ValueError("theory counts do not normalize")
    counts = [0] * theory["grid"]["bin_count"]
    values = []
    for ticket in data.numbers[:n]:
        integer_product = prod(int(x) for x in ticket)
        idx = product_bin(integer_product, theory["grid"])
        counts[idx] += 1
        values.append(log(integer_product))
    if sum(counts) != n:
        raise AssertionError("historical Log Product counts do not normalize")
    empirical_mean = fsum(values) / n
    empirical_var = fsum((v - empirical_mean) ** 2 for v in values) / n
    empirical_skew = (fsum((v - empirical_mean) ** 3 for v in values) / n /
                      empirical_var ** 1.5) if empirical_var > 0 else None
    bins = []
    for i, (actual, possible) in enumerate(zip(counts, theory["counts"])):
        expected = n * possible / theory["universe"]
        bins.append({
            "bin_index": i,
            "log_lower": theory["grid"]["edges"][i],
            "log_upper_exclusive": theory["grid"]["edges"][i + 1],
            "min_integer_product": theory["grid"]["integer_product_thresholds"][i],
            "max_integer_product": theory["grid"]["integer_product_thresholds"][i + 1] - 1,
            "theory_combinations": possible,
            "theory_probability": possible / theory["universe"],
            "observed_count": actual,
            "observed_probability": actual / n,
            "expected_count": expected,
            "residual_count": actual - expected,
            "deficit_count": expected - actual,
        })
    l2 = squared_l2_probability(counts, theory["counts"])
    return {
        "schema": SCHEMA,
        "stage": "LOG_PRODUCT_P0_EXACT_BINNED_THEORY_AND_FROZEN_HISTORY",
        "decision": "DESCRIPTIVE_LOG_PRODUCT_BASELINE_PASS; PREDICTIVE_ADVANTAGE_NOT_EVALUATED",
        "feature": "ln(prod(x1,x2,x3,x4,x5,x6)); main balls only; no bonus",
        "metric": {"name": METRIC, "weights": "equal weight per fixed log-product bin",
                   "normalization": "probabilities, not counts",
                   "zero_bin_policy": "unweighted squared L2, no per-bin inverse probability",
                   "fixed_before_selection": True},
        "source": source,
        "sum_baseline_reference": {
            "theory_checksum_sha256": sum_report["theory"]["checksum_sha256"],
            "squared_l2_probability": sum_report["observed"]["squared_l2_probability"],
            "historical_input_sha256": source["analysis_input_sha256"],
        },
        "theory": theory,
        "observed": {
            "counts": counts,
            "mean": empirical_mean, "variance": empirical_var, "skewness": empirical_skew,
            "squared_l2_probability": l2,
            "top_10_deficit_bin_indices": [r["bin_index"] for r in sorted(
                bins, key=lambda x: (-x["deficit_count"], x["bin_index"]))[:10]],
            "top_10_surplus_bin_indices": [r["bin_index"] for r in sorted(
                bins, key=lambda x: (-x["residual_count"], x["bin_index"]))[:10]],
        },
        "bins": bins,
        "interpretation": "Exact fair-theory binned counts and historical discrepancies; IID draws do not owe compensation to past deficits. No forecasting assertion.",
    }


def markdown_report(result: dict) -> str:
    """Full exact theory/historical bin table and explicitly bracketed quantiles."""
    s, theory, o = result["source"], result["theory"], result["observed"]
    grid, moments = theory["grid"], theory["moments"]
    lines = [
        "# Lotto649 v0.27 Issue #15 — Log Product P0 Theory and Historical Baseline", "",
        f"- **History:** {s['historical_draws']} draws through {s['cutoff']} / #{s['last_draw_id']}",
        f"- **Prefix SHA256:** `{s['analysis_input_sha256']}`",
        f"- **Log theory SHA256:** `{theory['checksum_sha256']}`",
        f"- **Exact universe:** C(49,6) = {theory['universe']:,} combinations (fully enumerated)",
        f"- **Bins:** {grid['bin_count']}; width={grid['width']} natural-log units; half-open [lower,upper)",
        "- **Boundary implementation:** fixed theoretical extremes, 80-digit Decimal exp/ceil, exact integer-product comparisons",
        f"- **Theoretical L mean:** {moments['mean']:.12g}",
        f"- **Theoretical L variance/SD:** {moments['variance']:.12g} / {moments['stddev']:.12g}",
        f"- **Theoretical L skewness:** {moments['skewness']:+.12g} ({'left skew' if moments['skewness'] < 0 else 'nonnegative skew'})",
        f"- **Log support:** [{theory['support_log_min']:.9f}, {theory['support_log_max']:.9f}]",
        f"- **Mode bins:** {', '.join(f'{grid['edges'][i]}–{grid['edges'][i+1]}' for i in theory['mode_bin_indices'])}",
        f"- **Strict interior local peaks:** {theory['strict_interior_local_peaks']}",
        f"- **Observed L mean/variance/skewness:** {o['mean']:.9f} / {o['variance']:.9f} / {o['skewness'] if o['skewness'] is not None else 'N/A'}",
        f"- **Frozen primary loss:** `{METRIC}` = {o['squared_l2_probability']:.12g}",
        f"- **Sum P0 loss (comparison only):** {result['sum_baseline_reference']['squared_l2_probability']:.12g}",
        "", "## Complete fixed-bin theoretical and historical counts", "",
        "| Log interval [lower,upper) | Exact fair combos | Fair P | Observed | Expected | Observed−Expected |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for row in result["bins"]:
        lines.append(
            f"| [{row['log_lower']},{row['log_upper_exclusive']}) | "
            f"{row['theory_combinations']:,} | {row['theory_probability']:.6%} | "
            f"{row['observed_count']} | {row['expected_count']:.3f} | {row['residual_count']:+.3f} |"
        )
    lines.extend(["", "## Largest deficits (expected − observed)", "",
                  "| Log interval | Observed | Expected | Deficit |", "|:---|---:|---:|---:|"])
    for row in sorted(result["bins"], key=lambda x: (-x["deficit_count"], x["bin_index"]))[:10]:
        lines.append(f"| [{row['log_lower']},{row['log_upper_exclusive']}) | "
                     f"{row['observed_count']} | {row['expected_count']:.3f} | "
                     f"{row['deficit_count']:+.3f} |")
    lines.extend(["", "## Exact-count quantile brackets (not point quantiles)", "",
                  "| q | Bin containing rank | Log lower | Log upper (exclusive) |",
                  "|---:|---:|---:|---:|"])
    for q, data in theory["quantile_brackets"].items():
        lines.append(f"| {q} | {data['bin_index']} | {data['log_lower']} | {data['log_upper_exclusive']} |")
    lines.extend(["", "## Reproduction", "", "```powershell",
                  "& .\\.venv\\Scripts\\python.exe -m lotto7x7.log_product_v027 "
                  "--csv-dir data/csv --cutoff 2026-11-06 "
                  "--out results/v027_log_product_p0.json --md-out results/v027_log_product_p0.md",
                  "```", "",
                  "Only the fixed bin counts and product boundaries are exact integers; log moments are evaluated numerically using independent finite-population formulas.",
                  "Third-party historical CSV provenance is inherited from the v0.27 sum baseline, not independently audited against official records.",
                  "**Scope:** this phase does not rank or buy tickets. Historical deficit repair is descriptive, not evidence that fair future draws self-correct.",
                  ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, default=Path("data/csv"))
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    parser.add_argument("--expected-prefix-sha256", default=EXPECTED_FROZEN_PREFIX,
                        help="default frozen #115000103 checksum; explicitly override for other research snapshots")
    parser.add_argument("--out", type=Path, default=Path("results/v027_log_product_p0.json"))
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args(argv)
    output_json, output_md = args.out, args.md_out or args.out.with_suffix(".md")
    if (output_json.resolve() == output_md.resolve() or
            output_json.exists() or output_md.exists()):
        raise FileExistsError("two distinct unused output paths required; never overwrite prior reports")
    result = historical_baseline(
        args.csv_dir, cutoff=args.cutoff, expected_prefix_sha256=args.expected_prefix_sha256)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        file.write("\n")
    output_md.write_text(markdown_report(result), encoding="utf-8")
    theory = result["theory"]
    print(f"[LOG-P0 PASS] {theory['universe']:,} combinations; exact {theory['grid']['bin_count']} bins; SHA256={theory['checksum_sha256']}")
    print(f"[MOMENTS] mean={theory['moments']['mean']:.12g}; SD={theory['moments']['stddev']:.12g}; skew={theory['moments']['skewness']:+.12g}")
    print(f"[HISTORY] {result['source']['historical_draws']} draws through {result['source']['cutoff']} #{result['source']['last_draw_id']}; prefix={result['source']['analysis_input_sha256']}")
    print(f"[LOSS] {METRIC}={result['observed']['squared_l2_probability']:.12g}")
    print(f"[REPORT] {output_json} / {output_md}")
    print("[SCOPE] Descriptive Log Product P0 only; dual-positive P1/P2 and predictive P3 pending")


if __name__ == "__main__":
    main()
