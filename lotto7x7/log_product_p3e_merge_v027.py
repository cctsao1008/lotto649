"""v0.27 Issue #15 P3e: merge five complete unmodified-P2 historical replay shards.

Five P3d runs cover the SAME 35 fixed target indices as P3b (warmup=300,
stride=64). Each shard calls the literal original P2 optimizer with all
original budgets. This module independently revalidates ticket legality,
official prize decisions, sealed chronology, source and report consistency,
then computes exact fair-binomial null and optional matched cached-P3b
comparison. This is a retrospective engineering evaluation, not GO.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .distribution_repair_portfolio_v027 import validate_four_disjoint
from .log_product_p3b_pilot_v027 import evaluate_draw
from .log_product_p3c_exact_iid_v027 import (
    audit_pilot, exact_binomial_tail, exact_fair_probability,
)
from .log_product_p3d_legacy_replay_v027 import (
    SCHEMA as P3D_SCHEMA, CONFIG, FINAL_CUTOFF, FROZEN_HASH, FROZEN_ID,
)

SCHEMA = "lotto7x7.log_product.v0.27.p3e_full_replay"
CHECKPOINTS = tuple(range(300, 2497, 64))
SHARD_COUNT = 5


def expected_shard_indices(i: int) -> tuple[int, ...]:
    if type(i) is not int or i not in range(SHARD_COUNT):
        raise ValueError("P3e shard index must be 0..4")
    return CHECKPOINTS[i::SHARD_COUNT]


def _audit_record(entry: dict, index: int) -> None:
    if (not isinstance(entry, dict) or
            entry.get("target_index_zero_based") != index or
            entry.get("training_count") != index or
            entry.get("original_P2_schema") != "lotto7x7.log_product.v0.27.p2" or
            entry.get("original_P2_config") != CONFIG):
        raise ValueError("original P2 record/index/config mismatched")
    if (type(entry.get("previous_draw_id")) is not int or
            type(entry.get("target_draw_id")) is not int or
            entry["previous_draw_id"] >= entry["target_draw_id"] or
            entry.get("previous_draw_date", "") >= entry.get("target_draw_date", "") or
            len(entry.get("source_prefix_sha256", "")) != 64):
        raise ValueError("prior-only chronological fingerprint missing")
    main = entry.get("target_main")
    bonus = entry.get("target_bonus")
    if (not isinstance(main, list) or len(main) != 6 or
            main != sorted(main) or any(type(v) is not int or not 1 <= v <= 49 for v in main) or
            len(set(main)) != 6 or type(bonus) is not int or
            not 1 <= bonus <= 49 or bonus in main):
        raise ValueError("held-out main/special numbers invalid")
    selected = entry.get("original_P2_portfolios", {})
    for name in ("primary", "distinct_sums"):
        row = selected.get(name)
        if not isinstance(row, dict):
            raise ValueError("original P2 strategy report missing")
        if row.get("selection_failure"):
            if row.get("outcome") is not None or row.get("tickets"):
                raise ValueError("failed selection must not present a bet or outcome")
            continue
        tickets = validate_four_disjoint(row.get("tickets"))
        if name == "distinct_sums" and len({sum(t) for t in tickets}) != 4:
            raise ValueError("four different sums constraint broken")
        if row.get("outcome") != evaluate_draw(tickets, main, bonus):
            raise ValueError("original P2 official prize outcome differs from independent audit")
    control = entry.get("random_control", {})
    four = validate_four_disjoint(control.get("tickets"))
    if control.get("outcome") != evaluate_draw(four, main, bonus):
        raise ValueError("random official prize outcome differs from independent audit")


def _check_shard(report: dict, i: int) -> list[dict]:
    expected = list(expected_shard_indices(i))
    if (not isinstance(report, dict) or
            report.get("schema") != P3D_SCHEMA or
            report.get("target_indices") != expected or
            report.get("checkpoint_count") != len(expected) or
            report.get("actual_original_optimizer") !=
                "lotto7x7.log_product_dual_portfolio_v027.optimize" or
            report.get("optimizer_parameters") != CONFIG):
        raise ValueError("shard is not an unchanged original P2 optimizer report")
    source = report.get("source", {})
    if (source.get("analysis_input_sha256") != FROZEN_HASH or
            source.get("last_draw_id") != FROZEN_ID or
            source.get("historical_draws") != 2497 or
            source.get("cutoff") != FINAL_CUTOFF):
        raise ValueError("frozen 2497-draw source fingerprint mismatch")
    records = report.get("individual_replays")
    if not isinstance(records, list) or len(records) != len(expected):
        raise ValueError("shard checkpoint record count mismatch")
    for t, row in zip(expected, records):
        _audit_record(row, t)
    for name in ("primary", "distinct_sums", "random"):
        selected = 0
        failures = 0
        wins = 0
        for row in records:
            item = (row["random_control"] if name == "random" else
                    row["original_P2_portfolios"][name])
            if item.get("selection_failure"):
                failures += 1
            else:
                selected += 1
                wins += item["outcome"]["any_award"]
        stat = report.get("empirical_results", {}).get(name, {})
        if (stat.get("selected") != selected or
                stat.get("selection_failures") != failures or
                stat.get("prize_events") != wins):
            raise ValueError("shard reported winner/selection totals altered")
    return records


def merge_reports(reports: list[dict], *, cached_p3b: dict | None = None) -> dict:
    if not isinstance(reports, list) or len(reports) != SHARD_COUNT:
        raise ValueError("requires exactly five shard reports")
    entries = {}
    bonus_hash = None
    for i, report in enumerate(reports):
        records = _check_shard(report, i)
        this_bonus = report.get("bonus_prefix_sha256")
        if not isinstance(this_bonus, str) or len(this_bonus) != 64:
            raise ValueError("historical bonus-prefix SHA256 invalid")
        if bonus_hash is None:
            bonus_hash = this_bonus
        elif this_bonus != bonus_hash:
            raise ValueError("inconsistent bonus-prefix SHA256 between shards")
        for row in records:
            t = row["target_index_zero_based"]
            if t in entries:
                raise ValueError("duplicate original P2 target index")
            entries[t] = row
    if tuple(sorted(entries)) != CHECKPOINTS:
        raise ValueError("missing or extra original P2 checkpoints")
    ordered = [entries[t] for t in CHECKPOINTS]
    fair = exact_fair_probability()
    counts = {}
    for name in ("primary", "distinct_sums", "random"):
        wins, selected, failures = 0, 0, 0
        for row in ordered:
            item = (row["random_control"] if name == "random" else
                    row["original_P2_portfolios"][name])
            if item.get("selection_failure"):
                failures += 1
            else:
                selected += 1
                wins += item["outcome"]["any_award"]
        counts[name] = {
            "awards": wins,
            "selected": selected,
            "failures": failures,
            "fair_expected_count": float(fair * selected),
            "exact_binomial": (
                exact_binomial_tail(len(CHECKPOINTS), wins, fair)
                if not failures else None),
        }
    if any(counts[k]["failures"] for k in ("primary", "distinct_sums")):
        adjusted = None
        decision = "ORIGINAL_P2_35_REPLAY_WITH_ADAPTIVE_SELECTION_FAILURES; PREDICTIVE_INCONCLUSIVE"
    else:
        adjusted = min(1., 2 * min(counts[k]["exact_binomial"]["upper_tail_probability"]
                                  for k in ("primary", "distinct_sums")))
        decision = "ORIGINAL_P2_35_REPLAY_VERIFIED; PREDICTIVE_NO_GO_OR_INCONCLUSIVE"
    comparison = None
    if cached_p3b is not None:
        audit = audit_pilot(cached_p3b)
        if (audit.get("decision") == "INCONCLUSIVE_ADAPTIVE_SELECTION_FAILURES" or
                cached_p3b.get("source", {}).get("analysis_input_sha256") != FROZEN_HASH):
            raise ValueError("cached P3b report not complete or frozen")
        cached_rows = cached_p3b.get("historical", {}).get("records", [])
        if [r["target_index_zero_based"] for r in cached_rows] != list(CHECKPOINTS):
            raise ValueError("P3b targets differ from same 35 indexed legacy targets")
        comparison = {}
        for name in ("primary", "distinct_sums"):
            identical = 0
            old_win = 0
            cached_win = 0
            disagreement = 0
            for old, newer in zip(ordered, cached_rows):
                if (old["target_main"] != newer.get("target_main") or
                        old["target_bonus"] != newer.get("target_bonus") or
                        old["training_count"] != newer.get("training_count")):
                    raise ValueError("historical target main, special or training differs from cached")
                first = old["original_P2_portfolios"][name]
                second = newer["decisions"][name]
                if first["selection_failure"] or "tickets" not in second:
                    continue
                identical += first["tickets"] == second["tickets"]
                a, b = first["outcome"]["any_award"], second["any_award"]
                old_win += a
                cached_win += b
                disagreement += a != b
            comparison[name] = {
                "identical_portfolios": identical,
                "original_awards_matched_targets": old_win,
                "cached_awards_matched_targets": cached_win,
                "different_award_outcomes": disagreement,
            }
    return {
        "schema": SCHEMA,
        "stage": "FULL_35_TARGET_ORIGINAL_P2_LITERAL_HISTORICAL_REPLAY",
        "decision": decision,
        "source": reports[0]["source"],
        "bonus_prefix_sha256": bonus_hash,
        "original_optimizer": "lotto7x7.log_product_dual_portfolio_v027.optimize",
        "original_optimizer_parameters": CONFIG,
        "checkpoint_indices": list(CHECKPOINTS),
        "checkpoint_count": len(CHECKPOINTS),
        "exact_fair_any_award_probability": float(fair),
        "exact_fair_any_award_rational": [str(fair.numerator), str(fair.denominator)],
        "empirical": counts,
        "two_policy_bonferroni_upper_tail_p": adjusted,
        "matched_p3b_cached_comparison": comparison,
        "records": ordered,
        "limitations": [
            "Every checkpoint uses original P2 and the strictly prior historical prefix, not P3b cached selection.",
            "Historical 2497-draw data was used earlier in model design; this is retrospective, not a prospectively sealed test.",
            "Two-policy Bonferroni does not correct the entire retrospectively explored model family.",
            "All fair-IID valid four-disjoint portfolios have the same official award probability independently of selector.",
            "No predictive GO, no change to previously purchased four ten-draw tickets.",
        ],
    }


def markdown_report(report: dict) -> str:
    lines = [
        "# v0.27 Issue #15 P3e — Full 35-Checkpoint Original P2 Replay",
        "",
        f"- Frozen source: {report['source']['analysis_input_sha256']}",
        f"- Original P2 code: {report['original_optimizer']}",
        f"- Chronological targets: {report['checkpoint_count']}",
        f"- Exact official fair award chance: {report['exact_fair_any_award_probability']:.12%}",
        "",
        "| Policy | Wins | Selected | Failures | Exact one-sided p |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("primary", "distinct_sums", "random"):
        r = report["empirical"][name]
        p = r["exact_binomial"]["upper_tail_probability"] if r["exact_binomial"] else None
        lines.append(f"| {name} | {r['awards']} | {r['selected']} | "
                     f"{r['failures']} | {format(p,'.8f') if p is not None else 'N/A'} |")
    lines.append("")
    lines.append(f"- Two-policy Bonferroni p: {report['two_policy_bonferroni_upper_tail_p']}")
    if report["matched_p3b_cached_comparison"] is not None:
        lines.extend(["", "## Exact matched-date comparison against cached P3b", ""])
        for name, row in report["matched_p3b_cached_comparison"].items():
            lines.append(f"- {name}: identical {row['identical_portfolios']}/35; "
                         f"original wins {row['original_awards_matched_targets']}; "
                         f"cached wins {row['cached_awards_matched_targets']}; "
                         f"discordant targets {row['different_award_outcomes']}")
    lines.extend(["", "## Prespecified true-draw targets", "",
                  "| Zero-based index | Date | Draw ID | Original Primary | Original Diverse | Random |",
                  "|---:|---|---:|---:|---:|---:|"])
    for entry in report["records"]:
        def hit(name):
            r = entry["original_P2_portfolios"][name]
            return str(r["outcome"]["any_award"]) if not r["selection_failure"] else "FAIL"
        lines.append(f"| {entry['target_index_zero_based']} | {entry['target_draw_date']} | "
                     f"{entry['target_draw_id']} | {hit('primary')} | {hit('distinct_sums')} | "
                     f"{entry['random_control']['outcome']['any_award']} |")
    lines.extend(["", "## Interpretation", ""])
    lines.extend("- " + note for note in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", action="append", type=Path, default=[])
    parser.add_argument("--cached-p3b", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.out.resolve() == args.md_out.resolve() or
            args.out.exists() or args.md_out.exists()):
        raise FileExistsError("two distinct unused report paths required")
    if len(args.shard) != SHARD_COUNT:
        raise ValueError("exactly five ordered --shard input files required")
    raw = [p.read_bytes() for p in args.shard]
    source = [json.loads(b) for b in raw]
    cached_raw = args.cached_p3b.read_bytes() if args.cached_p3b else None
    result = merge_reports(source, cached_p3b=json.loads(cached_raw) if cached_raw else None)
    result["shard_report_SHA256"] = [sha256(x).hexdigest() for x in raw]
    if cached_raw:
        result["cached_report_SHA256"] = sha256(cached_raw).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        out.write("\n")
    args.md_out.write_text(markdown_report(result), encoding="utf-8")
    print("[P3E MERGE PASS] original P2 35 fixed prior-only checkpoints, exact official prize audit")
    for name, info in result["empirical"].items():
        print(f"[{name.upper()}] wins={info['awards']} selected={info['selected']} "
              f"failed={info['failures']} fair_expected={info['fair_expected_count']:.4f}")
    print(f"[FAMILYWISE] two-policy Bonferroni p={result['two_policy_bonferroni_upper_tail_p']}")
    if result["matched_p3b_cached_comparison"] is not None:
        for name, d in result["matched_p3b_cached_comparison"].items():
            print(f"[CACHE COMPARISON {name}] same={d['identical_portfolios']}/35; "
                  f"original_wins={d['original_awards_matched_targets']}; "
                  f"cached_wins={d['cached_awards_matched_targets']}")
    print(f"[REPORT] {args.out} / {args.md_out}")
    print("[SCOPE] retrospective legacy optimizer history audit; not prospectively sealed and not predictive GO")


if __name__ == "__main__":
    main()
