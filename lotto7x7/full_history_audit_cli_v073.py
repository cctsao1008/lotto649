"""v0.7.3 CLI: independent raw-BIN CPU audit and multi-seed draw-level analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .full_history_audit_v073 import (
    audit_report, multiseed_report, read_verified_history,
)
from .research import DATA


def configure_parsers(action):
    a=action.add_parser("full-history-audit", help="Verify every v0.7.2 GPU target on independent CPU and both BIN formats")
    a.add_argument("--report", type=Path, required=True, help="Unmodified v0.7.2 full-history JSON")
    a.add_argument("--data", type=Path, default=DATA, help="Original image BIN")
    a.add_argument("--history-bin", type=Path, default=None, help="Optional history BIN; defaults to sibling of --data")
    a.add_argument("--manifest", type=Path, default=Path("data/bin/lotto649_source_manifest.json"))
    a.add_argument("--audit-seed", type=int, default=649, help="CPU bootstrap seed")
    a.add_argument("--allow-partial", action="store_true", help="Allow a --tail-targets smoke-test report")
    a.add_argument("--out", type=Path, required=True, help="New output file (never overwrite)")

    m=action.add_parser("full-history-multiseed", help="Aggregate pre-audited whole-history runs by draw rather than by seed")
    m.add_argument("--reports", type=Path, nargs="+", required=True,
                   help="Two or more GPU run JSON files, one unique seed per file")
    m.add_argument("--audits", type=Path, nargs="+", required=True,
                   help="Matching v0.7.3 audited JSON files, same order")
    m.add_argument("--seed", type=int, default=649, help="Draw-level bootstrap seed")
    m.add_argument("--out", type=Path, required=True, help="New output file (never overwrite)")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _save_new(path: Path, report: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write("\n")


def run(args: argparse.Namespace):
    if args.command == "full-history-audit":
        hist = read_verified_history(args.data,args.manifest,history_path=args.history_bin)
        report = audit_report(_load(args.report),hist,allow_partial=args.allow_partial,
                              audit_seed=args.audit_seed)
        _save_new(args.out, report)
        print(f'[DATA] draws={report["dataset"]["draws"]} first={report["dataset"]["first_draw_id"]} last={report["dataset"]["last_draw_id"]}')
        print(f'[SOURCE] {report["dataset"]["source_site"]}; source files and manifest SHA256 checked')
        print(f'[CPU PASS] targets={report["target_count"]} ticket sets={report["independent_cpu_scored_ticket_sets"]} (6 strategies per draw)')
        for name, values in report['summary'].items():
            print(f'[ARM] {name}: ge3={values["ge3_count"]}/{report["target_count"]} '
                  f'actual={values["ge3_rate"]:.3%} fair={values["fair_ge3_mean"]:.3%}')
        for method, values in report['s1_vs_b1_paired'].items():
            print(f'[PAIRED] {method}: S1-only={values["s1_only"]} B1-only={values["b1_only"]} '
                  f'delta={values["difference"]:+.3%} Holm-p={values["holm_adjusted_p"]:.6f}')
        print(f'[DONE] {args.out.resolve()}')
        print('[CAUTION] Historical exploratory audit; two BIN files share same third-party source')
        return
    if args.command == "full-history-multiseed":
        if len(args.reports) < 2 or len(args.reports) != len(args.audits):
            raise ValueError("provide >=2 corresponding report/audit files")
        reports = [_load(p) for p in args.reports]
        audits = [_load(p) for p in args.audits]
        if any(not audit.get('validated_full_history') for audit in audits):
            raise ValueError("all inputs must be fully audited, not partial smoke targets")
        result = multiseed_report(reports,audits,seed=args.seed)
        _save_new(args.out, result)
        print(f'[MATCHED] draws={result["draw_targets"]} seeds={result["seeds"]}')
        for method, row in result['draw_clustered_s1_minus_b1'].items():
            print(f'[S1-B1] {method}: observed={row["mean_observed_difference"]:+.3%} '
                  f'fair_delta={row["mean_fair_coverage_difference"]:+.3%} '
                  f'excess={row["mean_excess_over_fair"]:+.3%} '
                  f'draw-95CI={row["draw_level_excess_ci95"]}')
        print(f'[DONE] {args.out.resolve()}')
        print('[CAUTION] All seed replications use the SAME draw targets; no pseudo-replication')
        return
    raise ValueError("unknown v0.7.3 command")
