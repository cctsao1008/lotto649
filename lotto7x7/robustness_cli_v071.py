"""Additive CLI for v0.7.1 statistical robustness, without changing v0.7."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .research import DATA, load_history
from .robustness_v071 import audit_rolling, sweep_rolling


def configure_parsers(action):
    a = action.add_parser("robustness-audit", help="v0.7.1 paired S1 vs B1 stats from existing v0.7 JSON")
    a.add_argument("--input", type=Path, required=True,
                   help="Existing baseline-study report; no model recomputation")
    a.add_argument("--seed", type=int, default=649,
                   help="Deterministic bootstrap resampling seed")
    a.add_argument("--out", type=Path, default=Path("results/robustness_audit_v071.json"))

    s = action.add_parser("robustness-sweep", help="v0.7.1 matched seeds and N=1..5 sensitivity")
    s.add_argument("--data", type=Path, default=DATA)
    s.add_argument("--seeds", type=str, default="649,650,651")
    s.add_argument("--counts", type=str, default="1,3,5",
                   help="Comma separated ticket counts, each in 1..5")
    s.add_argument("--warmup", type=int, default=200)
    s.add_argument("--eval-draws", type=int, default=12)
    s.add_argument("--stride", type=int, default=5)
    s.add_argument("--size", type=int, default=100)
    s.add_argument("--families", type=int, default=20)
    s.add_argument("--out", type=Path, default=Path("results/robustness_sweep_v071.json"))


def run(args: argparse.Namespace):
    if args.command == "robustness-audit":
        source = json.loads(args.input.read_text(encoding="utf-8"))
        report = audit_rolling(source, seed=args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[AUDIT] {report['target_count']} matched draws; N={report['ticket_count_per_portfolio']}")
        for method, result in report["comparisons"].items():
            print(f"[PAIRED] {method:21s} S1-only={result['s1_only_ge3']:3d} "
                  f"B1-only={result['b1_only_ge3']:3d} "
                  f"observed={result['observed_difference']:+.2%} "
                  f"fair-delta={result['fair_coverage_difference']:+.3%} "
                  f"McNemar p={result['mcnemar_two_sided_exact_p']:.6f} "
                  f"Holm p={result['holm_familywise_adjusted_p']:.6f}")
        print(f"[DONE] {args.out.resolve()}")
        print("[CAUTION] Retrospective, exploratory paired statistics")
        return
    if args.command == "robustness-sweep":
        images, _, ids = load_history(args.data)
        report = sweep_rolling(
            images, ids, seeds=args.seeds, counts=args.counts,
            warmup=args.warmup, eval_draws=args.eval_draws,
            stride=args.stride, pool_size=args.size, family_count=args.families,
            on_progress=lambda message: print(message, flush=True))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[EVALUATED] {report['independent_target_draws']} distinct draw targets "
              f"over {len(report['seeds'])} seeds and {len(report['counts'])} N settings")
        for n, contrasts in report["draw_clustered_aggregate"].items():
            for method, result in contrasts.items():
                print(f"[S1-B1] {n} {method}: observed={result['mean_observed_difference']:+.2%} "
                      f"fair-delta={result['mean_fair_coverage_difference']:+.3%} "
                      f"draw-CI={result['draw_level_paired_bootstrap_95']}")
        print(f"[DONE] {args.out.resolve()}")
        print("[CAUTION] Multiple seeds reuse the same draws, not independent evidence")
        return
    raise ValueError("unsupported v0.7.1 CLI command")
