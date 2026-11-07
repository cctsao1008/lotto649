"""Argparse integration for the v0.7 comparison and prospective shadow ledger."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from .baseline_v07 import rolling_origin, shadow_snapshot, score_shadow
from .research import DATA, load_history


def configure_parsers(action):
    a = action.add_parser("baseline-study", help="v0.7 B0/B1/S1/B2 rolling-origin historical baselines")
    a.add_argument("--data", type=Path, default=DATA)
    a.add_argument("--count", type=int, default=5, help="Actual tickets per portfolio, 1..5")
    a.add_argument("--size", type=int, default=100, help="B1/S1 research pool size")
    a.add_argument("--families", type=int, default=20, help="S1 sampled historical families")
    a.add_argument("--seed", type=int, default=649)
    a.add_argument("--warmup", type=int, default=200)
    a.add_argument("--eval-draws", type=int, default=12)
    a.add_argument("--stride", type=int, default=1)
    a.add_argument("--out", type=Path, default=Path("results/baseline_study_v07.json"))

    p = action.add_parser("shadow-snapshot", help="v0.7 pre-draw SHA256-sealed selections; no target read")
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--cutoff-draw-id", type=int, default=None,
                   help="Explicit last known draw (must exist in dataset)")
    p.add_argument("--count", type=int, default=5)
    p.add_argument("--size", type=int, default=100)
    p.add_argument("--families", type=int, default=20)
    p.add_argument("--seed", type=int, default=649)
    p.add_argument("--commit", type=str, default=None,
                   help="Code commit; otherwise resolve local git HEAD")
    p.add_argument("--out", type=Path, required=True,
                   help="New file: existing path is never overwritten")

    s = action.add_parser("shadow-score", help="v0.7 post-draw scoring of a sealed snapshot")
    s.add_argument("--snapshot", type=Path, required=True)
    s.add_argument("--draw-id", type=int, required=True)
    s.add_argument("--numbers", type=str, required=True,
                   help="Six comma-separated main numbers; bonus excluded")
    s.add_argument("--out", type=Path, required=True,
                   help="New file: existing path is never overwritten")


def _write_new(path: Path, report: dict):
    """Exclusive creation protects previous pre-draw research commitments."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")


def run(args: argparse.Namespace) -> None:
    if args.command == "baseline-study":
        images, _, ids = load_history(args.data)
        report = rolling_origin(
            images, ids, warmup=args.warmup,
            eval_draws=args.eval_draws, stride=args.stride,
            count=args.count, pool_size=args.size,
            family_count=args.families, seed=args.seed,
            on_progress=lambda msg: print(msg, flush=True))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[EVALUATED] {report['target_count']} chronological draws; N={args.count}; M={args.size}")
        for name, value in report["summary"].items():
            print(f"[ARM] {name}: mean_hits={value['mean_max_hits']:.4f} "
                  f"observed_ge3={value['realized_any_ge3_rate']:.4%} "
                  f"fair_ge3={value['mean_theoretical_ge3']:.4%}")
        print(f"[DONE] {args.out.resolve()}")
        print("[CAUTION] Retrospective exploratory research, NOT a blind forecasting result")
    elif args.command == "shadow-snapshot":
        images, _, ids = load_history(args.data)
        if args.commit is None:
            git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, check=False)
            commit = git.stdout.strip() if git.returncode == 0 else "unknown"
        else:
            commit = args.commit
        report = shadow_snapshot(
            images, ids, cutoff_draw_id=args.cutoff_draw_id,
            count=args.count, pool_size=args.size,
            family_count=args.families, seed=args.seed, code_commit=commit)
        _write_new(args.out, report)
        print(f"[CUTOFF] {report['pre_draw']['history_cutoff_draw_id']}")
        print(f"[SNAPSHOT SHA256] {report['snapshot_sha256']}")
        print(f"[DONE] {args.out.resolve()}")
        print("[NOTICE] Publish timestamp/hash externally before the next draw to substantiate pre-draw commitment")
    elif args.command == "shadow-score":
        source = json.loads(args.snapshot.read_text(encoding="utf-8"))
        numbers = [int(n.strip()) for n in args.numbers.split(",")]
        report = score_shadow(source, actual_draw_id=args.draw_id,
                              actual_numbers=numbers)
        if args.out.resolve() == args.snapshot.resolve():
            raise ValueError("cannot overwrite the sealed snapshot")
        _write_new(args.out, report)
        print(f"[DRAW] {report['scored_draw_id']}; committed cutoff={report['cutoff_draw_id']}")
        for arm, methods in report["score"].items():
            for method, item in methods.items():
                print(f"[SCORE] {arm}/{method}: max_hits={item['max_hits']}")
        print(f"[DONE] {args.out.resolve()}")
    else:
        raise ValueError("unknown v0.7 action")