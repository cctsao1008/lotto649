"""CLI for full-history GPU scenario-coverage and fair 6/49 paired replay."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .full_history_v072 import run_full_history, resolve_device
from .research import DATA, load_history


def configure_parsers(action):
    gpu = action.add_parser('full-history-gpu', help='v0.7.2 all eligible draws; CUDA batched fair-scenario optimization')
    gpu.add_argument('--data', type=Path, default=DATA)
    gpu.add_argument('--warmup', type=int, default=200, help='Past-only prefix; evaluation count = source count - warmup')
    gpu.add_argument('--count', type=int, default=5, help='Distinct tickets per portfolio, 1..5')
    gpu.add_argument('--tail-targets', type=int, default=None, help='Optional last K eligible targets for a short smoke run; omit to evaluate all')
    gpu.add_argument('--size', type=int, default=100, help='B1/S1 research candidate pool size')
    gpu.add_argument('--families', type=int, default=20)
    gpu.add_argument('--seed', type=int, default=649)
    gpu.add_argument('--scenarios', type=int, default=8192, help='Independent fair 6/49 scenarios scored on GPU per pool')
    gpu.add_argument('--device', choices=('auto','cuda','cpu'), default='auto')
    gpu.add_argument('--checkpoint', type=Path, default=Path('results/full_history_v072_checkpoint.jsonl'))
    gpu.add_argument('--resume', action='store_true', help='Resume an existing checkpoint with identical data and arguments')
    gpu.add_argument('--out', type=Path, default=Path('results/full_history_v072.json'))
    diag = action.add_parser('gpu-check', help='Show installed PyTorch and CUDA capabilities')


def run(args):
    if args.command == 'gpu-check':
        print(f'[TORCH] version={torch.__version__}')
        print(f'[CUDA] available={torch.cuda.is_available()}')
        if torch.cuda.is_available():
            print(f'[CUDA] device={torch.cuda.get_device_name(0)}')
        else:
            print('[NOTICE] CPU-only PyTorch or no supported CUDA GPU. Follow official PyTorch installation instructions.')
        return
    if args.command != 'full-history-gpu':
        raise ValueError('unsupported full-history command')
    resolve_device(args.device)  # fail early if --device cuda was requested but unavailable
    images, _, ids = load_history(args.data)
    targets = len(images)-args.warmup if args.tail_targets is None else min(args.tail_targets,len(images)-args.warmup)
    print(f'[SOURCE] {len(images)} historical draws; warmup={args.warmup}; expected_targets={targets}',flush=True)
    report = run_full_history(
        images, ids, warmup=args.warmup, count=args.count, size=args.size,
        families=args.families, seed=args.seed, scenarios=args.scenarios,
        device=args.device, checkpoint=args.checkpoint, resume=args.resume,
        tail_targets=args.tail_targets,
        on_progress=lambda line: print(line, flush=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Atomic report output. Checkpoint is append-only, report may replace a summary.
    temporary = args.out.with_name(args.out.name+'.tmp')
    temporary.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    temporary.replace(args.out)
    print(f'[DEVICE] {report["compute"]["device"]} | scenario optimizer={report["compute"]["scenario_draws"]} fair scenarios',flush=True)
    for name, summary in report['summary'].items():
        print(f'[ARM] {name}: observed_ge3={summary["realized_ge3"]:.3%} '
              f'fair_ge3={summary["fair_expected_ge3"]:.3%} '
              f'mean_max_hits={summary["mean_max_hits"]:.4f}',flush=True)
    for name, comparison in report['s1_vs_b1_paired'].items():
        print(f'[PAIRED] {name}: delta={comparison["difference"]:+.3%} '
              f'fair_delta={comparison["fair_difference"]:+.3%} '
              f'p={comparison["exact_mcnemar_p"]:.4g} '
              f'Holm-p={comparison["holm_adjusted_p"]:.4g}',flush=True)
    print(f'[DONE] {args.out.resolve()}',flush=True)
    print('[CAUTION] Retrospective/exploratory; scenario-coverage is NOT original exact-greedy method',flush=True)
