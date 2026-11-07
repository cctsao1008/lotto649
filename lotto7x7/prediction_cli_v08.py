"""CLI integration for GPU v0.8 individual-number probability research."""
from __future__ import annotations

import json
from pathlib import Path

from .prediction_v08 import replay, METHODS
from .research import DATA, load_history


def configure_parsers(action):
    p = action.add_parser('predict-gpu', help='v0.8 GPU CNN-GRU 49-number probability walk-forward vs causal statistical baselines')
    p.add_argument('--data', type=Path, default=DATA)
    p.add_argument('--device', choices=('auto','cuda','cpu'), default='auto')
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--block-size', type=int, default=100)
    p.add_argument('--window', type=int, default=5)
    p.add_argument('--validation', type=int, default=32)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--learning-rate', type=float, default=.002)
    p.add_argument('--embedding', type=int, default=24)
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--seed', type=int, default=649)
    p.add_argument('--tail-targets', type=int, default=None, help='Smoke run last K targets; omit for all eligible targets')
    p.add_argument('--out', type=Path, default=Path('results/probability_gpu_v08.json'))


def run(args):
    if args.command != 'predict-gpu':
        raise ValueError('unknown v0.8 command')
    images, _, ids = load_history(args.data)
    print(f'[SOURCE] {len(images)} historical draws; targets={len(images)-args.warmup if args.tail_targets is None else min(len(images)-args.warmup,args.tail_targets)}',flush=True)
    report = replay(images, ids, warmup=args.warmup, block_size=args.block_size,
                    window=args.window, validation=args.validation,
                    epochs=args.epochs, patience=args.patience,
                    batch_size=args.batch_size, learning_rate=args.learning_rate,
                    embedding=args.embedding, hidden=args.hidden,
                    seed=args.seed, device=args.device, tail_targets=args.tail_targets,
                    on_progress=lambda line:print(line,flush=True))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    temp = args.out.with_name(args.out.name+'.tmp')
    temp.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(args.out)
    print(f'[DEVICE] {report["compute"]["device"]}',flush=True)
    for method in METHODS:
        row=report['summary'][method]
        print(f'[MODEL] {method:20s} Brier={row["mean_brier"]:.7f} '
              f'skill_vs_uniform={row["brier_skill_vs_iid"]:+.7f} '
              f'avg_top6_hits={row["mean_top6_hits"]:.4f} '
              f'ge3={row["top6_ge3_rate"]:.3%}',flush=True)
    print(f'[DONE] {args.out.resolve()}',flush=True)
    print('[CAUTION] Retrospective probability research, not a proven forecasting edge',flush=True)
