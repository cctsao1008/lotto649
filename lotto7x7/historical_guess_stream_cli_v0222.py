"""CLI for v0.22.2 fast, unlimited, Ctrl+C-stoppable historical matching."""
from __future__ import annotations

from pathlib import Path

from .historical_guess_stream_v0222 import run_stream


def configure_parsers(action):
    p = action.add_parser(
        'historical-hit-stream',
        help='Generate IID legal six-number guesses continuously at maximum speed until Ctrl+C; print final hits/guesses',
    )
    p.add_argument('--catalog', type=Path,
                   default=Path('results/v015_v016_p1_shape_catalog.npz'))
    p.add_argument('--universe', type=Path,
                   default=Path('results/v015_v016_p1_shape_catalog_universe.json'))
    p.add_argument('--target', choices=('all', 'holdout'), default='all')
    p.add_argument('--holdout-count', type=int, default=500)
    p.add_argument('--strategy', choices=('shape_score', 'uniform'), default='shape_score')
    p.add_argument('--batch-size', type=int, default=10000,
                   help='Computational chunk size 1..65536; NOT a total guesses limit')
    p.add_argument('--tau', type=float, default=384.0)
    p.add_argument('--seed', type=int, default=None,
                   help='Omit to use OS cryptographic random seed; use explicit integer to reproduce')
    p.add_argument('--checkpoint-seconds', type=float, default=10.0)
    p.add_argument('--show-all-hits', action='store_true',
                   help='Print every NEW distinct historic hit (may slow down terminal); repeats still count silently')
    p.add_argument('--max-batches', type=int, default=0,
                   help='Diagnostic optional stop after N batches; default 0 means unlimited until Ctrl+C')
    p.add_argument('--out', type=Path,
                   default=Path('results/v0222_historical_stream.json'))


def run(args):
    return run_stream(
        catalog=args.catalog, universe=args.universe, out=args.out,
        strategy=args.strategy, target=args.target,
        holdout_count=args.holdout_count, tau=args.tau,
        batch_size=args.batch_size, seed=args.seed,
        checkpoint_seconds=args.checkpoint_seconds,
        show_all_hits=args.show_all_hits, max_batches=args.max_batches,
        emit=lambda message: print(message, flush=True),
    )
