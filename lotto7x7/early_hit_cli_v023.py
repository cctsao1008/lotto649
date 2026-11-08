"""v0.23 chronological holdout paired early-hit statistical CLI."""
from __future__ import annotations

from pathlib import Path

from .early_hit_v023 import run_backtest, DEFAULT_CUTOFFS


def configure_parsers(action):
    p = action.add_parser('early-hit-backtest',
                          help='Compare early historical matches: identical IID pools, heldout recent draws, train-only shape score')
    p.add_argument('--catalog', type=Path, default=Path('results/v015_v016_p1_shape_catalog.npz'))
    p.add_argument('--universe', type=Path, default=Path('results/v015_v016_p1_shape_catalog_universe.json'))
    p.add_argument('--trials', type=int, default=1000, help='Number of independently seeded candidate pools')
    p.add_argument('--pool-size', type=int, default=10000)
    p.add_argument('--cutoffs', type=int, nargs='+', default=list(DEFAULT_CUTOFFS),
                   help='Increasing early K thresholds; space-delimited, e.g. 100 500 1000 5000')
    p.add_argument('--primary-cutoff', type=int, default=5000)
    p.add_argument('--holdout-count', type=int, default=500)
    p.add_argument('--tau', type=float, default=384.0)
    p.add_argument('--seed', type=int, default=None,
                   help='Reproducible root seed (default: secure OS-generated 128-bit seed)')
    p.add_argument('--bootstrap-reps', type=int, default=2000)
    p.add_argument('--progress-every', type=int, default=0,
                   help='Print periodic trial progress; default 0 is quiet')
    p.add_argument('--out', type=Path, default=Path('results/v023_early_hit_backtest.json'))


def run(args):
    return run_backtest(catalog=args.catalog, universe=args.universe, out=args.out,
                        trials=args.trials, pool_size=args.pool_size,
                        cutoffs=tuple(args.cutoffs), primary_cutoff=args.primary_cutoff,
                        holdout_count=args.holdout_count, tau=args.tau, seed=args.seed,
                        bootstrap_reps=args.bootstrap_reps,
                        progress_every=args.progress_every,
                        emit=lambda s: print(s, flush=True))
