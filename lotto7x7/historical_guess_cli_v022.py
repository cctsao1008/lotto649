"""v0.22.1 patched CLI: quiet, paced, 10,000 total historical guesses."""
from __future__ import annotations

from pathlib import Path

from .historical_guess_watch_v0221 import run_paced_experiment


def configure_parsers(action):
    p = action.add_parser('historical-hit-benchmark',
        help='20-minute paced historical 6/6 match check; 10,000 TOTAL guesses; print only hits')
    p.add_argument('--catalog', type=Path,
                   default=Path('results/v015_v016_p1_shape_catalog.npz'))
    p.add_argument('--universe', type=Path,
                   default=Path('results/v015_v016_p1_shape_catalog_universe.json'))
    p.add_argument('--minutes', type=float, default=20.0,
                   help='Pace entire search across this many minutes')
    p.add_argument('--max-guesses', '--per-round', dest='max_guesses',
                   type=int, default=10000,
                   help='Total unique guesses across the whole run (legacy --per-round accepted)')
    p.add_argument('--target', choices=('all', 'holdout'), default='all')
    p.add_argument('--holdout-count', type=int, default=500)
    p.add_argument('--tau', type=float, default=384.0)
    p.add_argument('--seed', type=int, default=649)
    p.add_argument('--strategy', choices=('shape_score', 'uniform'),
                   default='shape_score', help='Actual guess order; print uniform/shape counterfactual ranks')
    # Kept for compatibility with the existing user command, intentionally
    # ignored: console output is HIT-only regardless of this flag.
    p.add_argument('--progress-every', type=int, default=None,
                   help='Deprecated and ignored: no periodic progress printing')
    p.add_argument('--out', type=Path,
                   default=Path('results/v0221_historical_hits.json'))


def run(args):
    return run_paced_experiment(
        catalog=args.catalog, universe=args.universe, out=args.out,
        minutes=args.minutes, max_guesses=args.max_guesses,
        target=args.target, holdout_count=args.holdout_count,
        tau=args.tau, seed=args.seed, strategy=args.strategy,
        emit=lambda s: print(s, flush=True),
    )
