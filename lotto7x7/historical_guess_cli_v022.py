"""v0.22 timed historical exact-six-number retrieval benchmark CLI."""
from __future__ import annotations

from pathlib import Path

from .historical_guess_v022 import run_experiment


def configure_parsers(action):
    p=action.add_parser('historical-hit-benchmark',
        help='Run up to 10,000 distinct fair tickets per trial for a timed historical exact-match test; print hits immediately')
    p.add_argument('--catalog',type=Path,default=Path('results/v015_v016_p1_shape_catalog.npz'))
    p.add_argument('--universe',type=Path,default=Path('results/v015_v016_p1_shape_catalog_universe.json'))
    p.add_argument('--minutes',type=float,default=20.0,help='Time budget in minutes (may finish current trial)')
    p.add_argument('--per-round',type=int,default=10000,help='1 to 10000 unique guesses per independent trial')
    p.add_argument('--target',choices=('all','holdout'),default='all',help='Search all history, or held-out last N draws')
    p.add_argument('--holdout-count',type=int,default=500,help='Latest N historical draws excluded from model fitting for holdout mode')
    p.add_argument('--tau',type=float,default=384.)
    p.add_argument('--seed',type=int,default=649)
    p.add_argument('--progress-every',type=int,default=100)
    p.add_argument('--max-rounds',type=int,default=0,help='Optional trial limit, 0 means stop by clock only')
    p.add_argument('--out',type=Path,default=Path('results/v022_historical_hits.json'))


def run(args):
    return run_experiment(catalog=args.catalog,universe=args.universe,
        out=args.out,minutes=args.minutes,per_round=args.per_round,
        target=args.target,holdout_count=args.holdout_count,
        tau=args.tau,seed=args.seed,max_rounds=args.max_rounds,
        progress_every=args.progress_every,emit=lambda s:print(s,flush=True))
