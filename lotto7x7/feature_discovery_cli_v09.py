"""CLI for CUDA-first v0.9 multi-domain feature-periodicity discovery."""
from __future__ import annotations

import json
from pathlib import Path

from .feature_discovery_v09 import discover
from .research import DATA,load_history


def configure_parsers(action):
    p=action.add_parser("feature-discover-gpu",help="v0.9 GPU feature periodicity search with IID/shuffled max-null controls")
    p.add_argument("--data",type=Path,default=DATA)
    p.add_argument("--device",choices=("auto","cuda","cpu"),default="auto")
    p.add_argument("--holdout-draws",type=int,default=500)
    p.add_argument("--surrogates",type=int,default=199)
    p.add_argument("--surrogate-batch",type=int,default=8)
    p.add_argument("--recent-windows",type=int,nargs="+",default=(256,512,1024))
    p.add_argument("--z-half-lives",type=int,nargs="+",default=(0,64,256))
    p.add_argument("--min-period",type=float,default=2.)
    p.add_argument("--max-period",type=float,default=512.)
    p.add_argument("--top",type=int,default=20)
    p.add_argument("--seed",type=int,default=649)
    p.add_argument("--out",type=Path,required=True,help="New JSON file (exclusive creation)")


def run(args):
    if args.command!="feature-discover-gpu":
        raise ValueError("wrong command")
    images,_,ids=load_history(args.data)
    result=discover(images,ids,device=args.device,holdout_draws=args.holdout_draws,
                    surrogates=args.surrogates,surrogate_batch=args.surrogate_batch,
                    recent_windows=tuple(args.recent_windows),half_lives=tuple(args.z_half_lives),
                    min_period=args.min_period,max_period=args.max_period,top_count=args.top,
                    seed=args.seed,on_progress=lambda s:print(s,flush=True))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write("\n")
    print(f'[FEATURES] {result["feature_count"]} across {len(result["feature_families"])} families; '
          f'discovery={result["dataset"]["discovery_prefix_draws"]} '
          f'held_out={result["dataset"]["holdout_suffix_draws"]}',flush=True)
    print(f'[DEVICE] {result["device"]}',flush=True)
    for null,row in result['null_controls'].items():
        print(f'[NULL] {null}: best_global_p={row["global_max_p_for_best_observed"]:.6f}',flush=True)
    for x in result['top_candidates'][:8]:
        print(f'[CANDIDATE] {x["feature"]} period={x["period_draws"]:.3f} '
              f'window={x["window"]} z_half_life={x["z_half_life"]} '
              f'strength={x["strength"]:.6f} '
              f'p_iid={x["global_max_p_iid_legal_6of49"]:.6f} '
              f'p_shuffle={x["global_max_p_shuffled_historical_draws"]:.6f}',flush=True)
    print(f'[DONE] {args.out.resolve()}',flush=True)
    print('[CAUTION] Discovery only: no verified predictive/selection edge, no training or held-out scoring',flush=True)
