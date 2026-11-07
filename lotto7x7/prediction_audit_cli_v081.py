"""Register a v0.8.1 score-only audit: no GPU training or old unit suites."""
from __future__ import annotations

import json
from pathlib import Path

from .prediction_audit_v081 import audit


def configure_parsers(action):
    p=action.add_parser('predict-audit',help='v0.8.1 statistics from saved v0.8 prediction JSON; zero retraining')
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--seed',type=int,default=649)
    p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--out',type=Path,default=Path('results/probability_v081_audit.json'))


def run(args):
    if args.command!='predict-audit':
        raise ValueError('unsupported audit command')
    source=json.loads(args.report.read_text(encoding='utf-8'))
    result=audit(source,seed=args.seed,allow_partial=args.allow_partial)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as fh:
        json.dump(result,fh,indent=2,ensure_ascii=False,allow_nan=False)
        fh.write('\n')
    print(f'[AUDIT PASS] targets={result["target_count"]}; saved scores recomputed',flush=True)
    for name,row in result['comparisons'].items():
        print(f'[PAIRED] {name:20s} ge3 delta={row["mean_ge3_difference"]:+.3%} '
              f'McNemar p={row["mcnemar_two_sided_exact_p"]:.6f} '
              f'Holm p={row["holm_three_model_adjusted_p"]:.6f} '
              f'Brier skill={row["mean_brier_skill_vs_iid"]:+.8f}',flush=True)
    print(f'[DONE] {args.out.resolve()}',flush=True)
    print('[CAUTION] Retrospective. No retraining, no official-source authentication.',flush=True)
