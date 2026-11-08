"""v0.21 CLI: complete one-command selection research workflow."""
from __future__ import annotations

import json
from pathlib import Path

from .unified_selection_v021 import build_selection, selection_html, write_csv, OBJECTIVES


def configure_parsers(action):
    p=action.add_parser('shape-select-unified',help='v0.21 uniform candidate pool, exact portfolio coverage and frozen exploratory shape comparison')
    p.add_argument('--catalog',type=Path,default=Path('results/v015_v016_p1_shape_catalog.npz'))
    p.add_argument('--universe',type=Path,default=Path('results/v015_v016_p1_shape_catalog_universe.json'))
    p.add_argument('--freeze',type=Path,default=Path('docs/prospective_freeze_20261108.json'))
    p.add_argument('--pool-size',type=int,default=100)
    p.add_argument('--tickets',type=int,default=5)
    p.add_argument('--objective',choices=OBJECTIVES,default='3plus')
    p.add_argument('--seed',type=int,default=649)
    p.add_argument('--max-evaluations',type=int,default=10000)
    p.add_argument('--out',type=Path,default=Path('results/v021_unified_selection.json'))


def output_paths(json_path:Path)->dict[str,Path]:
    p=Path(json_path)
    if p.suffix.lower()!='.json':raise ValueError('--out must end with .json')
    root=p.with_suffix('')
    return {'JSON':p,'POOL_CSV':Path(str(root)+'_pool.csv'),
            'BETS_CSV':Path(str(root)+'_bets.csv'),
            'HTML':Path(str(root)+'.html')}


def _reserve(paths, inputs):
    targets=list(paths.values())
    sources={Path(x).resolve() for x in inputs}
    if len({p.resolve() for p in targets})!=len(targets):raise ValueError('duplicate output paths')
    if any(p.exists() or p.resolve() in sources for p in targets):
        raise FileExistsError('output would overwrite existing artifact or input; use a new --out filename')
    for p in targets:p.parent.mkdir(parents=True,exist_ok=True)


def run(args):
    if args.command!='shape-select-unified':raise ValueError('unsupported v0.21 command')
    paths=output_paths(args.out)
    _reserve(paths,[args.catalog,args.universe,args.freeze])
    report,pool,bets=build_selection(args.catalog,args.universe,args.freeze,
          pool_size=args.pool_size,tickets=args.tickets,
          objective=args.objective,seed=args.seed,max_evaluations=args.max_evaluations,
          on_progress=lambda m:print(m,flush=True))
    with paths['JSON'].open('x',encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    write_csv(paths['POOL_CSV'],pool)
    write_csv(paths['BETS_CSV'],bets)
    with paths['HTML'].open('x',encoding='utf-8') as f:f.write(selection_html(report))
    for method,data in report['strategies'].items():
        p=data['exact_fair_portfolio_coverage']['at_least']
        print(f'[STRATEGY] {method}: >=3 {p["3"]:.8f} >=4 {p["4"]:.8f} six {p["6"]:.10f}',flush=True)
    for name,p in paths.items():print(f'[DONE] {name}: {p.resolve()}',flush=True)
    print('[CAUTION] Shape frequency has no validated predictive edge. Every distinct N-ticket portfolio has equal fair 6/6 probability.',flush=True)
