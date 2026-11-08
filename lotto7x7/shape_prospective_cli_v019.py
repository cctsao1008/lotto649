"""v0.19 CLI: immutable preregistration and future-only prospective monitoring."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .shape_prospective_v019 import (make_freeze, check_freeze, read_updates,
  prospective_evaluate, file_sha256, report_html, _expand_csv)


def configure_parsers(action):
    p=action.add_parser('shape-prospective-freeze',help='Freeze the P2/P1 baseline before seeing any new lottery draw')
    p.add_argument('--catalog',type=Path,default=Path('results/v015_v016_p1_shape_catalog.npz'))
    p.add_argument('--universe',type=Path,default=Path('results/v015_v016_p1_shape_catalog_universe.json'))
    p.add_argument('--watchlist',type=Path,default=Path('results/v017_v018_p2_batch_future_watchlist.json'))
    p.add_argument('--tau',type=float,default=384.)
    p.add_argument('--out',type=Path,default=Path('results/v019_prospective_freeze.json'))

    p=action.add_parser('shape-prospective-update',help='Evaluate all *new* draws since a sealed preregistered baseline')
    p.add_argument('--freeze',type=Path,default=Path('docs/prospective_freeze_20261108.json'))
    p.add_argument('--catalog',type=Path,default=Path('results/v015_v016_p1_shape_catalog.npz'))
    p.add_argument('--universe',type=Path,default=Path('results/v015_v016_p1_shape_catalog_universe.json'))
    p.add_argument('--csv',nargs='*',default=[],help='One or more year CSV paths, directories, or quoted wildcard patterns; must have draw_id,n1..n6, and date for future draws')
    p.add_argument('--previous',type=Path,default=None,help='Optional earlier v0.19 report for append-only cross-check')
    p.add_argument('--out',type=Path,default=Path('results/v019_prospective_check.json'))


def _paths(json_path:Path):
    p=Path(json_path)
    if p.suffix.lower()!='.json':raise ValueError('output must end with .json')
    stem=p.with_suffix('')
    return {'JSON':p,'CSV':Path(str(stem)+'_events.csv'), 'HTML':Path(str(stem)+'.html')}


def _reserve(paths,inputs):
    targets=list(paths.values())
    if len({p.resolve() for p in targets})!=len(targets):raise ValueError('duplicate output paths')
    if any(p.exists() or p.resolve() in {Path(i).resolve() for i in inputs if i is not None} for p in targets):
        raise FileExistsError('outputs must not overwrite previous outputs or any input')
    for p in targets:p.parent.mkdir(parents=True,exist_ok=True)


def _write_json_new(path,data):
    with path.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def run(args):
    if args.command=='shape-prospective-freeze':
        _reserve({'JSON':args.out},[args.catalog,args.universe,args.watchlist])
        frozen=make_freeze(args.catalog,args.universe,args.watchlist,tau=args.tau)
        _write_json_new(args.out,frozen)
        print(f'[FREEZE] historical={frozen["source"]["historical_draw_count"]} cutoff={frozen["source"]["last_draw_id"]}',flush=True)
        print(f'[FREEZE] created_at={frozen["frozen_at_utc"]} sha256={frozen["self_checksum_sha256"]}',flush=True)
        print(f'[DONE] JSON: {args.out.resolve()}',flush=True)
        return
    if args.command!='shape-prospective-update':raise ValueError('wrong v0.19 subcommand')
    with args.freeze.open(encoding='utf-8') as f:frozen=json.load(f)
    ids,nums,tables,_=check_freeze(frozen,args.catalog,args.universe)
    source_files=_expand_csv(args.csv)
    paths=_paths(args.out)
    _reserve(paths,[args.freeze,args.catalog,args.universe,args.previous,*source_files])
    future,sources=read_updates(args.csv,freeze=frozen,reference_ids=ids,reference_numbers=nums)
    prior=None
    if args.previous is not None:
        prior=json.loads(args.previous.read_text(encoding='utf-8'))
        prior['_file_sha256']=file_sha256(args.previous)
    report,rows=prospective_evaluate(frozen,tables,future,sources,previous=prior)
    # Report content is still verified on every run against sealed freeze and source checksums.
    report['freeze_file_sha256']=file_sha256(args.freeze)
    report['source_integrity_checked']=True
    _write_json_new(paths['JSON'],report)
    fieldnames=['draw_id','date',*[f'n{i}' for i in range(1,7)]]
    for lev in ('component','partition','edge'):
        fieldnames += [f'{lev}_class',f'{lev}_q_before_draw',f'{lev}_fair_p',
                      f'{lev}_log_gain',f'{lev}_cumulative_log_e']
    fieldnames += ['rare_class_hit','rare_cumulative_hits','rare_log_gain',
                   'rare_cumulative_log_e','mixture_log_e','mixture_e','anytime_p_upper_bound']
    with paths['CSV'].open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames,extrasaction='raise');w.writeheader();w.writerows(rows)
    with paths['HTML'].open('x',encoding='utf-8') as f:f.write(report_html(report))
    print(f'[PROSPECTIVE] new_draws={len(rows)} status={report["status"]}',flush=True)
    print(f'[EVIDENCE] mixture_e={report["mixture_e"]:.6g} anytime_upper_p={report["anytime_p_upper_bound"]:.6g}',flush=True)
    for name,p in paths.items():print(f'[DONE] {name}: {p.resolve()}',flush=True)
    if not rows:print('[CAUTION] No future draws: ZERO prospective validation evidence',flush=True)
    else:print('[CAUTION] Verify draw records against primary lottery source; classification is not a betting advantage',flush=True)
