"""Single P1 Batch 2 entrypoint: v0.15 multiscale modes + v0.16 exhaustive theory."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .gaussian_shape_v014 import load_existing
from .shape_catalog_batch_p1 import analyze,html_report,rows_draws,rows_theory,rows_exact_families,_filesha
from .shape_theory_v016 import RULE_VERSION,UNIVERSE,exhaustive_class_counts


def configure_parsers(action):
    p=action.add_parser('shape-catalog-exact',
         help='P1 Batch 2: complete 7x7 shape catalog, empirical KDE modes, exhaustive 13,983,816 fair 6/49 taxonomy')
    p.add_argument('--matrix',type=Path,default=Path('results/v013_pairwise_similarity.npz'))
    p.add_argument('--assignments',type=Path,default=Path('results/v013_pairwise_similarity.csv'))
    p.add_argument('--device',choices=('auto','cuda','cpu'),default='auto')
    p.add_argument('--bandwidths',type=float,nargs='+',default=[0.35,0.55,0.8,1.1])
    p.add_argument('--radii',type=float,nargs='+',default=[0.70,0.75,0.80],
                   help='empirical observed-sample KDE peak neighborhood radii')
    p.add_argument('--assignment-radius',type=float,default=0.85)
    p.add_argument('--universe-batch-size',type=int,default=120000,
                   help='CPU vectorized bounded-memory enumeration batch')
    p.add_argument('--mc-replicates',type=int,default=5000)
    p.add_argument('--seed',type=int,default=649)
    p.add_argument('--baseline-cache',type=Path,default=None,
                   help='optional previously verified complete P1 universe JSON; avoids recounting 13.98M')
    p.add_argument('--gallery-modes',type=int,default=24)
    p.add_argument('--gallery-classes',type=int,default=20)
    p.add_argument('--gallery-shapes',type=int,default=16)
    p.add_argument('--out',type=Path,default=Path('results/v015_v016_p1_shape_catalog.json'))


def output_paths(out:Path)->dict[str,Path]:
    path=Path(out)
    if path.suffix.lower()!='.json':raise ValueError('--out must have .json suffix')
    base=path.with_suffix('')
    return {'JSON':path,'HTML':Path(str(base)+'.html'),'NPZ':Path(str(base)+'.npz'),
       'DRAWS_CSV':Path(str(base)+'_draws.csv'),
       'CATALOG_CSV':Path(str(base)+'_catalog.csv'),
       'THEORY_CSV':Path(str(base)+'_theory.csv'),
       'MODES_CSV':Path(str(base)+'_modes.csv'),
       'UNIVERSE_JSON':Path(str(base)+'_universe.json')}


def _write_csv(path:Path,rows:list[dict],headings=None):
    if not rows:raise ValueError('CSV requires nonempty records')
    fields=list(headings if headings is not None else rows[0].keys())
    with path.open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        for row in rows:writer.writerow(row)


def _verify_baseline(baseline:dict):
    if (baseline.get('schema')!=RULE_VERSION or not baseline.get('complete')
       or baseline.get('total')!=UNIVERSE or any(sum(int(v) for v in baseline['counts'][k].values())!=UNIVERSE for k in ('component','partition','edge'))):
        raise ValueError('baseline cache incomplete or invalid; exact full universe is required')


def run(args):
    if args.command!='shape-catalog-exact':raise ValueError('wrong subcommand')
    outputs=output_paths(args.out)
    paths=list(outputs.values())
    source=[args.matrix,args.assignments]+([args.baseline_cache] if args.baseline_cache else [])
    if (len({x.resolve() for x in paths})!=len(paths) or
         any(x.exists() for x in paths) or
         any(x.resolve() in {p.resolve() for p in source} for x in paths)):
        raise FileExistsError('P1 protects existing outputs and inputs; use a new --out path')
    if args.baseline_cache:
        baseline=json.loads(args.baseline_cache.read_text(encoding='utf-8'))
        _verify_baseline(baseline)
        print(f'[UNIVERSE] verified cached exhaustive graph classes={baseline["total"]}',flush=True)
    else:
        print(f'[UNIVERSE] exact enumeration C(49,6)={UNIVERSE}; no IID estimation',flush=True)
        baseline=exhaustive_class_counts(batch_size=args.universe_batch_size,
              progress=lambda x:print(x,flush=True))
        _verify_baseline(baseline)
    d,ids,nums=load_existing(args.matrix,args.assignments)
    report,arrays=analyze(d,ids,nums,baseline=baseline,
        device=args.device,bandwidths=args.bandwidths,radii=args.radii,
        assignment_radius=args.assignment_radius,
        mc_replicates=args.mc_replicates,seed=args.seed,
        on_progress=lambda x:print(x,flush=True))
    report['data']['shape_npz_file_sha256']=_filesha(args.matrix)
    report['data']['assignments_csv_file_sha256']=_filesha(args.assignments)
    report['data']['universe_rule_checksum']=baseline['rule_hash_sha256']
    for key,path in outputs.items():
        path.parent.mkdir(parents=True,exist_ok=True)
        if key=='JSON':
            with path.open('x',encoding='utf-8') as f:
                json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
        elif key=='HTML':
            with path.open('x',encoding='utf-8') as f:
                f.write(html_report(report,shown_modes=args.gallery_modes,
                     shown_classes=args.gallery_classes,shown_shapes=args.gallery_shapes))
        elif key=='NPZ':
            with path.open('xb') as f:np.savez_compressed(f,**arrays)
        elif key=='DRAWS_CSV':
            _write_csv(path,list(rows_draws(arrays)))
        elif key=='CATALOG_CSV':
            _write_csv(path,list(rows_exact_families(report)))
        elif key=='THEORY_CSV':
            _write_csv(path,list(rows_theory(report)))
        elif key=='MODES_CSV':
            _write_csv(path,report['kde_multiscale']['records'])
        else:
            with path.open('x',encoding='utf-8') as f:
                json.dump(baseline,f,indent=2,allow_nan=False);f.write('\n')
        print(f'[DONE] {key}: {path.resolve()}',flush=True)
    print('[CAUTION] Fixed geometry categories have exact fair combinatorial probabilities; none implies higher lottery odds',flush=True)
