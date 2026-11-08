"""CLI for GPU exact optimal six-pixel pair matching and distance taxonomy."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .pairwise_similarity_v013 import analyze, html_gallery, MODES, DEFAULT_QUANTILES
from .research import DATA, load_history


def configure_parsers(action):
    p=action.add_parser('pairwise-similarity-gpu',help='v0.13 all 7x7 image-pair distances, exact six-point transport and IID controls')
    p.add_argument('--data',type=Path,default=DATA)
    p.add_argument('--device',choices=('auto','cuda','cpu'),default='auto')
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--iid-pairs',type=int,default=30000)
    p.add_argument('--iid-quantiles',type=float,nargs='+',default=list(DEFAULT_QUANTILES))
    p.add_argument('--cluster-mode',choices=MODES,default='shape')
    p.add_argument('--control-draws',type=int,default=384)
    p.add_argument('--control-replicates',type=int,default=9)
    p.add_argument('--top-pairs',type=int,default=25)
    p.add_argument('--gallery-count',type=int,default=8)
    p.add_argument('--seed',type=int,default=649)
    p.add_argument('--out',type=Path,default=Path('results/v013_pairwise_similarity.json'))
    p.add_argument('--csv-out',type=Path,default=None)
    p.add_argument('--html-out',type=Path,default=None)
    p.add_argument('--matrix-out',type=Path,default=None,help='compressed NPZ containing each full NxN distance matrix')


def run(args):
    if args.command!='pairwise-similarity-gpu':raise ValueError('v0.13 CLI expected')
    out=Path(args.out)
    csv_out=Path(args.csv_out) if args.csv_out else out.with_suffix('.csv')
    html_out=Path(args.html_out) if args.html_out else out.with_suffix('.html')
    matrix_out=Path(args.matrix_out) if args.matrix_out else out.with_suffix('.npz')
    paths=(out,csv_out,html_out,matrix_out)
    if len({p.resolve() for p in paths})!=4 or any(p.exists() for p in paths):
        raise FileExistsError('Output JSON, CSV, HTML and NPZ must be new, distinct paths; never overwrite')
    images,_,ids=load_history(args.data)
    print(f'[SOURCE] {len(images)} actual legal 7x7 images; unordered_pairs={len(images)*(len(images)-1)//2}',flush=True)
    report,matrices=analyze(images,ids,device=args.device,batch_size=args.batch_size,
        iid_pairs=args.iid_pairs,iid_quantiles=tuple(args.iid_quantiles),cluster_mode=args.cluster_mode,
        control_draws=args.control_draws,control_replicates=args.control_replicates,
        top_pairs=args.top_pairs,gallery_count=args.gallery_count,seed=args.seed,
        on_progress=lambda msg:print(msg,flush=True))
    for m in MODES:
        stats=report['distance_distributions'][m]
        print(f'[DIST] {m}: history_mean={stats["mean"]:.5f} iid_mean={stats["iid_mean"]:.5f} pairs={stats["pair_count"]}',flush=True)
    for g in report['threshold_classes']:
        print(f'[CLASS] threshold_q={g["target_iid_quantile"]:.4g} d={g["threshold"]:.5f} '
              f'groups={g["multi_member_class_count"]} images={g["multi_member_images"]} '
              f'singletons={g["singleton_images"]}',flush=True)
    for path in paths:path.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:
        json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')
    with csv_out.open('x',encoding='utf-8',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['draw_id','n1','n2','n3','n4','n5','n6']+
            [f'class_q_{q}' for q in sorted(args.iid_quantiles)])
        assigned=report['assignments_at_threshold']
        for i,(image,draw_id) in enumerate(zip(images,ids)):
            nums=(np.flatnonzero(image.reshape(-1))+1).tolist()
            writer.writerow([int(draw_id),*nums]+[assigned[str(float(q))][i] for q in sorted(args.iid_quantiles)])
    with html_out.open('x',encoding='utf-8') as f:f.write(html_gallery(report))
    with matrix_out.open('xb') as f:
        np.savez_compressed(f,draw_ids=ids,**matrices)
    print('[DEVICE] '+report['device'],flush=True)
    for title,path in [('JSON',out),('CSV',csv_out),('HTML',html_out),('NPZ',matrix_out)]:
        print(f'[DONE] {title}: {path.resolve()}',flush=True)
    print('[CAUTION] Similarity taxonomy only; no verified next-draw prediction or selection advantage',flush=True)
