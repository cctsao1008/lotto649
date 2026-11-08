"""CLI to analyze cached v0.13 distances, without re-running historical matching."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np

from .gaussian_shape_v014 import (analyze, html_gallery, load_existing,
                                 DEFAULT_BANDWIDTHS, DEFAULT_COMPONENTS)


def configure_parsers(action):
    p=action.add_parser('shape-gaussian-gpu',help='v0.14 cached 7x7 shape KDE, density modes, MDS+GMM and legal IID controls')
    p.add_argument('--matrix',type=Path,default=Path('results/v013_pairwise_similarity.npz'))
    p.add_argument('--assignments',type=Path,default=Path('results/v013_pairwise_similarity.csv'))
    p.add_argument('--device',choices=('auto','cuda','cpu'),default='auto')
    p.add_argument('--bandwidths',type=float,nargs='+',default=list(DEFAULT_BANDWIDTHS))
    p.add_argument('--mode-bandwidth',type=float,default=0.55)
    p.add_argument('--peak-radius',type=float,default=0.6)
    p.add_argument('--assignment-radius',type=float,default=0.85)
    p.add_argument('--max-peaks',type=int,default=24)
    p.add_argument('--mds-dimensions',type=int,default=8)
    p.add_argument('--gmm-components',type=int,nargs='+',default=list(DEFAULT_COMPONENTS))
    p.add_argument('--control-draws',type=int,default=128)
    p.add_argument('--control-replicates',type=int,default=9)
    p.add_argument('--skip-control-gmm',action='store_true',help='skip Gaussian-mixture IID refits; KDE control remains')
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--gallery-count',type=int,default=8)
    p.add_argument('--seed',type=int,default=649)
    p.add_argument('--out',type=Path,default=Path('results/v014_shape_gaussian.json'))
    p.add_argument('--csv-out',type=Path,default=None)
    p.add_argument('--html-out',type=Path,default=None)
    p.add_argument('--embedding-out',type=Path,default=None)


def run(args):
    if args.command!='shape-gaussian-gpu':raise ValueError('wrong command')
    out=Path(args.out)
    csv_out=Path(args.csv_out) if args.csv_out else out.with_suffix('.csv')
    html_out=Path(args.html_out) if args.html_out else out.with_suffix('.html')
    embedding_out=Path(args.embedding_out) if args.embedding_out else out.with_suffix('.npz')
    paths=[out,csv_out,html_out,embedding_out]
    if len({p.resolve() for p in paths})!=4 or any(p.exists() for p in paths):
        raise FileExistsError('all four output paths must be new and distinct; no overwrite')
    matrix,ids,ns=load_existing(args.matrix,args.assignments)
    print(f'[SOURCE] cached v0.13 shape distances: draws={len(ids)} pairs={len(ids)*(len(ids)-1)//2}',flush=True)
    report,arrays=analyze(matrix,ids,ns,device=args.device,bandwidths=args.bandwidths,
         mode_bandwidth=args.mode_bandwidth,peak_radius=args.peak_radius,
         assignment_radius=args.assignment_radius,max_peaks=args.max_peaks,
         mds_dimensions=args.mds_dimensions,components=args.gmm_components,
         control_draws=args.control_draws,control_replicates=args.control_replicates,
         batch_size=args.batch_size,seed=args.seed,compare_gmm_controls=not args.skip_control_gmm,
         on_progress=lambda msg:print(msg,flush=True))
    print(f'[DEVICE] {report["device"]}',flush=True)
    for name,path in [('JSON',out),('CSV',csv_out),('HTML',html_out),('NPZ',embedding_out)]:
        path.parent.mkdir(parents=True,exist_ok=True)
        if name=='JSON':
            with path.open('x',encoding='utf-8') as f:
                json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
        elif name=='CSV':
            with path.open('x',encoding='utf-8',newline='') as f:
                writer=csv.writer(f)
                writer.writerow(['draw_id','n1','n2','n3','n4','n5','n6','kde_density',
                                 'shape_mode_class','shape_mode_nearest_distance','gmm_class','gmm_confidence'])
                bw_idx=int(np.argmin(abs(arrays['bandwidths']-args.mode_bandwidth)))
                for i,draw_id in enumerate(ids):
                    writer.writerow([int(draw_id),*map(int,ns[i]),float(arrays['density'][i,bw_idx]),
                         'NOISE' if int(arrays['mode_labels'][i])<0 else f"M{int(arrays['mode_labels'][i])+1:03d}",
                         float(arrays['mode_nearest_distances'][i]),
                         f"G{int(arrays['gmm_labels'][i])+1:03d}",float(arrays['gmm_confidence'][i])])
        elif name=='HTML':
            with path.open('x',encoding='utf-8') as f:f.write(html_gallery(report,arrays,gallery_count=args.gallery_count))
        else:
            with path.open('xb') as f:np.savez_compressed(f,**arrays)
        print(f'[DONE] {name}: {path.resolve()}',flush=True)
    print('[CAUTION] Gaussian shape density is descriptive, not evidence of improved draw odds',flush=True)
