"""P0 Batch 1 entrypoint: reusable v0.13 cache, fair-IID refit and sensitivity."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import numpy as np

from .gaussian_shape_v014 import load_existing
from .gaussian_validation_v0141 import (DEFAULT_DIMS,DEFAULT_K,DEFAULT_BW,
                  analyze,csv_records,html_report)


def configure_parsers(action):
    p=action.add_parser('shape-gaussian-validate',
           help='v0.14.1 P0 batch: matched fair-IID controls, GMM ARI/NMI, MDS dimensions, KDE bandwidth')
    p.add_argument('--matrix',type=Path,default=Path('results/v013_pairwise_similarity.npz'))
    p.add_argument('--assignments',type=Path,default=Path('results/v013_pairwise_similarity.csv'))
    p.add_argument('--device',choices=('cpu','cuda','auto'),default='auto')
    p.add_argument('--dimensions',type=int,nargs='+',default=list(DEFAULT_DIMS))
    p.add_argument('--comparison-dimension',type=int,default=8)
    p.add_argument('--gmm-components',type=int,nargs='+',default=list(DEFAULT_K))
    p.add_argument('--gmm-n-init',type=int,default=2)
    p.add_argument('--bandwidths',type=float,nargs='+',default=list(DEFAULT_BW))
    p.add_argument('--matched-size',type=int,default=384)
    p.add_argument('--matched-replicates',type=int,default=9)
    p.add_argument('--full-iid-replicates',type=int,default=1,
          help='independently refit full 2497-draw IID cohorts; 0 disables for quick smoke tests')
    p.add_argument('--stability-seeds',type=int,nargs='+',default=[649,650,651,652])
    p.add_argument('--stability-fraction',type=float,default=0.8)
    p.add_argument('--peak-radius',type=float,default=0.6)
    p.add_argument('--assignment-radius',type=float,default=0.85)
    p.add_argument('--max-peaks',type=int,default=24)
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--seed',type=int,default=649)
    p.add_argument('--out',type=Path,default=Path('results/v0141_p0_gaussian_validation.json'))
    p.add_argument('--csv-out',type=Path,default=None)
    p.add_argument('--html-out',type=Path,default=None)
    p.add_argument('--npz-out',type=Path,default=None)


def run(args):
    if args.command!='shape-gaussian-validate':raise ValueError('wrong command')
    out=args.out
    outputs={'JSON':out,'CSV':args.csv_out or out.with_suffix('.csv'),
             'HTML':args.html_out or out.with_suffix('.html'),
             'NPZ':args.npz_out or out.with_suffix('.npz')}
    paths=list(outputs.values())
    if len({p.resolve() for p in paths})!=4 or any(p.exists() for p in paths):
        raise FileExistsError('all four output paths must be distinct and new; existing files are protected')
    matrix,ids,ns=load_existing(args.matrix,args.assignments)
    print(f'[SOURCE] {len(ids)} historical draws, cached shape distance; no re-matching',flush=True)
    report,arrays=analyze(matrix,ids,ns,device=args.device,
        dimensions=args.dimensions,components=args.gmm_components,
        bandwidths=args.bandwidths,comparison_dimension=args.comparison_dimension,
        matched_size=args.matched_size,matched_replicates=args.matched_replicates,
        full_iid_replicates=args.full_iid_replicates,
        stability_seeds=args.stability_seeds,stability_fraction=args.stability_fraction,
        peak_radius=args.peak_radius,assignment_radius=args.assignment_radius,
        max_peaks=args.max_peaks,n_init=args.gmm_n_init,
        batch_size=args.batch_size,seed=args.seed,
        on_progress=lambda s:print(s,flush=True))
    def checksum(path):
        digest=hashlib.sha256()
        with Path(path).open('rb') as fh:
            for block in iter(lambda:fh.read(1024*1024),b''):
                digest.update(block)
        return digest.hexdigest()
    report['data']['shape_npz_file_sha256']=checksum(args.matrix)
    report['data']['assignments_csv_file_sha256']=checksum(args.assignments)
    print(f'[P0] complete — {args.matched_replicates} matched refits, {args.full_iid_replicates} full-IID refits, {len(args.dimensions)} dimensions, {len(args.bandwidths)} bandwidths',flush=True)
    columns=['kind','cohort','replicate','n','dimensions','bandwidth','best_k',
             'bic_gain','density_cv','silhouette','stress','rank_corr',
             'top10_jaccard','peak_count','unassigned','overlap','ari','nmi']
    for kind,p in outputs.items():
        p.parent.mkdir(parents=True,exist_ok=True)
        if kind=='JSON':
            with p.open('x',encoding='utf-8') as fh:
                json.dump(report,fh,ensure_ascii=False,indent=2,allow_nan=False);fh.write('\n')
        elif kind=='CSV':
            with p.open('x',newline='',encoding='utf-8') as fh:
                writer=csv.DictWriter(fh,fieldnames=columns,extrasaction='ignore');writer.writeheader()
                for row in csv_records(report):writer.writerow(row)
        elif kind=='HTML':
            with p.open('x',encoding='utf-8') as fh:fh.write(html_report(report))
        else:
            with p.open('xb') as fh:np.savez_compressed(fh,**arrays)
        print(f'[DONE] {kind}: {p.resolve()}',flush=True)
    print('[CAUTION] P0 diagnostics are descriptive; they do not validate lottery prediction',flush=True)
