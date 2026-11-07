"""v0.12 CLI: natural density and hierarchical shape clustering with galleries."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .natural_clustering_v012 import report, html_gallery, DEFAULT_K
from .research import DATA, load_history


def configure_parsers(action):
    p=action.add_parser('natural-cluster-gpu',help='v0.12 HDBSCAN + Ward + K-medoids clustering and 7x7 galleries')
    p.add_argument('--data',type=Path,default=DATA)
    p.add_argument('--device',choices=('auto','cuda','cpu'),default='auto')
    p.add_argument('--features',choices=('geometry','cnn','fused'),default='geometry')
    p.add_argument('--model',type=Path,default=None,help='frozen v0.10 CNN checkpoint for cnn/fused feature modes')
    p.add_argument('--k-values',type=int,nargs='+',default=list(DEFAULT_K),help='candidate Ward class counts; best selected by silhouette')
    p.add_argument('--min-cluster-size',type=int,default=30)
    p.add_argument('--min-samples',type=int,default=10)
    p.add_argument('--silhouette-sample',type=int,default=1200)
    p.add_argument('--stability-repeats',type=int,default=2)
    p.add_argument('--stability-fraction',type=float,default=.8)
    p.add_argument('--iid-replicates',type=int,default=9,help='full-refit legal 6/49 IID null replicates; raise for statistical precision')
    p.add_argument('--iid-samples',type=int,default=10000,help='fixed real-medoid fair-IID size proportions')
    p.add_argument('--gallery-count',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=1024)
    p.add_argument('--seed',type=int,default=649)
    p.add_argument('--out',type=Path,default=Path('results/v012_natural_clusters.json'))
    p.add_argument('--csv-out',type=Path,default=None)
    p.add_argument('--html-out',type=Path,default=None)


def run(args):
    if args.command!='natural-cluster-gpu':raise ValueError('unexpected command')
    out=Path(args.out); csv_path=Path(args.csv_out) if args.csv_out else out.with_suffix('.csv')
    html_path=Path(args.html_out) if args.html_out else out.with_suffix('.html')
    paths=[out,csv_path,html_path]
    if len({p.resolve() for p in paths})!=3 or any(p.exists() for p in paths):
        raise FileExistsError('Choose three unique paths; existing research outputs will not be overwritten')
    images,_,ids=load_history(args.data)
    print(f'[SOURCE] {len(images)} legal 7x7 images; features={args.features}; methods=hdbscan,ward,kmedoids',flush=True)
    result=report(images,ids,feature_mode=args.features,checkpoint=args.model,device=args.device,
                  k_values=tuple(args.k_values),min_cluster_size=args.min_cluster_size,
                  min_samples=args.min_samples,silhouette_sample=args.silhouette_sample,
                  stability_repeats=args.stability_repeats,stability_fraction=args.stability_fraction,
                  iid_replicates=args.iid_replicates,iid_samples=args.iid_samples,
                  batch_size=args.batch_size,gallery_count=args.gallery_count,seed=args.seed,
                  on_progress=lambda msg:print(msg,flush=True))
    for path in paths:path.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf8') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    with csv_path.open('x',encoding='utf8',newline='') as f:
        w=csv.writer(f);w.writerow(['draw_id','hdbscan_class','hdbscan_strength','ward_class','kmedoids_class','n1','n2','n3','n4','n5','n6'])
        groups=result['methods']
        for i in range(result['draw_count']):
            h=groups['hdbscan']['assignments'][i];a=groups['ward']['assignments'][i];b=groups['kmedoids']['assignments'][i]
            w.writerow([h['draw_id'],h['class_id'],h['membership_strength'],a['class_id'],b['class_id'],*h['numbers']])
    with html_path.open('x',encoding='utf8') as f:f.write(html_gallery(result))
    print(f'[WARD] K selection {result["ward_model_selection"]["selected_k"]} tested={args.k_values}',flush=True)
    for key,v in result['methods'].items():
        s=v['silhouette'];q=v['stability']['ari'];
        print(f'[METHOD] {key}: classes={v["class_count"]} noise={v["noise_count"]} coverage={v["coverage"]:.2%} silhouette={"n/a" if s is None else f"{s:.4f}"} stability_ARI={"n/a" if q is None else f"{q:.4f}"}',flush=True)
    c=result['iid_full_refit_controls'];
    print(f'[IID] refits={c["replicates"]} ward_p={c["ward_selected_best_silhouette_empirical_p"]} hdbscan_count_p={c["hdbscan_class_count_empirical_p"]}',flush=True)
    print(f'[DONE] JSON: {out.resolve()}\n[DONE] CSV: {csv_path.resolve()}\n[DONE] HTML: {html_path.resolve()}',flush=True)
    print('[CAUTION] Descriptive shape clustering only; no evidence of winning-number prediction',flush=True)
