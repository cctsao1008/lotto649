"""CLI: descriptive 7x7 image classification, count tables, and SVG gallery."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .image_classification_v011 import report, html_gallery
from .research import DATA, load_history


def configure_parsers(action):
    p=action.add_parser('image-classify',help='v0.11 D4 shape classification and CUDA-assisted 7x7 prototype groups')
    p.add_argument('--data',type=Path,default=DATA)
    p.add_argument('--device',choices=('auto','cuda','cpu'),default='auto')
    p.add_argument('--clusters',type=int,default=12)
    p.add_argument('--features',choices=('geometry','cnn','fused'),default='geometry')
    p.add_argument('--model',type=Path,default=None,help='v0.10 .pt checkpoint for CNN/fused mode; not retrained')
    p.add_argument('--iid-samples',type=int,default=10000)
    p.add_argument('--batch-size',type=int,default=2048)
    p.add_argument('--seed',type=int,default=649)
    p.add_argument('--out',type=Path,default=Path('results/v011_shape_classification.json'))
    p.add_argument('--csv-out',type=Path,default=None)
    p.add_argument('--html-out',type=Path,default=None)


def run(args):
    if args.command!='image-classify':
        raise ValueError('v0.11 CLI command expected')
    out=Path(args.out)
    csv_out=Path(args.csv_out) if args.csv_out else out.with_suffix('.csv')
    html_out=Path(args.html_out) if args.html_out else out.with_suffix('.html')
    paths=[out,csv_out,html_out]
    if len(set(x.resolve() for x in paths))!=3 or any(p.exists() for p in paths):
        raise FileExistsError('Output JSON, CSV and HTML must be different new paths; existing files will not be overwritten')
    images,_,ids=load_history(args.data)
    print(f'[SOURCE] {len(images)} historical six-hot 7x7 images; classes={args.clusters} features={args.features}',flush=True)
    data=report(images,ids,clusters=args.clusters,feature_mode=args.features,
                checkpoint=args.model,device=args.device,seed=args.seed,
                iid_samples=args.iid_samples,batch_size=args.batch_size)
    for path in paths:
        path.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')
    with csv_out.open('x',encoding='utf-8',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['draw_id','class_id','n1','n2','n3','n4','n5','n6','components_8'])
        for row in data['assignments']:
            writer.writerow([row['draw_id'],row['class_id'],*row['numbers'],row['components_8']])
    with html_out.open('x',encoding='utf-8') as f:
        f.write(html_gallery(data))
    e=data['exact_shape_classes'];network=data['five_of_six_similarity_network']
    print(f'[EXACT] shape_classes={e["shape_class_count"]} count_histogram={e["size_histogram"]}',flush=True)
    print(f'[MORPH] 8-connected components histogram={data["eight_neighbor_morphology"]["components_histogram"]}',flush=True)
    print(f'[SIMILAR] >=5-of-6 matched pairs={network["pair_count"]} images_with_neighbor={network["images_with_similar_neighbor"]}',flush=True)
    print(f'[DEVICE] {data["device"]}',flush=True)
    for item in data['cluster_summary']:
        print(f'[CLASS] {item["class_id"]}: n={item["count"]} history={item["historical_share"]:.2%} '
              f'fair_IID={item["iid_share_estimate"]:.2%} representative={item["representative_numbers"]}',flush=True)
    print(f'[DONE] JSON: {out.resolve()}',flush=True)
    print(f'[DONE] CSV: {csv_out.resolve()}',flush=True)
    print(f'[DONE] HTML: {html_out.resolve()}',flush=True)
    print('[CAUTION] Descriptive image groups and fixed-centroid IID comparisons; not winning-number predictions',flush=True)
