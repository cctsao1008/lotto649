"""Lotto649 v0.24 P0-P5 chronological 7x7 temporal-signal research CLI."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime,timezone
import json
import os
from pathlib import Path

import numpy as np

from .temporal_dataset_v024 import (load_csv_history,load_catalog_history,
                                    dataset_report,file_hash)
from .temporal_features_v024 import (temporal_features,transition_rows,p1_summary,
                                     p2_iid_acf_control,feature_acf_report,FEATURE_NAMES)
from .temporal_dsp_v024 import dsp_iid_report
from .temporal_predict_v024 import walk_forward

SCHEMA='lotto7x7.temporal_research.v0.24'


def configure_parsers(action):
    p=action.add_parser('temporal-research',help='P0-P5 causal 7x7 temporal image, DSP and walk-forward research')
    p.add_argument('--csv-root',type=Path,default=Path('data/csv'))
    p.add_argument('--image-bin',type=Path,default=Path('data/bin/lotto649_images_v2.bin'))
    p.add_argument('--catalog',type=Path,default=Path('results/v015_v016_p1_shape_catalog.npz'))
    p.add_argument('--universe',type=Path,default=Path('results/v015_v016_p1_shape_catalog_universe.json'))
    p.add_argument('--catalog-only',action='store_true',help='Explicitly waive date verification if CSV is unavailable')
    p.add_argument('--iid-reps',type=int,default=200)
    p.add_argument('--bootstrap-reps',type=int,default=1000)
    p.add_argument('--seed',type=int,default=649)
    p.add_argument('--out',type=Path,default=Path('results/v024_temporal_research.json'))


def _prefix_paths(out:Path)->dict[str,Path]:
    o=Path(out)
    if o.suffix.lower()!='.json':raise ValueError('--out must end with .json')
    stem=o.with_suffix('')
    return {'report':o,'transitions':Path(str(stem)+'_transitions.csv'),
            'features':Path(str(stem)+'_features.csv'),
            'forecast':Path(str(stem)+'_forecast.csv')}


def _write_csv(path:Path,fields:list[str],rows:list[dict]):
    with path.open('x',encoding='utf-8',newline='') as fh:
        writer=csv.DictWriter(fh,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        for row in rows:writer.writerow(row)
        fh.flush();os.fsync(fh.fileno())


def run_pipeline(*,csv_root:Path=Path('data/csv'),
                 image_bin:Path|None=Path('data/bin/lotto649_images_v2.bin'),
                 catalog:Path|None=Path('results/v015_v016_p1_shape_catalog.npz'),
                 universe:Path|None=Path('results/v015_v016_p1_shape_catalog_universe.json'),
                 catalog_only:bool=False,iid_reps:int=200,
                 bootstrap_reps:int=1000,seed:int=649,
                 out:Path=Path('results/v024_temporal_research.json'),
                 emit=print)->dict:
    if iid_reps<2 or bootstrap_reps<50 or seed<0:
        raise ValueError('IID replicas >=2, bootstrap reps >=50 and nonnegative seed required')
    dest=_prefix_paths(out)
    if any(p.exists() for p in dest.values()):
        raise FileExistsError('One or more output files already exist; use a fresh --out filename')
    dest['report'].parent.mkdir(parents=True,exist_ok=True)
    if catalog_only:
        if catalog is None or not Path(catalog).is_file():
            raise FileNotFoundError('explicit --catalog-only requires a valid --catalog NPZ')
        dataset=load_catalog_history(catalog)
    else:
        if not Path(csv_root).is_dir():
            raise FileNotFoundError('CSV source missing; --catalog-only permits dates-unverified research')
        if image_bin is None or not Path(image_bin).is_file():
            raise FileNotFoundError('strict P0 requires existing --image-bin for CSV/BIN parity')
        if catalog is None or not Path(catalog).is_file():
            raise FileNotFoundError('strict P0 requires existing --catalog for CSV/P1 parity')
        dataset=load_csv_history(csv_root,image_bin=image_bin,catalog=catalog)
    p0=dataset_report(dataset)
    emit(f'[P0 {p0["status"]}] {p0["draws"]} draws, {p0["source"]["date_verification"]}')
    transitions=transition_rows(dataset,full_shape=True)
    p1=p1_summary(transitions)
    emit(f'[P1 PASS] {len(transitions)} chronological transitions; mean overlap={p1["mean_retained"]:.4f} / IID {p1["iid_mean_retained"]:.4f}')
    m,_=temporal_features(dataset)
    p2={'schema':'lotto7x7.temporal_multilag.v0.24',
        'features':list(FEATURE_NAMES),'lags':[1,2,3,5,10,20],
        'per_feature_acf':feature_acf_report(m,max_lag=20),
        'iid_null':p2_iid_acf_control(dataset,replicates=iid_reps,seed=seed,max_lag=20)}
    emit(f'[P2 PASS] temporal feature matrix={m.shape}; IID ACF max-test p={p2["iid_null"]["iid_max_abs_acf_pvalue"]:.4f}')
    p3=dsp_iid_report(dataset,replicates=iid_reps,seed=seed+1)
    emit(f'[P3 PASS] DCT/FFT, three raster mappings, IID spectral max-test p={p3["familywise_iid_pvalue"]:.4f}')
    p5,forecast=walk_forward(dataset,universe=universe if universe and Path(universe).is_file() else None,
                            bootstrap_seed=seed+2,bootstrap_reps=bootstrap_reps)
    emit(f'[P4/P5 PASS] temporal Brier={p5["score_means"]["temporal_brier"]:.7f} vs fair={p5["score_means"]["fair_brier"]:.7f}; decision={p5["decision"]}')
    metadata={'schema':SCHEMA,'generated_at_utc':datetime.now(timezone.utc).isoformat(),
              'settings':{'iid_reps':iid_reps,'bootstrap_reps':bootstrap_reps,
                          'seed':int(seed),'catalog_only':catalog_only},
              'source_hashes':{'catalog':file_hash(catalog) if catalog and Path(catalog).is_file() else None,
                              'universe':file_hash(universe) if universe and Path(universe).is_file() else None},
              'issues':{'P0':2,'P1':3,'P2':4,'P3':5,'P4':6,'P5':7},
              'P0':p0,'P1':p1,'P2':p2,'P3':p3,'P4_P5':p5,
              'caution':'Exploratory retrospective analysis. NO proof of future ticket advantage. No integration unless independent preregistered future results.'}
    # All computation completes before making destination files, so a thrown
    # error does not leave an apparently successful partial research result.
    t_fields=list(transitions[0])
    _write_csv(dest['transitions'],t_fields,transitions)
    table=[]
    for i in range(len(dataset.ids)):
        row={'draw_id':int(dataset.ids[i]),'date':int(dataset.dates[i]) if dataset.dates is not None else ''}
        row.update({name:float(m[i,j]) if np.isfinite(m[i,j]) else '' for j,name in enumerate(FEATURE_NAMES)})
        table.append(row)
    _write_csv(dest['features'],['draw_id','date',*FEATURE_NAMES],table)
    names=['target_index','draw_id','date','actual_six_numbers',
           'fair_brier','static_brier','temporal_brier',
           'fair_bernoulli_logloss','static_bernoulli_logloss','temporal_bernoulli_logloss',
           'fair_ticket_logprob','static_ticket_logprob','temporal_ticket_logprob','shape_ticket_logprob',
           *(f'temporal_p_{k:02d}' for k in range(1,50))]
    frows=[]
    for row in forecast:
        o={k:v for k,v in row.items() if k in names}
        t=row['target_index']
        o['date']=int(dataset.dates[t]) if dataset.dates is not None else ''
        o['actual_six_numbers']=' '.join(f'{x:02d}' for x in dataset.numbers[t])
        o.update({f'temporal_p_{k:02d}':x for k,x in enumerate(row['temporal_49_parameters'],1)})
        frows.append(o)
    _write_csv(dest['forecast'],names,frows)
    temp=dest['report'].with_suffix('.json.tmp')
    try:
        with temp.open('w',encoding='utf-8') as f:
            json.dump(metadata,f,ensure_ascii=False,indent=2,allow_nan=False)
            f.write('\n');f.flush();os.fsync(f.fileno())
        temp.replace(dest['report'])
    finally:
        temp.unlink(missing_ok=True)
    for kind,path in dest.items():emit(f'[FILE] {kind}: {path}')
    return metadata


def run(args):
    return run_pipeline(csv_root=args.csv_root,image_bin=args.image_bin,
        catalog=args.catalog,universe=args.universe,catalog_only=args.catalog_only,
        iid_reps=args.iid_reps,bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,out=args.out,emit=lambda s:print(s,flush=True))
