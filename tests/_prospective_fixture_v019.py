"""Portable synthetic P1/P2 artifacts for v0.19 tests.

No user files, /mnt/data paths, external downloads or C(49,6) enumeration
needed. Frozen fair probabilities come from the existing checked-in P2 exact
universe fixture; generated numbers are explicitly synthetic tests.
"""
from __future__ import annotations

import csv
from pathlib import Path
import shutil
import json
import numpy as np
from hashlib import sha256

from lotto7x7.shape_taxonomy_v015 import classify_numbers
from lotto7x7.shape_theory_v016 import RULE_VERSION, classification_keys


def make_test_inputs(folder:Path):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    n=2497
    rng=np.random.default_rng(649)
    nums=np.sort(np.argsort(rng.random((n,49)),axis=1)[:,:6]+1,axis=1).astype(np.uint8)
    # Synthetic integer IDs with matching known source boundary, not a calendar claim.
    ids=np.linspace(93000001,115000103,n).astype(np.int64)
    assert len(np.unique(ids))==n and ids[-1]==115000103
    codes=classification_keys(classify_numbers(nums))
    cat=folder/'synthetic_p1_catalog.npz'
    np.savez_compressed(cat,draw_ids=ids,numbers=nums,**codes)
    src=Path(__file__).parent/'fixtures'/'p1_exact_universe_v1.json'
    assert src.is_file(), 'P2 exact-universe fixture must be available in repo'
    univ=folder/'synthetic_p1_exact_universe.json';shutil.copyfile(src,univ)
    watch=folder/'synthetic_p2_watchlist.json'
    meta={'schema':'lotto7x7.p2_future_only_watchlist.v0.17',
          'source_draw_id_last':int(ids[-1]),
          'frozen_rule_hash_sha256':sha256(RULE_VERSION.encode()).hexdigest(),
          'fixed_category':{'class':'P6_O1_D5','theoretical_combinations':744}}
    watch.write_text(json.dumps(meta),encoding='utf-8')
    csvfile=folder/'synthetic_historical.csv'
    with csvfile.open('w',newline='',encoding='utf-8') as out:
        w=csv.writer(out);w.writerow(['draw_id','n1','n2','n3','n4','n5','n6'])
        for k,row in zip(ids,nums):w.writerow([int(k),*map(int,row)])
    return {'catalog':cat,'universe':univ,'watchlist':watch,
            'historical_csv':csvfile,'ids':ids,'numbers':nums}
