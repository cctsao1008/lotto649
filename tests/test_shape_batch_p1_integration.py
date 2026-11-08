"""P1 cross-module integration with self-contained synthetic legal data."""
from math import comb
import numpy as np
import pytest

from lotto7x7.shape_theory_v016 import UNIVERSE,RULE_VERSION,classification_keys
from lotto7x7.shape_taxonomy_v015 import classify_numbers
from lotto7x7.shape_catalog_batch_p1 import analyze,html_report,rows_draws,rows_theory,rows_exact_families
from lotto7x7.shape_catalog_batch_p1_cli import output_paths


def fixture():
    rng=np.random.default_rng(2026)
    nums=np.stack([rng.choice(np.arange(1,50),6,replace=False) for _ in range(24)]).astype(np.uint8)
    ids=np.arange(1,25)
    # A symmetric synthetic pairwise metric; not used to assert lottery facts.
    x=np.sort(nums,axis=1).astype(float)
    d=np.sqrt(np.square(x[:,None,:]-x[None,:,:]).mean(axis=2)).astype(np.float32)
    np.fill_diagonal(d,0.)
    keys=classification_keys(classify_numbers(nums))
    counts={}
    for lvl,values in keys.items():
        uniq=np.unique(values)
        q=np.full(len(uniq),UNIVERSE//len(uniq),dtype=np.int64)
        q[0]+=UNIVERSE-int(q.sum())
        counts[lvl]={str(int(k)):int(v) for k,v in zip(uniq,q)}
    baseline={'schema':RULE_VERSION,'rule_hash_sha256':'synthetic','complete':True,
         'total':UNIVERSE,'counts':counts}
    return d,ids,nums,baseline


def test_cross_module_catalog_reporting_and_counts():
    d,ids,nums,baseline=fixture()
    report,arrays=analyze(d,ids,nums,baseline=baseline,device='cpu',
          bandwidths=(.3,.8),radii=(.6,1.),mc_replicates=99,seed=649)
    assert report['data']['draw_count']==len(ids)
    assert report['theory']['processed']==UNIVERSE
    assert sum(x['count'] for x in report['exact_shape_families'])==len(ids)
    assert report['kde_multiscale']['main_peak_count']==len(arrays['mode_peaks'])
    assert report['kde_multiscale']['main_assigned']+report['kde_multiscale']['main_unassigned']==len(ids)
    assert len(list(rows_draws(arrays)))==len(ids)
    assert len(list(rows_exact_families(report)))==len(report['exact_shape_families'])
    assert len(list(rows_theory(report)))==sum(len(z) for z in baseline['counts'].values())
    assert set(report['separation_diagnostics'])=={'component','partition','edge','empirical_modes'}
    assert 'Exact 6/49 Baseline' in html_report(report)


def test_integration_reject_non_symmetric_metric():
    d,ids,nums,baseline=fixture()
    d[0,1]=99
    with pytest.raises(ValueError):analyze(d,ids,nums,baseline=baseline,device='cpu',mc_replicates=99)


def test_integration_reject_partial_universe():
    d,ids,nums,baseline=fixture()
    baseline['complete']=False
    with pytest.raises(ValueError):analyze(d,ids,nums,baseline=baseline,device='cpu',mc_replicates=99)


def test_all_output_names_distinct():
    outputs=output_paths('results/test_P1.json')
    assert len(outputs)==8==len(set(outputs.values()))
    assert outputs['UNIVERSE_JSON'].name=='test_P1_universe.json'
