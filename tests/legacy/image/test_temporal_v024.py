"""P0..P5 regression and anti-leakage safeguards for v0.24."""
from __future__ import annotations

import csv
import json
from math import comb,log
from pathlib import Path
import numpy as np
import pytest

from lotto7x7.temporal_dataset_v024 import (
    TemporalDataset,load_catalog_history,load_csv_history,validate_arrays,
    chronological_split,rasterize,dataset_report)
from lotto7x7.temporal_features_v024 import (
    iid_overlap_probs,d4_translation_max_iou,temporal_features,
    transition_rows,p1_summary,autocorrelation,generate_iid_numbers,
    p2_iid_acf_control,FEATURE_NAMES)
from lotto7x7.temporal_dsp_v024 import (
    mapped_data,layout_indices,spatial_energy,periodogram_peak,dsp_iid_report)
from lotto7x7.temporal_predict_v024 import (
    _project_capped_simplex,exact_ticket_logprob,walk_forward,FAIR_P,FAIR_TICKET_LOG)
from lotto7x7.temporal_cli_v024 import run_pipeline,_prefix_paths


def sample(n:int=340,seed:int=25)->TemporalDataset:
    numbers=generate_iid_numbers(np.random.default_rng(seed),n)
    ids=np.arange(113000001,113000001+n,dtype=np.int64)
    return TemporalDataset(ids,numbers,rasterize(numbers),None,{'fixture':True})


def test_p0_rejects_duplicate_id_and_order():
    a=sample(5)
    for arr in ([113000001,113000001,113000003,113000004,113000005],
                [113000002,113000001,113000003,113000004,113000005]):
        with pytest.raises(ValueError,match='draw IDs'):
            validate_arrays(np.array(arr),a.numbers)


def test_p0_rejects_invalid_sixsets():
    a=sample(5)
    for value in (0,50,int(a.numbers[0,1])):
        bad=a.numbers.copy().astype(np.int64)
        bad[0,0]=value
        with pytest.raises(ValueError):validate_arrays(a.ids,bad)


def test_p0_rejects_calendar_date_order_and_bad_roc_year():
    a=sample(3)
    days=np.array([20240101,20240105,20240110])
    validate_arrays(a.ids,a.numbers,days)
    with pytest.raises(ValueError,match='chronological'):
        validate_arrays(a.ids,a.numbers,[20240101,20240101,20240110])
    with pytest.raises(ValueError,match='conflicts'):
        validate_arrays(a.ids,a.numbers,[20230101,20240105,20240110])


def test_p0_csv_stable_and_reject_duplicate_dates(tmp_path):
    root=tmp_path/'csv';root.mkdir()
    p=root/'lotto649_2024.csv'
    lines=['date,draw_id,n1,n2,n3,n4,n5,n6,bonus',
           '2024/01/02,113000001,01,04,08,16,30,44,03',
           '2024/01/05,113000002,02,09,12,17,36,41,03',
           '2024/01/09,113000003,03,08,10,22,34,47,01']
    p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    data=load_csv_history(root)
    assert len(data.ids)==3 and data.dates.tolist()==[20240102,20240105,20240109]
    assert data.images.sum()==18 and data.sources['mode']=='csv'
    p.write_text('\n'.join([lines[0],lines[1],lines[1],lines[3]])+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='out-of-order|duplicated'):
        load_csv_history(root)


def test_p0_validation_split_no_test_leak():
    s=chronological_split(2497)
    assert s['train_end']==1498 and s['validation_end']==1997
    assert s['test_end']==2497 and s['warmup']==21
    with pytest.raises(ValueError):chronological_split(120)


def test_p1_overlap_exact_iid_distribution():
    probs=iid_overlap_probs()
    assert sum(probs.values())==pytest.approx(1,abs=1e-12)
    assert sum(int(k)*p for k,p in probs.items())==pytest.approx(36/49,abs=1e-12)
    assert probs['6']==pytest.approx(1/comb(49,6))


def test_p1_shapes_transform_and_conservation():
    data=sample(12)
    identical=d4_translation_max_iou(data.images[1],data.images[1])
    assert identical==1.0
    transformed=np.fliplr(np.rot90(data.images[1]))
    assert d4_translation_max_iou(data.images[1],transformed)==1.
    rows=transition_rows(data)
    assert len(rows)==11
    assert all(r['removed']==6-r['retained'] and r['added']==r['removed'] and
               r['xor_pixels']==2*r['removed'] and r['d4_translation_iou']<=1 for r in rows)
    report=p1_summary(rows)
    assert sum(report['overlap_histogram'].values())==11


def test_p2_causal_feature_prefix_invariance():
    data=sample(75)
    a,_=temporal_features(data)
    changed=data.numbers.copy();changed[-1]=np.array([1,2,3,4,5,6])
    revised=TemporalDataset(data.ids,changed,rasterize(changed),None,{})
    b,_=temporal_features(revised)
    np.testing.assert_allclose(a[:-1],b[:-1],equal_nan=True)
    assert np.isnan(a[0,FEATURE_NAMES.index('overlap_lag_20')])
    assert np.isfinite(a[20,FEATURE_NAMES.index('overlap_lag_20')])
    assert np.isnan(a[9,FEATURE_NAMES.index('overlap_mean_10')])
    assert np.isfinite(a[10,FEATURE_NAMES.index('overlap_mean_10')])


def test_p2_iid_acf_surrogate_contains_rolling_artifact():
    data=sample(350)
    control=p2_iid_acf_control(data,max_lag=3,replicates=10,seed=10)
    assert 0<control['iid_max_abs_acf_pvalue']<=1
    # Adjacent 5-draw rolling windows overlap even under IID; ACF is large.
    assert control['iid_mean_acf_by_lag']['overlap_mean_5'][0]>0.5


def test_p2_acf_constant_and_invalid_short():
    a=autocorrelation(np.ones(60),max_lag=6)
    assert a.tolist()==[1.,0.,0.,0.,0.,0.,0.]
    with pytest.raises(ValueError):autocorrelation(np.ones(7),max_lag=6)


def test_p3_mapping_is_permutation_and_reproducible():
    for name in ('sequential','snake','permuted'):
        index=layout_indices(name,seed=91)
        assert np.array_equal(np.sort(index),np.arange(49))
        assert np.array_equal(index,layout_indices(name,seed=91))
        x=sample(21)
        y=mapped_data(x,name,seed=91)
        assert np.all(y.images.sum(axis=(1,2))==6)
        np.testing.assert_array_equal(y.images.reshape(21,49),
                                      x.images.reshape(21,49)[:,layout_indices(name,seed=91)])


def test_p3_parseval_and_dc_invariance():
    data=sample(20)
    stats=spatial_energy(data.images)
    assert stats['dc_power']==pytest.approx(36/49)
    assert stats['mean_total_energy']==pytest.approx(6)
    assert stats['mean_non_dc_energy']==pytest.approx(6-36/49,abs=1e-10)
    assert periodogram_peak(np.ones(100))['constant']


def test_p3_iid_full_mapping_spectral_test():
    data=sample(150)
    report=dsp_iid_report(data,replicates=8,seed=91)
    assert set(report['mapping_diagnostics'])=={'sequential','snake','permuted'}
    assert 0<report['familywise_iid_pvalue']<=1


def test_p4_projection_probability_sum_and_bounds():
    x=np.random.default_rng(3).normal(size=49)
    p=_project_capped_simplex(x)
    assert np.all(p>=0) and np.all(p<=1)
    assert p.sum()==pytest.approx(6,abs=1e-10)
    assert np.allclose(_project_capped_simplex(np.full(49,FAIR_P)),FAIR_P)


def test_p4_normalized_exact_ticket_iid_and_permutation():
    x=np.full(49,FAIR_P)
    p=exact_ticket_logprob(x,np.array([1,5,12,27,35,47]))
    assert p==pytest.approx(FAIR_TICKET_LOG,abs=1e-12)
    assert exact_ticket_logprob(x,np.array([47,35,27,12,5,1]))==pytest.approx(p)


def test_p5_walkforward_hyperparams_validation_only_and_no_go():
    data=sample(340)
    r,rows=walk_forward(data,alphas=(1.,100.),bootstrap_reps=50)
    assert r['test_target_count']==68 and len(rows)==68
    assert r['selected_alpha'] in (1.,100.)
    assert r['decision']=='NO_GO'
    assert len(rows[0]['temporal_49_parameters'])==49
    assert sum(rows[0]['temporal_49_parameters'])==pytest.approx(6)
    assert 0<=rows[0]['temporal_brier']<=1


def test_p5_future_target_perturbation_cannot_change_prior_prediction():
    data=sample(360,seed=2)
    r,rows=walk_forward(data,alphas=(1.,100.),bootstrap_reps=50)
    nums=data.numbers.copy()
    nums[-1]=np.array([1,2,3,4,5,6],dtype=np.uint8)
    b=TemporalDataset(data.ids,nums,rasterize(nums),None,{})
    r2,rows2=walk_forward(b,alphas=(1.,100.),bootstrap_reps=50)
    assert r['selected_alpha']==r2['selected_alpha']
    for original,revised in zip(rows,rows2):
        np.testing.assert_array_equal(original['temporal_49_parameters'],
                                      revised['temporal_49_parameters'])
    # Only final target was modified: earlier observed Brier must remain same.
    assert [x['temporal_brier'] for x in rows[:-1]]==[x['temporal_brier'] for x in rows2[:-1]]


def test_full_pipeline_json_and_csv_integrity_catalog_only(tmp_path):
    source=sample(330)
    catalog=tmp_path/'catalog.npz'
    np.savez(catalog,draw_ids=source.ids,numbers=source.numbers)
    out=tmp_path/'v024.json';logs=[]
    r=run_pipeline(catalog=catalog,universe=None,catalog_only=True,
                   iid_reps=6,bootstrap_reps=50,seed=17,out=out,emit=logs.append)
    assert r['P0']['draws']==330
    assert r['P4_P5']['test_target_count']==66
    assert r['P4_P5']['decision']=='NO_GO'
    assert r['P0']['first_date'] is None
    p=_prefix_paths(out)
    assert all(x.exists() for x in p.values())
    assert json.loads(out.read_text())['schema']=='lotto7x7.temporal_research.v0.24'
    with p['forecast'].open(newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==66
    assert sum(float(rows[0][f'temporal_p_{k:02d}']) for k in range(1,50))==pytest.approx(6)
    with pytest.raises(FileExistsError):
        run_pipeline(catalog=catalog,universe=None,catalog_only=True,
                     iid_reps=6,bootstrap_reps=50,seed=17,out=out,emit=lambda s:None)


def test_p0_catalog_mismatch_rejected(tmp_path):
    root=tmp_path/'csv';root.mkdir()
    p=root/'lotto649_2024.csv'
    p.write_text('date,draw_id,n1,n2,n3,n4,n5,n6,bonus\n'
                 '2024/01/02,113000001,01,04,08,16,30,44,03\n'
                 '2024/01/05,113000002,02,09,12,17,36,41,03\n'
                 '2024/01/09,113000003,03,08,10,22,34,47,01\n',encoding='utf-8')
    okay=load_csv_history(root)
    wrong=okay.numbers.copy();wrong[-1]=[1,2,3,4,5,6]
    out=tmp_path/'wrong_catalog.npz'
    np.savez(out,draw_ids=okay.ids,numbers=wrong)
    with pytest.raises(ValueError,match='P1 catalog mismatches'):
        load_csv_history(root,catalog=out)


def test_p0_csv_reverse_order_rejected(tmp_path):
    root=tmp_path/'csv';root.mkdir()
    (root/'lotto649_2024.csv').write_text(
        'date,draw_id,n1,n2,n3,n4,n5,n6,bonus\n'
        '2024/01/05,113000002,02,09,12,17,36,41,03\n'
        '2024/01/02,113000001,01,04,08,16,30,44,03\n'
        '2024/01/09,113000003,03,08,10,22,34,47,01\n',encoding='utf-8')
    with pytest.raises(ValueError,match='out-of-order'):
        load_csv_history(root)


def test_p0_catalog_only_dates_unavailable(tmp_path):
    a=sample(340)
    npz=tmp_path/'catalog.npz'
    np.savez(npz,draw_ids=a.ids,numbers=a.numbers)
    b=load_catalog_history(npz)
    assert b.dates is None
    assert dataset_report(b)['source']['date_verification'].startswith('UNAVAILABLE')
    np.testing.assert_array_equal(b.numbers,a.numbers)


def test_p4_p5_forecast_rows_only_targets_on_or_after_test_boundary():
    D=sample(360)
    report,forecasts=walk_forward(D,alphas=(1.,100.),bootstrap_reps=50)
    s=report['split']
    assert forecasts[0]['target_index']==s['validation_end']
    assert forecasts[-1]['target_index']==s['test_end']-1
    assert report['validation_target_count']==s['validation_end']-s['train_end']
    assert report['test_target_count']==s['test_end']-s['validation_end']
