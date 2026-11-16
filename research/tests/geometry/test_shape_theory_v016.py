"""P1 v0.16 universe enumerator, exact orbit accounting and null calibration."""
import json
from math import comb
import numpy as np
import pytest

from lotto7x7.shape_theory_v016 import (
    UNIVERSE,exhaustive_class_counts,classification_keys,category_name,
    historical_comparison,shape_family_baseline,_bh_fdr,_bonferroni)
from lotto7x7.shape_catalog_batch_p1_cli import output_paths,_verify_baseline
from lotto7x7.shape_taxonomy_v015 import classify_numbers


def test_universe_size():
    assert UNIVERSE==13_983_816==comb(49,6)


def test_first_small_enumeration_matches_combinations():
    from itertools import islice,combinations
    x=exhaustive_class_counts(batch_size=13,max_items=110,allow_partial=True)
    a=np.asarray(list(islice(combinations(range(1,50),6),110)),dtype=np.uint8)
    p=classification_keys(classify_numbers(a))
    for level,vals in p.items():
        keys,counts=np.unique(vals,return_counts=True)
        expected={str(int(k)):int(v) for k,v in zip(keys,counts)}
        assert x['counts'][level]==expected
    assert x['total']==110 and not x['complete']


def test_partial_is_never_labeled_exhaustive():
    with pytest.raises(ValueError):exhaustive_class_counts(max_items=10)
    with pytest.raises(ValueError):_verify_baseline(exhaustive_class_counts(max_items=10,allow_partial=True))


def test_one_row_population_valid():
    a=np.array([[1,2,3,4,5,6]])
    v=classification_keys(classify_numbers(a))
    assert len(v)==3 and all(t.shape==(1,) for t in v.values())


def test_bh_and_global_adjustment():
    assert _bonferroni([.1,.2,.7])==pytest.approx([.3,.6,1])
    q=_bh_fdr([.01,.04,.03])
    assert q==pytest.approx([.03,.04,.04])


def test_exact_family_mass_no_posthoc_significance():
    f=[{'shape_id':'S1','theory_combinations':12},{'shape_id':'S2','theory_combinations':24}]
    r=shape_family_baseline(f,2497)
    assert r['theoretical_total_in_unseen_exact_shapes']==UNIVERSE-36
    assert r['theoretical_total_in_seen_exact_shapes']==36
    with pytest.raises(ValueError):shape_family_baseline([f[0],f[0]],2)


def test_category_names():
    assert category_name('component',6)=='C6'
    with pytest.raises(ValueError):category_name('bad',1)


def test_mock_full_baseline_significance_mechanism():
    # All probability mass on one legal shape class: degenerate mock for
    # verifying multinomial and binomial computations, not real universe.
    x=np.repeat([[1,2,3,4,5,6]],8,axis=0)
    v=classification_keys(classify_numbers(x[:1]))
    baseline={'complete':True,'total':UNIVERSE,
       'counts':{k:{str(int(vals[0])):UNIVERSE} for k,vals in v.items()}}
    r=historical_comparison(x,baseline,replicates=99,seed=649)
    for q in r['levels'].values():
        assert q['empirical_multinomial_p']==1
        assert q['rows'][0]['binomial_p']==1
        assert q['bonferroni_global_p']==1


def test_no_universe_partial_allowed_in_model():
    x=exhaustive_class_counts(batch_size=32,max_items=32,allow_partial=True)
    with pytest.raises(ValueError):historical_comparison(np.repeat([[1,2,3,4,5,6]],8,axis=0),x,replicates=99)


def test_cli_output_protection_schema():
    p=output_paths('results/p1_result.json')
    assert len(set(p.values()))==8
    assert str(p['CATALOG_CSV']).endswith('_catalog.csv')
    with pytest.raises(ValueError):output_paths('results/foo.csv')
