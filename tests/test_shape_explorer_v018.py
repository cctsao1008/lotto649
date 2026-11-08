"""Offline interactive UI validation and exact class mapping checks."""
from math import comb
import numpy as np
import pytest

from lotto7x7.shape_explorer_v018 import gallery_data,offline_shape_explorer
from lotto7x7.shape_theory_v016 import RULE_VERSION,UNIVERSE,classification_keys
from lotto7x7.shape_taxonomy_v015 import classify_numbers


def fixture():
    x=np.array([[1,3,5,29,31,33],[1,8,16,22,24,30],[17,22,23,25,31,33]],dtype=np.uint8)
    q=classification_keys(classify_numbers(x))
    counts={}
    for k in ('component','partition','edge'):
        values=sorted(map(int,set(q[k])));masses={str(v):UNIVERSE//len(values) for v in values}
        masses[str(values[0])]+=UNIVERSE-sum(masses.values())
        counts[k]=masses
    return np.arange(3),x,{'schema':RULE_VERSION,'complete':True,'total':UNIVERSE,'counts':counts}


def test_explorer_data_and_structured_labels():
    ids,x,u=fixture();z=gallery_data(ids,x,u)
    assert z['count']==3
    assert z['records'][0][:7]==[0,1,3,5,29,31,33]
    assert 'edge' in z['counts']


def test_complete_offline_html_has_no_remote_resources():
    ids,x,u=fixture();h=offline_shape_explorer(ids,x,u)
    assert '<script id="history-data" type="application/json">' in h
    assert 'const BITCOUNT' in h and 'shapeDistance(qVariants,record)' in h
    assert '<input id="query"' in h and 'addEventListener' in h
    assert 'http://' not in h and 'https://' not in h
    assert 'Lotto649 v0.18' in h


def test_explorer_rejects_bad_mode_alignment():
    ids,x,u=fixture()
    with pytest.raises(ValueError):gallery_data(ids,x,u,mode_labels=np.array([1,2]))


def test_explorer_rejects_incomplete_universe():
    ids,x,u=fixture();u['complete']=False
    with pytest.raises(ValueError):offline_shape_explorer(ids,x,u)


def test_kde_is_only_optional_display_context():
    ids,x,u=fixture();h=offline_shape_explorer(ids,x,u,density=np.array([.1,.2,.3]),
       mode_labels=np.array([-1,1,1]));assert '0.3' in h
