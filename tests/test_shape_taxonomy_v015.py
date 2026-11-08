"""P1 v0.15 fixed image shape codes and descriptive empirical KDE modes."""
import numpy as np
import pytest
from itertools import combinations

from lotto7x7.shape_taxonomy_v015 import (
    classify_numbers,decode_partition,partition_name,edge_name,shape_key,
    exact_shape_families,sample_local_modes,nearest_mode_assignments,
    mode_stability_multiscale,pairwise_diameter_metrics,UNIVERSE)


def brute_graph(nums):
    points=[divmod(int(x)-1,7) for x in nums]
    adj={i:set() for i in range(6)}
    cardinal=diagonal=0
    for i,j in combinations(range(6),2):
        dy=abs(points[i][0]-points[j][0]); dx=abs(points[i][1]-points[j][1])
        if dy<=1 and dx<=1:
            adj[i].add(j); adj[j].add(i)
            if dy+dx==1:cardinal+=1
            elif dy==dx==1:diagonal+=1
    seen=set();part=[]
    for i in range(6):
        if i in seen:continue
        stack=[i];seen.add(i);v=0
        while stack:
            j=stack.pop();v+=1
            for k in adj[j]:
                if k not in seen:seen.add(k);stack.append(k)
        part.append(v)
    return tuple(sorted(part,reverse=True)),cardinal,diagonal


def code_data(numbers):
    v=classify_numbers(np.asarray(numbers,dtype=np.uint8).reshape(1,6))
    return int(v['n_components'][0]),decode_partition(int(v['partition_code'][0])),int(v['cardinal_edges'][0]),int(v['diagonal_edges'][0])


def test_six_aligned_cardinal_edges():
    assert code_data([1,2,3,4,5,6])==(1,(6,),5,0)


def test_six_diagonal_edges():
    assert code_data([1,9,17,25,33,41])==(1,(6,),0,5)


def test_six_separate():
    assert code_data([1,3,22,24,36,49])==(6,(1,1,1,1,1,1),0,0)


def test_random_brute_graph_1000():
    rng=np.random.default_rng(649)
    a=np.stack([rng.choice(np.arange(1,50),6,replace=False) for _ in range(1000)]).astype(np.uint8)
    y=classify_numbers(a)
    for i,row in enumerate(a):
        sizes,o,d=brute_graph(row)
        assert tuple(sizes)==decode_partition(int(y['partition_code'][i]))
        assert o==y['cardinal_edges'][i]
        assert d==y['diagonal_edges'][i]
        assert len(sizes)==y['n_components'][i]


def test_rotations_and_reflection_preserve_exact_codes():
    a=np.array([2,8,11,22,27,40]);points=np.stack(divmod(a-1,7),axis=1)
    reflect=7*(6-points[:,0])+points[:,1]+1
    rotate=7*points[:,1]+(6-points[:,0])+1
    for x in (reflect,rotate):
        assert code_data(a)==code_data(x)
        assert shape_key(a)==shape_key(x)


def test_integer_translation_preserves_classes_and_shape():
    base=np.array([1,3,12,16,24,33]);shifted=base+8
    assert code_data(base)==code_data(shifted)
    assert shape_key(base)==shape_key(shifted)


def test_parse_signature():
    assert partition_name(int(classify_numbers(np.array([[1,2,3,4,5,6]]))['partition_code'][0]))=='P6'
    assert edge_name(int(classify_numbers(np.array([[1,2,3,4,5,6]]))['edge_code'][0]))=='P6_O5_D0'
    with pytest.raises(ValueError):decode_partition(0)


@pytest.mark.parametrize('bad',[[1,1,3,4,5,6],[0,2,3,4,5,6],[1,2,3,4,5,50],[-1,2,3,4,5,6]])
def test_illegal_six_hot(bad):
    with pytest.raises(ValueError): classify_numbers(np.array([bad]))


def test_illegal_shape_key():
    with pytest.raises(ValueError):shape_key([1,1,2,3,4,5])


def test_exact_family_groups_and_orbit_mass():
    nums=np.array([[1,2,3,4,5,6],[8,9,10,11,12,13],[1,3,22,24,36,49]],dtype=np.uint8)
    f,labels=exact_shape_families(nums,np.array([101,102,103]))
    assert len(f)==2 and labels[0]==labels[1]!=labels[2]
    assert sum(z['theory_combinations'] for z in f)<UNIVERSE
    assert sorted(z['count'] for z in f)==[1,2]


def test_empirical_discrete_modes_are_not_capped():
    d=np.array([[0.,.2,3.,3.],[.2,0.,3.,3.],[3.,3.,0.,.2],[3.,3.,.2,0.]],dtype=np.float32)
    peak=sample_local_modes(d,np.array([.7,.5,.2,.1]),radius=.3)
    assert set(peak['peaks'])=={0,2}
    labels,md=nearest_mode_assignments(d,peak['peaks'],.4)
    assert np.all(labels>=0)
    assert np.max(md)<=.2+1e-6


def test_density_modes_change_with_radius():
    d=np.array([[0.,.5,1.4],[.5,0.,.5],[1.4,.5,0.]],dtype=np.float32)
    y=np.array([.1,.2,.3])
    assert len(sample_local_modes(d,y,.3)['peaks'])==3
    assert len(sample_local_modes(d,y,.6)['peaks'])==1


def test_multiscale_mode_status():
    d=np.array([[0.,.5,1.4],[.5,0.,.5],[1.4,.5,0.]],dtype=np.float32)
    y=np.stack([np.array([.2,.3,.4]),np.array([.3,.4,.2])],axis=1)
    report,arr=mode_stability_multiscale(d,y,(.4,.8),(.3,.6),.7)
    assert len(report['records'])==4
    assert len(arr['labels'])==3
    assert all(0<=x['presence_rate']<=1 for x in arr['persistence'])


def test_pairwise_diameter_in_original_metric():
    d=np.array([[0.,.1,2.],[.1,0.,1.9],[2.,1.9,0.]],dtype=np.float32)
    out=pairwise_diameter_metrics(d,np.array([0,0,1]))
    assert len(out)==2
    assert out[0]['diameter']==pytest.approx(.1)
    assert out[0]['nearest_other_min']==pytest.approx(1.9)
