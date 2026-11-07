"""Only new focused v0.11 tests, no rerun of historical model experiments."""
import argparse
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from lotto7x7.image_classification_v011 import (
    validate, canonical, orbit_size, morphology, classify_exact,
    similarity_five_graph, kmeans_gpu, iid_legal_images, report, html_gallery,
)
from lotto7x7.image_classification_cli_v011 import configure_parsers


def fixture(n=100, seed=7):
    rng=np.random.default_rng(seed)
    arr=iid_legal_images(n,rng)
    ids=np.arange(100000001,100000001+n,dtype=np.int64)
    return arr,ids


class ClassificationV011Tests(unittest.TestCase):
    def test_validate_six_hot_and_ids(self):
        x,ids=fixture()
        self.assertEqual(validate(x,ids)[0].shape,(100,7,7))
        bad=x.copy();bad[0]=0
        with self.assertRaises(ValueError):validate(bad,ids)
        with self.assertRaises(ValueError):validate(x,ids[::-1])

    def test_key_is_translation_d4_invariant(self):
        pts=np.array([(0,0),(0,1),(1,3),(2,0),(2,1),(2,3)])
        moved=np.rot90(np.pad(np.array([[1,1,0,0],[0,0,0,1],[1,1,0,1]],dtype=int),((0,0),(0,0)))).astype(np.uint8)
        self.assertEqual(canonical(pts),canonical(np.argwhere(moved)))
        self.assertEqual(canonical(pts),canonical(pts+5))

    def test_orbit_reference(self):
        # A straight row of six has 2 directions, 7 horizontal positions and 2 vertical positions.
        pts=np.array([(0,i) for i in range(6)])
        self.assertEqual(orbit_size(canonical(pts)),28)

    def test_exact_shape_groups_and_same_numbers(self):
        a=np.array([(0,0),(0,1),(1,3),(2,0),(2,1),(2,3)])
        x=np.zeros((3,7,7),dtype=np.uint8)
        x[0,a[:,0],a[:,1]]=1
        b=a+1
        x[1,b[:,0],b[:,1]]=1
        x[2]=x[1]
        result=classify_exact(x,np.array([1,2,3]))
        self.assertEqual(result['shape_class_count'],1)
        self.assertEqual(result['exact_ticket_unique'],2)
        self.assertEqual(result['size_histogram'],{'3':1})

    def test_morphology_mirrors_and_translations(self):
        x=np.zeros((3,7,7),dtype=np.uint8)
        pts=np.array([(0,0),(0,1),(1,3),(2,0),(2,1),(2,3)])
        x[0,pts[:,0],pts[:,1]]=1
        x[1,pts[:,0]+2,pts[:,1]+1]=1
        x[2]=np.rot90(x[1])
        features,details=morphology(x)
        np.testing.assert_allclose(features[0],features[1],atol=1e-6)
        np.testing.assert_allclose(features[0],features[2],atol=1e-6)
        self.assertEqual(details[0]['components_8'],details[2]['components_8'])

    def test_similarity_network_has_links_but_not_classes(self):
        a=np.zeros((3,7,7),dtype=np.uint8)
        a[0,0,[0,1,2,3,4,5]]=1
        a[1,0,[0,1,2,3,4,6]]=1
        a[2,2,[0,1,2,3,4,5]]=1
        s=similarity_five_graph(a)
        self.assertEqual(s['images_with_similar_neighbor'],3)
        self.assertGreaterEqual(s['pair_count'],2)
        self.assertIn('NOT pairwise',s['warning'])

    def test_iid_sampling_legal(self):
        arr=iid_legal_images(1000,np.random.default_rng(2))
        self.assertEqual(arr.shape,(1000,7,7))
        self.assertTrue(np.all(arr.sum(axis=(1,2))==6))

    def test_kmeans_clusters_and_stability(self):
        x=np.vstack((np.zeros((12,2)),np.ones((12,2))*10)).astype('float32')
        l,c,d=kmeans_gpu(x,2,device='cpu',seed=13)
        self.assertEqual(d,'cpu')
        self.assertEqual(sorted(np.bincount(l).tolist()),[12,12])
        self.assertTrue(np.array_equal(l,kmeans_gpu(x,2,device='cpu',seed=13)[0]))

    def test_full_geometry_report_and_gallery(self):
        x,ids=fixture(80)
        r=report(x,ids,clusters=5,device='cpu',iid_samples=100,seed=10)
        self.assertEqual(r['draw_count'],80)
        self.assertEqual(sum(c['count'] for c in r['cluster_summary']),80)
        self.assertEqual(sum(r['eight_neighbor_morphology']['components_histogram'].values()),80)
        self.assertEqual(len(r['assignments']),80)
        self.assertIn('<svg',html_gallery(r))
        self.assertIn('not predicted winning probabilities',html_gallery(r))

    def test_morphology_extremes(self):
        x=np.zeros((2,7,7),dtype=np.uint8)
        x[0,0,0:6]=1
        x[1,[0,0,0,3,3,3],[0,3,6,0,3,6]]=1
        _,details=morphology(x)
        self.assertEqual(details[0]['components_8'],1)
        self.assertEqual(details[0]['adjacent_pairs_8'],5)
        self.assertEqual(details[1]['components_8'],6)

    def test_frozen_encoder_loading_without_training(self):
        from lotto7x7.deep_feature_v010 import DeepFeatureNet
        x,ids=fixture(65)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'frozen.pt'
            model=DeepFeatureNet(embedding=8,hidden=8)
            torch.save({'schema':'lotto7x7.deep_feature_checkpoint.v0.10',
                        'architecture':{'embedding':8,'hidden':8,'window':64},
                        'state_dict':model.state_dict()},path)
            result=report(x,ids,clusters=3,feature_mode='cnn',checkpoint=path,
                          device='cpu',iid_samples=100,batch_size=20)
            self.assertEqual(result['number_of_classes'],3)
            self.assertEqual(result['feature_mode'],'cnn')

    def test_checkpoint_only_feature_modes(self):
        x,ids=fixture(80)
        with self.assertRaises(ValueError):report(x,ids,feature_mode='cnn',device='cpu')
        with self.assertRaises(ValueError):report(x,ids,feature_mode='fused',device='cpu')

    def test_cli_parser(self):
        p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command')
        configure_parsers(sub)
        args=p.parse_args(['image-classify','--device','cuda','--clusters','16','--features','fused','--model','results/model.pt'])
        self.assertEqual(args.clusters,16)
        self.assertEqual(args.features,'fused')


if __name__=='__main__':unittest.main()
