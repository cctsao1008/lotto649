"""New v0.12 isolated tests. The previous passed v0.8--v0.11 suites are not rerun."""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from lotto7x7.natural_clustering_v012 import (
    SCHEMA,adjusted_silhouette,device_name,feature_space,pairwise_gpu,
    ward_candidates,density_hdbscan,kmedoids,class_summary,stability_subsamples,
    report,html_gallery,
)
from lotto7x7.image_classification_v011 import iid_legal_images
from lotto7x7.natural_clustering_cli_v012 import configure_parsers,run


def sample(n=100,seed=11):
    rng=np.random.default_rng(seed)
    return iid_legal_images(n,rng),np.arange(93000001,93000001+n,dtype=np.int64)


class NaturalClusteringV012Tests(unittest.TestCase):
    def test_device_is_strict(self):
        self.assertEqual(device_name('cpu'),'cpu')
        with self.assertRaises(ValueError):device_name('tpu')
        if not torch.cuda.is_available():
            with self.assertRaises(ValueError):device_name('cuda')

    def test_feature_space_geometry_is_d4_invariant(self):
        a=np.zeros((3,7,7),dtype=np.uint8)
        a[0,0,[0,1,3]]=1;a[0,2,[1,2,3]]=1
        a[1]=np.rot90(a[0]);a[2]=np.fliplr(a[0])
        x,ref=feature_space(a,feature_mode='geometry',checkpoint=None,device='cpu')
        np.testing.assert_allclose(x[0],x[1],rtol=0,atol=1e-5)
        np.testing.assert_allclose(x[0],x[2],rtol=0,atol=1e-5)
        self.assertEqual(x.shape,(3,40))
        with self.assertRaises(ValueError):feature_space(a,feature_mode='cnn',checkpoint=None,device='cpu')

    def test_pairwise_gpu_distance_symmetry_and_duplicates(self):
        z=np.array([[0,0],[3,4],[0,0]],dtype=np.float32)
        d=pairwise_gpu(z,device='cpu',batch_size=2)
        self.assertAlmostEqual(float(d[0,1]),5,places=5)
        self.assertEqual(d[0,2],0.)
        self.assertTrue(np.array_equal(d,d.T))
        self.assertTrue(np.all(np.diag(d)==0))

    def test_ward_tests_multiple_k_and_selection_rule(self):
        rng=np.random.default_rng(2)
        x=np.concatenate([rng.normal(loc=v,scale=.05,size=(12,2)) for v in (0,3,7)]).astype(np.float32)
        d=pairwise_gpu(x,device='cpu')
        w,labels=ward_candidates(x,d,(2,3,5),sample=36,seed=5)
        self.assertEqual(w['selected_k'],3)
        self.assertEqual(sorted(labels),[2,3,5])
        self.assertEqual(len(np.unique(labels[3])),3)
        self.assertGreater(w['selected_silhouette'],.9)
        with self.assertRaises(ValueError):ward_candidates(x,d,(36,40),sample=30,seed=5)

    def test_hdbscan_finds_density_clusters_and_noise(self):
        rng=np.random.default_rng(12)
        x=np.vstack((rng.normal(size=(40,2))*.015,
                     rng.normal(loc=3,size=(40,2))*.015,
                     np.array([[100,100]],dtype=float)))
        labels,prob=density_hdbscan(pairwise_gpu(x,device='cpu'),min_cluster_size=9,min_samples=4)
        self.assertGreaterEqual(len(set(labels[labels>=0])),2)
        self.assertEqual(len(prob),len(x))
        self.assertEqual(labels[-1],-1)

    def test_hdbscan_can_return_all_noise(self):
        x=np.eye(15,dtype=np.float32)
        labels,_=density_hdbscan(pairwise_gpu(x,device='cpu'),min_cluster_size=14,min_samples=5)
        self.assertTrue(np.all(labels==-1) or len(set(labels[labels>=0]))<=1)

    def test_kmedoids_selects_actual_points(self):
        x=np.array([[0,0],[0,.1],[10,10],[10,10.1]],dtype=np.float32)
        d=pairwise_gpu(x,device='cpu')
        labels,medoids=kmedoids(d,2,seed=50)
        self.assertEqual(len(set(medoids)),2)
        self.assertEqual(sorted(np.bincount(labels).tolist()),[2,2])
        self.assertTrue(np.array_equal(labels,kmedoids(d,2,seed=50)[0]))

    def test_silhouette_rejects_one_group(self):
        d=pairwise_gpu(np.eye(7,dtype=np.float32),device='cpu')
        self.assertIsNone(adjusted_silhouette(d,np.zeros(7,dtype=int),sample=7,seed=1))

    def test_class_summary_has_true_medoid_and_no_forced_noise(self):
        x,ids=sample(12)
        feat,_=feature_space(x,feature_mode='geometry',checkpoint=None,device='cpu')
        d=pairwise_gpu(feat,device='cpu')
        labels=np.array([-1,-1,0,0,0,0,1,1,1,1,1,1])
        classes,assign=class_summary(x,ids,d,labels,method='hdbscan',gallery_count=2)
        self.assertEqual(sum(g['count'] for g in classes),10)
        self.assertEqual(sum(a['class_id']=='NOISE' for a in assign),2)
        self.assertEqual(len(classes[0]['gallery_nearest']),2)
        self.assertEqual(len(classes[0]['gallery_boundary']),2)
        self.assertIn(classes[0]['representative_draw_id'],set(ids.tolist()))

    def test_stability_is_ari_of_common_indices(self):
        rng=np.random.default_rng(44)
        x=np.vstack((rng.normal(loc=0,scale=.05,size=(40,2)),rng.normal(loc=5,scale=.05,size=(40,2)))).astype(np.float32)
        d=pairwise_gpu(x,device='cpu')
        w,labels=ward_candidates(x,d,(2,),sample=80,seed=1)
        stable=stability_subsamples(x,labels[2],method='ward',k=2,min_cluster_size=10,
                                   min_samples=4,repeats=2,fraction=.75,seed=11,device='cpu')
        self.assertEqual(stable['valid_comparisons'],2)
        self.assertGreater(stable['ari'],.95)

    def test_full_report_counts_and_full_refit_null(self):
        x,ids=sample(80)
        r=report(x,ids,feature_mode='geometry',device='cpu',k_values=(3,5,8),
                 min_cluster_size=8,min_samples=3,silhouette_sample=60,
                 stability_repeats=1,iid_replicates=2,iid_samples=100,
                 gallery_count=3,batch_size=40,seed=71)
        self.assertEqual(r['schema'],SCHEMA)
        self.assertEqual(set(r['methods']),{'ward','hdbscan','kmedoids'})
        for m in r['methods'].values():
            self.assertEqual(sum(c['count'] for c in m['classes'])+m['noise_count'],80)
            self.assertEqual(len(m['assignments']),80)
        self.assertEqual(len(r['iid_full_refit_controls']['null_summary']),2)
        self.assertTrue(0<=r['iid_full_refit_controls']['ward_selected_best_silhouette_empirical_p']<=1)
        self.assertIn('Nearest historical images',html_gallery(r))
        self.assertIn('Class boundary images',html_gallery(r))

    def test_report_no_hidden_random_change(self):
        x,ids=sample(65)
        kw=dict(device='cpu',k_values=(3,5),min_cluster_size=6,min_samples=3,
                silhouette_sample=60,stability_repeats=1,iid_replicates=1,
                iid_samples=100,gallery_count=2,seed=649)
        a=report(x,ids,**kw);b=report(x,ids,**kw)
        self.assertEqual(a['ward_model_selection']['selected_k'],b['ward_model_selection']['selected_k'])
        self.assertEqual(a['methods']['ward']['assignments'],b['methods']['ward']['assignments'])

    def test_output_validation(self):
        x,ids=sample(55)
        with self.assertRaises(ValueError):report(x,ids,device='cpu',k_values=(90,),iid_replicates=0)
        with self.assertRaises(ValueError):report(x,ids,device='cpu',k_values=(3,),min_samples=200)

    def test_cli_parser_paths_and_modes(self):
        p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command');configure_parsers(sub)
        a=p.parse_args(['natural-cluster-gpu','--device','cuda','--features','fused',
                        '--k-values','4','8','16','--model','results/frozen.pt'])
        self.assertEqual(a.k_values,[4,8,16]);self.assertEqual(a.features,'fused')
        with tempfile.TemporaryDirectory() as folder:
            existing=Path(folder)/'x.json';existing.write_text('{}')
            a.command='natural-cluster-gpu';a.out=existing;a.csv_out=None;a.html_out=None
            with self.assertRaises(FileExistsError):run(a)

    def test_frozen_cnn_checkpoint_without_training(self):
        from lotto7x7.deep_feature_v010 import DeepFeatureNet
        x,ids=sample(50)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'model.pt'
            net=DeepFeatureNet(embedding=8,hidden=8)
            torch.save({'schema':'lotto7x7.deep_feature_checkpoint.v0.10',
                        'architecture':{'embedding':8,'hidden':8,'window':64},
                        'state_dict':net.state_dict()},p)
            z,_=feature_space(x,feature_mode='fused',checkpoint=p,device='cpu',batch_size=16)
            self.assertEqual(z.shape,(50,48))


if __name__=='__main__':unittest.main()
