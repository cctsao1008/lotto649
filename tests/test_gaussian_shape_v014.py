"""Only new v0.14 tests; no old GPU backtests or full regression suite."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lotto7x7.gaussian_shape_v014 import (
  SCHEMA, analyze, kde_density, density_metrics, density_modes,
  mds_embedding, gaussian_mixture, matched_null_controls, load_existing, html_gallery)


def points_distance(n=20,seed=649):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(n,3))
    return np.linalg.norm(x[:,None,:]-x[None,:,:],axis=-1).astype(np.float32)


class TestGaussianShape(unittest.TestCase):
    def test_kde_matches_manual_excluding_self(self):
        x=np.array([[0.,1.,2.],[1.,0.,3.],[2.,3.,0.]],dtype=np.float32)
        y=kde_density(x,[1.],device='cpu')[:,0]
        np.testing.assert_allclose(y,[ (np.exp(-.5)+np.exp(-2))/2,
                 (np.exp(-.5)+np.exp(-4.5))/2,
                 (np.exp(-2)+np.exp(-4.5))/2],rtol=1e-6)

    def test_kde_multiscale_increases_with_bandwidth(self):
        d=points_distance();y=kde_density(d,[.2,.5,1.5],device='cpu')
        self.assertEqual(y.shape,(20,3))
        self.assertTrue(np.all(np.diff(y,axis=1)>=0))

    def test_reject_invalid_bandwidth(self):
        with self.assertRaises(ValueError):kde_density(points_distance(),[0.])

    def test_density_metrics(self):
        r=density_metrics(np.array([.1,.2,.3]))
        self.assertAlmostEqual(r['mean'],.2)
        self.assertLess(r['p90'],r['max'])

    def test_density_modes_reject_outliers(self):
        d=points_distance(20);v=np.linspace(1,.1,20)
        x=density_modes(d,v,peak_radius=.3,assignment_radius=.05,max_peaks=3)
        self.assertLessEqual(len(x['peaks']),3)
        self.assertTrue(x['unassigned']>0)
        for k,p in enumerate(x['peaks']):self.assertEqual(x['labels'][p],k)

    def test_mds_euclidean(self):
        d=points_distance(28)
        emb,diag=mds_embedding(d,3,seed=7)
        self.assertEqual(emb.shape,(28,3))
        self.assertLess(diag['sample_relative_stress'],1e-4)

    def test_mds_reject_oversized(self):
        with self.assertRaises(ValueError):mds_embedding(points_distance(12),12)

    def test_gaussian_mixture_bic_soft_labels(self):
        d=points_distance(36)
        emb,diag=mds_embedding(d,3)
        r=gaussian_mixture(emb,d,[1,2,3],seed=7,n_init=1)
        self.assertIn(r['best_k'],(1,2,3))
        self.assertEqual(r['posterior'].shape,(36,r['best_k']))
        np.testing.assert_allclose(r['posterior'].sum(axis=1),1,atol=1e-5)
        self.assertEqual(sum(c['count'] for c in r['classes']),36)

    def test_fair_null_control_real_legal(self):
        d=points_distance(16)
        result=matched_null_controls(d,.6,draws=8,repeats=1,device='cpu',batch_size=16)
        self.assertEqual(len(result['replicates']),1)
        self.assertTrue(np.isfinite(result['mean_delta_cv']))

    def test_zero_controls_without_null_recalculation(self):
        d=points_distance(16)
        result=matched_null_controls(d,.6,draws=8,repeats=0,device='cpu')
        self.assertIsNone(result['mean_delta_cv'])

    def test_loader_integrity_and_alignment(self):
        with tempfile.TemporaryDirectory() as t:
            a=Path(t)/'d.npz';b=Path(t)/'d.csv';ids=np.arange(10,18)
            np.savez(a,draw_ids=ids,shape=points_distance(8))
            with b.open('w',newline='',encoding='utf-8') as f:
                w=csv.writer(f);w.writerow(['draw_id',*[f'n{i}' for i in range(1,7)]])
                for idx in ids[::-1]:w.writerow([idx,1,2,3,4,5,int(idx-3)])
            m,newids,ns=load_existing(a,b)
            self.assertEqual(ns[0].tolist(),[1,2,3,4,5,7])
            self.assertEqual(newids[0],10)
            self.assertEqual(m.shape,(8,8))

    def test_loader_reject_ids_mismatch(self):
        with tempfile.TemporaryDirectory() as t:
            a=Path(t)/'d.npz';b=Path(t)/'d.csv'
            np.savez(a,draw_ids=np.arange(10,18),shape=points_distance(8))
            with b.open('w',newline='',encoding='utf-8') as f:
                w=csv.writer(f);w.writerow(['draw_id',*[f'n{i}' for i in range(1,7)]])
                for idx in range(11,19):w.writerow([idx,1,2,3,4,5,6])
            with self.assertRaises(ValueError):load_existing(a,b)

    def test_full_analysis_and_html(self):
        n=20;d=points_distance(n)
        ids=np.arange(93000001,93000001+n)
        ns=np.tile([1,5,11,19,26,45],(n,1))
        report,arrays=analyze(d,ids,ns,device='cpu',bandwidths=[.4,.6],
          mode_bandwidth=.6,max_peaks=5,mds_dimensions=3,components=[1,2],
          control_draws=8,control_replicates=0,seed=649)
        self.assertEqual(report['schema'],SCHEMA)
        self.assertEqual(len(arrays['gmm_labels']),n)
        self.assertEqual(len(arrays['mode_labels']),n)
        self.assertIn('svg',html_gallery(report,arrays,gallery_count=3))

    def test_repeatable_analysis(self):
        n=20;d=points_distance(n);ids=np.arange(93000001,93000001+n)
        ns=np.tile([1,5,11,19,26,45],(n,1))
        opts=dict(device='cpu',bandwidths=[.6],mode_bandwidth=.6,
            max_peaks=4,mds_dimensions=3,components=[1,2],control_draws=8,
            control_replicates=0,seed=649)
        ra,aa=analyze(d,ids,ns,**opts)
        rb,ab=analyze(d,ids,ns,**opts)
        np.testing.assert_allclose(aa['density'],ab['density'])
        np.testing.assert_array_equal(aa['gmm_labels'],ab['gmm_labels'])

if __name__=='__main__':unittest.main()
