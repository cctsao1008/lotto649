"""Focused P0 Gaussian validation tests; do not re-run legacy prediction suites."""
from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest
import numpy as np

from lotto7x7.gaussian_validation_v0141 import (
 SCHEMA,validate_parameters,fit_gmm_grid,embed_once,dimension_comparison,
 bandwidth_comparison,iid_distance_sample,matched_controls,classification_stability,
 analyze,csv_records,html_report,full_size_iid_controls)
from lotto7x7.gaussian_validation_cli_v0141 import configure_parsers
from lotto7x7.gaussian_shape_v014 import load_existing
from lotto7x7.image_classification_v011 import iid_legal_images
from lotto7x7.pairwise_similarity_v013 import point_coordinates


def euclidean_distance(n=44,seed=7):
    x=np.random.default_rng(seed).normal(size=(n,5))
    d=np.linalg.norm(x[:,None]-x[None,:],axis=-1).astype(np.float32)
    np.fill_diagonal(d,0)
    return d


class TestBatchP0(unittest.TestCase):
    def test_schema(self):self.assertEqual(SCHEMA,'lotto7x7.gaussian_shape_validation.v0.14.1')

    def test_invalid_nonsymmetric(self):
        x=euclidean_distance();x[1,0]=99
        with self.assertRaises(ValueError):validate_parameters(x,[2,3],[1,2],[.5],24,1,0,[1,2],.8,3,8)

    def test_invalid_component_grid(self):
        with self.assertRaises(ValueError):validate_parameters(euclidean_distance(),[2,3],[2,3],[.5],24,1,0,[1,2],.8,3,8)

    def test_invalid_dimension_vs_sample(self):
        with self.assertRaises(ValueError):validate_parameters(euclidean_distance(),[2,23],[1,2],[.5],24,1,0,[1,2],.8,2,8)

    def test_bad_bandwidth(self):
        with self.assertRaises(ValueError):validate_parameters(euclidean_distance(),[2,3],[1,2],[0],24,1,0,[1,2],.8,2,8)

    def test_duplicate_stability_seed_rejected(self):
        with self.assertRaises(ValueError):validate_parameters(euclidean_distance(),[2,3],[1,2],[.5],24,1,0,[1,1],.8,2,8)

    def test_gmm_bic_table_and_labels(self):
        x=np.random.default_rng(2).normal(size=(48,3))
        f=fit_gmm_grid(x,[1,2,3],seed=4,n_init=1)
        self.assertEqual(len(f['bic_table']),3)
        self.assertIn(f['best_k'],(1,2,3))
        self.assertEqual(len(f['labels']),48)
        self.assertTrue(np.isfinite(f['mean_confidence']))

    def test_gmm_reject_grid_above_n_over_4(self):
        with self.assertRaises(ValueError):fit_gmm_grid(np.ones((24,2)),[1,10],seed=1)

    def test_mds_reuses_one_basis_for_different_dims(self):
        d=euclidean_distance()
        emb,rows,fits=dimension_comparison(d,[2,4],[1,2],seed=7,n_init=1)
        self.assertEqual(emb.shape[1],4)
        self.assertEqual([z['dimensions'] for z in rows],[2,4])
        self.assertLess(rows[1]['relative_stress'],rows[0]['relative_stress']+1e-3)

    def test_bandwidth_rank_self_match(self):
        d=euclidean_distance()
        scores,rows=bandwidth_comparison(d,[.4,1.],device='cpu',peak_radius=.8,
                                 assignment_radius=1.2,max_peaks=10)
        self.assertEqual(scores.shape,(44,2))
        self.assertAlmostEqual(rows[0]['spearman_vs_smallest_bandwidth'],1.)
        self.assertAlmostEqual(rows[0]['top10pct_jaccard_vs_smallest'],1.)
        self.assertTrue(rows[1]['density']['mean']>=rows[0]['density']['mean'])

    def test_legal_iid_sample_is_six_hot(self):
        rng=np.random.default_rng(7)
        x=iid_legal_images(24,rng)
        self.assertTrue(np.all(x.sum((1,2))==6))
        self.assertTrue(np.all((x==0)|(x==1)))

    def test_independent_iid_distance_symmetric(self):
        d=iid_distance_sample(24,rng=np.random.default_rng(10),device='cpu',batch_size=32)
        self.assertEqual(d.shape,(24,24))
        self.assertTrue(np.allclose(d,d.T))
        self.assertTrue(np.allclose(np.diag(d),0))

    def test_matched_iid_identical_n_dim_k_grid(self):
        d=euclidean_distance()
        runs=matched_controls(d,[3],[1,2,3],[.4,.7],matched_size=24,
                  repeats=1,device='cpu',batch_size=32,seed=7,n_init=1)
        a,b=runs[0]['cohorts']['historical'],runs[0]['cohorts']['iid']
        self.assertEqual(a['n'],b['n'])
        self.assertEqual(a['mds_dimensions'],b['mds_dimensions'])
        self.assertEqual([x['k'] for x in a['bic_table']],[x['k'] for x in b['bic_table']])
        self.assertEqual(list(a['density']),list(b['density']))

    def test_stability_permutation_invariant(self):
        emb=np.random.default_rng(44).normal(size=(84,4)).astype(np.float32)
        o=classification_stability(emb,2,fraction=.75,seeds=[9,10],dimension=4,n_init=1)
        self.assertEqual(len(o['runs']),2)
        self.assertTrue(all(-1<=z['ari']<=1 and 0<=z['nmi']<=1.01 for z in o['runs']))
        self.assertTrue(all(z['intersection_size']>20 for z in o['runs']))

    def test_stability_deterministic(self):
        emb=np.random.default_rng(44).normal(size=(84,4)).astype(np.float32)
        a=classification_stability(emb,2,fraction=.75,seeds=[9,10],dimension=4,n_init=1)
        b=classification_stability(emb,2,fraction=.75,seeds=[9,10],dimension=4,n_init=1)
        self.assertEqual(a,b)

    def test_full_iid_controls_zero_is_safe(self):
        x=full_size_iid_controls({'best_k':1,'bic_gain_vs_k1':0.},44,[2],[1,2],[.5],
            repeats=0,seed=7,device='cpu',batch_size=32,n_init=1)
        self.assertEqual(x,[])

    def test_full_size_iid_nonzero_equal_comparison_n(self):
        f={'best_k':2,'bic_gain_vs_k1':3.}
        rows=full_size_iid_controls(f,24,[2],[1,2],[.5],repeats=1,
             seed=649,device='cpu',batch_size=32,n_init=1)
        self.assertEqual(rows[0]['n'],24)
        self.assertEqual(rows[0]['mds_dimensions'],2)
        self.assertEqual([x['k'] for x in rows[0]['iid_bic_table']],[1,2])
        self.assertEqual(rows[0]['historical_best_k'],2)
        self.assertIn(rows[0]['iid_best_k'],(1,2))

    def test_complete_small_batch_generates_all_sections(self):
        d=euclidean_distance(44)
        ids=np.arange(93000001,93000045)
        ns=np.tile([1,8,15,22,29,36],(44,1))
        report,arrays=analyze(d,ids,ns,device='cpu',dimensions=[2,3],
            comparison_dimension=3,components=[1,2,3],bandwidths=[.4,.8],
            matched_size=24,matched_replicates=1,full_iid_replicates=0,
            stability_seeds=[7,8],stability_fraction=.75,max_peaks=8,
            n_init=1,batch_size=32,seed=7)
        self.assertEqual(report['schema'],SCHEMA)
        self.assertEqual(len(report['matched_iid_controls']),1)
        self.assertEqual(len(report['dimension_sensitivity']),2)
        self.assertEqual(len(report['bandwidth_sensitivity']),2)
        self.assertEqual(arrays['embedding_maxdim'].shape,(44,3))
        self.assertIn('Matched-size',html_report(report))
        self.assertIn('stability',html_report(report).lower())
        self.assertGreater(len(list(csv_records(report))),5)

    def test_cli_registration(self):
        import argparse
        p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command')
        configure_parsers(sub)
        args=p.parse_args(['shape-gaussian-validate','--device','cpu','--full-iid-replicates','0'])
        self.assertEqual(args.comparison_dimension,8)
        self.assertEqual(args.full_iid_replicates,0)

    def test_loader_identifies_csv_npz_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz=Path(tmp)/'a.npz';csvpath=Path(tmp)/'b.csv'
            np.savez_compressed(npz,shape=euclidean_distance(44),draw_ids=np.arange(44)+100)
            with csvpath.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['draw_id',*['n'+str(i) for i in range(1,7)]])
                for i in range(44):w.writerow([i+101,1,8,15,22,29,36])
            with self.assertRaises(ValueError):load_existing(npz,csvpath)


if __name__=='__main__':unittest.main()
