"""v0.13-only tests: distances, geometric invariance, IID controls and reports."""
import unittest

import numpy as np
from scipy.optimize import linear_sum_assignment
from lotto7x7.pairwise_similarity_v013 import (
    SCHEMA, MODES, point_coordinates, pair_distances, pairwise_matrices,
    _d4_coordinates, _threshold_classes, _closest_pairs, iid_reference_pairs,
    distance_stats, analyze, html_gallery,
)
from lotto7x7.image_classification_v011 import iid_legal_images


def images_from_points(*items):
    x=np.zeros((len(items),7,7),dtype=np.uint8)
    for i,pts in enumerate(items):
        for r,c in pts:x[i,r,c]=1
    return x


class SimilarityV013Tests(unittest.TestCase):
    def test_legal_coordinate_extraction(self):
        x=images_from_points([(0,i) for i in range(6)],[(i,0) for i in range(6)])
        p=point_coordinates(x)
        self.assertEqual(p.shape,(2,6,2))
        self.assertTrue(np.all(p[0,:,0]==0))
        with self.assertRaises(ValueError):point_coordinates(np.ones((2,7,7),dtype=np.uint8))

    def test_exact_best_assignment_against_scipy_hungarian(self):
        rng=np.random.default_rng(14)
        draws=iid_legal_images(8,rng)
        p=point_coordinates(draws)
        ds=pair_distances(p[:4],p[4:],device='cpu',batch_size=2)
        for i in range(4):
            costs=np.sqrt(((p[i,:,None,:]-p[4+i,None,:,:])**2).sum(axis=2))
            # The objective is squared Euclidean matching, not sum of norms.
            squared=((p[i,:,None,:]-p[4+i,None,:,:])**2).sum(axis=2)
            rows,cols=linear_sum_assignment(squared)
            self.assertAlmostEqual(float(ds['position'][i]),float(np.sqrt(squared[rows,cols].sum()/6)),places=4)

    def test_identical_and_reversed_inputs(self):
        rng=np.random.default_rng(122)
        p=point_coordinates(iid_legal_images(5,rng))
        z=pair_distances(p,p,device='cpu')
        self.assertTrue(all(np.all(v<1e-4) for v in z.values()))
        a=pair_distances(p[:3],p[1:4],device='cpu')
        b=pair_distances(p[1:4],p[:3],device='cpu')
        for name in MODES:np.testing.assert_allclose(a[name],b[name],atol=3e-4)

    def test_translation_invariant_shape_not_position(self):
        p=np.array([(0,0),(0,1),(1,3),(2,0),(2,1),(2,3)])
        x=images_from_points(p,p+2)
        q=point_coordinates(x)
        ds=pair_distances(q[:1],q[1:],device='cpu')
        self.assertGreater(float(ds['position'][0]),0.1)
        self.assertLess(float(ds['shape'][0]),1e-4)
        self.assertLess(float(ds['shape_hausdorff'][0]),1e-4)

    def test_d4_rotation_invariance_no_translations(self):
        p=np.array([(0,0),(0,1),(1,3),(2,0),(2,1),(2,3)])
        r=np.column_stack((p[:,1],6-p[:,0]))
        x=images_from_points(p,r)
        v=point_coordinates(x)
        ds=pair_distances(v[:1],v[1:],device='cpu')
        self.assertGreater(float(ds['position'][0]),0.01)
        self.assertLess(float(ds['d4'][0]),1e-4)
        self.assertLess(float(ds['shape'][0]),1e-4)

    def test_strict_nonnegative_order_and_diagonal(self):
        rng=np.random.default_rng(11)
        pts=point_coordinates(iid_legal_images(11,rng))
        matrices=pairwise_matrices(pts,device='cpu',batch_size=8)
        for name,m in matrices.items():
            self.assertEqual(m.shape,(11,11))
            np.testing.assert_allclose(m,m.T,atol=1e-4)
            self.assertTrue(np.all(np.diag(m)==0))
            self.assertTrue(np.all(m>=0))
        self.assertTrue(np.all(matrices['position']+1e-4>=matrices['d4']))
        self.assertTrue(np.all(matrices['d4']+1e-4>=matrices['shape']))

    def test_symmetry_of_hausdorff(self):
        p=point_coordinates(iid_legal_images(8,np.random.default_rng(2)))
        a=pair_distances(p[:4],p[4:],device='cpu')
        b=pair_distances(p[4:],p[:4],device='cpu')
        np.testing.assert_allclose(a['shape_hausdorff'],b['shape_hausdorff'],atol=1e-4)

    def test_iid_null_legal_and_finite(self):
        refs=iid_reference_pairs(100,seed=17,device='cpu',batch_size=16)
        for name in MODES:
            self.assertEqual(refs[name].shape,(100,))
            self.assertTrue(np.isfinite(refs[name]).all())
            self.assertTrue(np.all(refs[name]>=0))

    def test_complete_linkage_class_diameter_bounded(self):
        x=images_from_points([(0,i) for i in range(6)],[(1,i) for i in range(6)],
            [(0,i) for i in range(5)]+[(6,6)],[(i,0) for i in range(6)])
        pts=point_coordinates(x)
        d=pairwise_matrices(pts,device='cpu',batch_size=8,modes=('shape',))['shape']
        result,labels=_threshold_classes(d,np.arange(1,5),pts,[(.1,.01),(.5,1.0)],gallery_count=2)
        self.assertEqual(result[0]['multi_member_images'],3)
        self.assertEqual(result[0]['singleton_images'],1)
        self.assertTrue(all(g['diameter']<.01+1e-4 for g in result[0]['classes_shown']))
        self.assertEqual(len(labels['0.1']),4)

    def test_identical_shape_not_identical_numbers(self):
        p=np.array([(0,0),(0,1),(0,2),(1,0),(1,1),(2,0)])
        pts=point_coordinates(images_from_points(p,p+4))
        d=pairwise_matrices(pts,device='cpu',modes=('shape',))['shape']
        best=_closest_pairs(d,pts,np.array([19,20]),top=1)
        self.assertEqual(best[0]['shared_number_count'],0)
        self.assertAlmostEqual(best[0]['distance'],0.,places=4)

    def test_full_report_smoke_with_controls(self):
        x=iid_legal_images(30,np.random.default_rng(23))
        ids=np.arange(90000001,90000031)
        reports,matrices=analyze(x,ids,device='cpu',batch_size=16,iid_pairs=150,
            iid_quantiles=(.01,.05),control_draws=15,control_replicates=1,
            top_pairs=3,gallery_count=2,seed=11)
        self.assertEqual(reports['schema'],SCHEMA)
        self.assertEqual(reports['unordered_pair_count'],435)
        self.assertTrue(all(reports['sanity'].values()))
        self.assertEqual(len(reports['threshold_classes']),2)
        self.assertEqual(len(reports['iid_matched_subset_refits']),1)
        self.assertIn('shape_hausdorff',matrices)
        self.assertIn('Common actual numbers',html_gallery(reports))

    def test_invalid_mode_and_quantiles_rejected(self):
        x=iid_legal_images(20,np.random.default_rng(13))
        ids=np.arange(1,21)
        for kwargs in [{'cluster_mode':'not_a_distance'},{'iid_quantiles':(0.,.01)},
                       {'control_draws':21},{'iid_pairs':1}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                analyze(x,ids,device='cpu',**kwargs)

    def test_fixed_seed_for_iid(self):
        a=iid_reference_pairs(120,seed=3,device='cpu',batch_size=24)
        b=iid_reference_pairs(120,seed=3,device='cpu',batch_size=17)
        for mode in MODES:np.testing.assert_allclose(a[mode],b[mode],atol=1e-4)

    def test_distance_statistics_thresholds_monotonic(self):
        x=iid_legal_images(8,np.random.default_rng(101))
        m=pairwise_matrices(point_coordinates(x),device='cpu',batch_size=8,modes=('shape',))['shape']
        null=iid_reference_pairs(120,seed=4,device='cpu',batch_size=32,modes=('shape',))['shape']
        s=distance_stats(m,iid_distances=null,iqs=(.001,.01,.1))
        self.assertEqual(s['pair_count'],28)
        self.assertEqual(len(s['predeclared_iid_thresholds']),3)
        self.assertLessEqual(s['predeclared_iid_thresholds'][0]['distance'],s['predeclared_iid_thresholds'][1]['distance'])

    def test_shape_overlay_best_alignment_and_gallery(self):
        from lotto7x7.pairwise_similarity_v013 import _aligned_shape_pair, _overlay_svg
        p=np.array([(0,0),(0,1),(0,2),(1,0),(1,1),(2,0)])
        q=p+4
        a=_aligned_shape_pair(p.astype(np.float32),q.astype(np.float32))
        self.assertLess(a['distance'],1e-8)
        self.assertEqual(len(a['aligned_points']),6)
        self.assertIn('Aligned', _overlay_svg(a).replace('shape-aligned', 'Aligned'))

    def test_cuda_flag_failure_when_unavailable(self):
        import torch
        if not torch.cuda.is_available():
            pts=point_coordinates(iid_legal_images(2,np.random.default_rng(10)))
            with self.assertRaises(ValueError):pair_distances(pts[:1],pts[1:],device='cuda')


if __name__=='__main__':unittest.main()
