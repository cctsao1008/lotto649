"""Focused v0.9 tests only; no reruns of frozen 216 previous cases."""
import argparse
import unittest
import numpy as np
import torch

from lotto7x7.feature_discovery_v09 import (
    validate_images,_dct_basis,_feature_bank,_scan_periods,iid_six_hot,
    shuffled_history,discover,resolve_device,
)
from lotto7x7.feature_discovery_cli_v09 import configure_parsers


def sample(n=150,seed=8):
    rng=np.random.default_rng(seed)
    a=np.zeros((n,49),dtype=np.uint8)
    for i in range(n):
        a[i,rng.choice(49,6,replace=False)]=1
    return a.reshape(n,7,7)


class FeatureDiscoveryV09Tests(unittest.TestCase):
    def test_requires_legal_images(self):
        x=sample()
        np.testing.assert_array_equal(validate_images(x),x)
        x[20,0,0]=2
        with self.assertRaises(ValueError):validate_images(x)

    def test_dct_basis_is_orthogonal(self):
        k=_dct_basis(device=torch.device('cpu'))
        torch.testing.assert_close(k@k.T,torch.eye(49),atol=1e-5,rtol=1e-5)

    def test_feature_families_and_finite(self):
        b,names,fams=_feature_bank(torch.as_tensor(sample(90),dtype=torch.float32)[None])
        self.assertEqual(b.shape,(1,88,len(names)))
        self.assertEqual(fams.count('raw'),49)
        self.assertEqual(fams.count('signed_diff'),49)
        self.assertEqual(fams.count('central_second_diff'),49)
        self.assertEqual(fams.count('spatial_dct'),48)
        self.assertEqual(fams.count('spatial_fft_power'),27)
        self.assertEqual(fams.count('pair_motif'),15)
        self.assertTrue(torch.isfinite(b).all())

    def test_historical_triplet_forward_and_backward_diff(self):
        x=torch.as_tensor(sample(80),dtype=torch.float32)
        feats,names,_=_feature_bank(x[None])
        a=x.reshape(80,49)
        for i in range(49):
            self.assertEqual(float(feats[0,0,names.index(f'signed_diff/number_{i+1:02}')]),float(a[2,i]-a[1,i]))
            self.assertEqual(float(feats[0,0,names.index(f'central_second_diff/number_{i+1:02}')]),float(a[2,i]-2*a[1,i]+a[0,i]))

    def test_iid_samples_are_uniform_legal_masks(self):
        dev=torch.device('cpu')
        a=iid_six_hot(3,41,generator=torch.Generator(device=dev).manual_seed(99),device=dev)
        b=iid_six_hot(3,41,generator=torch.Generator(device=dev).manual_seed(99),device=dev)
        self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.all(a.sum((-1,-2))==6))

    def test_shuffle_preserves_complete_draw_multiset(self):
        a=torch.as_tensor(sample(80),dtype=torch.float32)
        s=shuffled_history(a,3,generator=torch.Generator().manual_seed(11))
        original=sorted(tuple(x.flatten().tolist()) for x in a)
        for row in s:
            self.assertEqual(original,sorted(tuple(x.flatten().tolist()) for x in row))

    def test_finite_z_spectrum_finds_constructed_period(self):
        n=256
        t=torch.arange(n,dtype=torch.float32)
        feature=(torch.sin(2*torch.pi*t/8.) + .03*torch.sin(2*torch.pi*t/13.)).reshape(1,n,1)
        stats,winners=_scan_periods(feature,['synthetic'],recent_windows=(128,),
                                    half_lives=(0,32),min_period=2,max_period=64,top_count=10)
        self.assertEqual(stats.shape,(1,))
        self.assertAlmostEqual(winners[0]['period_draws'],8.,delta=.02)

    def test_null_controls_and_heldout_excluded(self):
        x=sample(115)
        r=discover(x,np.arange(1,116),device='cpu',holdout_draws=20,
                   surrogates=9,surrogate_batch=3,recent_windows=(32,64),
                   half_lives=(0,16),max_period=32,top_count=5,seed=22)
        self.assertEqual(r['dataset']['discovery_prefix_draws'],95)
        self.assertEqual(r['dataset']['holdout_suffix_draws'],20)
        self.assertEqual(len(r['top_candidates']),5)
        for null in r['null_controls'].values():
            self.assertGreaterEqual(null['global_max_p_for_best_observed'],.1)
        with self.assertRaises(ValueError):
            discover(x,np.arange(1,116),holdout_draws=100, surrogates=9)

    def test_periodic_legal_image_sequence_gives_large_signal(self):
        n=120
        x=np.zeros((n,49),dtype=np.float32)
        for i in range(n):
            x[i,((np.arange(6)*7+i%8*5)%49).astype(int)]=1
        images=x.reshape(n,7,7)
        result=discover(images,np.arange(1,n+1),device='cpu',holdout_draws=0,
                        surrogates=19,surrogate_batch=4,recent_windows=(64,),
                        half_lives=(0,),min_period=2,max_period=40,top_count=4,seed=9)
        self.assertGreater(result['periodicity_scan']['global_observed_max'],.2)
        self.assertLessEqual(result['null_controls']['iid_legal_6of49']['global_max_p_for_best_observed'],.1)

    def test_cli_options(self):
        p=argparse.ArgumentParser()
        sub=p.add_subparsers(dest='command')
        configure_parsers(sub)
        args=p.parse_args(['feature-discover-gpu','--device','cuda','--surrogates','199',
                           '--holdout-draws','500','--out','results/v09.json'])
        self.assertEqual(args.surrogates,199)
        self.assertEqual(args.device,'cuda')

    def test_wrong_device(self):
        with self.assertRaises(ValueError):resolve_device('apple')


if __name__=='__main__':unittest.main()
