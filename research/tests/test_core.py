"""New 7x7 pixel and spectral mathematics tests only."""
import unittest
import numpy as np
from lotto7x7.core import (decode, encode, iid_expected, iid_images, invert_spectrum,
                          layout, signature, similarity, spectrum, validate)


class ImageTests(unittest.TestCase):
    def test_mapping_round_trip(self):
        for kind in ("sequential", "snake", "permuted"):
            for six in ([1,2,3,4,5,6], [5,9,26,32,41,48]):
                np.testing.assert_equal(decode(encode(np.array(six), mapping=kind), mapping=kind), six)

    def test_exactly_six_pixels(self):
        image = encode(np.array([1,8,15,22,29,36]))
        self.assertEqual(int(image.sum()), 6)
        self.assertEqual(image[:,0].tolist(), [1,1,1,1,1,1,0])

    def test_bad_six_rejected(self):
        for draw in ([1,2,3,4,5,5], [0,2,3,4,5,6], [1,2,3,4,5,50], [1,2,3,4,5]):
            with self.assertRaises(ValueError): encode(np.asarray(draw))

    def test_bad_image_rejected(self):
        for image in (np.zeros((7,7)),np.ones((7,7)),np.full((7,7),np.nan)):
            with self.assertRaises(ValueError): validate(image)

    def test_fft_and_dct_are_reversible(self):
        image = encode(np.array([5,9,26,32,41,48]))
        for domain in ("dct", "fft"):
            np.testing.assert_allclose(invert_spectrum(spectrum(image,domain),domain),image,atol=1e-12)

    def test_same_image_score_one(self):
        image = encode(np.array([1,2,3,8,9,15]))
        self.assertAlmostEqual(similarity(image,image)['composite'],1.0)

    def test_same_shape_different_numbers(self):
        a=encode(np.array([1,2,3,8,9,15]))
        b=encode(np.array([25,26,27,32,33,39]))
        self.assertEqual(similarity(a,b)['pixels'],0.0)
        self.assertEqual(similarity(a,b)['shape'],1.0)

    def test_symmetry_bounds(self):
        rng=np.random.default_rng(123)
        masks=[encode(rng.choice(np.arange(1,50),6,replace=False)) for _ in range(12)]
        for a,b in zip(masks[:-1], masks[1:]):
            p,q=similarity(a,b),similarity(b,a)
            for name in p:
                self.assertTrue(0<=p[name]<=1)
                self.assertAlmostEqual(p[name],q[name])

    def test_iid_conditional_baseline_reproducible(self):
        controls=iid_images(25,seed=5)
        np.testing.assert_array_equal(controls,iid_images(25,seed=5))
        self.assertTrue(0<=iid_expected(encode(np.array([1,2,3,4,5,6])),controls)<=1)

    def test_signature_finite(self):
        x=signature(encode(np.array([1,2,3,4,5,6])))
        self.assertEqual(x.shape,(24,))
        self.assertTrue(np.isfinite(x).all())


if __name__=='__main__': unittest.main()
