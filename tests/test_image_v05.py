"""Image-first v0.5: morphology, spectrum, causal forecast, and mapping controls."""
from __future__ import annotations

from unittest.mock import patch
import unittest
import numpy as np
import torch

from lotto7x7.core import encode, decode, iid_images
from lotto7x7.image_structure_v05 import (
    SCHEMA as STRUCTURE_SCHEMA, MORPHOLOGY_NAMES, SPECTRAL_NAMES,
    check_images, map_images, describe_images, temporal_null, image_study,
    image_gallery,
)
from lotto7x7.image_compare_v05 import (
    SCHEMA as COMPARE_SCHEMA, MODELS, exact_iid_expected_jaccard,
    standardized_features, prepare_windows, train_image_arm,
    infer_image_arm, score_image_logits, image_compare, image_compare_gallery,
)


class ImageResearchV05Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.images = iid_images(120, seed=317)
        cls.ids = np.arange(107000001,107000121,dtype=np.int64)

    def test_invalid_image_cardinality_and_shapes(self):
        with self.assertRaises(ValueError):
            check_images(np.zeros((20,7,7)))
        with self.assertRaises(ValueError):
            check_images(np.zeros((7,7)))
        with self.assertRaises(ValueError):
            check_images(np.ones((10,7,7)))

    def test_mapping_roundtrip_and_pixel_overlap_layout_invariance(self):
        original = self.images
        sample = np.sum(original[1:] * original[:-1], axis=(1,2))
        for kind in ("sequential","snake","permuted"):
            x = map_images(original,kind=kind,seed=21)
            self.assertEqual(x.shape,original.shape)
            from lotto7x7.core import layout
            inverse = np.argsort(layout(kind,21).ravel())
            recovered = x.reshape(-1,49)[:,inverse].reshape(-1,7,7)
            np.testing.assert_array_equal(recovered, original)
            np.testing.assert_array_equal(
                np.sum(x[1:]*x[:-1],axis=(1,2)),sample)
        with self.assertRaises(ValueError):
            map_images(original,kind="unknown")

    def test_morphology_and_spectra_have_finite_fixed_dimensions(self):
        desc = describe_images(self.images)
        self.assertEqual(desc["morphology"].shape,
                         (120,len(MORPHOLOGY_NAMES)))
        self.assertEqual(desc["spectral"].shape,
                         (120,len(SPECTRAL_NAMES)))
        self.assertTrue(np.isfinite(desc["morphology"]).all())
        self.assertTrue(np.isfinite(desc["spectral"]).all())
        self.assertTrue(np.all(desc["spectral"][:,8] >= 0))
        self.assertTrue(np.all(desc["spectral"][:,8] <= 1+1e-10))
        self.assertTrue(np.all(desc["spectral"][:,9] >= 0))
        self.assertTrue(np.all(desc["spectral"][:,9] <= 1+1e-10))

    def test_components_and_point_geometry(self):
        a = np.stack([encode(np.array([1,2,3,4,5,6]))] * 2)
        d = describe_images(a)["morphology"][0]
        self.assertEqual(d[7], 1/6)  # all six points connected (8-neighborhood)
        self.assertEqual(d[5], 6/49)  # one row by six columns
        self.assertEqual(d[6], 5/15)  # adjacent edges in one row

    def test_lag_null_reproducible_within_year_only(self):
        years = np.array([2018]*60+[2019]*60)
        a=temporal_null(self.images,years,lags=(1,2),permutations=19,seed=42)
        b=temporal_null(self.images,years,lags=(1,2),permutations=19,seed=42)
        self.assertEqual(a,b)
        self.assertEqual(a["1"]["pairs"],118)
        self.assertIn("bonferroni_p",a["1"]["low_dct_cosine"])
        with self.assertRaises(ValueError):
            temporal_null(self.images,years[:-1],permutations=19)

    def test_periodic_order_null_detects_big_image_similarity_signal(self):
        samples=np.stack([encode(np.array(a)) for a in (
            [1,2,3,4,5,6], [8,9,10,11,12,13],
            [15,16,17,18,19,20], [22,23,24,25,26,27])])
        series=samples[np.arange(120)%4]
        control=temporal_null(series,np.full(120,2018),lags=(4,),
                              permutations=29,seed=120)
        self.assertGreater(control["4"]["pixel_overlap"]["delta"],4)

    def test_image_study_report_and_offline_svg_gallery(self):
        fake=(self.images,np.zeros(120,dtype=np.int64),self.ids)
        with patch("lotto7x7.image_structure_v05.load_history",return_value=fake):
            result=image_study(permutations=9,null_images=100,
                               mapping_seeds=1,seed=11,lags=(1,2))
        self.assertEqual(result["schema"],STRUCTURE_SCHEMA)
        self.assertEqual(len(result["mapping_comparisons"]),3)
        self.assertEqual(len(result["latest_images"]),20)
        html=image_gallery(result)
        self.assertIn("Lotto7x7 v0.5",html)
        self.assertIn("<svg",html)
        self.assertNotIn("http://",html)
        with self.assertRaises(ValueError):
            image_gallery({})

    def test_exact_random_comparison(self):
        self.assertLess(exact_iid_expected_jaccard(),0.2)
        self.assertGreater(exact_iid_expected_jaccard(),0.05)

    def test_feature_scaler_excludes_validation_and_test(self):
        a,b,metadata=standardized_features(self.images,train_end=72)
        modified=self.images.copy()
        modified[72:]=self.images[0]
        c,d,meta2=standardized_features(modified,train_end=72)
        np.testing.assert_array_equal(a[:72],c[:72])
        np.testing.assert_array_equal(b[:72],d[:72])
        self.assertEqual(metadata,meta2)
        self.assertEqual(a.shape[0],len(self.images))

    def test_feature_windows_do_not_contain_target_frame(self):
        first=prepare_windows(self.images,window=3,train_end=72)
        changed=self.images.copy();changed[96:]=self.images[0]
        second=prepare_windows(changed,window=3,train_end=72)
        for i in (0,1,2):
            np.testing.assert_array_equal(first[i][96-3].numpy(),
                                          second[i][96-3].numpy())

    def test_train_and_infer_four_image_models(self):
        windows=prepare_windows(self.images,window=3,train_end=72)[:4]
        for kind in MODELS:
            model,fit=train_image_arm(windows,kind=kind,seed=10,
                                      train_end=72,validation_end=96,
                                      window=3,embedding=8,hidden=12,
                                      epochs=1,patience=1,batch_size=16)
            logits=infer_image_arm(model,windows,targets=np.arange(96,101),
                                   window=3,batch_size=2)
            self.assertEqual(logits.shape,(5,7,7))
            self.assertGreater(fit["parameter_count"],0)
            summary,arrays=score_image_logits(logits,self.images[96:101])
            self.assertEqual(sum(summary["hit_distribution"].values()),5)
            self.assertEqual(arrays["masks"].sum(),5*6)
            self.assertEqual(len(fit["history"]),2)

    def test_compare_report_and_gallery_with_synthetic(self):
        fake=(self.images,np.zeros(120,dtype=np.int64),self.ids)
        with patch("lotto7x7.image_compare_v05.load_history",return_value=fake):
            result=image_compare(seeds=1,seed=21,window=3,epochs=1,
                                 patience=1,batch_size=16,embedding=8,hidden=12,
                                 bootstrap=10)
        self.assertEqual(result["schema"],COMPARE_SCHEMA)
        self.assertEqual(result["evaluation_draws"],24)
        self.assertEqual(len(result["summary"]),4)
        self.assertEqual(len(result["examples"]),16)
        self.assertEqual(len(result["paired_intervals"]),6)
        for name in MODELS:
            self.assertEqual(len(result["seed_results"][name]),1)
            self.assertAlmostEqual(result["summary"][name]["mean_hits"],
                                   np.mean(result["seed_results"][name][0]
                                           ["hits_per_target"]))
        html=image_compare_gallery(result)
        self.assertIn("image_fusion_gru",html)
        self.assertIn("<svg",html)
        with self.assertRaises(ValueError):
            image_compare_gallery({})

    def test_reject_bad_comparison_configuration(self):
        with self.assertRaises(ValueError):
            image_compare(seeds=0)
        with self.assertRaises(ValueError):
            image_compare(bootstrap=1)
        with self.assertRaises(ValueError):
            standard_features=standardized_features(self.images,train_end=120)

if __name__ == "__main__":
    unittest.main()
