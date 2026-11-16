"""v0.5.1 visual dynamics, number-position semantics and report integrity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from lotto7x7.core import decode, encode, iid_images
from lotto7x7.image_dynamics_v051 import (
    SCHEMA, IID_NON_DC_POWER, IID_OVERLAP, IID_OCCUPANCY,
    _distribution, _layouts, _model_gallery,
    _spectral_from_dct, image_dynamics, image_dynamics_gallery,
    similarity_matrices, transition,
)


class VisualImageDynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.images=iid_images(125,seed=420)
        cls.draw_ids=np.arange(110000001,110000126,dtype=np.int64)
        cls.data=(cls.images,np.zeros(125,dtype=np.int64),cls.draw_ids)

    def test_transition_retained_lost_added_conserve_six(self):
        before=encode(np.array([1,2,3,4,5,6]))
        after=encode(np.array([1,2,10,20,30,40]))
        t=transition(before,after)
        self.assertEqual([t[k] for k in ("retained","removed","added")],[2,4,4])
        self.assertAlmostEqual(t["exact_pixel_jaccard"],.2)
        c=np.asarray(t["transition_code"])
        self.assertEqual(np.count_nonzero(c==2),2)
        self.assertEqual(np.count_nonzero(c==3),4)
        self.assertEqual(np.count_nonzero(c==4),4)
        np.testing.assert_array_equal(before,np.logical_or(c==2,c==3).astype(np.uint8))
        np.testing.assert_array_equal(after,np.logical_or(c==2,c==4).astype(np.uint8))

    def test_tolerant_shape_does_not_mean_number_hits(self):
        a=encode(np.array([1,2,3,4,5,6]))
        b=encode(np.array([8,9,10,11,12,13]))
        t=transition(a,b)
        self.assertEqual(t["retained"],0)
        self.assertEqual(t["exact_pixel_jaccard"],0)
        self.assertAlmostEqual(t["d4_translation_shape_similarity"],1.0)

    def test_reject_bad_transition(self):
        with self.assertRaises(ValueError):
            transition(np.zeros((7,7)),self.images[0])

    def test_histogram_probabilities_and_descriptive_z(self):
        obs=np.arange(1,101,dtype=float)
        ctrl=np.arange(3,203,dtype=float)
        r=_distribution(obs,ctrl,bins=11)
        self.assertAlmostEqual(sum(r["observed_hist"]),1)
        self.assertAlmostEqual(sum(r["iid_hist"]),1)
        self.assertEqual(len(r["edges"]),12)
        self.assertLess(r["delta"],0)
        self.assertIsNotNone(r["descriptive_z_mc"])

    def test_constant_histogram_is_finite(self):
        r=_distribution(np.ones(8),np.ones(10))
        self.assertIsNone(r["descriptive_z_mc"])
        self.assertEqual(r["delta"],0)
        with self.assertRaises(ValueError):_distribution(np.ones((2,2)),np.ones(10))

    def test_exact_iid_spectral_mean_of_non_dc(self):
        self.assertAlmostEqual(IID_NON_DC_POWER,258/2352)
        self.assertAlmostEqual(IID_OVERLAP,36/49)
        self.assertAlmostEqual(IID_OCCUPANCY,6/49)
        sample=iid_images(4096,17)
        from lotto7x7.image_structure_v05 import describe_images
        dct=describe_images(sample)["dct"]
        power=np.mean(dct*dct,axis=0)
        self.assertAlmostEqual(np.sum(power)-power[0,0],258/49,places=10)
        self.assertLess(abs(np.mean(power[1:,1:])-IID_NON_DC_POWER),.01)
        stats=_spectral_from_dct(dct,dct)
        self.assertEqual(stats["observed_non_dc_power"][0][0],0)
        self.assertEqual(stats["delta_non_dc_power"][0][0],0)
        self.assertTrue(np.allclose(stats["delta_non_dc_power"],0))

    def test_spatial_permutation_has_morphological_sensitivity(self):
        iid=iid_images(256,123)
        layouts=_layouts(self.images,iid,seed=649,mapping_seeds=1,on_progress=None)
        self.assertEqual([x["mapping"] for x in layouts],
                         ["sequential","snake","permuted"])
        for row in layouts:
            self.assertEqual(len(row["morphology"]),6)
            self.assertEqual(row["occupancy_z_unadjusted"][0][0],
                             row["occupancy_z_unadjusted"][0][0])
            self.assertEqual(row["spectral"]["excluded_dc"],True)
        a=np.array(layouts[0]["morphology"]["radial_spread"]["observed_hist"])
        b=np.array(layouts[2]["morphology"]["radial_spread"]["observed_hist"])
        self.assertFalse(np.array_equal(a,b))

    def test_similarity_matrix_symmetry_and_diagonal(self):
        m=similarity_matrices(self.images,self.draw_ids,count=5,seed=3,null_pairs=16)
        overlap=np.asarray(m["exact_common_pixels"])
        shape=np.asarray(m["shape_similarity_d4_translation"])
        np.testing.assert_array_equal(overlap,overlap.T)
        np.testing.assert_array_equal(np.diag(overlap),np.full(5,6))
        np.testing.assert_array_equal(shape,shape.T)
        np.testing.assert_allclose(np.diag(shape),np.ones(5))
        self.assertEqual(m["draw_ids"],self.draw_ids[-5:].tolist())
        self.assertGreater(m["iid_sample_mean_shape_similarity"],0)

    def test_reject_bad_matrix_arguments(self):
        with self.assertRaises(ValueError):
            similarity_matrices(self.images,self.draw_ids[:-1],count=4)
        with self.assertRaises(ValueError):
            similarity_matrices(self.images,self.draw_ids,count=1)

    def fake_report(self, **kwargs):
        with patch("lotto7x7.image_dynamics_v051.load_history",return_value=self.data):
            return image_dynamics(seed=42,null_images=200,mapping_seeds=1,
                                  transitions=3,matrix_count=5,null_shape_pairs=16,
                                  **kwargs)

    def test_report_schema_and_html_offline(self):
        result=self.fake_report()
        self.assertEqual(result["schema"],SCHEMA)
        self.assertEqual(len(result["recent_transitions"]),3)
        self.assertEqual(len(result["layouts"]),3)
        html=image_dynamics_gallery(result)
        self.assertIn("non-DC",html)
        self.assertIn("Changed pixels",html)
        self.assertIn("Shape similarity",html)
        self.assertIn("<svg",html)
        self.assertIn("Observed − IID",html)
        self.assertNotIn("https://",html)
        self.assertNotIn("http://",html)
        self.assertEqual(result["first_draw_id"],int(self.draw_ids[0]))
        self.assertEqual(result["last_draw_id"],int(self.draw_ids[-1]))

    def test_reproducible_random_null_and_no_input_mutation(self):
        original=self.images.copy()
        a=self.fake_report()
        b=self.fake_report()
        self.assertEqual(a,b)
        np.testing.assert_array_equal(self.images,original)

    def test_reject_invalid_configuration(self):
        with self.assertRaises(ValueError):
            image_dynamics(null_images=10)
        with self.assertRaises(ValueError):
            image_dynamics(mapping_seeds=9)
        with self.assertRaises(ValueError):
            image_dynamics(matrix_count=1)
        with self.assertRaises(ValueError):
            image_dynamics_gallery({})

    def test_optional_saved_prediction_integration_and_consistency(self):
        im=self.images[-1]
        p=self.images[-2]
        scored=transition(p,im)
        digest=hashlib.sha256(self.images.tobytes()+self.draw_ids.tobytes()).hexdigest()
        prev={"schema":"lotto7x7.image_compare.v0.5","data_sha256":digest,
              "evaluation_draws":10,"examples":[{
                "draw_id":int(self.draw_ids[-1]),"bitmap":im.tolist(),
                "predictions":{"image_cnn_gru":{"bitmap":p.tolist(),
                    "numbers":decode(p),"hits":scored["retained"]}}}]}
        with tempfile.TemporaryDirectory() as folder:
            file=Path(folder)/"comparison.json"
            file.write_text(json.dumps(prev),encoding="utf-8")
            report=self.fake_report(comparison_report=file)
            examples=report["prediction_gallery"]["examples"]
            self.assertEqual(len(examples),1)
            self.assertEqual(examples[0]["methods"]["image_cnn_gru"]["exact_hits"],scored["retained"])
            html=image_dynamics_gallery(report)
            self.assertIn("Previously saved v0.5 image predictions",html)
            wrong=prev.copy();wrong["data_sha256"]="wrong"
            file.write_text(json.dumps(wrong),encoding="utf-8")
            with self.assertRaises(ValueError):self.fake_report(comparison_report=file)
            wrong=prev.copy();wrong["examples"]=[dict(prev["examples"][0],bitmap=self.images[0].tolist())]
            file.write_text(json.dumps(wrong),encoding="utf-8")
            with self.assertRaises(ValueError):self.fake_report(comparison_report=file)

    def test_optional_prediction_rejects_forged_hit_count(self):
        im=self.images[-1];p=self.images[-2]
        digest=hashlib.sha256(self.images.tobytes()+self.draw_ids.tobytes()).hexdigest()
        x={"schema":"lotto7x7.image_compare.v0.5","data_sha256":digest,
           "examples":[{"draw_id":int(self.draw_ids[-1]),"bitmap":im.tolist(),
                        "predictions":{"cnn":{"bitmap":p.tolist(),
                                                 "numbers":decode(p),"hits":6}}}]}
        with self.assertRaises(ValueError):_model_gallery(x,self.images,self.draw_ids,digest)


if __name__=="__main__":unittest.main()
