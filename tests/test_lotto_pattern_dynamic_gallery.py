"""Phase 7C dynamic 7x7 visual report tests."""
from __future__ import annotations

import unittest

from lotto_pattern_dynamic_gallery import render_gallery


class DynamicGalleryTests(unittest.TestCase):
    def test_rejects_old_phase7b_report_and_invalid_limit(self):
        with self.assertRaises(ValueError):
            render_gallery({"schema": "lotto649.phase7b.pattern_similarity.v1"})
        with self.assertRaises(ValueError):
            render_gallery({
                "schema": "lotto649.phase7c.dynamic_pattern_replay.v1",
            }, max_draws=0)

    def test_valid_schema_renders_score_table_and_four_images(self):
        six = ["######."] + ["......."] * 6
        score = {
            "mean_raw": {"composite": 0.55},
            "mean_iid_expected": {"composite": 0.50},
            "mean_chance_adjusted": {"composite": 0.05},
            "diversity": {
                "unique_pixel_patterns": 1,
                "unique_shape_classes_d4_translate": 1,
                "top_pattern_fraction": 1.0,
            },
        }
        pred = {
            "predicted_main_six": [1, 2, 3, 4, 5, 6],
            "predicted_pattern": six,
            "adjusted_similarity": {"composite": 0.05},
            "raw_similarity": {"shape_d4": 1.0},
        }
        report = {
            "schema": "lotto649.phase7c.dynamic_pattern_replay.v1",
            "year": "demo",
            "evaluated_draws": 1,
            "methods": ["last_pattern", "centroid_flow", "analog_fusion"],
            "validation_selected": "analog_fusion",
            "evaluation_summary": {
                name: score for name in
                ("last_pattern", "centroid_flow", "analog_fusion")
            },
            "temporal_order_audit": {
                "segments": {"historical_test": {
                    "lags": {"1": {
                        "observed_mean_cosine": 0.4,
                        "shuffled_mean_cosine": 0.1,
                        "empirical_two_sided_tail": 0.1,
                    }},
                }},
            },
            "individual_draws": [{
                "draw_id": 1,
                "date": 20180101,
                "actual_main_six": [1, 2, 3, 4, 5, 6],
                "actual_pattern": six,
                "methods": {
                    name: pred for name in
                    ("last_pattern", "centroid_flow", "analog_fusion")
                },
            }],
        }
        html = render_gallery(report, max_draws=3)
        self.assertIn("Dynamic 7×7", html)
        self.assertIn("Temporal", html)
        self.assertIn("analog_fusion", html)
        self.assertEqual(html.count("<svg "), 4)


if __name__ == "__main__":
    unittest.main()
