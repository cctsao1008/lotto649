"""Static gallery tests for the Phase 7B visual six-point reports."""
from __future__ import annotations

import unittest

import numpy as np

from lotto_image_repr import encode_draws
from lotto_pattern_gallery import render_gallery, svg_mask


class PatternGalleryTests(unittest.TestCase):
    def test_valid_image_renders_49_svg_rectangles(self):
        mask = encode_draws([1, 7, 14, 21, 28, 49])
        rows = ["".join("#" if cell else "." for cell in row) for row in mask]
        html = svg_mask(rows, 'Example "quoted"')
        self.assertEqual(html.count("<rect "), 49)
        self.assertEqual(html.count("<circle "), 6)
        self.assertIn("&quot;", html)

    def test_invalid_image_rejected(self):
        with self.assertRaises(ValueError):
            svg_mask(["......."] * 7, "Empty")
        with self.assertRaises(ValueError):
            svg_mask(["."] * 7, "Wrong")

    def test_renderer_requires_new_phase7b_schema(self):
        with self.assertRaises(ValueError):
            render_gallery({"schema": "lotto649.phase7.image_experiment.v1"},
                           method="temporal_analog")


if __name__ == "__main__":
    unittest.main()
