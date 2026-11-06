"""Phase 7 reversible 7x7 images, causal windows and spectral invariants."""
from __future__ import annotations

import unittest

import numpy as np

from lotto_image_repr import (DOMAINS, MAPPINGS, ascii_image, cell_to_number,
                              decode_images, encode_draws, inverse_transform,
                              temporal_windows, transform_images)


class ImageRepresentationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(729)
        self.draws = np.stack([np.sort(rng.choice(49, 6, replace=False) + 1)
                               for _ in range(50)])

    def test_every_layout_is_bijective_and_reproducible(self):
        for name in MAPPINGS:
            x = cell_to_number(name, seed=2026)
            self.assertEqual(x.shape, (7, 7))
            self.assertEqual(sorted(x.ravel().tolist()), list(range(1, 50)))
            np.testing.assert_array_equal(x, cell_to_number(name, seed=2026))
        self.assertFalse(np.array_equal(cell_to_number("permuted", seed=4),
                                        cell_to_number("permuted", seed=7)))

    def test_single_and_batch_six_number_roundtrip(self):
        for name in MAPPINGS:
            images = encode_draws(self.draws, mapping=name)
            self.assertEqual(images.shape, (50, 7, 7))
            self.assertTrue(np.all(images.sum(axis=(1, 2)) == 6))
            np.testing.assert_array_equal(decode_images(images, mapping=name), self.draws)
            single = self.draws[0][::-1]
            np.testing.assert_array_equal(decode_images(encode_draws(single, mapping=name),
                                                        mapping=name), self.draws[0])

    def test_number_to_pixel_orientation(self):
        image = encode_draws(np.array([1, 8, 15, 22, 29, 36]))
        self.assertTrue(np.array_equal(image[:, 0], np.array([1, 1, 1, 1, 1, 1, 0])))
        self.assertEqual(int(image.sum()), 6)

    def test_reject_invalid_draws_and_binary_masks(self):
        for source in ([1, 2, 3, 4, 5, 5], [1, 2, 3, 4, 5, 50],
                       [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6.0]):
            with self.assertRaises(ValueError):
                encode_draws(np.array(source))
        valid = encode_draws(self.draws[0])
        for bad in (np.zeros((7, 7)), valid.astype(float) * 0.5,
                    np.full((7, 7), np.nan)):
            with self.assertRaises(ValueError):
                decode_images(bad)
        with self.assertRaises(ValueError):
            cell_to_number("unknown")

    def test_every_domain_is_reconstructable_in_double_precision(self):
        masks = encode_draws(self.draws)
        for domain in DOMAINS:
            coeff = transform_images(masks, domain=domain)
            restored = inverse_transform(coeff, domain=domain)
            np.testing.assert_allclose(restored, masks, rtol=0, atol=1e-12)
            if domain == "fft":
                self.assertEqual(coeff.shape, (50, 2, 7, 7))
            else:
                self.assertEqual(coeff.shape, (50, 7, 7))

    def test_frequency_full_energy_is_invariant_for_six_active_points(self):
        masks = encode_draws(self.draws)
        dct = transform_images(masks, domain="dct")
        fft = transform_images(masks, domain="fft")
        np.testing.assert_allclose((dct ** 2).sum(axis=(1, 2)),
                                   np.full(50, 6.0), atol=1e-12)
        np.testing.assert_allclose((fft ** 2).sum(axis=(1, 2, 3)),
                                   np.full(50, 6.0), atol=1e-12)

    def test_reject_wrong_transform_shapes(self):
        with self.assertRaises(ValueError):
            transform_images(np.zeros((6, 6)), domain="dct")
        with self.assertRaises(ValueError):
            transform_images(np.zeros((7, 7)), domain="invalid")
        with self.assertRaises(ValueError):
            inverse_transform(np.zeros((7, 7)), domain="fft")
        with self.assertRaises(ValueError):
            inverse_transform(np.zeros((2, 7, 7)), domain="invalid")

    def test_temporal_window_shapes_and_causal_alignment(self):
        for domain, expected_channels in (("spatial", 5), ("dct", 5), ("fft", 10)):
            x, targets, origins = temporal_windows(self.draws, window=5,
                                                   domain=domain)
            self.assertEqual(x.shape, (45, expected_channels, 7, 7))
            np.testing.assert_array_equal(origins, np.arange(5, 50))
            np.testing.assert_array_equal(targets, self.draws[5:])
        x, _, _ = temporal_windows(self.draws, window=5, as_volume=True)
        self.assertEqual(x.shape, (45, 1, 5, 7, 7))

    def test_target_mutation_cannot_change_same_date_input(self):
        base = self.draws.copy()
        x, _, _ = temporal_windows(base, window=5, domain="fft")
        changed = base.copy()
        replacement = [1, 2, 3, 4, 5, 6]
        if np.array_equal(base[5], replacement):
            replacement = [2, 3, 4, 5, 6, 7]
        changed[5] = replacement
        y, labels, _ = temporal_windows(changed, window=5, domain="fft")
        np.testing.assert_array_equal(x[0], y[0])
        self.assertFalse(np.array_equal(x[1], y[1]))
        np.testing.assert_array_equal(labels[0], replacement)

    def test_invalid_volumes_and_windows_rejected(self):
        with self.assertRaises(ValueError):
            temporal_windows(self.draws, window=0)
        with self.assertRaises(ValueError):
            temporal_windows(self.draws, window=50)
        with self.assertRaises(ValueError):
            temporal_windows(self.draws, domain="fft", as_volume=True)

    def test_ascii_image_has_exactly_seven_rows_and_six_marks(self):
        output = ascii_image(self.draws[0])
        self.assertEqual(len(output.splitlines()), 7)
        self.assertEqual(output.count("#"), 6)


if __name__ == "__main__":
    unittest.main()
