"""v0.6 Phase 1: mathematical, geometry and non-lookahead regression tests."""
from __future__ import annotations

import copy
from math import comb
import unittest
import numpy as np

from lotto7x7.shape_engine_v06 import (
    UNIVERSE_SIZE, bitmap, canonical_points, enumerate_placements,
    history_shape_families, image_points, orbit_size, orientations,
    parse_shape_key, shape_id, shape_key, validate_images,
)
from lotto7x7.candidate_pool_v06 import (
    SCHEMA, create_shape_pool, pool_digest, random_sample, shape_pool_gallery,
)


def _fixture():
    rng = np.random.default_rng(267)
    images = np.stack([bitmap(tuple(int(x) for x in
                              rng.choice(49, 6, replace=False) + 1))
                       for _ in range(145)])
    ids = np.arange(115000001, 115000146, dtype=np.int64)
    return images, ids


class ShapeEngineV06Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.images, cls.ids = _fixture()

    def test_exact_6_of_49_universe(self):
        self.assertEqual(UNIVERSE_SIZE, 13983816)
        self.assertEqual(UNIVERSE_SIZE, comb(49, 6))

    def test_bitmap_exact_six_number_identity(self):
        nums = (2, 4, 10, 22, 34, 49)
        im = bitmap(nums)
        self.assertEqual(int(im.sum()), 6)
        self.assertEqual(tuple(np.flatnonzero(im) + 1), nums)
        self.assertEqual(image_points(im), tuple(sorted(((n-1)//7, (n-1)%7)
                                                        for n in nums)))

    def test_bitmap_rejects_duplicates_out_of_range_and_nonintegers(self):
        for numbers in [(1,1,2,3,4,5), (1,2,3,4,5,50), (0,2,3,4,5,6),
                        (1,2,3,4,5,6.0), (1,2,3,4,5), ()]:
            with self.subTest(numbers=numbers), self.assertRaises(ValueError):
                bitmap(numbers)

    def test_shape_key_is_translation_invariant(self):
        a = bitmap((1, 2, 3, 8, 9, 10))
        b = bitmap((9, 10, 11, 16, 17, 18))
        self.assertEqual(shape_key(a), shape_key(b))
        self.assertEqual(shape_id(shape_key(a)), shape_id(shape_key(b)))

    def test_shape_key_invariant_under_all_d4(self):
        img = bitmap((2, 8, 13, 23, 24, 42))
        k = shape_key(img)
        for r in range(4):
            for changed in (np.rot90(img, r), np.fliplr(np.rot90(img, r))):
                self.assertEqual(shape_key(changed), k)

    def test_canonical_is_translation_anchored(self):
        k = shape_key(bitmap((9, 10, 11, 16, 17, 18)))
        p = parse_shape_key(k)
        self.assertEqual(min(r for r, _ in p), 0)
        self.assertEqual(min(c for _, c in p), 0)
        self.assertEqual(canonical_points(p), p)

    def test_reject_uncanonical_key_and_duplicate_points(self):
        for key in ["1,1;1,2;1,3;2,1;2,2;2,3", "0,0;0,0;0,1;0,2;0,3;0,4",
                    "0,0;1,1", "not-a-shape", "0,0;0,1;0,2;1,0;1,1;1,20"]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                parse_shape_key(key)

    def test_rectangular_2_by_3_orbit_is_60(self):
        key = shape_key(bitmap((1, 2, 3, 8, 9, 10)))
        self.assertEqual(len(orientations(parse_shape_key(key))), 2)
        self.assertEqual(orbit_size(key), 60)
        places = enumerate_placements(key)
        self.assertEqual(len(places), 60)
        self.assertEqual(len({x.numbers for x in places}), 60)

    def test_straight_six_orbit_is_28(self):
        key = shape_key(bitmap((1, 2, 3, 4, 5, 6)))
        self.assertEqual(orbit_size(key), 28)
        self.assertEqual(len(enumerate_placements(key)), 28)

    def test_orbit_placements_all_in_bounds_and_same_shape(self):
        for img in self.images[:15]:
            key = shape_key(img)
            positions = enumerate_placements(key)
            self.assertEqual(len(positions), orbit_size(key))
            self.assertEqual(len({p.numbers for p in positions}), len(positions))
            for p in positions:
                self.assertEqual(len(p.numbers), 6)
                self.assertTrue(all(1 <= n <= 49 for n in p.numbers))
                self.assertEqual(shape_key(bitmap(p.numbers)), key)

    def test_different_shape_families_have_disjoint_orbits(self):
        keys = [shape_key(self.images[i]) for i in range(8)]
        observed = {}
        for key in sorted(set(keys)):
            for p in enumerate_placements(key):
                if p.numbers in observed:
                    self.assertEqual(observed[p.numbers], key)
                observed[p.numbers] = key

    def test_image_validation_and_bad_cardinality(self):
        self.assertEqual(validate_images(self.images).shape, self.images.shape)
        for images in (np.zeros((3,7,7)), np.ones((3,7,7)),
                       np.ones((7,7)), np.ones((3,6,7))):
            with self.assertRaises(ValueError):
                validate_images(images)

    def test_history_families_frequency_consistency(self):
        fams = history_shape_families(self.images, self.ids)
        self.assertEqual(sum(f['historical_count'] for f in fams), len(self.images))
        self.assertEqual(len({f['shape_key'] for f in fams}), len(fams))
        self.assertTrue(all(f['orbit_size'] >= 1 for f in fams))
        self.assertTrue(all(f['first_draw_id'] <= f['last_draw_id'] for f in fams))

    def test_history_rejects_non_chronological_ids(self):
        with self.assertRaises(ValueError):
            history_shape_families(self.images, self.ids[::-1])
        with self.assertRaises(ValueError):
            history_shape_families(self.images, self.ids[:-1])


class CandidatePoolV06Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.images, cls.ids = _fixture()
        cls.pool = create_shape_pool(cls.images, cls.ids,
                                     seed=649, pool_size=100, family_count=20)

    def test_default_pool_has_100_distinct_six_number_sets(self):
        pool = self.pool
        self.assertEqual(pool['schema'], SCHEMA)
        self.assertTrue(pool['research_only'])
        self.assertEqual(pool['candidate_count'], 100)
        self.assertEqual(len({tuple(x['numbers']) for x in pool['candidates']}), 100)
        self.assertEqual(pool['candidate_sha256'], pool_digest(pool['candidates']))
        self.assertEqual(pool['source']['cutoff_draw_id'], int(self.ids[-1]))

    def test_family_counts_and_candidate_orbit_identity(self):
        pool = self.pool
        self.assertEqual(len(pool['selected_families']), 20)
        self.assertEqual(sum(x['generated_count'] for x in pool['selected_families']), 100)
        by_id = {x['shape_id']: x for x in pool['selected_families']}
        for row in pool['candidates']:
            self.assertEqual(shape_key(bitmap(row['numbers'])), row['shape_key'])
            self.assertEqual(row['shape_id'], by_id[row['shape_id']]['shape_id'])
            self.assertEqual(row['shape_key'], by_id[row['shape_id']]['shape_key'])
            self.assertEqual(int(np.asarray(row['bitmap']).sum()), 6)
            self.assertEqual(np.asarray(row['bitmap']).tolist(), bitmap(row['numbers']).tolist())

    def test_pool_is_seed_reproducible_and_immutable_input(self):
        old = self.images.copy()
        first = self.pool
        second = create_shape_pool(self.images, self.ids, seed=649)
        self.assertEqual(first, second)
        self.assertTrue(np.array_equal(self.images, old))
        changed = create_shape_pool(self.images, self.ids, seed=650)
        self.assertNotEqual(first['candidate_sha256'], changed['candidate_sha256'])

    def test_sampling_count_one_through_five(self):
        for n in range(1, 6):
            sample = random_sample(self.pool, count=n, seed=42)
            self.assertEqual(len(sample), n)
            self.assertEqual(len({x['pool_index'] for x in sample}), n)
            for item in sample:
                self.assertEqual(item['numbers'],
                                 self.pool['candidates'][item['pool_index']]['numbers'])
        self.assertEqual(random_sample(self.pool, count=5, seed=42),
                         random_sample(self.pool, count=5, seed=42))
        self.assertEqual(random_sample(self.pool, count=3, seed=42),
                         random_sample(self.pool, count=5, seed=42)[:3])

    def test_previews_are_not_additional_tickets(self):
        sample = self.pool['random_preview']
        self.assertEqual(len(sample), 5)
        self.assertEqual(self.pool['fair_independent_reference']['preview_exact_six_coverage'],
                         5 / comb(49, 6))
        self.assertEqual(self.pool['fair_independent_reference']['candidate_pool_exact_six_coverage'],
                         100 / comb(49, 6))

    def test_reject_excess_tickets_and_pool_size_parameters(self):
        invalid = [(0,1,1), (3,4,1), (3,2,4), (3,2,0)]
        for pool_size,families,sample in invalid:
            with self.subTest(args=(pool_size,families,sample)), self.assertRaises(ValueError):
                create_shape_pool(self.images,self.ids,pool_size=pool_size,
                                  family_count=families,sample_count=sample)
        for n in (0,6,7):
            with self.assertRaises(ValueError):
                random_sample(self.pool,count=n)

    def test_history_frequency_and_orbit_weighted_policies_work(self):
        for policy in ('orbit-weighted', 'historical-frequency'):
            pool = create_shape_pool(self.images, self.ids, seed=18,
                                     family_policy=policy, pool_size=28,
                                     family_count=7, sample_count=1)
            self.assertEqual(len(pool['candidates']), 28)
            self.assertEqual(pool['configuration']['family_policy'], policy)
            self.assertEqual(len(pool['random_preview']), 1)

    def test_incorrect_history_cutoff_or_data_changes_digest(self):
        smaller = create_shape_pool(self.images[:-1], self.ids[:-1])
        self.assertNotEqual(self.pool['source']['history_sha256'],
                            smaller['source']['history_sha256'])
        self.assertLess(smaller['source']['cutoff_draw_id'],
                        self.pool['source']['cutoff_draw_id'])

    def test_html_contains_shape_gallery_without_external_urls(self):
        html = shape_pool_gallery(self.pool)
        self.assertTrue(html.startswith('<!doctype html>'))
        self.assertIn('Selected canonical shapes',html)
        self.assertIn('All candidate images',html)
        self.assertIn('Pool #',html)
        self.assertEqual(html.count('aria-label="7 by 7 six point bitmap"'),120)
        self.assertNotIn('https://',html)
        self.assertNotIn('<script',html)

    def test_detect_tampered_pool(self):
        damaged = copy.deepcopy(self.pool)
        damaged['candidates'][1]['numbers'] = damaged['candidates'][0]['numbers']
        with self.assertRaises(ValueError):
            random_sample(damaged)
        with self.assertRaises(ValueError):
            shape_pool_gallery(damaged)


if __name__ == '__main__':
    unittest.main()
