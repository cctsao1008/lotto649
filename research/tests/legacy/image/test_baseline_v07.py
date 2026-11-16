"""v0.7 baseline fairness, data leakage, reproducibility and shadow sealing."""
from __future__ import annotations

from copy import deepcopy
import json
from math import comb
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from lotto7x7.baseline_v07 import (
    ARMS, SCHEMA, SCORE_SCHEMA, SNAPSHOT_SCHEMA, make_arms, random_pool,
    random_sets, rolling_origin, score_shadow, shadow_snapshot,
)
from lotto7x7.coverage_v06 import UNIVERSE, normalized_bets, exact_coverage
from lotto7x7.candidate_pool_v06 import pool_digest
from lotto7x7.shape_engine_v06 import bitmap


def fixture():
    rng = np.random.default_rng(701)
    images = np.stack([bitmap(tuple(sorted(int(v) for v in rng.choice(49, 6, replace=False) + 1)))
                       for _ in range(85)])
    ids = np.arange(115000001, 115000086, dtype=np.int64)
    return images, ids


class BaselineV07Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.images, cls.ids = fixture()

    def test_b0_and_b2_ticket_counts_disjoint_and_seeded(self):
        for n in range(1, 6):
            b0 = random_sets(n, 31)
            b2 = random_sets(n, 37, disjoint=True)
            self.assertEqual(b0, random_sets(n, 31))
            self.assertEqual(b2, random_sets(n, 37, disjoint=True))
            self.assertEqual(len(normalized_bets(b0)), n)
            self.assertEqual(len(normalized_bets(b2)), n)
            self.assertEqual(len(set(k for row in b2 for k in row)), n*6)
        for n in (0, 6):
            with self.assertRaises(ValueError):
                random_sets(n, 9)

    def test_b2_five_disjoint_exact_coverage_reference(self):
        b2 = random_sets(5, 100, disjoint=True)
        self.assertEqual(exact_coverage(b2)['at_least_counts']['3'], 1299120)
        self.assertEqual(exact_coverage(b2)['at_least_counts']['6'], 5)

    def test_random_pool_digest_unique_and_shape_identity(self):
        from lotto7x7.shape_engine_v06 import shape_key, shape_id
        pool = random_pool(100, 2026, int(self.ids[-1]))
        self.assertEqual(pool['candidate_count'], 100)
        self.assertEqual(pool['candidate_sha256'], pool_digest(pool['candidates']))
        self.assertEqual(len(set(tuple(x['numbers']) for x in pool['candidates'])), 100)
        for row in pool['candidates']:
            self.assertEqual(row['shape_key'], shape_key(bitmap(row['numbers'])))
            self.assertEqual(row['shape_id'], shape_id(row['shape_key']))
        self.assertEqual(pool, random_pool(100, 2026, int(self.ids[-1])))
        self.assertNotEqual(pool['candidate_sha256'], random_pool(100, 2027, int(self.ids[-1]))['candidate_sha256'])

    def test_four_arms_and_identical_number_of_tickets(self):
        result = make_arms(self.images, self.ids, pool_size=18,
                           family_count=4, count=3, seed=55)
        self.assertEqual(list(result['arms']), list(ARMS))
        self.assertEqual(result['history_cutoff_draw_id'], int(self.ids[-1]))
        for arm in ARMS:
            self.assertEqual(len(result['arms'][arm]['strategies']),
                             1 if arm in (ARMS[0], ARMS[3]) else 3)
            for value in result['arms'][arm]['strategies'].values():
                self.assertEqual(len(value['bets']), 3)
                self.assertEqual(value['exact_fair_coverage']['at_least_counts']['6'], 3)
                self.assertEqual(value['exact_fair_coverage']['at_least']['6'], 3/comb(49,6))
        self.assertEqual(result, make_arms(self.images, self.ids,
                                            pool_size=18, family_count=4,
                                            count=3, seed=55))

    def test_rolling_origin_excludes_target_and_is_reproducible(self):
        opts = dict(warmup=50, eval_draws=1, count=2, pool_size=12,
                    family_count=4, seed=9)
        a = rolling_origin(self.images, self.ids, **opts)
        self.assertEqual(a['schema'], SCHEMA)
        self.assertEqual(a['target_count'], 1)
        row = a['targets'][0]
        self.assertEqual(row['target_draw_id'], int(self.ids[-1]))
        self.assertEqual(row['cutoff_draw_id'], int(self.ids[-2]))
        self.assertLess(row['cutoff_draw_id'], row['target_draw_id'])
        self.assertEqual(len(a['summary']), 8)
        self.assertEqual(a, rolling_origin(self.images, self.ids, **opts))
        # Alter the held-out future target only, without modifying the prefix.
        changed = self.images.copy()
        changed[-1] = bitmap((1, 2, 3, 4, 5, 6))
        b = rolling_origin(changed, self.ids, **opts)
        self.assertEqual(a['targets'][0]['history_sha256'], b['targets'][0]['history_sha256'])
        self.assertEqual(a['targets'][0]['selection_commitments'],
                         b['targets'][0]['selection_commitments'])
        self.assertEqual(b['targets'][0]['actual'], [1,2,3,4,5,6])

    def test_snapshot_exact_digest_post_draw_scoring_and_no_overwrite(self):
        snapshot = shadow_snapshot(self.images[:-1], self.ids[:-1],
                                   count=2, pool_size=12, family_count=4,
                                   seed=21, code_commit='test-commit',
                                   created_at_utc='2026-11-08T00:00:00Z')
        self.assertEqual(snapshot['schema'], SNAPSHOT_SCHEMA)
        self.assertEqual(snapshot, shadow_snapshot(self.images[:-1], self.ids[:-1],
                     count=2, pool_size=12, family_count=4,
                     seed=21, code_commit='test-commit',
                     created_at_utc='2026-11-08T00:00:00Z'))
        result = score_shadow(snapshot, actual_draw_id=int(self.ids[-1]),
                              actual_numbers=tuple(np.flatnonzero(self.images[-1]) + 1))
        self.assertEqual(result['schema'], SCORE_SCHEMA)
        self.assertEqual(len(result['score']), 4)
        for arm in ARMS:
            for strategy in result['score'][arm].values():
                self.assertEqual(len(strategy['ticket_hits']), 2)
        for mismatch in (int(self.ids[-2]), 1):
            with self.assertRaises(ValueError):
                score_shadow(snapshot, actual_draw_id=mismatch,
                             actual_numbers=(1,2,3,4,5,6))
        changed = deepcopy(snapshot)
        changed['pre_draw']['arms'][ARMS[0]]['strategies']['random']['bets'][0]['numbers'] = [1,2,3,4,5,6]
        with self.assertRaises(ValueError):
            score_shadow(changed, actual_draw_id=int(self.ids[-1]),
                         actual_numbers=(1,2,3,4,5,6))

    def test_explicit_cutoff_is_only_history_prefix(self):
        cutoff = int(self.ids[-6])
        snap = shadow_snapshot(self.images, self.ids, cutoff_draw_id=cutoff,
                               count=1, pool_size=12, family_count=4,
                               created_at_utc='2026-11-08T00:00:00Z')
        self.assertEqual(snap['pre_draw']['history_cutoff_draw_id'], cutoff)
        self.assertEqual(snap['pre_draw']['history_draw_count'], len(self.images)-5)
        with self.assertRaises(ValueError):
            shadow_snapshot(self.images,self.ids,cutoff_draw_id=99999999)

    def test_invalid_pool_count_and_history_rejected(self):
        for bad in [dict(count=6), dict(pool_size=2, count=3),
                    dict(family_count=19, pool_size=18), dict(seed=-1)]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                make_arms(self.images,self.ids,**bad)
        with self.assertRaises(ValueError):
            make_arms(self.images, self.ids[::-1], count=1)

    def test_command_line_three_commands_smoke(self):
        """Actual installed repository test; uses generated BIN v2, not stock results."""
        x, ids = self.images, self.ids
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root/'synthetic.bin'
            hdr = struct.pack('<8sII', b'L649IMG2', len(x), 53)
            payload = b''.join(struct.pack('<I49B', int(draw_id), *image.ravel().tolist())
                               for image, draw_id in zip(x, ids))
            dataset.write_bytes(hdr+payload)
            rolling = root/'baseline.json'
            snap = root/'snapshot.json'
            scored = root/'score.json'
            for command in [
                ['baseline-study','--data',str(dataset),'--eval-draws','1',
                 '--warmup','50','--count','1','--size','12','--families','4',
                 '--out',str(rolling)],
                ['shadow-snapshot','--data',str(dataset),'--count','1','--size','12',
                 '--families','4','--out',str(snap)],
                ['shadow-score','--snapshot',str(snap),'--draw-id',str(int(ids[-1])+1),
                 '--numbers','1,2,3,4,5,6','--out',str(scored)],
            ]:
                subprocess.run([sys.executable,'-m','lotto7x7',*command], check=True,
                               capture_output=True,text=True)
            self.assertEqual(json.loads(rolling.read_text())['target_count'],1)
            self.assertEqual(json.loads(snap.read_text())['schema'],SNAPSHOT_SCHEMA)
            self.assertEqual(json.loads(scored.read_text())['schema'],SCORE_SCHEMA)
            # Refuse to overwrite sealed snapshots on repeated creation.
            repeat = subprocess.run([sys.executable,'-m','lotto7x7','shadow-snapshot',
                                     '--data',str(dataset),'--count','1','--size','12',
                                     '--families','4','--out',str(snap)],
                                    capture_output=True,text=True)
            self.assertNotEqual(repeat.returncode,0)


if __name__ == '__main__':
    unittest.main()