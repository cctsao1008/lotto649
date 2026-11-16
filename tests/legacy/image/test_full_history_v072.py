"""GPU full-history replay: deterministic fair scenarios, no target leakage, parity."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
from unittest import TestCase, mock

import numpy as np
import torch

from lotto7x7.coverage_v06 import exact_coverage
from lotto7x7.full_history_v072 import (NAMES, PrefixShapePool,
    ScenarioEngine, _disjoint_option, _theory, resolve_device, run_full_history,
    score_batch, uniform_pool)


class GPUFullHistoryTests(TestCase):
    def test_device_cpu_and_missing_cuda(self):
        self.assertEqual(str(resolve_device('cpu')), 'cpu')
        with self.assertRaises(ValueError):
            resolve_device('metal')
        with mock.patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CUDA requested'):
                resolve_device('cuda')

    def test_scenario_reproducible_and_ticket_constraints(self):
        pool, digest = uniform_pool(60, 123)
        self.assertEqual(len(pool), 60)
        self.assertEqual(len(set(pool)), 60)
        a = ScenarioEngine(device='cpu', seed=51, scenarios=512)
        b = ScenarioEngine(device='cpu', seed=51, scenarios=512)
        self.assertTrue(torch.equal(a.draws, b.draws))
        idx = a.choose(pool, count=5, seed=123)
        self.assertEqual(idx, b.choose(pool, count=5, seed=123))
        self.assertEqual(len(set(idx)), 5)
        self.assertTrue(all(0 <= i < len(pool) for i in idx))
        self.assertAlmostEqual(_theory([pool[i] for i in idx])['6'], 5/13983816)

    def test_disjoint_reference_exact(self):
        pool = [tuple(range(i*6+1,i*6+7)) for i in range(5)]
        self.assertEqual(_disjoint_option(pool,5,list(range(5))),list(range(5)))
        self.assertEqual(_theory(pool),exact_coverage(pool)['at_least'])
        self.assertAlmostEqual(_theory(pool)['3'],.09290168,places=6)

    def test_batch_score_parity(self):
        first = [[1,2,3,4,5,6],[10,11,12,13,14,15]]
        others = [[2,3,4,10,20,30],[1,11,22,33,44,49]]
        rows = [dict(actual=[1,2,3,4,5,6], bets={name:first for name in NAMES}),
                dict(actual=[3,4,5,6,7,8], bets={name:others for name in NAMES})]
        matrix = score_batch(rows, device='cpu')
        self.assertEqual(matrix.shape,(2,6))
        self.assertTrue(np.all(matrix[0]==6))
        self.assertTrue(np.all(matrix[1]==2))

    def test_uniform_pool_matches_existing_v07_sampler(self):
        from lotto7x7.baseline_v07 import random_pool
        actual, digest = uniform_pool(100, 777)
        old = random_pool(100, 777, 115000101)
        self.assertEqual(actual, [tuple(row['numbers']) for row in old['candidates']])
        self.assertEqual(digest, old['candidate_sha256'])

    def test_catalog_uses_only_previous_draws(self):
        cat = PrefixShapePool()
        image1 = np.zeros((7,7), dtype=np.uint8)
        image1.flat[:6] = 1
        cat.add(image1,101)
        self.assertEqual(cat.count,1)
        with self.assertRaises(ValueError):
            cat.add(image1,101)
        with self.assertRaises(ValueError):
            cat.make(size=10,families=2,seed=22)

    def test_full_replay_cutoff_checkpoint_resume_no_target_leak(self):
        rng=np.random.default_rng(18)
        arr=np.zeros((53,7,7), dtype=np.uint8)
        for i in range(len(arr)):
            arr[i].flat[rng.choice(49,6,replace=False)]=1
        ids=np.arange(115000001,115000054,dtype=np.int64)
        class FastCatalog:
            def __init__(self): self.history=[]
            def add(self, image, draw_id): self.history.append(int(draw_id))
            def make(self, size, families, seed):
                # Candidate generation sees only the preceding cutoff IDs.
                self_owner.assertEqual(self.history[-1], ids[len(self.history)-1])
                return uniform_pool(size,seed)
        self_owner=self
        with tempfile.TemporaryDirectory() as td, mock.patch(
                'lotto7x7.full_history_v072.PrefixShapePool', FastCatalog):
            path=Path(td)/'progress.jsonl'
            report=run_full_history(arr,ids,warmup=50,count=2,size=20,families=5,
                                    seed=649,scenarios=512,device='cpu',checkpoint=path)
            self.assertEqual(report['source_draws'],53)
            self.assertEqual(report['evaluated_draws'],3)
            self.assertEqual(len(report['targets']),3)
            self.assertEqual(len(path.read_text().splitlines()),3)
            repeated=run_full_history(arr,ids,warmup=50,count=2,size=20,
                                    families=5,seed=649,scenarios=512,
                                    device='cpu',checkpoint=path,resume=True)
            self.assertEqual(report['summary'],repeated['summary'])
            self.assertEqual(report['targets'],repeated['targets'])
            with self.assertRaises(FileExistsError):
                run_full_history(arr,ids,warmup=50,count=2,size=20,families=5,
                                 seed=649,scenarios=512,device='cpu',checkpoint=path)
            tampered=arr.copy()
            tampered[-1]=0
            tampered[-1].flat[8:14]=1
            # Same historical information before last target -> same tickets.
            changed=run_full_history(tampered,ids,warmup=50,count=2,size=20,
                                     families=5,seed=649,scenarios=512,device='cpu')
            self.assertEqual(changed['targets'][-1]['bets'],report['targets'][-1]['bets'])
            self.assertNotEqual(changed['targets'][-1]['actual'],report['targets'][-1]['actual'])


    def test_incremental_pool_matches_original_phase1_at_each_cutoff(self):
        """Run on the real repo: cached chronological library must match v0.6."""
        from lotto7x7 import candidate_pool_v06
        if not hasattr(candidate_pool_v06, 'random_sample'):
            self.skipTest('local isolated stub does not include full v0.6 implementation')
        rng = np.random.default_rng(71)
        images = np.zeros((90, 7, 7), dtype=np.uint8)
        for row in images:
            row.flat[rng.choice(49, 6, replace=False)] = 1
        ids = np.arange(115000001, 115000091, dtype=np.int64)
        catalog = PrefixShapePool()
        for t, (im, draw_id) in enumerate(zip(images, ids), start=1):
            catalog.add(im, int(draw_id))
            if t in (50, 75, 90):
                actual, digest = catalog.make(size=40, families=8, seed=649)
                expected = candidate_pool_v06.create_shape_pool(
                    images[:t], ids[:t], pool_size=40, family_count=8,
                    family_policy='uniform-family', sample_count=1, seed=649)
                self.assertEqual(actual, [tuple(row['numbers']) for row in expected['candidates']])
                self.assertEqual(digest, expected['candidate_sha256'])

    def test_parser_accepts_gpu_diagnostics_and_tail_targets(self):
        import argparse
        from lotto7x7.full_history_cli_v072 import configure_parsers
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest='command', required=True)
        configure_parsers(sub)
        obj = parser.parse_args(['full-history-gpu','--device','cuda',
                                 '--tail-targets','5','--warmup','200'])
        self.assertEqual(obj.tail_targets,5)
        self.assertEqual(obj.device,'cuda')
        obj2 = parser.parse_args(['gpu-check'])
        self.assertEqual(obj2.command,'gpu-check')
    def test_invalid_history_and_scenario_count(self):
        with self.assertRaises(ValueError):
            ScenarioEngine(device='cpu',seed=1,scenarios=100)
        a=np.zeros((55,7,7),dtype=np.uint8)
        a[:,:,:6]=1  # invalid: 42 pixels
        ids=np.arange(115000001,115000056)
        with self.assertRaises(ValueError):
            run_full_history(a,ids)
