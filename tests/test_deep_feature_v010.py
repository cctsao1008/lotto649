"""Focused v0.10 verification; never reruns old 216+ regression cases."""
from __future__ import annotations

import argparse
import json
import contextlib
import io
import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest

import numpy as np
import torch

from lotto7x7.deep_feature_v010 import (
    DeepFeatureNet, causal_rolling_mean, next_windows, deep_bank,
    motif_scan, train_prefix, discover_deep,
)
from lotto7x7.feature_discovery_v09 import _scan_periods, iid_six_hot
from lotto7x7.deep_feature_cli_v010 import configure_parsers, run


def fixture(n: int, seed: int = 100) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 49), dtype=np.float32)
    for i in range(n):
        x[i, rng.choice(49, size=6, replace=False)] = 1.
    return x.reshape(n, 7, 7), np.arange(93000001, 93000001 + n, dtype=np.int64)


class DeepFeatureV010Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_01_causal_rolling_never_sees_future(self):
        x = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1)
        y = causal_rolling_mean(x, 4)
        self.assertEqual(y.shape, x.shape)
        self.assertAlmostEqual(float(y[0, 4, 0]), 2.5)
        modified = x.clone()
        modified[:, 7:] = 9999
        torch.testing.assert_close(causal_rolling_mean(modified, 4)[:, :7], y[:, :7])

    def test_02_next_window_excludes_target(self):
        a, _ = fixture(90)
        tensor = torch.from_numpy(a)
        inds = torch.tensor([16, 22])
        w = next_windows(tensor, inds, 8)
        torch.testing.assert_close(w[0], tensor[8:16])
        changed = tensor.clone()
        changed[16:] = 0
        torch.testing.assert_close(next_windows(changed, torch.tensor([16]), 8), w[:1])

    def test_03_ae_gru_dimensions(self):
        net = DeepFeatureNet(embedding=8, hidden=12)
        x = torch.rand(3, 8, 7, 7)
        dec, pred = net(x)
        self.assertEqual(tuple(dec.shape), (3, 8, 49))
        self.assertEqual(tuple(pred.shape), (3, 49))
        self.assertEqual(tuple(net.encode(x).shape), (3, 8, 8))

    def test_04_multiscale_inputs_only_past(self):
        net = DeepFeatureNet(8, 12)
        z = torch.randn(2, 8, 8)
        altered = z.clone()
        altered[:, 6:] *= 10
        torch.testing.assert_close(net.multiscale(z)[:, :6], net.multiscale(altered)[:, :6])
        self.assertEqual(net.multiscale(z).shape, (2, 8, 40))

    def test_05_frozen_representation_future_mutation_is_causal(self):
        images, _ = fixture(95)
        net = DeepFeatureNet(8, 12)
        s = torch.from_numpy(images[:50])[None]
        features, names = deep_bank(net, s, window=8, chunk=12)
        changed = s.clone()
        changed[:, 35:] = 0
        other, _ = deep_bank(net, changed, window=8, chunk=12)
        torch.testing.assert_close(other[:, :35-8+1], features[:, :35-8+1])
        self.assertEqual(len(names), 20)

    def test_06_motif_search_detects_planted_repetition(self):
        # Two exact nonoverlapping 6-step motif subsequences among noisy features.
        torch.manual_seed(22)
        bank = torch.randn(1, 80, 6)
        bank[:, 49:57] = bank[:, 10:18]
        names = [f"cnn_latent/feature_{i}" for i in range(3)] + [f"gru_state/feature_{i}" for i in range(3)]
        score, found = motif_scan(bank, names, (8,), top_count=5)
        self.assertGreater(float(score[0]), 0.99)
        self.assertEqual(found[0]["subsequence_length"], 8)
        self.assertEqual({found[0]["start_index"], found[0]["match_index"]}, {10, 49})

    def test_07_latent_spectral_scan_planted_period(self):
        n = 96
        t = torch.arange(n, dtype=torch.float32)
        signal = torch.cos(t * 2 * np.pi / 8)
        bank = torch.stack((signal, torch.sin(t * 2 * np.pi / 8)), dim=-1)[None]
        stats, rows = _scan_periods(bank, ["cnn_latent/f0", "cnn_latent/f1"],
                                    recent_windows=(64,), half_lives=(0,),
                                    min_period=2., max_period=24., top_count=3)
        self.assertGreater(float(stats[0]), 0.9)
        self.assertAlmostEqual(rows[0]["period_draws"], 8, places=2)

    def test_08_iid_null_is_legal(self):
        gen = torch.Generator().manual_seed(123)
        a = iid_six_hot(6, 33, generator=gen, device=torch.device('cpu'))
        self.assertEqual(a.shape, (6, 33, 7, 7))
        self.assertTrue(bool((a.sum(dim=(-1, -2)) == 6).all()))

    def test_09_training_is_prefix_only(self):
        x, _ = fixture(120)
        data = torch.from_numpy(x)
        net, rec = train_prefix(data, train_draws=70, validation=8, window=8,
                                embedding=8, hidden=10, epochs=2,
                                patience=2, batch_size=16, learning_rate=.002,
                                reconstruction_weight=.7, seed=8)
        self.assertEqual(rec['training_targets'], [8, 61])
        self.assertEqual(rec['validation_targets'], [62, 69])
        self.assertTrue(1 <= rec['best_epoch'] <= 2)
        self.assertTrue(np.isfinite(rec['best_validation_brier']))

    def test_10_end_to_end_score_nulls_and_splits(self):
        x, ids = fixture(150)
        report, trained = discover_deep(x, ids, device="cpu", train_draws=80, holdout_draws=30,
                               window=8, validation=8, embedding=8, hidden=8,
                               epochs=2, patience=2, batch_size=16,
                               recent_windows=(32,), half_lives=(0, 16),
                               motif_windows=(4, 8), min_period=2, max_period=16,
                               surrogates=9, surrogate_batch=3, top_count=3,
                               seed=42, return_model=True)
        self.assertEqual(trained.embedding, 8)
        self.assertEqual(trained.hidden, 8)
        self.assertEqual(len(trained.state_dict()), len(DeepFeatureNet(8, 8).state_dict()))
        self.assertEqual(report['schema'], 'lotto7x7.deep_feature.v0.10')
        self.assertEqual(report['splits']['representation_train'], [0, 79])
        self.assertEqual(report['splits']['discovery_scan'], [80, 119])
        self.assertEqual(report['splits']['heldout_evaluation'], [120, 149])
        self.assertEqual(report['deep_feature_channels'], 16)
        self.assertEqual(report['heldout_next_draw']['evaluated_targets'], 30)
        for null in report['nulls'].values():
            self.assertEqual(len(null['period_null_maxima']), 9)
            self.assertEqual(len(null['motif_null_maxima']), 9)
            self.assertGreaterEqual(null['period_global_p'], 0.1)
            self.assertLessEqual(null['motif_global_p'], 1)
        json.dumps(report, allow_nan=False)

    def test_11_reject_invalid_splits(self):
        a, ids = fixture(150)
        for config in ({'train_draws': 119, 'holdout_draws': 30},
                       {'train_draws': 80, 'holdout_draws': 30, 'motif_windows': (30,)},
                       {'train_draws': 80, 'holdout_draws': 30, 'surrogates': 4}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                discover_deep(a, ids, window=8, validation=8, **config)

    def test_12_cli_options(self):
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest='command')
        configure_parsers(sub)
        args = parser.parse_args(['deep-feature-gpu', '--device', 'cuda',
                                  '--surrogates', '99', '--window', '64',
                                  '--motif-windows', '8', '16', '32'])
        self.assertEqual(args.command, 'deep-feature-gpu')
        self.assertEqual(args.surrogates, 99)
        self.assertEqual(args.motif_windows, [8, 16, 32])

    def test_13_cli_writes_loadable_checkpoint_without_overwrite(self):
        imgs, ids = fixture(150)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'audit.json'
            checkpoint = Path(tmp) / 'encoder.pt'
            parser = argparse.ArgumentParser()
            action = parser.add_subparsers(dest='command')
            configure_parsers(action)
            args = parser.parse_args([
                'deep-feature-gpu', '--device', 'cpu',
                '--train-draws', '80', '--holdout-draws', '30',
                '--window', '8', '--validation', '8', '--embedding', '8',
                '--hidden', '8', '--epochs', '1', '--patience', '1',
                '--batch-size', '16', '--surrogates', '9', '--surrogate-batch', '3',
                '--recent-windows', '32', '--z-half-lives', '0',
                '--motif-windows', '4', '8', '--max-period', '16',
                '--top', '3', '--out', str(out), '--model-out', str(checkpoint)
            ])
            with patch('lotto7x7.deep_feature_cli_v010.load_history',
                       return_value=(imgs, np.zeros(len(imgs)), ids)):
                with contextlib.redirect_stdout(io.StringIO()):
                    run(args)
                report = json.loads(out.read_text(encoding='utf-8'))
                saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                loaded = DeepFeatureNet(8, 8)
                loaded.load_state_dict(saved['state_dict'])
                self.assertEqual(report['model_checkpoint']['trained_draws'], 80)
                self.assertEqual(saved['schema'], 'lotto7x7.deep_feature_checkpoint.v0.10')
                with self.assertRaises(FileExistsError):
                    run(args)


if __name__ == '__main__':
    unittest.main()
