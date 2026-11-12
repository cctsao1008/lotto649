"""v0.31 H2: causal, marginal-proper-score geometry increment regression."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from lotto7x7.distribution_dynamics_v030 import _uniform_history
from lotto7x7.geometry_convergence_v029 import _validate_draws
from lotto7x7.geometry_repair_v029 import P0_SHA256
from lotto7x7.geometry_v029 import catalog
from lotto7x7.log_product_v027 import product_bin
from lotto7x7.incremental_geometry_v031 import (
    ALPHA_DEFAULT, FAIR_P, categorical_features, geometry_features,
    legal_marginal_projection, model_features, one_step_scores,
    analyze_incremental_geometry, forecast_marginals, main,
)
from lotto7x7.log_product_v027 import _grid, theoretical_log_moments
from lotto7x7.distribution_dynamics_v030 import exact_null_dynamics
from lotto7x7.shape_taxonomy_v015 import classify_numbers


def fixture(n=90, seed=649):
    return _validate_draws(_uniform_history(np.random.default_rng(seed), n))


class TestIncrementalGeometryV031(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid = _grid(49, 6, "0.5")
        cls.log_theory = theoretical_log_moments()
        cls.geo_theory = exact_null_dynamics()

    def test_frozen_geometry_version_and_exact_log_bin(self):
        self.assertEqual(catalog()["definition_sha256"], P0_SHA256)
        draws = fixture(90)
        self.assertEqual(len(classify_numbers(draws)["edge_code"]), len(draws))
        for row in draws[:5]:
            p = int(np.prod(row.astype(np.int64)))
            self.assertGreaterEqual(product_bin(p, self.grid), 0)

    def test_same_l3_sum_logbin_different_geometry(self):
        certificate = np.asarray([[3, 10, 19, 23, 31, 46],
                                  [4, 8, 12, 27, 34, 47]], dtype=np.int16)
        a = np.vstack((certificate, np.tile(certificate[0], (23, 1))))
        shape = classify_numbers(a)
        self.assertEqual(int(shape["edge_code"][0]), int(shape["edge_code"][1]))
        self.assertEqual(int(a[0].sum()), int(a[1].sum()))
        prod = np.prod(a.astype(np.int64), axis=1)
        bins = np.searchsorted(self.grid["integer_product_thresholds"], prod, side="right") - 1
        self.assertEqual(int(bins[0]), int(bins[1]))
        values = geometry_features(a, self.geo_theory)
        self.assertFalse(np.allclose(values[0], values[1]))

    def test_features_are_frozen_shapes_and_finite(self):
        x = model_features(fixture(90), self.grid, self.log_theory, self.geo_theory)
        self.assertEqual(x["m0_compressed"].shape, (90, 13))
        self.assertEqual(x["geometry"].shape, (90, 4))
        self.assertEqual(x["combined"].shape, (90, 16))
        self.assertTrue(all(np.isfinite(y).all() for y in x.values()))
        self.assertTrue(np.allclose(x["m0_compressed"][:, 0], 1))
        self.assertTrue(np.allclose(x["geometry"][:, 0], 1))

    def test_projection_enforces_legal_six_of_49_marginals(self):
        rng = np.random.default_rng(812)
        raw = rng.normal(0.1, 0.8, size=(8, 49))
        projected = legal_marginal_projection(raw)
        self.assertTrue(np.all(projected > 0))
        self.assertTrue(np.all(projected < 1))
        self.assertTrue(np.allclose(projected.sum(axis=1), 6, atol=1e-9))
        baseline = legal_marginal_projection(np.full((1, 49), FAIR_P))
        self.assertTrue(np.allclose(baseline, FAIR_P, atol=1e-10))
        with self.assertRaises(ValueError):
            legal_marginal_projection(np.zeros((49,)))
        with self.assertRaises(ValueError):
            legal_marginal_projection(np.full((1, 49), np.nan))

    def test_causality_prediction_origin_excludes_target_and_future(self):
        draws = fixture(90)
        one = one_step_scores(draws, np.array([55, 65, 75]), ALPHA_DEFAULT,
                              self.grid, self.log_theory, self.geo_theory, detailed=True)
        changed = draws.copy()
        changed[60:] = fixture(30, 8888)
        two = one_step_scores(changed, np.array([55, 65, 75]), ALPHA_DEFAULT,
                              self.grid, self.log_theory, self.geo_theory, detailed=True)
        self.assertEqual(one["per_target"][0], two["per_target"][0])
        self.assertNotEqual(one["per_target"][1], two["per_target"][1])
        self.assertEqual(one["per_target"][0]["training_pairs"], 54)

    def test_fair_brier_matches_analytic_proper_score(self):
        draws = fixture(90)
        targets = np.arange(55, 90, 5, dtype=int)
        a = one_step_scores(draws, targets, ALPHA_DEFAULT, self.grid,
                            self.log_theory, self.geo_theory, detailed=False)
        theoretical = FAIR_P * (1.0 - FAIR_P)
        self.assertAlmostEqual(a["mean_brier"]["iid"], theoretical, places=13)
        self.assertTrue(a["forecasts_sum_to_six"])
        self.assertEqual(len(a["brier_improvements"]), 4)

    def test_training_targets_shift_causally_with_exact_expanding_pairs(self):
        draws = fixture(90)
        feats = model_features(draws, self.grid, self.log_theory, self.geo_theory)["geometry"]
        y = np.zeros((len(draws), 49))
        y[np.arange(len(draws))[:, None], draws-1] = 1.0
        early = forecast_marginals(feats, y, np.array([60]), ALPHA_DEFAULT)
        altered = y.copy()
        altered[60:] = 0
        same = forecast_marginals(feats, altered, np.array([60]), ALPHA_DEFAULT)
        self.assertTrue(np.allclose(early, same, atol=1e-14))
        altered[59] = 0
        different = forecast_marginals(feats, altered, np.array([60]), ALPHA_DEFAULT)
        self.assertFalse(np.allclose(early, different))

    def test_whole_pipeline_is_seeded_and_controls_retrained(self):
        draws = fixture(90)
        a = analyze_incremental_geometry(draws, start_draw=50, step=10, iid_reps=3)
        b = analyze_incremental_geometry(draws, start_draw=50, step=10, iid_reps=3)
        self.assertEqual(a, b)
        self.assertEqual(a["targets"], 4)
        self.assertEqual(a["IID_full_pipeline"]["replicates"], 3)
        self.assertEqual(a["primary"]["name"], "combined_vs_m0")
        self.assertGreaterEqual(a["IID_full_pipeline"]["familywise_max_Z_upper_tail_p"], 0.25)
        self.assertIn("EXPLORATORY", a["decision"])

    def test_invalid_history_and_configuration_fail_closed(self):
        draws = fixture(90)
        cases = [dict(start_draw=90), dict(start_draw=20), dict(step=0),
                 dict(step=101), dict(iid_reps=2), dict(seed=-1), dict(ridge=0)]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                analyze_incremental_geometry(draws, **kwargs)
        bad = draws.copy()
        bad[0, 0] = bad[0, 1]
        with self.assertRaises(ValueError):
            analyze_incremental_geometry(bad)

    def test_cli_audit_and_exclusive_output(self):
        draws = fixture(90)
        with TemporaryDirectory(dir="results") as td:
            output = str(Path(td) / "audit.json")
            with patch("lotto7x7.incremental_geometry_v031.audit_repository_inputs",
                       return_value=(draws, {"status": "TEST_SOURCE_GATE"})):
                main(["--start-draw", "50", "--step", "10", "--iid-reps", "3", "--out", output])
                with self.assertRaises(FileExistsError):
                    main(["--start-draw", "50", "--step", "10", "--iid-reps", "3", "--out", output])
            data = json.loads(Path(output).read_text())
            self.assertEqual(data["source_gate"]["status"], "TEST_SOURCE_GATE")
            self.assertEqual(data["targets"], 4)
            with patch("lotto7x7.incremental_geometry_v031.audit_repository_inputs",
                       side_effect=ValueError("source mismatch")):
                with self.assertRaises(ValueError):
                    main(["--start-draw", "50", "--step", "10", "--iid-reps", "3",
                          "--out", str(Path(td)/"failed.json")])
                self.assertFalse((Path(td)/"failed.json").exists())


if __name__ == "__main__":
    unittest.main()
