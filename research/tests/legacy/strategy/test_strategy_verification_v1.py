"""Independent strategy verifier: successful audit and deliberate failure cases."""
from __future__ import annotations

import copy
import unittest

import numpy as np

from lotto7x7.portfolio_benchmark_v030 import analyze_portfolios
from lotto7x7.strategy_verification_v1 import (
    _features, _pool, _theory, _ticket, _unique_count, _masks,
    verify_selection,
)


class TestStrategyVerificationV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(20261112)
        numbers = np.arange(1, 50, dtype=np.int16)
        cls.history = np.sort(rng.permuted(
            np.broadcast_to(numbers, (90, 49)), axis=1)[:, :6], axis=1)
        cls.report = analyze_portfolios(cls.history, pool_size=128,
                                        random_portfolios=24,
                                        min_hybrid_unique=22,
                                        iid_reps=3, seed=649)
        cls.provenance = {"status": "SYNTHETIC_FOR_TESTS_ONLY"}
        cls.report["source_gate"] = cls.provenance

    def audit(self, report=None, source_gate=None):
        return verify_selection(copy.deepcopy(self.report if report is None else report),
                                self.history, source_gate=(self.provenance if source_gate is None
                                                           else source_gate))

    def test_01_synthetic_selection_passes_independent_replay(self):
        result = self.audit()
        self.assertEqual(result["decision"], "STRATEGY_COMPLIANCE_PASS", result["failures"])
        self.assertGreaterEqual(len(result["checks"]), 20)
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["per_arm"]["coverage"]["unique_numbers"], 24)

    def test_02_ticket_legality_enforced(self):
        for row in ([1, 2, 3, 4, 5, 5], [1, 2, 3, 4, 5, 50],
                    [2, 1, 3, 4, 5, 6], [1, 2, 3, 4, 5, True]):
            with self.subTest(ticket=row), self.assertRaises(ValueError):
                _ticket(row)

    def test_03_reported_objective_tampering_fails(self):
        tampered = copy.deepcopy(self.report)
        tampered["observed"]["arms"]["hybrid"]["counterfactual_energy_repair_gain"] += 1
        result = self.audit(tampered)
        self.assertEqual(result["decision"], "STRATEGY_COMPLIANCE_FAIL")
        self.assertFalse(result["checks"]["hybrid.objective"])

    def test_04_candidate_mutation_fails(self):
        tampered = copy.deepcopy(self.report)
        tampered["observed"]["arms"]["repair"]["research_only_tickets"][0] = [1, 2, 3, 4, 5, 6]
        result = self.audit(tampered)
        self.assertEqual(result["decision"], "STRATEGY_COMPLIANCE_FAIL")
        self.assertFalse(result["checks"]["repair.selection_trace"])

    def test_05_config_mutation_fails(self):
        tampered = copy.deepcopy(self.report)
        tampered["configuration"]["min_hybrid_unique"] = 21
        result = self.audit(tampered)
        self.assertEqual(result["decision"], "STRATEGY_COMPLIANCE_FAIL")
        self.assertFalse(result["checks"]["config_digest"])

    def test_06_pool_hash_mutation_fails(self):
        tampered = copy.deepcopy(self.report)
        tampered["observed"]["pool_sha256"] = "0" * 64
        result = self.audit(tampered)
        self.assertEqual(result["decision"], "STRATEGY_COMPLIANCE_FAIL")
        self.assertFalse(result["checks"]["pool_digest"])

    def test_07_source_provenance_mutation_fails(self):
        tampered = copy.deepcopy(self.report)
        tampered["source_gate"]["status"] = "UNVERIFIED"
        result = self.audit(tampered)
        self.assertEqual(result["decision"], "STRATEGY_COMPLIANCE_FAIL")
        self.assertFalse(result["checks"]["source_gate"])

    def test_08_no_provenance_is_not_pass(self):
        result = verify_selection(copy.deepcopy(self.report), self.history)
        self.assertEqual(result["decision"], "STRATEGY_COMPLIANCE_FAIL")
        self.assertFalse(result["checks"]["source_gate"])

    def test_09_random_controls_mutation_fails(self):
        tampered = copy.deepcopy(self.report)
        tampered["observed"]["random_portfolio_controls"]["repair_gain_median"] += 0.5
        result = self.audit(tampered)
        self.assertEqual(result["decision"], "STRATEGY_COMPLIANCE_FAIL")
        self.assertFalse(result["checks"]["random_controls"])

    def test_10_independent_pair_features_match_frozen_reference(self):
        from lotto7x7.geometry_repair_v029 import (
            exact_structural_covariance, _whitened_features,
        )
        original = exact_structural_covariance()
        mean, std, w = _theory()
        self.assertTrue(np.allclose(mean, original["mean"], atol=1e-9))
        self.assertTrue(np.allclose(w, original["whitener"], atol=1e-10))
        self.assertTrue(np.allclose(_features(self.history, mean, std, w),
            _whitened_features(self.history, mean, std, w), atol=1e-10))

    def test_11_pool_deterministic_legal(self):
        a = _pool(np.random.default_rng(649), 128)
        b = _pool(np.random.default_rng(649), 128)
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(len(set(map(tuple, a))), 128)
        self.assertEqual(_unique_count(np.arange(4), _masks(a)),
                         len(set(a[:4].ravel().tolist())))

    def test_12_invalid_missing_fields_fail_closed(self):
        tampered = copy.deepcopy(self.report)
        del tampered["observed"]["arms"]["coverage"]["unique_numbers"]
        result = self.audit(tampered)
        self.assertEqual(result["decision"], "STRATEGY_COMPLIANCE_FAIL")
        self.assertTrue(result["failures"])


if __name__ == "__main__":
    unittest.main()