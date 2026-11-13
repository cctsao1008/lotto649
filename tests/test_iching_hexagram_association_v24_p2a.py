"""Synthetic/adversarial P2A checks; no historical source mutation."""
from __future__ import annotations

import csv
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lotto7x7 import iching_hexagram_association_v24_p2a as m


class HexagramAssociationP2ATests(unittest.TestCase):
    @staticmethod
    def data():
        rows = []
        for sky in range(64):
            for j in range(3):
                code = (sky + j * 13) % 64
                d = date(2024 if j < 2 else 2025, 1, 1) + timedelta(days=sky * 2 + j)
                rows.append({"date": str(d), "year": d.year, "weekday": d.weekday(),
                    "sky": sky, "outcome": code, "sum": 50 + sky + j,
                    "odd": code.bit_count(), "shape": 1 + ((sky + j) % 6),
                    "hamming": (sky ^ code).bit_count()})
        return rows

    def test_exact_parity_universe(self):
        p = m.exact_parity_masses()
        self.assertEqual(len(p), 64)
        self.assertEqual(sum(p), 13983816)
        self.assertEqual(min(p), 134596)
        self.assertEqual(max(p), 296010)
        self.assertNotEqual(len(set(p)), 1)

    def test_fair_moments(self):
        t = m.fair_reference()
        self.assertEqual(t["sum"], {"mean": 150, "variance": 1075.0})
        self.assertAlmostEqual(t["odd_count"]["mean"], 150 / 49)
        self.assertEqual(sum(t["shape_L1_components"]["exact_universe_counts_components_1_to_6"]), m.UNIVERSE)

    def test_gray_cyclic_neighbors(self):
        bits = [m.gray3(i) for i in range(8)]
        self.assertEqual(len(set(bits)), 8)
        self.assertTrue(all((bits[k] ^ bits[(k + 1) % 8]).bit_count() == 1 for k in range(8)))

    def test_sorted_parity_is_postdraw_label(self):
        self.assertEqual(m.outcome_code((2, 4, 6, 8, 10, 12)), 0)
        self.assertEqual(m.outcome_code((1, 3, 5, 7, 9, 11)), 63)
        self.assertEqual(m.outcome_code((2, 3, 8, 13, 22, 49)), 42)
        for x in [(1, 1, 2, 3, 4, 5), (7, 6, 5, 4, 3, 2), (1, 2, 3, 4, 5, 50)]:
            with self.assertRaises(ValueError): m.outcome_code(x)

    def test_original_7x7_eight_neighbor_graph(self):
        self.assertEqual(m.shape_l1((1, 2, 3, 4, 5, 6)), 1)
        self.assertEqual(m.shape_l1((1, 9, 17, 25, 33, 41)), 1)
        self.assertEqual(m.shape_l1((1, 7, 22, 28, 43, 49)), 6)

    def test_64_states_covered_by_synthetic(self):
        rows = self.data()
        design = m.build_null_design(rows)
        self.assertEqual(len(rows), 192)
        self.assertEqual(sum(design["sky_counts"]), 192)
        self.assertEqual(len(design["strata"]), design["strata_year_weekday"])
        self.assertAlmostEqual(sum(design["expected_hex"]), 192)
        self.assertAlmostEqual(sum(design["expected_shape"]), 192)
        self.assertEqual(len(design["expected_hex"]), 4096)
        self.assertEqual(len(design["expected_shape"]), 384)

    def test_conditional_permutation_preserves_each_year_weekday(self):
        rows = self.data()
        design = m.build_null_design(rows)
        perm = m.independent_permutation(len(rows), design["strata"], random.Random(651))
        self.assertEqual(sorted(perm), list(range(len(rows))))
        for group in design["strata"]:
            self.assertEqual(sorted(perm[i] for i in group), sorted(group))
        self.assertTrue(any(i != j for i, j in enumerate(perm)))

    def test_conditional_expectation_matches_exact_hamming(self):
        rows = self.data()
        d = m.build_null_design(rows)
        expected_manual = 0
        for group in d["strata"]:
            for i in group:
                for j in group:
                    expected_manual += (rows[i]["sky"] ^ rows[j]["outcome"]).bit_count() / len(group)
        self.assertAlmostEqual(d["hamming_expected_sum"], expected_manual)

    def test_all_six_statistics_nonnegative_and_finite(self):
        rows = self.data()
        d = m.build_null_design(rows)
        stats = m.stats_for_assignment(rows, d, list(range(len(rows))))
        self.assertEqual(set(stats), set(m.ENDPOINTS))
        self.assertTrue(all(0 <= x < 1e8 for x in stats.values()))

    def test_small_calibration_reproducibility_and_bonferroni(self):
        rows = self.data()
        a, d = m.calibrate(rows, reps=99, seed=649)
        b, _ = m.calibrate(rows, reps=99, seed=649)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 6)
        for entry in a.values():
            p = entry["permutation_p_plus_one"]
            self.assertGreaterEqual(p, 0.01)
            self.assertLessEqual(p, 1.0)
            self.assertEqual(entry["bonferroni_six_global_p"], min(1., 6 * p))
        with self.assertRaises(ValueError):
            m.calibrate(rows, reps=98)

    def test_different_rng_seed_changes_permutation(self):
        d = m.build_null_design(self.data())
        self.assertNotEqual(m.independent_permutation(d["n"], d["strata"], random.Random(13)),
                            m.independent_permutation(d["n"], d["strata"], random.Random(17)))

    def test_descriptive_not_predictive(self):
        rows = self.data()
        states, summary = m.describe_states(rows, m.fair_reference())
        self.assertEqual(len(states), 64)
        self.assertTrue(all(s["draw_count"] == 3 for s in states))
        self.assertTrue(all(s["report_scope"] == "RETROSPECTIVE_DESCRIPTIVE_NOT_PREDICTIVE" for s in states))
        self.assertTrue(0 <= summary["fair_iid_exact_parity_mass_weighted_hamming_expected"] <= 6)

    def test_header_duplicates_fail(self):
        with self.assertRaises(ValueError):
            m.read_csv_rows(b"date,date\n2026-01-01,2026-01-02\n")

    def test_frozen_source_sha_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "fake.csv"
            audit = Path(tmp) / "fake.json"
            bad.write_bytes(b"not-the-archived-data")
            audit.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                m.load_frozen(bad, audit)

    def test_exclusive_output_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frozen.csv"
            m.write_exclusive_bytes(path, b"first")
            with self.assertRaises(FileExistsError): m.write_exclusive_bytes(path, b"other")
            self.assertEqual(path.read_bytes(), b"first")


if __name__ == "__main__":
    unittest.main()
