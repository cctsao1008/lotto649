"""Historical full-six replay tests using fabricated, never official, draw data."""
from __future__ import annotations

import csv
import hashlib
from contextlib import redirect_stdout
from io import StringIO
import math
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from lotto_neural import IID_LOGP
from lotto_replay import (
    TOTAL_SUBSETS, expected_iid_overlap_distribution, lag_midrank_exact,
    export_draw_csv, midrank_fraction_mc, print_exact_six_report, run_replay, top_subset, uniform_subset_pool,
    validate_replay_split,
)
from lotto_synthetic import generate


def official_layout_fixture(path: Path, draws: np.ndarray) -> None:
    """Create a 2018 archive with six main numbers and a separate special."""
    dtype = np.dtype([
        ("draw_id", "<u4"), ("date", "<u4"),
        ("numbers", "u1", (6,)), ("special", "u1"),
    ])
    arr = np.empty(len(draws), dtype=dtype)
    for i, numbers in enumerate(draws):
        dt = date(2018, 1, 1) + timedelta(days=3 * i)
        arr[i]["draw_id"] = 107000001 + i
        arr[i]["date"] = int(dt.strftime("%Y%m%d"))
        arr[i]["numbers"] = numbers[::-1]
        arr[i]["special"] = next(v for v in range(1, 50) if v not in numbers)
    np.save(path, arr, allow_pickle=False)


class HistoricalTopSixReplayTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "fixture.npy"
        self.draws = generate(count=120, mode="iid", seed=990)[0]
        official_layout_fixture(self.path, self.draws)

    def run_fixture(self, **options):
        arguments = {
            "source": self.path, "year": 2018,
            "seed": 23, "window": 3,
            "hidden": 6, "epochs": 2, "patience": 2,
            "batch_size": 32, "train_end": 72,
            "validation_end": 96, "rank_samples": 750,
        }
        arguments.update(options)
        return run_replay(**arguments)

    def test_fixed_split_remains_fixed_when_future_archive_grows(self):
        a = validate_replay_split(120, window=3, train_end=72,
                                  validation_end=96)
        b = validate_replay_split(125, window=3, train_end=72,
                                  validation_end=96)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        self.assertEqual(list(a[2]), list(range(96, 120)))

    def test_invalid_split_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_replay_split(120, window=5, train_end=72,
                                  validation_end=119)
        with self.assertRaises(ValueError):
            validate_replay_split(120, window=5, train_end=3,
                                  validation_end=96)

    def test_iid_overlap_theory_exactly_normalizes(self):
        p = expected_iid_overlap_distribution()
        self.assertAlmostEqual(sum(p.values()), 1.0, places=14)
        self.assertAlmostEqual(
            sum(int(j) * q for j, q in p.items()), 36 / 49, places=14)
        self.assertEqual(TOTAL_SUBSETS, 13983816)
        self.assertAlmostEqual(p["6"], 1 / TOTAL_SUBSETS, places=18)

    def test_uniform_pool_is_reproducible_and_distinct(self):
        p1 = uniform_subset_pool(samples=1000, seed=190)
        p2 = uniform_subset_pool(samples=1000, seed=190)
        np.testing.assert_array_equal(p1, p2)
        self.assertEqual(p1.shape, (1000, 6))
        self.assertTrue(np.all((p1 >= 0) & (p1 < 49)))
        self.assertTrue(np.all(np.diff(p1, axis=1) > 0))

    def test_iid_rank_ties_midpoint_and_zero_uncertainty_for_exact_class(self):
        pool = uniform_subset_pool(samples=200, seed=12)
        frac, error = midrank_fraction_mc(np.zeros(49),
                                          np.array([1, 2, 3, 4, 5, 6]), pool)
        self.assertEqual(frac, 0.5)
        self.assertTrue(error > 0)
        self.assertAlmostEqual(
            0.5 + TOTAL_SUBSETS * frac, (TOTAL_SUBSETS + 1) / 2)

    def test_clear_best_subset_has_near_top_mc_rank(self):
        w = np.full(49, -100.0)
        w[:6] = 5.0
        pool = uniform_subset_pool(samples=1000, seed=1)
        f, _ = midrank_fraction_mc(w, np.arange(1, 7), pool)
        self.assertLessEqual(f, 0.002)

    def test_lag_midrank_exact_direction_and_ties(self):
        self.assertEqual(lag_midrank_exact(beta=0.0, overlap=6), 0.5)
        self.assertAlmostEqual(
            lag_midrank_exact(beta=2.0, overlap=6),
            0.5 / TOTAL_SUBSETS, places=14)
        self.assertAlmostEqual(
            0.5 + TOTAL_SUBSETS * lag_midrank_exact(beta=2.0, overlap=6),
            1.0, places=12)
        self.assertAlmostEqual(
            lag_midrank_exact(beta=-2.0, overlap=6),
            1.0 - 0.5 / TOTAL_SUBSETS, places=14)
        self.assertAlmostEqual(
            lag_midrank_exact(beta=2.0, overlap=0) +
            lag_midrank_exact(beta=-2.0, overlap=0), 1.0)
        with self.assertRaises(ValueError):
            lag_midrank_exact(beta=1.0, overlap=7)

    def test_top_subset_uses_seeded_tie_breaking_without_target(self):
        zeros = np.zeros(49)
        top1 = top_subset(zeros, seed=5, draw_id=115000100, model_index=0)
        top2 = top_subset(zeros, seed=5, draw_id=115000100, model_index=0)
        self.assertEqual(top1, top2)
        self.assertEqual(len(top1), 6)
        self.assertEqual(len(set(top1)), 6)
        unique = np.full(49, -100.0)
        unique[:6] = np.arange(6, 0, -1)
        self.assertEqual(top_subset(unique, seed=5, draw_id=1, model_index=1),
                         [1, 2, 3, 4, 5, 6])

    def test_replay_uses_test_only_and_is_read_only(self):
        old_hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        report = self.run_fixture()
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), old_hash)
        self.assertEqual(report["source_sha256"], old_hash)
        self.assertEqual(report["evaluated_draws"], 24)
        self.assertEqual(report["split_boundaries"]["train"], [3, 71])
        self.assertEqual(report["split_boundaries"]["validation"], [72, 95])
        self.assertEqual(report["split_boundaries"]["test_start"], 96)
        self.assertEqual(len(report["individual_draws"]), 24)
        self.assertEqual(set(report["summary_by_model"]), {"iid", "static", "lag", "mlp"})
        self.assertEqual(set(report["training"]["validation_scores"]),
                         {"iid", "static", "lag", "mlp"})
        self.assertAlmostEqual(
            report["iid_theory"]["expected_main_numbers_matched_per_draw"],
            36/49)
        self.assertAlmostEqual(
            report["summary_by_model"]["iid"]["mean_log_score_lift_vs_iid_nats"], 0.0)
        for item in report["individual_draws"]:
            self.assertEqual(len(item["actual_main_six"]), 6)
            for label, value in item["models"].items():
                self.assertEqual(len(value["diagnostic_top_six"]), 6)
                self.assertGreaterEqual(value["matched_main_numbers"], 0)
                self.assertLessEqual(value["matched_main_numbers"], 6)
                self.assertEqual(
                    len(set(value["diagnostic_top_six"]).intersection(
                        item["actual_main_six"])), value["matched_main_numbers"])
                self.assertEqual(value["exact_six_match"],
                                 value["matched_main_numbers"] == 6)
                self.assertTrue(math.isfinite(value["observed_set_log_probability"]))
                self.assertTrue(0.0 <= value["rank_fraction_lower_is_better"] <= 1.0)
                self.assertGreaterEqual(value["observed_set_midrank"], 1.0)
                self.assertLessEqual(value["observed_set_midrank"], TOTAL_SUBSETS)
                if label in ("iid", "lag"):
                    self.assertEqual(value["rank_fraction_mc_95pct_halfwidth"], 0.0)
        for label, row in report["summary_by_model"].items():
            self.assertEqual(sum(row["matched_main_number_histogram"].values()), 24)
            self.assertEqual(row["exact_six_matches"],
                             row["matched_main_number_histogram"]["6"])

    def test_future_result_change_does_not_change_first_forecast_top_six(self):
        a = self.run_fixture()
        data = np.load(self.path, allow_pickle=False)
        original = sorted(map(int, data[96]["numbers"]))
        replacement = [1, 2, 3, 4, 5, 6]
        if original == replacement:
            replacement = [2, 3, 4, 5, 6, 7]
        data[96]["numbers"] = replacement
        data[96]["special"] = next(v for v in range(1, 50) if v not in replacement)
        np.save(self.path, data, allow_pickle=False)
        b = self.run_fixture()
        first_a = a["individual_draws"][0]["models"]
        first_b = b["individual_draws"][0]["models"]
        for name in ("iid", "static", "lag", "mlp"):
            self.assertEqual(first_a[name]["diagnostic_top_six"],
                             first_b[name]["diagnostic_top_six"])
        self.assertNotEqual(a["individual_draws"][0]["actual_main_six"],
                            b["individual_draws"][0]["actual_main_six"])

    def test_csv_export_has_one_row_per_draw_and_candidate(self):
        report = self.run_fixture()
        output = self.path.parent / "replay.csv"
        export_draw_csv(report, output)
        with output.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 24 * 4)
        self.assertEqual(set(row["model"] for row in rows),
                         {"iid", "static", "lag", "mlp"})
        self.assertEqual(rows[0]["actual_main_six"],
                         " ".join(f"{v:02d}" for v in
                                  report["individual_draws"][0]["actual_main_six"]))
        self.assertEqual(rows[0]["model"], "iid")

    def test_exact_six_report_treats_partial_matches_as_failure(self):
        report = {
            "individual_draws": [
                {"date": 20250103, "draw_id": 114000001,
                 "actual_main_six": [1, 2, 3, 4, 5, 6],
                 "models": {"mlp": {"diagnostic_top_six": [1, 2, 3, 4, 5, 7],
                                    "exact_six_match": False}}},
                {"date": 20250107, "draw_id": 114000002,
                 "actual_main_six": [8, 9, 10, 11, 12, 13],
                 "models": {"mlp": {"diagnostic_top_six": [8, 9, 10, 11, 12, 13],
                                    "exact_six_match": True}}},
            ],
            "summary_by_model": {
                name: {"exact_six_matches": 1 if name == "mlp" else 0,
                       "draws": 2}
                for name in ("iid", "static", "lag", "mlp")
            },
        }
        output = StringIO()
        with redirect_stdout(output):
            print_exact_six_report(report, model="mlp", show_draws=True)
        content = output.getvalue()
        self.assertIn("only 6/6 counts as PASS", content)
        self.assertIn("20250103", content)
        self.assertIn("20250107", content)
        self.assertEqual(sum(line.rstrip().endswith("FAIL")
                             for line in content.splitlines()), 1)
        self.assertEqual(sum(line.rstrip().endswith("PASS")
                             for line in content.splitlines()), 1)
        with self.assertRaises(ValueError):
            print_exact_six_report(report, model="unsupported")

    def test_no_year_in_test_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no observations"):
            self.run_fixture(year=2025)


if __name__ == "__main__":
    unittest.main()
