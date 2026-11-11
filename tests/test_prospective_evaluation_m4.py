"""M4 read-only shadow evaluation: empty, partial, complete, tampering."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from lotto7x7 import prospective_evaluation_m4 as m4

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = [
    {"draw_id": 115000105 + i, "date": day}
    for i, day in enumerate((
        "2026-11-13", "2026-11-17", "2026-11-20",
        "2026-11-24", "2026-11-27", "2026-12-01",
        "2026-12-04", "2026-12-08", "2026-12-11",
    ))
]
GEOMETRY_WIN = [7, 10, 12, 19, 30, 34]
CONTROL_WIN = [3, 6, 19, 22, 27, 42]
NO_AWARD = [1, 2, 13, 14, 15, 16]


def draw_for(n, numbers=GEOMETRY_WIN, bonus=49):
    return {"draw_id": SCHEDULE[n]["draw_id"], "date": SCHEDULE[n]["date"],
            "main": list(numbers), "bonus": bonus}


class TestProspectiveEvaluationM4(unittest.TestCase):
    def test_frozen_seal_is_original_blob(self):
        seal = m4.verify_p5a_seal()
        self.assertEqual("v029-p5a-geometry-shadow-static9-20261113",
                         seal["experiment_id"])
        self.assertEqual(9, seal["experiment"]["fixed_draw_count"])

    def test_current_archive_pending_empty_with_no_invented_draw(self):
        data, evidence = m4.read_archived_verified_prefix(ROOT / "data/csv",
                                                          ROOT / "data/bin")
        self.assertEqual([], data)
        self.assertEqual(0, evidence["admitted_draws"])
        report = m4.evaluate_verified_rows(data)
        self.assertEqual("PENDING_NO_EARLY_SIGNIFICANCE_OR_GO", report["status"])
        self.assertEqual(0, report["observed_draw_count"])
        self.assertEqual(9, report["pending_draw_count"])
        self.assertEqual(115000105, report["next_expected_draw"]["draw_id"])
        self.assertIsNone(report["final_one_sided_exact_p"])
        self.assertFalse(report["winner_declared"])
        self.assertFalse(report["predictive_go"])
        self.assertEqual([], report["observed_draws"])

    def test_one_verified_draw_is_descriptive_not_inferential(self):
        report = m4.evaluate_verified_rows([draw_for(0)])
        self.assertEqual(1, report["observed_draw_count"])
        self.assertEqual(8, report["pending_draw_count"])
        self.assertEqual(1, report["paired_event_counts_observed"]["geometry_only"])
        self.assertEqual(0, report["paired_event_counts_observed"]["control_only"])
        self.assertEqual(1, report["prespecified_primary_arm_difference_observed"])
        self.assertIsNone(report["final_one_sided_exact_p"])
        self.assertIsNone(report["realized_profit_twd"])

    def test_full_nine_draw_paired_exact_one_sided_result(self):
        rows = [draw_for(i, GEOMETRY_WIN if i < 5 else CONTROL_WIN)
                for i in range(9)]
        report = m4.evaluate_verified_rows(rows)
        self.assertEqual(0, report["pending_draw_count"])
        self.assertEqual(9, report["observed_draw_count"])
        self.assertEqual("COMPLETE_DESCRIPTIVE_OR_INCONCLUSIVE_NO_PREDICTIVE_GO",
                         report["status"])
        self.assertEqual(5, report["paired_event_counts_observed"]["geometry_only"])
        self.assertEqual(4, report["paired_event_counts_observed"]["control_only"])
        self.assertEqual(.5, report["final_one_sided_exact_p"])
        self.assertEqual(1, report["prespecified_primary_arm_difference_observed"])
        self.assertFalse(report["predictive_go"])

    def test_exact_binomial_edge_conditions(self):
        self.assertEqual(1.0, m4.exact_paired_one_sided_p(0, 0))
        self.assertEqual(.5, m4.exact_paired_one_sided_p(1, 0))
        self.assertEqual(.125, m4.exact_paired_one_sided_p(3, 0))
        self.assertEqual(.5, m4.exact_paired_one_sided_p(3, 2))
        for invalid in ((-1, 3), (3, -1), (10, 0), (True, 0)):
            with self.assertRaises(ValueError):
                m4.exact_paired_one_sided_p(*invalid)

    def test_no_award_still_has_correct_paired_neither_count(self):
        report = m4.evaluate_verified_rows([draw_for(0, NO_AWARD, bonus=49)])
        self.assertEqual(1, report["paired_event_counts_observed"]["neither"])
        self.assertEqual(0, report["paired_event_counts_observed"]["both"])
        self.assertEqual(0, report["prespecified_primary_arm_difference_observed"])

    def test_future_draw_prefix_not_permitted_to_skip(self):
        for illegal in (
            [draw_for(1)],
            [draw_for(0), draw_for(2)],
            [draw_for(0), draw_for(0)],
            [draw_for(0), {**draw_for(1), "date": "2026-11-18"}],
            [draw_for(0), {**draw_for(1), "draw_id": 115000114}],
            [draw_for(0)] * 10,
        ):
            with self.assertRaises(ValueError):
                m4.evaluate_verified_rows(illegal)

    def test_invalid_bonus_or_main_rejected(self):
        for row in (
            {**draw_for(0), "bonus": 7},
            {**draw_for(0), "main": [7, 7, 12, 19, 30, 34]},
            {**draw_for(0), "main": [7, 10, 12, 19, 30, 50]},
            {**draw_for(0), "main": [7, 10, 12, 19, 30, True]},
        ):
            with self.assertRaises(ValueError):
                m4.evaluate_verified_rows([row])

    def test_future_modification_cannot_relabel_first_draw(self):
        original = m4.evaluate_verified_rows([draw_for(0)])
        later = m4.evaluate_verified_rows([draw_for(0), draw_for(1, CONTROL_WIN)])
        self.assertEqual(original["observed_draws"][0], later["observed_draws"][0])
        self.assertIsNone(later["final_one_sided_exact_p"])

    def test_p5a_seal_tampering_is_detected_by_original_git_sha(self):
        original = m4.verify_p5a_seal()
        modified = copy.deepcopy(original)
        modified["strategies"]["geometry_p3b_counterfactual"]["tickets"][0][0] = 1
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "shadow.json"
            fake.write_text(json.dumps(modified), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Git blob SHA1 changed"):
                m4.verify_p5a_seal(fake)

    def test_future_archive_requires_saved_pages_and_matching_draw(self):
        with mock.patch.object(m4, "audit_source", return_value={
            "total_draws": 2499
        }), mock.patch.object(m4, "generate_yearly_csv",
                             return_value=({Path("lotto649_2026.csv"): b"placeholder"}, {})), \
                mock.patch.object(m4, "_from_csv_bytes",
                                  return_value=[draw_for(0)]), \
                mock.patch.object(m4, "verify_local_evidence",
                                  side_effect=FileNotFoundError("no evidence")):
            with self.assertRaisesRegex(FileNotFoundError, "no evidence"):
                m4.read_archived_verified_prefix()
        with mock.patch.object(m4, "audit_source", return_value={
            "total_draws": 2499
        }), mock.patch.object(m4, "generate_yearly_csv",
                             return_value=({Path("lotto649_2026.csv"): b"placeholder"}, {})), \
                mock.patch.object(m4, "_from_csv_bytes",
                                  return_value=[draw_for(0)]), \
                mock.patch.object(m4, "verify_local_evidence",
                                  return_value={"draw": draw_for(0, CONTROL_WIN),
                                                "page_sha256": {},
                                                "source_index_sha256": "a" * 64}):
            with self.assertRaisesRegex(ValueError, "contradicts"):
                m4.read_archived_verified_prefix()

    def test_export_json_no_predictive_or_settlement_fields_claimed(self):
        report = m4.evaluate_verified_rows([])
        self.assertIsInstance(json.dumps(report, allow_nan=False), str)
        self.assertEqual("PROSPECTIVE_SHADOW_NO_REAL_WAGERS", report["study_type"])
        self.assertFalse(report["winner_declared"])
        self.assertFalse(report["predictive_go"])
        self.assertIsNone(report["realized_payout_twd"])


if __name__ == "__main__":
    unittest.main()
