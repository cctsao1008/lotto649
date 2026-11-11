"""P5A pre-draw immutable shadow artifact integrity and no-outcome tests."""
import json
from pathlib import Path
import tempfile
import unittest

from lotto7x7 import geometry_shadow_v029 as sealcheck

ROOT = Path(__file__).resolve().parents[1]
SEAL = ROOT / sealcheck.SEAL_PATH
ORIGINAL = ROOT / sealcheck.ORIGINAL_PATH


class TestGeometryShadowV029P5A(unittest.TestCase):
    def verify_rewritten(self, alter):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            seal = json.loads(SEAL.read_text(encoding="utf-8"))
            alter(seal)
            target = folder / "shadow.json"
            target.write_text(json.dumps(seal), encoding="utf-8")
            return sealcheck.verify_shadow(target, ORIGINAL)

    def test_original_shadow_seal_valid_and_verifiable(self):
        result = sealcheck.verify_shadow(SEAL, ORIGINAL)
        self.assertEqual("v029-p5a-geometry-shadow-static9-20261113",
                         result["experiment_id"])
        self.assertEqual("PROSPECTIVE_PILOT_SEALED_NO_RESULTS_EVALUATED",
                         result["decision"])
        self.assertEqual(9, result["draw_count"])
        self.assertEqual(2, result["strategies"])
        self.assertEqual(4, result["tickets_per_strategy"])
        self.assertTrue(result["no_purchases_authorized"])

    def test_future_dates_and_source_cutoff_are_frozen(self):
        result = sealcheck.verify_shadow(SEAL, ORIGINAL)
        self.assertEqual({"date": "2026-11-13", "draw_id": 115000105},
                         result["first_future_draw"])
        self.assertEqual({"date": "2026-12-11", "draw_id": 115000113},
                         result["last_future_draw"])
        with self.assertRaisesRegex(ValueError, "historical input cutoff"):
            self.verify_rewritten(lambda obj: obj["historical_cutoff"].update(
                {"last_draw_id": 115000105}))

    def test_geometry_four_ticket_pool_matches_original_source(self):
        seal = json.loads(SEAL.read_text(encoding="utf-8"))
        tickets = sealcheck.validate_tickets(
            seal["strategies"]["geometry_p3b_counterfactual"]["tickets"])
        self.assertEqual(sealcheck.P3B_FROZEN_FOUR, tickets)
        self.assertEqual(24, len(set(n for ticket in tickets for n in ticket)))
        self.assertEqual(0.122731458961909,
                         seal["experiment"]["fair_iid_context"][
                             "marginal_any_prize_probability_for_four_disjoint_tickets"
                         ])

    def test_control_exactly_reuses_previously_frozen_public_control(self):
        old = json.loads(ORIGINAL.read_text(encoding="utf-8"))
        shadow = json.loads(SEAL.read_text(encoding="utf-8"))
        self.assertEqual(
            old["strategies"]["history_blind_random_control"]["tickets"],
            shadow["strategies"]["history_blind_random_control"]["tickets"],
        )
        self.assertEqual(sealcheck.ORIGINAL_BLOB_SHA1,
                         sealcheck._git_blob_sha1(ORIGINAL.read_bytes().replace(b"\r\n", b"\n")))
        with self.assertRaisesRegex(ValueError, "random control differs"):
            self.verify_rewritten(lambda o: o["strategies"]["history_blind_random_control"][
                "tickets"][0].__setitem__(0, 1))

    def test_outcomes_cannot_be_written_to_prespecified_seal(self):
        with self.assertRaisesRegex(ValueError, "cannot contain future outcomes"):
            self.verify_rewritten(lambda obj: obj.update({"outcomes": []}))

    def test_unknown_shadow_strategy_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unexpected shadow strategy"):
            self.verify_rewritten(lambda o: o["strategies"].update({
                "user_paid_tickets": {"tickets": []}
            }))

    def test_source_provenance_fingerprint_rejected_if_changed(self):
        with self.assertRaisesRegex(ValueError, "source artifact JSON"):
            self.verify_rewritten(lambda obj: obj["source_evidence"][
                "geometry_optimizer"].update({"report_json_sha256": "0" * 64}))
        with self.assertRaisesRegex(ValueError, "algorithm parameters"):
            self.verify_rewritten(lambda obj: obj["source_evidence"][
                "geometry_optimizer"]["optimizer_params"].update({"seed": 650}))

    def test_no_wager_authorization(self):
        with self.assertRaisesRegex(ValueError, "cannot authorize"):
            self.verify_rewritten(lambda o: o["experiment"].update(
                {"purchase_authorized": True}))
        with self.assertRaisesRegex(ValueError, "schema"):
            self.verify_rewritten(lambda o: o.update({"status": "PAID"}))

    def test_ticket_duplicates_or_overlap_rejected(self):
        for bad in (
            [[1, 2, 3, 4, 5, 5]] * 4,
            [[1, 2, 3, 4, 5, 6]] * 4,
            [[1, 2, 3, 4, 5, 50]] * 4,
        ):
            with self.assertRaises(ValueError):
                sealcheck.validate_tickets(bad)
        with self.assertRaisesRegex(ValueError, "numbers must"):
            self.verify_rewritten(lambda o: o["strategies"][
                "geometry_p3b_counterfactual"]["tickets"][0].__setitem__(0, 11))

    def test_fixed_schedule_cannot_shift_or_start_in_history(self):
        with self.assertRaisesRegex(ValueError, "future nine-draw schedule"):
            self.verify_rewritten(lambda o: o["experiment"]["planned_draws"][0].update(
                {"draw_id": 115000104, "date": "2026-11-10"}))
        with self.assertRaisesRegex(ValueError, "future nine-draw schedule"):
            self.verify_rewritten(lambda o: o["experiment"].update(
                {"fixed_draw_count": 10}))


if __name__ == "__main__":
    unittest.main()
