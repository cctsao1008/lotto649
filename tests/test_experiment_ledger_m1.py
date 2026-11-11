"""No-network M1 private ledger and frozen public artifact tests."""
import json
from pathlib import Path
import tempfile
import unittest

from lotto7x7 import experiment_ledger as ledger


BASE_SHA = "d865e1d241b607488e58a00b22fb595734f804595afd9d8850718808d6e8b7f8"


def fixture():
    return {
        "schema": ledger.PAID_SCHEMA,
        "experiment_id": "paid-test-four-lines",
        "strategy": "synthetic_fixture",
        "registration_status": "RETROSPECTIVE_UNSEALED",
        "purchase_date_reported": "2026-11-09",
        "source_evidence": "USER_REPORTED",
        "historical_cutoff": {"date": "2026-11-06", "last_draw_id": 115000103,
                              "draws": 2497, "prefix_sha256": BASE_SHA},
        "draw_schedule": [
            {"draw_id": 115000104, "date": "2026-11-10"},
            {"draw_id": 115000105, "date": "2026-11-13"},
        ],
        "tickets": [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22, 23, 24],
        ],
        "disjoint_tickets": True,
        "cost_per_line_twd": 50,
        "total_cost_twd": 400,
    }


class TestExperimentLedgerM1(unittest.TestCase):
    def test_public_repository_anchors(self):
        path = Path(__file__).resolve().parents[1] / "experiments" / "ledger_registry_v1.json"
        count = ledger.verify_public_registry(path)
        self.assertGreaterEqual(count, 3)
        entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
        ids = {entry["experiment_id"] for entry in entries}
        self.assertTrue({
            "v027-p3f-shadow-static10-20261110",
            "M0_new_four_block_composite_20261110",
            "v029-p5a-geometry-shadow-static9-20261113",
        }.issubset(ids))

    def test_register_and_verify(self):
        with tempfile.TemporaryDirectory() as folder:
            record = fixture()
            path, digest = ledger.register_paid(record, Path(folder),
                                                registered_at_utc="2026-11-11T03:00:00+00:00")
            self.assertTrue(path.is_file())
            self.assertEqual(64, len(digest))
            self.assertEqual(1, ledger.verify_private_ledger(folder))
            with self.assertRaises(FileExistsError):
                ledger.register_paid(record, Path(folder))

    def test_private_checksum_detects_modification(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = ledger.register_paid(fixture(), Path(folder))
            entry = json.loads(path.read_text("utf-8"))
            entry["registration"]["tickets"][0][0] = 2
            path.write_text(json.dumps(entry), encoding="utf-8")
            with self.assertRaises(ValueError):
                ledger.verify_private_ledger(folder)

    def test_draw_id_gap_rejected(self):
        record = fixture()
        record["draw_schedule"][1]["draw_id"] += 1
        with self.assertRaisesRegex(ValueError, "missing draw ID"):
            ledger.validate_paid_registration(record)

    def test_draw_year_mismatch_rejected(self):
        record = fixture()
        record["draw_schedule"][0]["date"] = "2025-11-10"
        with self.assertRaises(ValueError):
            ledger.validate_paid_registration(record)

    def test_duplicate_number_and_ticket_overlap_rejected(self):
        record = fixture()
        record["tickets"][0][1] = 1
        with self.assertRaisesRegex(ValueError, "six-number"):
            ledger.validate_paid_registration(record)
        record = fixture()
        record["tickets"][1] = [1, 8, 9, 10, 11, 12]
        with self.assertRaisesRegex(ValueError, "pairwise disjoint"):
            ledger.validate_paid_registration(record)

    def test_cost_mismatch_rejected(self):
        record = fixture()
        record["total_cost_twd"] = 350
        with self.assertRaisesRegex(ValueError, "cost mismatch"):
            ledger.validate_paid_registration(record)

    def test_no_outcome_or_unsupported_status(self):
        record = fixture()
        record["outcomes"] = []
        with self.assertRaises(ValueError):
            ledger.validate_paid_registration(record)
        record = fixture()
        record["registration_status"] = "PROVEN_PRE_DRAW"
        with self.assertRaisesRegex(ValueError, "cannot claim"):
            ledger.validate_paid_registration(record)

    def test_public_registry_rejects_paid_or_tampered_source(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "experiments").mkdir()
            (root / "prospective").mkdir()
            source = root / "prospective" / "test.json"
            content = b'{"experiment_id":"shadow-synthetic","strategies":{},"experiment":{}}\n'
            source.write_bytes(content)
            registry = {"schema": ledger.PUBLIC_SCHEMA, "entries": [{
                "experiment_id": "shadow-synthetic", "role": "FROZEN_SHADOW",
                "artifact": "prospective/test.json",
                "git_blob_sha1": ledger._git_blob_sha1(content),
            }]}
            path = root / "experiments" / "ledger_registry_v1.json"
            path.write_text(json.dumps(registry), encoding="utf-8")
            self.assertEqual(1, ledger.verify_public_registry(path))
            registry["entries"][0]["role"] = "PAID_TICKET"
            path.write_text(json.dumps(registry), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not store paid"):
                ledger.verify_public_registry(path)
            registry["entries"][0]["role"] = "FROZEN_SHADOW"
            path.write_text(json.dumps(registry), encoding="utf-8")
            source.write_bytes(content + b" ")
            with self.assertRaisesRegex(ValueError, "differs from frozen"):
                ledger.verify_public_registry(path)

    def test_public_anchor_accepts_windows_crlf(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "experiments").mkdir()
            (root / "prospective").mkdir()
            src = b'{\n  "experiment_id": "shadow",\n  "strategies": {},\n  "experiment": {}\n}\n'
            (root / "prospective" / "test.json").write_bytes(src.replace(b"\n", b"\r\n"))
            registry = {"schema": ledger.PUBLIC_SCHEMA, "entries": [{
                "experiment_id": "shadow", "role": "FROZEN_SHADOW",
                "artifact": "prospective/test.json", "git_blob_sha1": ledger._git_blob_sha1(src),
            }]}
            path = root / "experiments" / "ledger_registry_v1.json"
            path.write_text(json.dumps(registry), encoding="utf-8")
            self.assertEqual(1, ledger.verify_public_registry(path))

    def test_private_directory_absent(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(0, ledger.verify_private_ledger(Path(root) / "missing"))


if __name__ == "__main__":
    unittest.main()
