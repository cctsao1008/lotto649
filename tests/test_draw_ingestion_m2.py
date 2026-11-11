"""M2 source-integrity, published anchor and safe future staging checks."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from lotto7x7 import draw_ingestion_m2 as m2

ROOT = Path(__file__).resolve().parents[1]


class TestDrawIngestionM2(unittest.TestCase):
    def _audit(self):
        return m2.audit_source(ROOT / "data/csv", ROOT / "data/bin")

    @staticmethod
    def example_stage(audit):
        future = {"draw_id": 115000105, "date": "2026-11-13",
                  "main": [1, 9, 17, 22, 39, 48], "bonus": 3}
        def source(publisher, kind, hostname, checksum):
            return {"publisher": publisher, "kind": kind,
                    "url": f"https://{hostname}/lotto/2026-11-13",
                    "page_sha256": checksum,
                    "observed_draw": future.copy()}
        return {
            "schema": m2.STAGE_SCHEMA,
            "base_manifest_sha256": audit["source_fingerprints"]["manifest_sha256"],
            "records": [{"draw": future,
                         "sources": [
                             source("Operator", "lottery_operator", "official.example.org", "a"*64),
                             source("Press", "other_publisher", "news.example.net", "b"*64),
                         ]}],
        }

    def test_archive_and_independent_published_anchor(self):
        result = self._audit()
        self.assertEqual("PASS_ARCHIVE_INTEGRITY_AND_SINGLE_PUBLISHED_DRAW_CROSSCHECK",
                         result["status"])
        self.assertEqual(2498, result["total_draws"])
        self.assertEqual(23, result["annual_csv_files"])
        self.assertEqual(115000104, result["last_draw_id"])
        self.assertEqual("2026-11-10", result["last_date"])
        self.assertTrue(result["crosscheck"]["exact_match"])
        self.assertEqual([11, 23, 30, 32, 40, 41],
                         result["crosscheck"]["anchor"]["main"])
        self.assertEqual(28, result["crosscheck"]["anchor"]["bonus"])
        self.assertIn("NOT individually", result["source_authenticity_scope"])

    def test_no_source_data_mutated_by_audit(self):
        originals = [
            ROOT / "data/bin/lotto649_history_v1.bin",
            ROOT / "data/bin/lotto649_images_v2.bin",
            ROOT / "data/bin/lotto649_source_manifest.json",
            ROOT / "data/csv/lotto649_2026.csv",
        ]
        before = [path.read_bytes() for path in originals]
        self._audit()
        self.assertEqual(before, [p.read_bytes() for p in originals])

    def test_strict_six_main_and_special_constraints(self):
        correct = dict(m2.ANCHOR)
        self.assertEqual(correct, m2.normalize_draw(correct))
        for k, value in [
            ("bonus", 11), ("bonus", 50), ("bonus", True),
            ("main", [11, 11, 23, 32, 40, 41]),
            ("main", [11, 23, 30, 32, 40, 50]),
            ("main", [11, 23, 30, 32, 41, 40]),
            ("draw_id", 114000104),
            ("date", "2026-02-30"),
            ("date", "2026/11/10"),
        ]:
            broken = dict(correct)
            broken[k] = value
            with self.assertRaises(ValueError, msg=k):
                m2.normalize_draw(broken)

    def test_extra_fields_and_malformed_six_rejected(self):
        draw = {**m2.ANCHOR, "jackpot": 10}
        with self.assertRaisesRegex(ValueError, "unexpected draw fields"):
            m2.normalize_draw(draw)
        for invalid in ([], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, "6"]):
            with self.assertRaises(ValueError):
                m2.normalize_draw({**m2.ANCHOR, "main": invalid})

    def test_staged_future_schema_accepts_without_source_authenticity_claim(self):
        audit = self._audit()
        stage = self.example_stage(audit)
        result = m2.validate_staged_batch(stage, audit)
        self.assertEqual(1, result["staged_draws"])
        self.assertEqual(115000105, result["first_staged_draw"]["draw_id"])
        self.assertFalse(result["ready_to_mutate_archive"])
        self.assertIn("SOURCE_PAGES_NOT_FETCHED", result["status"])
        self.assertIsInstance(json.dumps(result), str)

    def test_stale_manifest_and_missing_evidence_rejected(self):
        audit = self._audit()
        stage = self.example_stage(audit)
        stage["base_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "base manifest SHA"):
            m2.validate_staged_batch(stage, audit)
        stage = self.example_stage(audit)
        stage["records"][0]["sources"] = stage["records"][0]["sources"][:1]
        with self.assertRaisesRegex(ValueError, "two separately"):
            m2.validate_staged_batch(stage, audit)

    def test_no_independent_publisher_or_fake_official_rejected(self):
        audit = self._audit()
        stage = self.example_stage(audit)
        stage["records"][0]["sources"][0]["kind"] = "other_publisher"
        with self.assertRaisesRegex(ValueError, "operator or licensed bank"):
            m2.validate_staged_batch(stage, audit)
        stage = self.example_stage(audit)
        stage["records"][0]["sources"][1]["url"] = "https://official.example.org/another"
        with self.assertRaisesRegex(ValueError, "distinct published hosts"):
            m2.validate_staged_batch(stage, audit)

    def test_source_contradiction_and_invalid_hash_rejected(self):
        audit = self._audit()
        stage = self.example_stage(audit)
        stage["records"][0]["sources"][1]["observed_draw"]["bonus"] = 49
        with self.assertRaisesRegex(ValueError, "disagrees"):
            m2.validate_staged_batch(stage, audit)
        stage = self.example_stage(audit)
        stage["records"][0]["sources"][1]["page_sha256"] = "Not-a-hash"
        with self.assertRaisesRegex(ValueError, "SHA256"):
            m2.validate_staged_batch(stage, audit)
        stage = self.example_stage(audit)
        stage["records"][0]["sources"][0]["url"] = "http://example.org/not-https"
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            m2.validate_staged_batch(stage, audit)

    def test_stage_gap_rollback_and_nonchronological_order_rejected(self):
        audit = self._audit()
        for number, day in (
            (115000104, "2026-11-10"),
            (115000106, "2026-11-13"),
            (115000105, "2026-11-10"),
        ):
            stage = self.example_stage(audit)
            draw = stage["records"][0]["draw"]
            draw["draw_id"], draw["date"] = number, day
            for source in stage["records"][0]["sources"]:
                source["observed_draw"] = draw.copy()
            with self.assertRaises(ValueError):
                m2.validate_staged_batch(stage, audit)

    def test_append_sequential_two_future_draws_and_year_rollover(self):
        audit = self._audit()
        stage = self.example_stage(audit)
        next_row = json.loads(json.dumps(stage["records"][0]))
        next_row["draw"]["draw_id"] = 115000106
        next_row["draw"]["date"] = "2026-11-17"
        next_row["draw"]["main"] = [2, 6, 14, 22, 39, 47]
        next_row["draw"]["bonus"] = 9
        for source in next_row["sources"]:
            source["observed_draw"] = next_row["draw"].copy()
        stage["records"].append(next_row)
        self.assertEqual(2, m2.validate_staged_batch(stage, audit)["staged_draws"])

        year_end = dict(audit)
        year_end["last_draw_id"] = 115000115
        year_end["last_date"] = "2026-12-29"
        rollover = self.example_stage(year_end)
        row = rollover["records"][0]["draw"]
        row["draw_id"], row["date"] = 116000001, "2027-01-01"
        for source in rollover["records"][0]["sources"]:
            source["observed_draw"] = row.copy()
        self.assertEqual(116000001,
                         m2.validate_staged_batch(rollover, year_end)["last_staged_draw"]["draw_id"])

    def test_tampered_yearly_csv_detected_before_reporting(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            csv_dir = root / "csv"
            csv_dir.mkdir()
            for path in (ROOT / "data/csv").glob("lotto649_*.csv"):
                shutil.copyfile(path, csv_dir / path.name)
            target = csv_dir / "lotto649_2026.csv"
            text = target.read_text("ascii")
            self.assertIn("11,23,30,32,40,41,28", text)
            target.write_text(text.replace("11,23,30,32,40,41,28",
                                           "11,23,30,32,40,41,29"), encoding="ascii")
            with self.assertRaises(ValueError):
                m2.audit_source(csv_dir, ROOT / "data/bin")


if __name__ == "__main__":
    unittest.main()
