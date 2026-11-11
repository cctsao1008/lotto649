"""M2 append preview: strict two-page evidence, isolated full rebuild, no writes."""
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from lotto7x7 import draw_append_preview_m2 as preview
from lotto7x7 import prospective_evaluation_m4 as m4
from lotto7x7 import official_draw_evidence_m2 as evidence

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "data"
NEXT_DRAW = {
    "draw_id": 115000105, "date": "2026-11-13",
    "main": [1, 9, 17, 22, 39, 48], "bonus": 3,
}
CTBC_HTML = (
    "<html><body><h1>大樂透</h1>"
    "<div>遊 戲 大樂透 期 別 第 115000105 期"
    " 開獎日期 民國 115 年 11 月 13 日"
    " 本期中獎號碼 依大小順序排列：01 09 17 22 39 48"
    " 特別號：03 依開出順序排列： 39 09 48 01 22 17</div>"
    "<div>遊 戲 今彩539 期 別 第 115000001 期</div></body></html>"
).encode("utf-8")
INDEPENDENT_HTML = (
    "<html><body><h1>大樂透開獎號碼查詢</h1>"
    "<p>115/11/13 115000105頭獎:0 01 09 17 22 39 48 03</p>"
    "<p>115/11/10 115000104頭獎:0 11 23 30 32 40 41 28</p>"
    "</body></html>"
).encode("utf-8")


def make_fake_source_files(folder: Path, captured_at="2026-11-13T14:00:00+00:00"):
    folder.mkdir(parents=True, exist_ok=True)
    pages = []
    for provider, data in (("ctbc", CTBC_HTML), ("independent", INDEPENDENT_HTML)):
        h = sha256(data).hexdigest()
        (folder / (h + ".html")).write_bytes(data)
        pages.append({
            "provider": provider,
            "url": evidence.TRUSTED[provider],
            "sha256": h,
            "captured_at_utc": captured_at,
        })
    item = {"schema": evidence.SCHEMA,
            "draw": NEXT_DRAW, "pages": pages}
    (folder / "115000105.json").write_text(json.dumps(item), encoding="utf-8")
    return item


class TestDrawAppendPreviewM2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "results").mkdir(exist_ok=True)

    def test_two_source_synthetic_fixture_parsers(self):
        self.assertEqual(NEXT_DRAW, evidence.parse_ctbc(CTBC_HTML))
        self.assertEqual(NEXT_DRAW, evidence.parse_independent(INDEPENDENT_HTML))

    def test_source_capture_must_not_precede_advertised_draw(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            make_fake_source_files(p, captured_at="2026-11-11T08:00:00+00:00")
            with self.assertRaisesRegex(ValueError, "before the advertised draw date"):
                evidence.verify_local_evidence(115000105, p)

    def test_full_verified_preview_does_not_mutate_current_archive(self):
        original = [
            CURRENT / "bin/lotto649_history_v1.bin",
            CURRENT / "bin/lotto649_images_v2.bin",
            CURRENT / "bin/lotto649_source_manifest.json",
            CURRENT / "csv/lotto649_2026.csv",
        ]
        fingerprints = [sha256(p.read_bytes()).hexdigest() for p in original]
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as scratch:
            # Destination itself must be nested under ignored results.
            temp = Path(scratch)
            evidence_root = temp / "proof"
            make_fake_source_files(evidence_root)
            candidate = temp / "new-archive"
            result = preview.build_preview(
                115000105, candidate, csv_root=CURRENT / "csv",
                bin_root=CURRENT / "bin", evidence_root=evidence_root,
            )
            self.assertEqual(2499, result["new_draw_count"])
            self.assertFalse(result["active_repository_files_changed"])
            self.assertFalse(result["ready_to_commit_to_main"])
            self.assertTrue((candidate / "data/bin/lotto649_history_v1.bin").is_file())
            self.assertTrue((candidate / "data/bin/lotto649_images_v2.bin").is_file())
            self.assertTrue((candidate / "data/csv/lotto649_2026.csv").is_file())
            self.assertEqual(104 + 1,
                len((candidate / "data/csv/lotto649_2026.csv").read_text().splitlines()) - 1)
            drawn, gate = m4.read_archived_verified_prefix(
                candidate / "data/csv", candidate / "data/bin",
                candidate / "evidence",
            )
            self.assertEqual([NEXT_DRAW], drawn)
            self.assertEqual(1, gate["admitted_draws"])
            evaluation = m4.evaluate_verified_rows(drawn)
            self.assertEqual(1, evaluation["observed_draw_count"])
            self.assertEqual(8, evaluation["pending_draw_count"])
            self.assertIsNone(evaluation["final_one_sided_exact_p"])
            with self.assertRaises(FileExistsError):
                preview.build_preview(115000105, candidate,
                                      csv_root=CURRENT / "csv",
                                      bin_root=CURRENT / "bin",
                                      evidence_root=evidence_root)
        self.assertEqual(fingerprints, [
            sha256(p.read_bytes()).hexdigest() for p in original
        ])

    def test_missing_evidence_or_future_draw_gap_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as scratch:
            temp = Path(scratch)
            evidence_root = temp / "proof"
            make_fake_source_files(evidence_root)
            with self.assertRaisesRegex(ValueError, "immediate next"):
                preview.build_preview(
                    115000106, temp / "bad-gap",
                    csv_root=CURRENT / "csv", bin_root=CURRENT / "bin",
                    evidence_root=evidence_root,
                )
            with self.assertRaises(FileNotFoundError):
                preview.build_preview(
                    115000105, temp / "bad-evidence",
                    csv_root=CURRENT / "csv", bin_root=CURRENT / "bin",
                    evidence_root=temp / "missing",
                )
            self.assertFalse((temp / "bad-gap").exists())
            self.assertFalse((temp / "bad-evidence").exists())

    def test_no_writes_allowed_outside_ignored_results(self):
        with tempfile.TemporaryDirectory() as scratch:
            with self.assertRaisesRegex(ValueError, "below results"):
                preview.build_preview(
                    115000105, Path(scratch) / "bad",
                    csv_root=CURRENT / "csv", bin_root=CURRENT / "bin",
                )


if __name__ == "__main__":
    unittest.main()
