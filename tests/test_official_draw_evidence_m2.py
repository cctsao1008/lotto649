"""Offline source-page proof tests: no live HTTP, no archive writes."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from lotto7x7 import official_draw_evidence_m2 as evidence


DRAW = {
    "draw_id": 115000104, "date": "2026-11-10",
    "main": [11, 23, 30, 32, 40, 41], "bonus": 28,
}
CTBC = (
    "<html><body><section>遊 戲 威力彩 期 別 第 115000090 期 "
    "開獎日期 民國 115 年 11 月 9 日 依大小順序排列： 01 02 03 04 05 06 特別號： 07"
    "</section><section>遊 戲 大樂透 期 別 第 115000104 期 "
    "開獎日期 民國 115 年 11 月 10 日 "
    "依大小順序排列： 11 23 30 32 40 41 特別號： 28 "
    "依開出順序排列： 23 41 40 30 32 11</section>"
    "<section>遊 戲 今彩539 期 別 第 115000272 期</section></body></html>"
).encode("utf-8")
INDEPENDENT = (
    "<html><body>大樂透開獎 <p>115/11/10 115000104頭獎:0 "
    "11 23 30 32 40 41 28</p>"
    "<div>115/11/6 115000103頭獎:1 05 09 26 32 41 48 23</div>"
    "</body></html>"
).encode("utf-8")


def create_evidence(root, official=CTBC, independent=INDEPENDENT):
    root = Path(root)
    index = []
    for name, payload in (("ctbc", official), ("independent", independent)):
        digest = sha256(payload).hexdigest()
        (root / (digest + ".html")).write_bytes(payload)
        index.append({
            "provider": name,
            "url": evidence.TRUSTED[name],
            "sha256": digest,
            "captured_at_utc": "2026-11-11T07:50:00+00:00",
        })
    report = {"schema": evidence.SCHEMA, "draw": dict(DRAW), "pages": index}
    (root / "115000104.json").write_text(json.dumps(report), encoding="utf-8")
    return report


class TestOfficialDrawEvidenceM2(unittest.TestCase):
    def test_ctbc_parser_is_game_specific(self):
        self.assertEqual(DRAW, evidence.parse_ctbc(CTBC))
        self.assertEqual(DRAW, evidence.parse_independent(INDEPENDENT))

    def test_ctbc_rejects_wrong_game_or_missing_bonus(self):
        with self.assertRaisesRegex(ValueError, "recognized Chinese Lotto649 publication|大樂透 issue"):
            evidence.parse_ctbc(CTBC.replace("遊 戲 大樂透".encode("utf-8"),
                                             "遊 戲 威力彩".encode("utf-8")))
        with self.assertRaisesRegex(ValueError, "bonus unavailable"):
            evidence.parse_ctbc(CTBC.replace("特別號： 28".encode("utf-8"),
                                             "未公布特別號".encode("utf-8")))

    def test_independent_different_issue_is_not_the_same_draw(self):
        different = evidence.parse_independent(
            INDEPENDENT.replace(b"115000104", b"115000105"))
        self.assertNotEqual(DRAW, different)

    def test_all_snapshot_evidence_checks_pass(self):
        with tempfile.TemporaryDirectory() as root:
            index = create_evidence(root)
            actual = evidence.verify_local_evidence(115000104, Path(root))
            self.assertEqual(DRAW, actual["draw"])
            self.assertEqual("PASS_TWO_SAVED_SOURCE_PAGES_CONTENT_AGREES",
                             actual["status"])
            self.assertEqual(2, len(actual["page_sha256"]))
            self.assertEqual(64, len(actual["source_index_sha256"]))
            self.assertIn("cannot cryptographically", actual["scope_warning"])

    def test_snapshot_content_conflict_fails(self):
        wrong = INDEPENDENT.replace(b"11 23 30 32 40 41 28",
                                     b"11 23 30 32 40 41 29")
        with tempfile.TemporaryDirectory() as root:
            create_evidence(root, independent=wrong)
            with self.assertRaisesRegex(ValueError, "contradicts"):
                evidence.verify_local_evidence(115000104, Path(root))

    def test_hash_tampering_fails(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = create_evidence(root)
            path = Path(root) / (bundle["pages"][0]["sha256"] + ".html")
            path.write_bytes(CTBC + b"added")
            with self.assertRaisesRegex(ValueError, "bytes disagree"):
                evidence.verify_local_evidence(115000104, Path(root))

    def test_pinned_origin_and_missing_second_provider_fail(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = create_evidence(root)
            bundle["pages"][0]["url"] = "https://untrusted.example.org/"
            (Path(root) / "115000104.json").write_text(json.dumps(bundle), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pinned HTTPS origin"):
                evidence.verify_local_evidence(115000104, Path(root))
            bundle["pages"] = bundle["pages"][:1]
            (Path(root) / "115000104.json").write_text(json.dumps(bundle), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly two"):
                evidence.verify_local_evidence(115000104, Path(root))

    def test_source_index_cannot_claim_wrong_draw(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = create_evidence(root)
            bundle["draw"]["bonus"] = 27
            (Path(root) / "115000104.json").write_text(json.dumps(bundle), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contradicts"):
                evidence.verify_local_evidence(115000104, Path(root))

    def test_invalid_capture_timestamp_or_index_id_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = create_evidence(root)
            bundle["pages"][0]["captured_at_utc"] = "2026-11-11"
            (Path(root) / "115000104.json").write_text(json.dumps(bundle), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "UTC offset"):
                evidence.verify_local_evidence(115000104, Path(root))
            (Path(root) / "115000103.json").write_text(
                json.dumps(bundle), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                evidence.verify_local_evidence(115000103, Path(root))

    def test_no_real_network_dependency_for_verification(self):
        with tempfile.TemporaryDirectory() as root:
            create_evidence(root)
            with mock.patch.object(evidence, "urlopen",
                                            side_effect=AssertionError("network forbidden")):
                self.assertEqual(DRAW, evidence.verify_local_evidence(
                    115000104, Path(root))["draw"])


if __name__ == "__main__":
    unittest.main()
