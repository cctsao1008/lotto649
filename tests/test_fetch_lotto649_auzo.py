"""Offline sanity checks for AUZO single-source importer."""
from datetime import date
import importlib.util
from pathlib import Path
import sys
import unittest

PATH = Path(__file__).resolve().parents[1] / "scripts" / "fetch_lotto649_auzo.py"
spec = importlib.util.spec_from_file_location("auzo_fetch_test", PATH)
assert spec and spec.loader
auzo = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = auzo
spec.loader.exec_module(auzo)

def fixture(issue, date_string, main, bonus):
    anchors = "".join(f"<a>{number:02d}</a>" for number in main)
    html = (f"<html><table><tr><td>{issue}</td><td>{date_string} (星期五)</td>"
            f"<td>落球順序：{anchors} 大小順序：{anchors}</td>"
            f"<td>{bonus:02d} $100,000,000</td></tr></table></html>")
    return html.encode("utf-8")

class TestAuzoScraper(unittest.TestCase):
    def test_original_short_id_and_history_binary(self):
        rows = auzo.parse_year(2004, fixture("093001", "2004-01-05",
                                           (6, 9, 12, 13, 33, 39), 21))
        self.assertEqual((93000001, date(2004, 1, 5)), (rows[0].id, rows[0].day))
        self.assertEqual(21, rows[0].bonus)
        hist = auzo.binary(rows, "history")
        self.assertEqual((b"L649HIS1", 1, 15), auzo.HEAD.unpack_from(hist))
        self.assertEqual(31, len(hist))
        img = auzo.binary(rows, "image")
        self.assertEqual((b"L649IMG2", 1, 53), auzo.HEAD.unpack_from(img))
        self.assertEqual(6, sum(img[20:]))

    def test_official_nine_digit_id(self):
        rows = auzo.parse_year(2026, fixture("115000001", "2026-01-02",
                                           (3, 7, 16, 19, 40, 42), 12))
        self.assertEqual(115000001, rows[0].id)

    def test_reject_special_number_reuse(self):
        with self.assertRaises(ValueError):
            auzo.parse_year(2004, fixture("093001", "2004-01-05",
                                         (6, 9, 12, 13, 33, 39), 6))

    def test_reject_number_out_of_range(self):
        with self.assertRaises(ValueError):
            auzo.parse_year(2004, fixture("093001", "2004-01-05",
                                         (6, 9, 12, 13, 33, 50), 21))

if __name__ == "__main__":
    unittest.main()
