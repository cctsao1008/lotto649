"""Offline tests for single-source Biga Lotto649 scraper."""
from datetime import date
import importlib.util
from pathlib import Path
import sys
import unittest
PATH = Path(__file__).resolve().parents[1] / "scripts" / "fetch_lotto649_biga.py"
spec = importlib.util.spec_from_file_location("biga_fetch_test", PATH)
assert spec and spec.loader
biga = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = biga
spec.loader.exec_module(biga)
def row(day, issue, balls, bonus):
    order = "".join(f"<span>{x:02d}</span>" for x in balls)
    sorted_numbers = "".join(f"<span>{x:02d}</span>" for x in sorted(balls))
    return (f"<tr><td>{day}</td><td>星期五</td><td>{issue}</td><td>202644</td>"
            f"<td>2</td><td>甲申</td><td>{order}</td><td>{bonus:02d}</td>"
            f"<td>{sorted_numbers}</td></tr>")
def fixture(rows):
    return ("<html><body><h1>大樂透開獎號碼一覽表</h1>"
            "<table><thead><tr><th>落球順序</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></body></html>").encode("utf-8")
class TestBiga(unittest.TestCase):
    def test_2004_source_issue_mapping(self):
        draws = biga.parse_page(fixture([row("2004/01/05", "093001", (33, 12, 6, 9, 39, 13), 21)]))
        self.assertEqual(1, len(draws))
        self.assertEqual((93000001, date(2004, 1, 5)), (draws[0].draw_id, draws[0].day))
        self.assertEqual((6, 9, 12, 13, 33, 39), draws[0].main)
    def test_2005_missing_in_auzo_exists_in_biga(self):
        draws = biga.parse_page(fixture([row("2005/02/03", "094010", (48, 19, 43, 7, 30, 8), 49)]))
        self.assertEqual(94000010, draws[0].draw_id)
    def test_2026_binary_formats(self):
        draws = biga.parse_page(fixture([row("2026/11/06", "115103", (48, 32, 5, 41, 26, 9), 23)]))
        full = biga.build_binary(draws, "history")
        img = biga.build_binary(draws, "image")
        self.assertEqual((b"L649HIS1", 1, 15), biga.HEAD.unpack_from(full))
        self.assertEqual((b"L649IMG2", 1, 53), biga.HEAD.unpack_from(img))
        self.assertEqual(6, sum(img[20:]))
    def test_invalid_bonus_is_rejected(self):
        with self.assertRaises(ValueError):
            biga.parse_page(fixture([row("2026/11/06", "115103", (48, 32, 5, 41, 26, 9), 5)]))
if __name__ == "__main__":
    unittest.main()
