"""Tests for new Lotto649 versioned CSV-to-BIN conversion only."""
from __future__ import annotations

from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from lotto7x7.bin_dataset import (
    DTYPE, HEADER, MAGIC, RECORD, Draw, build, decode_bin, discover_sources,
    encode_bin, load_csv_directory, parse_csv,
)


SAMPLE = ("\ufeff遊戲名稱,期別,開獎日期,銷售總額,獎號1,獎號2,獎號3,獎號4,獎號5,獎號6,特別號,  \n"
          "大樂透,107000001,2018/01/02,123456,02,06,12,18,36,46,47,,,,\n"
          "大樂透,107000002,2018/01/05,123456,04,13,17,29,31,44,14,,,,\n")


class BinaryDatasetTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "csv"
        d = self.root / "2018"
        d.mkdir(parents=True)
        self.csv = d / "大樂透_201801_201812.csv"
        self.csv.write_text(SAMPLE, encoding="utf-8")

    def test_bom_and_trailing_columns_are_accepted(self):
        rows = parse_csv(self.csv)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].main, (2, 6, 12, 18, 36, 46))
        self.assertEqual(rows[0].special, 47)

    def test_unpadded_month_and_day_in_csv_are_valid(self):
        year_dir = self.root / "2024"
        year_dir.mkdir()
        path = year_dir / "大樂透_2024.csv"
        path.write_text(
            "遊戲名稱,期別,開獎日期,獎號1,獎號2,獎號3,獎號4,獎號5,獎號6,特別號\n"
            "大樂透,113000110,2024/12/3,5,6,7,12,17,35,45\n",
            encoding="utf-8",
        )
        values = parse_csv(path)
        self.assertEqual(values[0].date, 20241203)
        self.assertEqual(values[0].draw_id, 113000110)

    def test_binary_record_is_eleven_bytes(self):
        self.assertEqual(RECORD.size, 11)
        self.assertEqual(DTYPE.itemsize, 11)
        self.assertEqual(HEADER.size, 16)

    def test_binary_exact_roundtrip(self):
        rows = parse_csv(self.csv)
        raw = encode_bin(rows)
        self.assertEqual(len(raw), 16 + 11 * 2)
        self.assertEqual(raw[:8], MAGIC)
        self.assertEqual(raw[16:20], struct.pack("<I", 107000001))
        decoded = decode_bin(raw)
        self.assertEqual(int(decoded[0]["draw_id"]), 107000001)
        self.assertEqual(decoded[0]["numbers"].tolist(), [2, 6, 12, 18, 36, 46])
        self.assertEqual(int(decoded[0]["special"]), 47)

    def test_build_persists_and_verifies(self):
        out = self.root.parent / "out" / "lotto649_draws_v1.bin"
        report = build(self.root, out)
        self.assertEqual(report["records"], 2)
        self.assertEqual(report["source_files"], 1)
        self.assertEqual(report["bytes"], 38)
        self.assertEqual(report["counts_by_year"], {"2018": 2})
        self.assertEqual(len(decode_bin(out.read_bytes())), 2)
        self.assertEqual(build(self.root, out)["sha256"], report["sha256"])

    def test_duplicate_draw_id_rejected(self):
        self.csv.write_text(SAMPLE + SAMPLE.splitlines()[-1] + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_csv_directory(self.root)

    def test_repeated_main_or_special_rejected(self):
        self.csv.write_text(SAMPLE.replace("36,46,47", "36,46,46"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "seven distinct"):
            load_csv_directory(self.root)

    def test_truncated_and_corrupted_bin_rejected(self):
        raw = encode_bin(parse_csv(self.csv))
        for data in (raw[:10], raw[:-1], b"BADMAGIC" + raw[8:],
                     raw[:8] + struct.pack("<II", 3, 11) + raw[16:]):
            with self.assertRaises(ValueError):
                decode_bin(data)

    def test_bad_record_data_rejected(self):
        raw = bytearray(encode_bin(parse_csv(self.csv)))
        raw[20] = 50
        with self.assertRaises(ValueError):
            decode_bin(bytes(raw))

    def test_missing_year_in_sequence_rejected(self):
        other = self.root / "2020"
        other.mkdir()
        (other / "大樂透_202001_202012.csv").write_text(
            SAMPLE.replace("2018", "2020").replace("1070000", "1090000"),
            encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing calendar"):
            discover_sources(self.root)

    def test_reject_unrelated_game_csv(self):
        (self.root / "2018" / "大樂透加開獎項_201801_201812.csv").write_text(
            SAMPLE, encoding="utf-8")
        self.assertEqual(len(discover_sources(self.root)), 1)


if __name__ == "__main__":
    unittest.main()