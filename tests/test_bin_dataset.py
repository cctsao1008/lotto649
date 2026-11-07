"""Standalone Lotto7x7 BIN v2 storage and source-to-image tests."""
from __future__ import annotations

from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from lotto7x7.bin_dataset import (
    DTYPE, HEADER, MAGIC, RECORD, build, decode_bin, discover_sources,
    encode_bin, load_csv_directory, parse_csv, verify,
)
from lotto7x7.core import decode, encode

SAMPLE = ("\ufeff遊戲名稱,期別,開獎日期,銷售總額,獎號1,獎號2,獎號3,獎號4,獎號5,獎號6,特別號,  \n"
          "大樂透,107000001,2018/01/02,123456,02,06,12,18,36,46,47,,,,\n"
          "大樂透,107000002,2018/01/05,123456,04,13,17,29,31,44,14,,,,\n")


class BinaryImageDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "csv"
        (self.root / "2018").mkdir(parents=True)
        self.csv = self.root / "2018" / "大樂透_201801_201812.csv"
        self.csv.write_text(SAMPLE, encoding="utf-8")

    def test_record_is_exactly_53_bytes_and_header_16(self):
        self.assertEqual(HEADER.size, 16)
        self.assertEqual(RECORD.size, 53)
        self.assertEqual(DTYPE.itemsize, 53)
        self.assertEqual(MAGIC, b"L649IMG2")
        self.assertEqual(DTYPE.names, ("draw_id", "image"))

    def test_csv_bom_extra_columns_and_missing_special_supported(self):
        first = parse_csv(self.csv)
        self.assertEqual([row.main for row in first], [
            (2, 6, 12, 18, 36, 46), (4, 13, 17, 29, 31, 44)])
        self.csv.write_text(SAMPLE.replace(",特別號,  ", "")
                            .replace(",47,,,,", "").replace(",14,,,,", ""),
                            encoding="utf-8")
        removed = parse_csv(self.csv)
        self.assertEqual(first, removed)

    def test_special_field_does_not_change_bin(self):
        original = encode_bin(parse_csv(self.csv))
        self.csv.write_text(SAMPLE.replace(",47,,,,", ",05,,,,")
                            .replace(",14,,,,", ",49,,,,"), encoding="utf-8")
        modified = encode_bin(parse_csv(self.csv))
        self.assertEqual(original, modified)

    def test_unpadded_date_accepted_but_not_stored(self):
        p = self.root / "2024" / "大樂透_2024.csv"
        p.parent.mkdir()
        p.write_text(
            "遊戲名稱,期別,開獎日期,獎號1,獎號2,獎號3,獎號4,獎號5,獎號6,特別號\n"
            "大樂透,113000110,2024/12/3,5,6,7,12,17,35,45\n",
            encoding="utf-8")
        result = parse_csv(p)
        self.assertEqual(result[0].date, 20241203)
        self.assertEqual(result[0].draw_id, 113000110)

    def test_binary_exact_roundtrip_sequential_bitmap(self):
        draws = parse_csv(self.csv)
        binary = encode_bin(draws)
        self.assertEqual(len(binary), 16 + 2 * 53)
        self.assertEqual(binary[:8], MAGIC)
        self.assertEqual(binary[8:16], struct.pack("<II", 2, 53))
        self.assertEqual(binary[16:20], struct.pack("<I", 107000001))
        first_pixels = np.frombuffer(binary[20:69], dtype=np.uint8).reshape(7, 7)
        np.testing.assert_array_equal(first_pixels, encode(np.array(draws[0].main)))
        records = decode_bin(binary)
        self.assertEqual(list(records.dtype.names), ["draw_id", "image"])
        self.assertEqual(records["image"].shape, (2, 7, 7))
        self.assertEqual(records["image"].dtype, np.uint8)
        self.assertEqual(decode(records["image"][0]), [2, 6, 12, 18, 36, 46])
        self.assertEqual(decode(records["image"][1]), [4, 13, 17, 29, 31, 44])
        self.assertTrue(np.all(records["image"].sum(axis=(1, 2)) == 6))

    def test_different_source_column_order_produces_same_image(self):
        first = encode_bin(parse_csv(self.csv))
        self.csv.write_text(SAMPLE.replace("02,06,12,18,36,46", "46,36,18,12,06,02")
                            .replace("04,13,17,29,31,44", "44,31,29,17,13,04"),
                            encoding="utf-8")
        self.assertEqual(first, encode_bin(parse_csv(self.csv)))

    def test_build_and_independent_csv_verification(self):
        out = Path(self.temp.name) / "bin" / "images.bin"
        first = build(self.root, out)
        self.assertEqual(first["records"], 2)
        self.assertEqual(first["source_files"], 1)
        self.assertEqual(first["bytes"], 122)
        self.assertEqual(first["counts_by_year"], {"2018": 2})
        self.assertEqual(verify(out, self.root)["sha256"], first["sha256"])
        self.assertEqual(build(self.root, out)["sha256"], first["sha256"])
        self.csv.write_text(SAMPLE.replace("02,06,12,18,36,46", "02,06,12,18,36,45"),
                            encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs"):
            verify(out, self.root)

    def test_duplicate_id_rejected(self):
        self.csv.write_text(SAMPLE + SAMPLE.splitlines()[-1] + "\n",
                            encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_csv_directory(self.root)

    def test_duplicate_main_and_out_of_range_rejected(self):
        self.csv.write_text(SAMPLE.replace("36,46,47", "36,36,47"),
                            encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "six main"):
            parse_csv(self.csv)
        self.csv.write_text(SAMPLE.replace("36,46,47", "36,50,47"),
                            encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "six main"):
            parse_csv(self.csv)

    def test_bad_magic_truncated_and_incorrect_count_rejected(self):
        binary = encode_bin(parse_csv(self.csv))
        damaged = (
            binary[:10], binary[:-1], b"L649BIN1" + binary[8:],
            b"BADMAGIC" + binary[8:],
            binary[:8] + struct.pack("<II", 3, 53) + binary[16:],
            binary[:8] + struct.pack("<II", 2, 11) + binary[16:],
        )
        for payload in damaged:
            with self.subTest(length=len(payload)):
                with self.assertRaises(ValueError):
                    decode_bin(payload)

    def test_corrupt_binary_pixel_rejected(self):
        binary = bytearray(encode_bin(parse_csv(self.csv)))
        binary[20] = 2
        with self.assertRaisesRegex(ValueError, "exactly six"):
            decode_bin(bytes(binary))
        binary = bytearray(encode_bin(parse_csv(self.csv)))
        binary[20] = 1 if binary[20] == 0 else 0
        with self.assertRaisesRegex(ValueError, "exactly six"):
            decode_bin(bytes(binary))

    def test_bin_out_of_order_or_duplicate_draw_id_rejected(self):
        binary = bytearray(encode_bin(parse_csv(self.csv)))
        binary[69:73] = binary[16:20]
        with self.assertRaisesRegex(ValueError, "unordered"):
            decode_bin(bytes(binary))

    def test_missing_year_detected(self):
        other = self.root / "2020"
        other.mkdir()
        (other / "大樂透_202001_202012.csv").write_text(
            SAMPLE.replace("2018", "2020").replace("1070000", "1090000"),
            encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing calendar"):
            discover_sources(self.root)

    def test_exclude_unrelated_lottery_csvs(self):
        (self.root / "2018" / "大樂透加開獎項_201801_201812.csv").write_text(
            SAMPLE, encoding="utf-8")
        self.assertEqual(len(discover_sources(self.root)), 1)

    def test_wrong_date_year_rejected(self):
        self.csv.write_text(SAMPLE.replace("2018/01/02", "2017/01/02"),
                            encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "year mismatch"):
            parse_csv(self.csv)


if __name__ == "__main__":
    unittest.main()
