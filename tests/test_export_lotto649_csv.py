"""Offline tests for canonical yearly CSV export (no network, no third-party packages)."""
import csv
from datetime import date
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_lotto649_csv.py"
spec = importlib.util.spec_from_file_location("lotto649_csv_export", PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def pack_records(records, *, image=False):
    row_format = module.IMAGE if image else module.HISTORY
    magic = b"L649IMG2" if image else b"L649HIS1"
    encoded = bytearray(module.HEAD.pack(magic, len(records), row_format.size))
    for draw_id, yyyymmdd, six, bonus in records:
        if image:
            pixels = [int(i in six) for i in range(1, 50)]
            encoded.extend(module.IMAGE.pack(draw_id, *pixels))
        else:
            encoded.extend(module.HISTORY.pack(draw_id, yyyymmdd, *six, bonus))
    return bytes(encoded)


class TestYearlyCsvExport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin = self.root / "bin"
        self.csv = self.root / "csv"
        self.bin.mkdir()
        self.records = [
            (93000001, 20040105, (6, 9, 12, 13, 33, 39), 21),
            (93000002, 20040108, (1, 3, 10, 11, 22, 44), 49),
        ]
        self.write_source()

    def write_source(self):
        files = {}
        for name, value in [
            ("lotto649_history_v1.bin", pack_records(self.records)),
            ("lotto649_images_v2.bin", pack_records(self.records, image=True)),
        ]:
            (self.bin / name).write_bytes(value)
            files[name] = {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
        manifest = {
            "dataset": "lotto649.single-source.biga.v1",
            "total_draws": len(self.records),
            "first_draw_id": self.records[0][0],
            "last_draw_id": self.records[-1][0],
            "by_year": {"2004": len(self.records)},
            "files": files,
        }
        (self.bin / "lotto649_source_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")

    def test_minimal_nine_column_csv_and_id_padding(self):
        counts = module.export(self.bin, self.csv)
        self.assertEqual({2004: 2}, counts)
        path = self.csv / "2004" / "大樂透_2004.csv"
        with path.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.reader(file))
        self.assertEqual(list(module.HEADER), rows[0])
        self.assertEqual(9, len(rows[1]))
        self.assertEqual(
            ["2004/01/05", "093000001", "06", "09", "12", "13", "33", "39", "21"],
            rows[1])
        module.export(self.bin, self.csv, verify=True)

    def test_tampered_csv_fails_verification(self):
        module.export(self.bin, self.csv)
        path = self.csv / "2004" / "大樂透_2004.csv"
        path.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "CSV content differs"):
            module.export(self.bin, self.csv, verify=True)

    def test_mismatching_image_bin_fails_before_export(self):
        raw = bytearray((self.bin / "lotto649_images_v2.bin").read_bytes())
        raw[-1] ^= 1
        (self.bin / "lotto649_images_v2.bin").write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
            module.export(self.bin, self.csv)
        self.assertFalse(self.csv.exists())

    def test_bonus_collision_rejected(self):
        self.records[1] = (93000002, 20040108, (1, 3, 10, 11, 22, 44), 44)
        self.write_source()
        with self.assertRaisesRegex(ValueError, "invalid main/special"):
            module.export(self.bin, self.csv)


if __name__ == "__main__":
    unittest.main()
