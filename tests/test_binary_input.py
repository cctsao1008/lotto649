"""End-to-end BIN v2 data loading for image learning and temporal research."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lotto7x7.bin_dataset import build, decode_bin
from lotto7x7.__main__ import gallery
from lotto7x7.learning import learned_replay
from lotto7x7.research import DATA, calendar_years, load_history, research


class BinaryImageInputTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        csv_root = self.root / "data" / "csv" / "2018"
        csv_root.mkdir(parents=True)
        self.csv = csv_root / "大樂透_201801_201812.csv"
        self.binary = self.root / "data" / "bin" / "lotto649_images_v2.bin"
        rng = np.random.default_rng(2050)
        self.nums = np.stack([
            np.sort(rng.choice(np.arange(1, 50), size=6, replace=False))
            for _ in range(120)
        ])
        rows = [
            "遊戲名稱,期別,開獎日期,獎號1,獎號2,獎號3,獎號4,獎號5,獎號6,特別號"
        ]
        for i, six in enumerate(self.nums):
            day = date(2018, 1, 1) + timedelta(days=3 * i)
            bonus = next(n for n in range(1, 50) if n not in six)
            rows.append(",".join(("大樂透", str(107000001 + i),
                                  day.strftime("%Y/%m/%d"),
                                  *(str(n) for n in six), str(bonus))))
        self.csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
        build(csv_root.parent, self.binary)

    def test_bin_is_the_default_learning_source(self):
        self.assertEqual(DATA, Path("data/bin/lotto649_images_v2.bin"))

    def test_bin_has_no_special_or_dates_or_duplicate_numbers(self):
        records = decode_bin(self.binary.read_bytes())
        self.assertEqual(records.dtype.names, ("draw_id", "image"))
        self.assertEqual(records["image"].shape, (120, 7, 7))
        self.assertTrue(np.all(records["image"].sum(axis=(1, 2)) == 6))

    def test_training_source_reads_images_without_any_number_conversion(self):
        images, dates, ids = load_history(self.binary)
        self.assertEqual(images.dtype, np.uint8)
        self.assertEqual(images.shape, (120, 7, 7))
        self.assertTrue(np.all(dates == 0))
        self.assertTrue(np.array_equal(calendar_years(ids), np.full(120, 2018)))
        self.assertEqual(ids[0], 107000001)
        self.assertEqual(ids[-1], 107000120)
        np.testing.assert_array_equal(images[0].ravel().nonzero()[0] + 1,
                                      self.nums[0])

    def test_legacy_source_can_be_compared_but_not_used_by_default(self):
        from lotto7x7.core import encode
        structured = np.zeros(120, dtype=np.dtype([
            ("date", "<u4"), ("draw_id", "<u4"),
            ("numbers", "u1", (6,)), ("special", "u1"),
        ]))
        for i, six in enumerate(self.nums):
            day = date(2018, 1, 1) + timedelta(days=3 * i)
            structured["date"][i] = int(day.strftime("%Y%m%d"))
            structured["draw_id"][i] = 107000001 + i
            structured["numbers"][i] = six
            structured["special"][i] = next(n for n in range(1, 50)
                                                    if n not in six)
        npy = self.root / "comparison.npy"
        np.save(npy, structured, allow_pickle=False)
        bin_images, bin_dates, bin_ids = load_history(self.binary)
        old_images, old_dates, old_ids = load_history(npy)
        np.testing.assert_array_equal(bin_images, old_images)
        np.testing.assert_array_equal(bin_ids, old_ids)
        self.assertTrue(np.all(bin_dates == 0))
        self.assertTrue(np.all(old_dates > 20000000))
        np.testing.assert_array_equal(bin_images[0], encode(self.nums[0]))

    def test_year_filter_and_gallery_use_year_not_fabricated_date(self):
        report, model = learned_replay(
            self.binary, year=2018, window=3, epochs=2, patience=2,
            batch_size=32, embedding=8, hidden=12, baseline=2, seed=33)
        self.assertEqual(report["evaluation_draws"], 24)
        self.assertTrue(all(r["date"] is None for r in report["rows"]))
        self.assertTrue(all(r["year"] == 2018 for r in report["rows"]))
        self.assertIn("learned_image", report["summary"])
        html = gallery(report, count=3)
        self.assertIn("Year 2018", html)
        self.assertNotIn("20180000", html)
        self.assertIn("<svg ", html)

    def test_historical_temporal_audit_years_are_derived_from_ids(self):
        report = research(self.binary, year=2018, window=3,
                          baseline=2, shuffles=3, seed=33)
        self.assertEqual(report["evaluated_draws"], 24)
        self.assertEqual(len(report["rows"]), 24)
        self.assertTrue(all(r["date"] is None for r in report["rows"]))
        self.assertTrue(all(r["year"] == 2018 for r in report["rows"]))
        self.assertEqual(report["temporal_order"]["within_year_pairs"], 23)

    def test_corrupt_bin_is_rejected_before_model_training(self):
        invalid = self.root / "bad.bin"
        content = bytearray(self.binary.read_bytes())
        content[20] = 2
        invalid.write_bytes(content)
        with self.assertRaisesRegex(ValueError, "exactly six"):
            load_history(invalid)


if __name__ == "__main__":
    unittest.main()
