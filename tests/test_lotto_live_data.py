"""Phase 6.1 API snapshot parsing and isolated shadow-data bridge tests."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lotto_live_data import (build_shadow_source, expected_regular_days_after,
                             fetch_month, parse_api_result, parse_response)
from lotto_synthetic import generate


def regular_dates(end: date, count: int) -> list[date]:
    days = []
    cursor = end
    while len(days) < count:
        if cursor.weekday() in (1, 4):
            days.append(cursor)
        cursor -= timedelta(days=1)
    return sorted(days)


def fixture_archive(path: Path) -> np.ndarray:
    draws = generate(count=60, mode="iid", seed=140)[0]
    days = regular_dates(date(2026, 10, 30), 60)
    dtype = np.dtype([
        ("draw_id", "<u4"), ("date", "<u4"),
        ("sales", "<u8"), ("tickets", "<u4"), ("prize", "<u8"),
        ("numbers", "u1", (6,)), ("special", "u1"),
    ])
    records = np.zeros(60, dtype=dtype)
    for i, (numbers, dt) in enumerate(zip(draws, days)):
        records[i]["date"] = int(dt.strftime("%Y%m%d"))
        records[i]["draw_id"] = 115000042 + i
        records[i]["numbers"] = numbers
        records[i]["special"] = next(v for v in range(1, 50) if v not in numbers)
        records[i]["sales"] = 1000 + i
    np.save(path, records, allow_pickle=False)
    return records


def api_draw(period: str, dt: str, main: list[int], special: int) -> dict:
    return {
        "period": period,
        "lotteryDate": dt + "T00:00:00",
        "drawNumberSize": main + [special],
    }


def snapshot(path: Path, records: list[dict]) -> None:
    path.write_text(json.dumps({"content": {"lotto649Res": records}}), encoding="utf-8")


class RecentAPIDataTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.old = self.root / "official.npy"
        self.original = fixture_archive(self.old)
        self.old_sha = hashlib.sha256(self.old.read_bytes()).hexdigest()
        self.raw = self.root / "api.json"
        self.new = self.root / "shadow.npy"
        self.manifest = self.root / "manifest.json"
        self.api = [
            api_draw("115000102", "2026-11-03", [2, 23, 24, 25, 29, 39], 31),
            api_draw("115000103", "2026-11-06", [5, 9, 26, 32, 41, 48], 23),
        ]
        snapshot(self.raw, self.api)

    def build(self):
        return build_shadow_source(
            official_archive=self.old, api_snapshots=[self.raw],
            out=self.new, manifest=self.manifest, as_of=date(2026, 11, 7),
        )

    def test_strict_official_api_format_and_sorted_numbers(self):
        records = parse_response(json.loads(self.raw.read_text(encoding="utf-8")))
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["draw_id"], 115000102)
        self.assertEqual(records[1]["date"], 20261106)
        self.assertEqual(records[1]["numbers"], [5, 9, 26, 32, 41, 48])

    def test_reject_format_drift_and_duplicate_periods(self):
        with self.assertRaises(ValueError):
            parse_api_result({"period": 115000102, "lotteryDate": "2026-11-03"})
        with self.assertRaises(ValueError):
            parse_api_result(api_draw("115000102", "2026-11-03",
                                      [1, 1, 2, 3, 4, 5], 7))
        with self.assertRaises(ValueError):
            parse_response({"content": {"lotto649Res": [self.api[0], self.api[0]]}})
        with self.assertRaises(ValueError):
            parse_api_result(api_draw("114000102", "2026-11-03",
                                      [1, 2, 3, 4, 5, 6], 7))

    def test_non_destructive_bridge_continuity_and_evidence(self):
        manifest = self.build()
        self.assertEqual(hashlib.sha256(self.old.read_bytes()).hexdigest(), self.old_sha)
        self.assertEqual(manifest["original_draws"], 60)
        self.assertEqual(manifest["staged_draws"], 2)
        self.assertEqual(manifest["new_total"], 62)
        staged = np.load(self.new, allow_pickle=False)
        np.testing.assert_array_equal(staged[:60], self.original)
        self.assertEqual(int(staged[-1]["draw_id"]), 115000103)
        self.assertEqual(int(staged[-1]["date"]), 20261106)
        np.testing.assert_array_equal(staged[-1]["numbers"], [5, 9, 26, 32, 41, 48])
        self.assertEqual(int(staged[-1]["special"]), 23)
        self.assertEqual(int(staged[-1]["sales"]), 0)
        self.assertEqual(int(staged[0]["sales"]), 1000)
        self.assertEqual(manifest["shadow_sha256"],
                         hashlib.sha256(self.new.read_bytes()).hexdigest())
        self.assertTrue(self.manifest.is_file())

    def test_gap_is_not_silently_filled(self):
        snapshot(self.raw, [self.api[1]])
        with self.assertRaisesRegex(ValueError, "continuous regular"):
            self.build()
        self.assertFalse(self.new.exists())
        self.assertFalse(self.manifest.exists())

    def test_duplicate_old_period_conflict_rejected(self):
        conflicting = [
            api_draw("115000101", "2026-10-30", [1, 2, 3, 4, 5, 6], 7),
            *self.api,
        ]
        snapshot(self.raw, conflicting)
        with self.assertRaisesRegex(ValueError, "disagree"):
            self.build()

    def test_incomplete_archive_or_existing_outputs_fail_closed(self):
        self.build()
        with self.assertRaises(FileExistsError):
            self.build()

    def test_future_dated_api_record_rejected(self):
        with self.assertRaisesRegex(ValueError, "future-dated"):
            build_shadow_source(
                official_archive=self.old, api_snapshots=[self.raw],
                out=self.new, manifest=self.manifest, as_of=date(2026, 11, 4),
            )

    def test_expected_calendar_tuesday_friday(self):
        expected = expected_regular_days_after(date(2026, 10, 30),
                                               date(2026, 11, 6))
        self.assertEqual(expected, [date(2026, 11, 3), date(2026, 11, 6)])

    def test_network_fetch_is_mocked_and_writes_exact_bytes(self):
        class FakeResponse:
            content = json.dumps({"content": {"lotto649Res": self.api}}).encode("utf-8")
            def raise_for_status(self):
                pass
            def json(self):
                return json.loads(self.content)
        out = self.root / "fetched.json"
        with patch("lotto_live_data.requests.get", return_value=FakeResponse()) as call:
            meta = fetch_month(month="2026-11", out=out)
        self.assertEqual(out.read_bytes(), FakeResponse.content)
        self.assertEqual(meta["response_draw_count"], 2)
        self.assertEqual(meta["latest_response_date"], 20261106)
        self.assertEqual(call.call_args.kwargs["params"]["month"], "2026-11")


if __name__ == "__main__":
    unittest.main()
