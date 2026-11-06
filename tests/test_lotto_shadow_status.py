"""Read-only Phase 6 readiness tests with synthetic official-layout archives."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lotto_shadow import (SCHEMA, make_seal, observed_prefix_hash,
                          write_once)
from lotto_shadow_status import inspect_status
from lotto_synthetic import generate

SEAL_TIME = datetime(2026, 11, 7, 1, 43, tzinfo=timezone(timedelta(hours=8)))
BEFORE = datetime(2026, 11, 7, 2, 0, tzinfo=timezone(timedelta(hours=8)))
AFTER = datetime(2026, 11, 10, 21, 0, tzinfo=timezone(timedelta(hours=8)))


def write_archive(path: Path, draws: np.ndarray, days: list[date]) -> None:
    dtype = np.dtype([
        ("draw_id", "<u4"), ("date", "<u4"),
        ("sales", "<u8"), ("tickets", "<u4"), ("prize", "<u8"),
        ("numbers", "u1", (6,)), ("special", "u1"),
    ])
    arr = np.zeros(len(draws), dtype=dtype)
    for i, (numbers, day) in enumerate(zip(draws, days)):
        arr[i]["draw_id"] = 115000041 + i
        arr[i]["date"] = int(day.strftime("%Y%m%d"))
        arr[i]["numbers"] = numbers
        arr[i]["special"] = next(x for x in range(1, 50) if x not in numbers)
    np.save(path, arr, allow_pickle=False)


class ShadowReadinessTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.base = self.root / "sealed_source.npy"
        self.candidate = self.root / "lotto649_recent_20261110.npy"
        self.seal = self.root / "forecast.json"
        self.manifest = self.candidate.with_suffix(".manifest.json")
        self.draws = generate(count=61, mode="iid", seed=567)[0]
        self.cutoff = date(2026, 11, 6)
        self.target = date(2026, 11, 10)
        self.days = [self.cutoff - timedelta(days=59-i) for i in range(60)]
        write_archive(self.base, self.draws[:60], self.days)
        data = np.load(self.base, allow_pickle=False)
        payload = {
            "schema": SCHEMA,
            "target_date": self.target.isoformat(),
            "deadline_utc": "2026-11-10T11:00:00+00:00",
            "prepared_at_utc": SEAL_TIME.isoformat(),
            "default_model": "iid",
            "code_commit": "a"*40,
            "forecasts": {name: {"logits": [0.0]*49}
                          for name in ("iid", "static", "lag", "mlp")},
            "source": {
                "last_date": self.cutoff.isoformat(),
                "last_draw_id": int(data[-1]["draw_id"]),
                "draw_count": 60,
                "sha256": hashlib.sha256(self.base.read_bytes()).hexdigest(),
                "observed_prefix_sha256": observed_prefix_hash(
                    data["numbers"].astype(np.int64),
                    data["date"].astype(np.int64),
                    data["draw_id"].astype(np.int64)),
            },
        }
        write_once(self.seal, make_seal(payload))

    def write_candidate(self, date_of_new: date | None = None):
        day = date_of_new or self.target
        write_archive(self.candidate,
                      self.draws,
                      self.days + [day])

    def inspect(self, **kwargs):
        args = {"seal_path": self.seal, "candidate": self.candidate, "now": BEFORE}
        args.update(kwargs)
        return inspect_status(**args)

    def test_named_future_file_without_new_draw_is_waiting(self):
        self.candidate.write_bytes(self.base.read_bytes())
        result = self.inspect()
        self.assertEqual(result["status"], "WAITING_FOR_TARGET_DRAW")
        self.assertEqual(result["candidate"]["new_draw_count"], 0)
        self.assertTrue(result["candidate"]["same_file_sha256_as_sealed_source"])
        self.assertEqual(result["candidate"]["last_recorded_date"], 20261106)

    def test_nonexistent_candidate_has_no_target_data(self):
        result = self.inspect()
        self.assertEqual(result["status"], "NO_CANDIDATE_SOURCE")
        self.assertIsNone(result["candidate"])

    def test_valid_target_still_blocked_before_deadline(self):
        self.write_candidate()
        result = self.inspect()
        self.assertEqual(result["status"], "BLOCKED_RESEARCH_DEADLINE_NOT_PASSED")
        self.assertFalse(result["clock"]["research_deadline_passed"])

    def test_valid_target_is_ready_after_deadline_not_automatically_scored(self):
        self.write_candidate()
        result = self.inspect(now=AFTER)
        self.assertEqual(result["status"],
                         "READY_FOR_SCORING_PENDING_OFFICIAL_RESULT_CHECK")
        self.assertEqual(result["candidate"]["first_new_date"], 20261110)
        self.assertTrue(result["candidate"]["source_prefix_sha256_verified"])

    def test_wrong_first_new_date_is_blocked(self):
        self.write_candidate(date(2026, 11, 9))
        result = self.inspect(now=AFTER)
        self.assertEqual(result["status"], "BLOCKED_TARGET_NOT_FIRST_NEW_DRAW")

    def test_modified_historical_prefix_is_rejected(self):
        self.write_candidate()
        data = np.load(self.candidate, allow_pickle=False)
        data[0]["numbers"] = [1, 2, 3, 4, 5, 6]
        data[0]["special"] = 7
        np.save(self.candidate, data, allow_pickle=False)
        with self.assertRaisesRegex(ValueError, "history differs"):
            self.inspect(now=AFTER)

    def test_manifest_and_raw_api_bytes_are_verified(self):
        self.write_candidate()
        snapshot = self.root / "raw.json"
        body = {"content": {"lotto649Res": [{
            "period": "115000104",
            "lotteryDate": "2026-11-10T00:00:00",
            "drawNumberSize": self.draws[60].astype(int).tolist() + [
                next(x for x in range(1, 50) if x not in self.draws[60])
            ],
        }]}}
        snapshot.write_text(json.dumps(body), encoding="utf-8")
        manifest = {
            "shadow_sha256": hashlib.sha256(self.candidate.read_bytes()).hexdigest(),
            "original_draws": 60,
            "staged_draws": 1,
            "new_total": 61,
            "last_added": {"date": 20261110, "draw_id": 115000104},
            "api_snapshots": [{
                "file": str(snapshot),
                "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                "records": 1,
            }],
        }
        self.manifest.write_text(json.dumps(manifest), encoding="utf-8")
        result = self.inspect(now=AFTER)
        self.assertEqual(result["manifest"]["status"], "VERIFIED")
        self.assertEqual(result["manifest"]["api_snapshots"][0]["last_draw_date"],
                         20261110)
        self.assertTrue(result["manifest"]["api_snapshots"][0]["sha256_verified"])

    def test_manifest_mismatch_refused(self):
        self.write_candidate()
        self.manifest.write_text(
            json.dumps({
                "shadow_sha256": "0"*64, "original_draws": 60,
                "staged_draws": 1, "new_total": 61,
                "api_snapshots": [{"file": "unused", "sha256": "0"*64, "records": 1}],
            }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest shadow SHA-256"):
            self.inspect(now=AFTER)

    def test_git_commit_byte_verification_is_opt_in_but_enforced(self):
        self.candidate.write_bytes(self.base.read_bytes())
        with patch("lotto_shadow_status.verify_committed_content") as verifier:
            result = self.inspect(seal_commit="b"*40)
            self.assertTrue(result["forecast"]["commit_content_verified"])
            verifier.assert_called_once_with(self.seal, "b"*40)
        with patch("lotto_shadow_status.verify_committed_content") as verifier:
            result = self.inspect()
            self.assertFalse(result["forecast"]["commit_content_verified"])
            verifier.assert_not_called()


if __name__ == "__main__":
    unittest.main()
