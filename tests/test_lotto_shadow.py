"""Phase 6 unit tests: seal integrity, future-only cutoffs and post-draw scoring.

Synthetic fixture archives mimic the official structured data schema.
No network, current lottery draws, or ticket selections are required.
"""
from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from lotto_neural import IID_LOGP
from lotto_shadow import (digest, make_seal, prepare, score,
                          validate_forecast_payload, verify_seal_file)
from lotto_synthetic import generate

TPE = ZoneInfo("Asia/Taipei")


def make_fixture(path: Path, draws: np.ndarray, dates: list[date]) -> None:
    """Write a minimal valid official-layout archive with ordered dates."""
    dtype = np.dtype([
        ("draw_id", "<u4"), ("date", "<u4"),
        ("numbers", "u1", (6,)), ("special", "u1"),
    ])
    archive = np.empty(len(draws), dtype=dtype)
    for i, row in enumerate(draws):
        archive[i]["draw_id"] = 111000001 + i
        archive[i]["date"] = int(dates[i].strftime("%Y%m%d"))
        archive[i]["numbers"] = row[::-1]
        archive[i]["special"] = next(v for v in range(1, 50) if v not in row)
    np.save(path, archive, allow_pickle=False)


class ShadowSealTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "source.npy"
        self.seal = self.root / "forecast.json"
        self.report = self.root / "score.json"
        self.draws = generate(count=100, mode="iid", seed=649)[0]
        self.first = date(2026, 1, 1)
        self.dates = [self.first + timedelta(days=i) for i in range(100)]
        make_fixture(self.path, self.draws, self.dates)
        self.last = self.dates[-1]
        self.target = self.last + timedelta(days=2)
        self.now = datetime.combine(self.last + timedelta(days=1),
                                    datetime.min.time(), tzinfo=TPE)
        self.deadline = datetime.combine(self.target,
                                         datetime.min.time(), tzinfo=TPE).replace(hour=18)

    def prep(self, **overrides):
        args = {
            "source": self.path,
            "target_date": self.target.isoformat(),
            "deadline": self.deadline.isoformat(),
            "out": self.seal,
            "now": self.now,
            "code_commit": "a" * 40,
            "seed": 7,
            "window": 3,
            "hidden": 6,
            "epochs": 2,
            "patience": 2,
            "batch_size": 32,
        }
        args.update(overrides)
        return prepare(**args)

    def test_prepared_seal_is_complete_and_correct(self):
        result = self.prep()
        stored = verify_seal_file(self.seal)
        self.assertEqual(result, stored)
        payload = stored["payload"]
        self.assertEqual(payload["default_model"], "iid")
        self.assertEqual(payload["source"]["draw_count"], 100)
        self.assertEqual(set(payload["forecasts"]), {"iid", "static", "lag", "mlp"})
        self.assertEqual(len(payload["forecasts"]["mlp"]["logits"]), 49)
        self.assertTrue(all(v == 0.0 for v in payload["forecasts"]["iid"]["logits"]))
        self.assertEqual(stored["payload_sha256"], digest(payload))
        self.assertNotIn("winning_numbers", str(payload))

    def test_file_is_create_once_only(self):
        self.prep()
        with self.assertRaises(FileExistsError):
            self.prep()

    def test_detect_modified_seal_even_when_valid_json(self):
        self.prep()
        content = json.loads(self.seal.read_text(encoding="utf-8"))
        content["payload"]["forecasts"]["mlp"]["logits"][0] += 1.0
        self.seal.write_text(json.dumps(content), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            verify_seal_file(self.seal)

    def test_reject_stale_source_and_missed_deadline(self):
        with self.assertRaisesRegex(ValueError, "outdated"):
            self.prep(now=self.now + timedelta(days=10),
                      target_date=(self.target + timedelta(days=11)).isoformat(),
                      deadline=(self.deadline + timedelta(days=11)).isoformat())
        with self.assertRaises(ValueError):
            self.prep(now=self.deadline + timedelta(hours=1))

    def test_reject_past_or_wrong_deadline_timezone(self):
        with self.assertRaises(ValueError):
            self.prep(target_date=self.last.isoformat())
        with self.assertRaises(ValueError):
            self.prep(deadline=(self.deadline + timedelta(days=1)).isoformat())

    def test_score_exact_first_future_draw_without_altering_seal(self):
        seal = self.prep()
        initial_bytes = self.seal.read_bytes()
        new_number = generate(count=2, mode="iid", seed=11)[0][0]
        expanded = np.vstack((self.draws, new_number))
        updated = self.root / "updated.npy"
        make_fixture(updated, expanded, self.dates + [self.target])
        result = score(seal_path=self.seal, updated_source=updated,
                       seal_commit="b" * 40, out=self.report,
                       now=self.deadline + timedelta(hours=1),
                       skip_git_verification=True)
        self.assertEqual(result["target_date"], self.target.isoformat())
        self.assertEqual(result["default_model"], "iid")
        self.assertEqual(set(result["scored_models"]), {"iid", "static", "lag", "mlp"})
        self.assertAlmostEqual(result["scored_models"]["iid"]["log_probability"],
                               IID_LOGP, places=11)
        self.assertEqual(result["scored_models"]["iid"]["lift_vs_iid_nats"], 0.0)
        self.assertEqual(self.seal.read_bytes(), initial_bytes)
        with self.assertRaises(FileExistsError):
            score(seal_path=self.seal, updated_source=updated,
                  seal_commit="b" * 40, out=self.report,
                  now=self.deadline + timedelta(hours=1),
                  skip_git_verification=True)
        self.assertEqual(seal["payload_sha256"], result["seal_payload_sha256"])

    def test_scoring_rejects_stale_or_rewritten_draw_history(self):
        self.prep()
        with self.assertRaisesRegex(ValueError, "not been appended"):
            score(seal_path=self.seal, updated_source=self.path,
                  seal_commit="b" * 40, out=self.report,
                  now=self.deadline + timedelta(days=1),
                  skip_git_verification=True)
        updated = self.root / "updated.npy"
        added = generate(count=2, mode="iid", seed=13)[0][0]
        make_fixture(updated, np.vstack((self.draws, added)),
                     self.dates + [self.target + timedelta(days=1)])
        with self.assertRaisesRegex(ValueError, "target date"):
            score(seal_path=self.seal, updated_source=updated,
                  seal_commit="b" * 40, out=self.report,
                  now=self.deadline + timedelta(days=1),
                  skip_git_verification=True)
        changed = self.draws.copy()
        changed[0] = generate(count=2, mode="iid", seed=717)[0][0]
        make_fixture(updated, np.vstack((changed, added)),
                     self.dates + [self.target])
        with self.assertRaisesRegex(ValueError, "prefix differs"):
            score(seal_path=self.seal, updated_source=updated,
                  seal_commit="b" * 40, out=self.report,
                  now=self.deadline + timedelta(days=1),
                  skip_git_verification=True)

    def test_cannot_score_before_draw_deadline(self):
        self.prep()
        extra = generate(count=2, mode="iid", seed=11)[0][0]
        updated = self.root / "updated.npy"
        make_fixture(updated, np.vstack((self.draws, extra)),
                     self.dates + [self.target])
        with self.assertRaisesRegex(ValueError, "before"):
            score(seal_path=self.seal, updated_source=updated,
                  seal_commit="b" * 40, out=self.report,
                  now=self.now,
                  skip_git_verification=True)

    def test_reject_nonuniform_default_or_corrupted_logits(self):
        result = self.prep()
        payload = copy.deepcopy(result["payload"])
        payload["default_model"] = "mlp"
        with self.assertRaises(ValueError):
            validate_forecast_payload(payload)
        payload = copy.deepcopy(result["payload"])
        payload["forecasts"]["iid"]["logits"][0] = 0.1
        with self.assertRaises(ValueError):
            make_seal(payload)
        payload = copy.deepcopy(result["payload"])
        payload["forecasts"]["static"]["logits"][0] = math.inf
        with self.assertRaises(ValueError):
            make_seal(payload)


if __name__ == "__main__":
    unittest.main()
