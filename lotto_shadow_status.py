"""Read-only Phase 6 readiness audit for a sealed one-step forecast.

A filename is not evidence that its contents include a future draw. This
module inspects actual draw dates and content hashes without modifying the
original archive, derived shadow source, forecast seal, or Git history.

Status READY_FOR_SCORING still requires an independently verified official
draw result. Git issue creation time and Git commit metadata are not a
cryptographically trusted timestamp for the initial forecast's content.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lotto_historical import load_official_main_draws
from lotto_live_data import parse_response
from lotto_shadow import (observed_prefix_hash, parse_date, parse_deadline,
                          verify_committed_content, verify_seal_file)

VERSION = "0.1.0"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_manifest(path: Path, candidate: Path, actual_count: int) -> dict:
    """Verify a source manifest and the saved raw-API evidence it references."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest must contain an object")
    if manifest.get("shadow_sha256") != file_sha256(candidate):
        raise ValueError("manifest shadow SHA-256 does not match candidate bytes")
    if manifest.get("new_total") != actual_count:
        raise ValueError("manifest new_total does not match candidate record count")
    original_count = manifest.get("original_draws")
    staged_count = manifest.get("staged_draws")
    if (not isinstance(original_count, int) or not isinstance(staged_count, int)
            or original_count + staged_count != actual_count):
        raise ValueError("manifest record counts are inconsistent")
    entries = manifest.get("api_snapshots")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest is missing API evidence")
    snapshots = []
    for item in entries:
        if not isinstance(item, dict) or not isinstance(item.get("file"), str):
            raise ValueError("invalid API snapshot manifest entry")
        # Manifest paths may originate on Windows, including backslash separators.
        snapshot_path = Path(item["file"].replace("\\", "/"))
        if not snapshot_path.is_file():
            raise ValueError(f"manifest API snapshot missing: {snapshot_path}")
        if file_sha256(snapshot_path) != item.get("sha256"):
            raise ValueError(f"API snapshot SHA-256 mismatch: {snapshot_path}")
        results = parse_response(json.loads(snapshot_path.read_text(encoding="utf-8")))
        if len(results) != item.get("records"):
            raise ValueError(f"API snapshot record count mismatch: {snapshot_path}")
        snapshots.append({
            "file": str(snapshot_path),
            "sha256_verified": True,
            "record_count": len(results),
            "last_draw_date": results[-1]["date"] if results else None,
        })
    return {
        "status": "VERIFIED",
        "path": str(path),
        "original_draws": original_count,
        "staged_draws": staged_count,
        "new_total": actual_count,
        "last_added": manifest.get("last_added"),
        "api_snapshots": snapshots,
    }


def inspect_status(*, seal_path: Path, candidate: Path | None = None,
                   manifest_path: Path | None = None,
                   seal_commit: str | None = None,
                   now: datetime | None = None) -> dict:
    """Inspect immutable forecast content and the actual candidate draw data."""
    seal = verify_seal_file(seal_path)
    if seal_commit is not None:
        verify_committed_content(seal_path, seal_commit)
    payload = seal["payload"]
    target = int(parse_date(payload["target_date"]).strftime("%Y%m%d"))
    deadline = parse_deadline(payload["deadline_utc"])
    current = now if now is not None else datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("inspection time must be timezone-aware")
    current = current.astimezone(timezone.utc)
    report = {
        "schema": "lotto649.shadow.readiness.v1",
        "status": "NO_CANDIDATE_SOURCE",
        "forecast": {
            "target_date": payload["target_date"],
            "last_sealed_date": payload["source"]["last_date"],
            "last_sealed_draw_id": payload["source"]["last_draw_id"],
            "sealed_draw_count": payload["source"]["draw_count"],
            "payload_sha256": seal["payload_sha256"],
            "default_model": payload["default_model"],
            "commit_content_verified": seal_commit is not None,
        },
        "clock": {
            "observed_utc": current.isoformat(),
            "deadline_utc": deadline.isoformat(),
            "research_deadline_passed": current > deadline,
            "note": "Local clock is not an independently attested timestamp",
        },
        "candidate": None,
        "manifest": None,
        "warnings": [
            "A valid Git seal and server-side issue time are different evidence types.",
            "READY_FOR_SCORING requires independent official result verification.",
            "This audit never selects numbers or scores the forecast.",
        ],
    }
    if candidate is None or not candidate.exists():
        report["status"] = "NO_CANDIDATE_SOURCE"
        return report
    numbers, dates, ids = load_official_main_draws(candidate)
    prefix_len = payload["source"]["draw_count"]
    if len(numbers) < prefix_len:
        raise ValueError("candidate has fewer historical draws than sealed source")
    prefix_sha = observed_prefix_hash(numbers[:prefix_len], dates[:prefix_len],
                                      ids[:prefix_len])
    if prefix_sha != payload["source"]["observed_prefix_sha256"]:
        raise ValueError("candidate history differs from the sealed forecast source")
    candidate_sha = file_sha256(candidate)
    next_date = int(dates[prefix_len]) if len(numbers) > prefix_len else None
    next_id = int(ids[prefix_len]) if len(numbers) > prefix_len else None
    report["candidate"] = {
        "path": str(candidate),
        "file_sha256": candidate_sha,
        "same_file_sha256_as_sealed_source": (
            candidate_sha == payload["source"]["sha256"]),
        "record_count": len(numbers),
        "new_draw_count": len(numbers) - prefix_len,
        "last_recorded_date": int(dates[-1]),
        "last_recorded_draw_id": int(ids[-1]),
        "first_new_date": next_date,
        "first_new_draw_id": next_id,
        "source_prefix_sha256_verified": True,
    }
    if manifest_path is None:
        inferred = candidate.with_suffix(".manifest.json")
        if inferred.is_file():
            manifest_path = inferred
    if manifest_path is not None:
        if not manifest_path.is_file():
            raise ValueError(f"manifest not found: {manifest_path}")
        report["manifest"] = inspect_manifest(manifest_path, candidate, len(numbers))
        if report["manifest"]["new_total"] != len(numbers):
            raise AssertionError("manifest count verification failed")
    if len(numbers) == prefix_len:
        report["status"] = "WAITING_FOR_TARGET_DRAW"
    elif next_date != target:
        report["status"] = "BLOCKED_TARGET_NOT_FIRST_NEW_DRAW"
    elif current <= deadline:
        report["status"] = "BLOCKED_RESEARCH_DEADLINE_NOT_PASSED"
    else:
        report["status"] = "READY_FOR_SCORING_PENDING_OFFICIAL_RESULT_CHECK"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only shadow forecast readiness audit")
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--seal-commit",
                        help="Verify forecast file bytes against its original commit")
    args = parser.parse_args()
    result = inspect_status(
        seal_path=args.seal, candidate=args.candidate,
        manifest_path=args.manifest, seal_commit=args.seal_commit)
    print(f"Lotto649 Shadow Readiness v{VERSION}")
    print(f"[STATUS] {result['status']}")
    print(f"[FORECAST] Target {result['forecast']['target_date']} "
          f"sealed draws={result['forecast']['sealed_draw_count']}")
    print(f"[CLOCK] Research deadline passed: "
          f"{result['clock']['research_deadline_passed']}")
    if result["candidate"] is not None:
        x = result["candidate"]
        print(f"[CANDIDATE] {x['record_count']} draws, "
              f"new={x['new_draw_count']}, last={x['last_recorded_date']}, "
              f"first_new={x['first_new_date']}")
        print(f"[HASH] Same candidate bytes as original seal source: "
              f"{x['same_file_sha256_as_sealed_source']}")
    if result["manifest"] is not None:
        print(f"[MANIFEST] {result['manifest']['status']}, "
              f"API snapshots={len(result['manifest']['api_snapshots'])}")
    print("[CAUTION] Read-only audit; official result and timing still need external validation")


if __name__ == "__main__":
    main()
