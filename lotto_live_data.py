"""Lotto649 Phase 6.1: stage recent official-API results without editing history.

The official annual archive is updated monthly, so a prospective one-step
study needs a distinct recent-draw source. This module obtains the Taiwan
Lottery site's undocumented read-only Lotto649Result JSON, validates dates,
periods, six main numbers and the special number, and creates a separate
shadow-only composite NPY with an evidence manifest.

The internal API has no compatibility guarantee. Never guess missing fields,
fill draw gaps or claim that local retrieval time proves external provenance.
All unavailable financial fields in newly appended shadow-only records are
explicitly zero placeholders, not official audited values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import requests

from lotto_historical import load_official_main_draws

VERSION = "0.1.0"
API_URL = "https://api.taiwanlottery.com/TLCAPIWeB/Lottery/Lotto649Result"
TPE = ZoneInfo("Asia/Taipei")
MONTH_RE = re.compile(r"^20\d{2}-(?:0[1-9]|1[0-2])$")
NUM_RE = re.compile(r"^(?:[0-9]{1,2})$")
PERIOD_RE = re.compile(r"^[0-9]{9}$")
DATE_RE = re.compile(r"^20\d{2}-\d{2}-\d{2}")


def sha256_file(path: Path) -> str:
    """Return the byte-level file digest without modifying the source."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integer(value: object, *, low: int, high: int) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not an integer")
    if isinstance(value, (int, np.integer)):
        number = int(value)
    elif isinstance(value, str) and NUM_RE.fullmatch(value.strip()):
        number = int(value)
    else:
        raise ValueError(f"invalid draw number: {value!r}")
    if not low <= number <= high:
        raise ValueError(f"number out of range: {number}")
    return number


def parse_api_result(value: object) -> dict:
    """Decode a single official API record; fail closed on format drift."""
    if not isinstance(value, dict):
        raise ValueError("API draw record must be an object")
    period = value.get("period")
    if isinstance(period, int) and not isinstance(period, bool):
        period_str = str(period)
    elif isinstance(period, str):
        period_str = period.strip()
    else:
        raise ValueError("API period is missing or invalid")
    if not PERIOD_RE.fullmatch(period_str):
        raise ValueError(f"invalid nine-digit period {period!r}")
    raw_date = value.get("lotteryDate")
    if not isinstance(raw_date, str) or not DATE_RE.match(raw_date):
        raise ValueError("API lotteryDate is missing or invalid")
    result_date = date.fromisoformat(raw_date[:10])
    if int(period_str[:3]) != result_date.year - 1911:
        raise ValueError("period ROC year and drawing date disagree")
    values = value.get("drawNumberSize")
    if not isinstance(values, list) or len(values) != 7:
        raise ValueError("expected six main numbers followed by one special number")
    parsed = [integer(v, low=1, high=49) for v in values]
    if len(set(parsed)) != 7:
        raise ValueError("duplicate draw or special numbers")
    return {
        "date": int(result_date.strftime("%Y%m%d")),
        "draw_id": int(period_str),
        "numbers": sorted(parsed[:6]),
        "special": parsed[6],
    }


def parse_response(payload: object) -> list[dict]:
    """Parse the API result envelope with a strict documented research key."""
    if not isinstance(payload, dict) or not isinstance(payload.get("content"), dict):
        raise ValueError("API must provide a content object")
    records = payload["content"].get("lotto649Res")
    if not isinstance(records, list):
        raise ValueError("API content.lotto649Res must be a list")
    result = [parse_api_result(record) for record in records]
    seen_ids = set()
    seen_dates = set()
    for item in result:
        if item["draw_id"] in seen_ids or item["date"] in seen_dates:
            raise ValueError("duplicate period or date in API response")
        seen_ids.add(item["draw_id"])
        seen_dates.add(item["date"])
    return sorted(result, key=lambda x: x["date"])


def expected_regular_days_after(cutoff: date, latest: date) -> list[date]:
    """Conservative Tuesday/Friday schedule, without exceptional draw overrides."""
    if latest < cutoff:
        raise ValueError("latest date precedes cutoff")
    days = []
    day = cutoff + timedelta(days=1)
    while day <= latest:
        if day.weekday() in (1, 4):
            days.append(day)
        day += timedelta(days=1)
    return days


def fetch_month(*, month: str, out: Path, timeout: int = 20) -> dict:
    """Fetch and preserve the API JSON bytes exactly once for traceability."""
    if not MONTH_RE.fullmatch(month) or timeout <= 0:
        raise ValueError("month must be YYYY-MM and timeout must be positive")
    if out.exists():
        raise FileExistsError(f"API snapshot already exists: {out}")
    response = requests.get(API_URL,
                            params={"period": "", "month": month,
                                    "pageNum": 1, "pageSize": 50},
                            headers={"User-Agent": "lotto649-phase6-research/0.1"},
                            timeout=timeout)
    response.raise_for_status()
    if len(response.content) > 4_000_000:
        raise ValueError("API response exceeds 4 MB safety limit")
    payload = response.json()
    observations = parse_response(payload)
    if not observations:
        raise ValueError("official API returned no draws for the requested month")
    if any(str(record["date"])[:6] != month.replace("-", "") for record in observations):
        raise ValueError("API returned a draw outside the requested month")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as handle:
        handle.write(response.content)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "api": API_URL,
        "requested_month": month,
        "retrieved_at_utc_local_clock_only": datetime.now(timezone.utc).isoformat(),
        "file": str(out),
        "file_sha256": sha256_file(out),
        "response_draw_count": len(observations),
        "latest_response_date": observations[-1]["date"],
        "note": "Undocumented official-site API; locally observed data, not independent timestamp proof",
    }


def build_shadow_source(*, official_archive: Path,
                        api_snapshots: list[Path], out: Path,
                        manifest: Path, as_of: date | None = None) -> dict:
    """Append only strictly verified recent official-API draws to a new file."""
    if not api_snapshots:
        raise ValueError("at least one API snapshot is required")
    if out.exists() or manifest.exists():
        raise FileExistsError("shadow output or evidence manifest already exists")
    today = as_of if as_of is not None else datetime.now(TPE).date()
    archive = np.load(official_archive, allow_pickle=False)
    numbers, dates, ids = load_official_main_draws(official_archive)
    cutoff_date = date.fromisoformat(str(int(dates[-1]))[:4] + "-" +
                                    str(int(dates[-1]))[4:6] + "-" +
                                    str(int(dates[-1]))[6:8])
    received = []
    source_evidence = []
    for path in api_snapshots:
        data = json.loads(path.read_text(encoding="utf-8"))
        records = parse_response(data)
        received.extend(records)
        source_evidence.append({
            "file": str(path),
            "sha256": sha256_file(path),
            "records": len(records),
        })
    unique_ids = set()
    unique_dates = set()
    for entry in received:
        if entry["date"] in unique_dates or entry["draw_id"] in unique_ids:
            raise ValueError("duplicate API draw across monthly snapshots")
        unique_dates.add(entry["date"])
        unique_ids.add(entry["draw_id"])
    historical_by_id = {int(k): (int(d), list(map(int, s)))
                        for k, d, s in zip(ids, dates, numbers)}
    additions = []
    for entry in received:
        dt = date.fromisoformat(str(entry["date"])[:4] + "-" +
                               str(entry["date"])[4:6] + "-" +
                               str(entry["date"])[6:8])
        if dt > today:
            raise ValueError("future-dated result in API response")
        if int(entry["draw_id"]) in historical_by_id:
            archived_dt, archived_numbers = historical_by_id[entry["draw_id"]]
            if archived_dt != entry["date"] or archived_numbers != entry["numbers"]:
                raise ValueError("API and official archive disagree on an existing period")
        if dt > cutoff_date:
            additions.append(entry)
    additions.sort(key=lambda x: x["date"])
    if not additions:
        raise ValueError("no newer draw available in supplied API responses")
    required = expected_regular_days_after(
        cutoff_date,
        date.fromisoformat(str(additions[-1]["date"])[:4] + "-" +
                           str(additions[-1]["date"])[4:6] + "-" +
                           str(additions[-1]["date"])[6:8]),
    )
    actual = [
        date.fromisoformat(str(v["date"])[:4] + "-" +
                            str(v["date"])[4:6] + "-" +
                            str(v["date"])[6:8])
        for v in additions
    ]
    if actual != required:
        raise ValueError(
            f"expected continuous regular Tue/Fri draw dates {required}; received {actual}. "
            "Exceptional draw schedules need a separately reviewed source update."
        )
    if any(int(item["draw_id"]) in set(map(int, ids)) for item in additions):
        raise ValueError("newer API period already exists in base archive")
    if any(b["draw_id"] - a["draw_id"] != 1
           for a, b in zip(additions, additions[1:])
           if a["draw_id"] // 1_000_000 == b["draw_id"] // 1_000_000):
        raise ValueError("gap in sequential API draw IDs")
    result = np.empty(len(archive) + len(additions), dtype=archive.dtype)
    result[:len(archive)] = archive
    for i, item in enumerate(additions, start=len(archive)):
        result[i] = np.zeros((), dtype=archive.dtype)
        result[i]["draw_id"] = item["draw_id"]
        result[i]["date"] = item["date"]
        result[i]["numbers"] = item["numbers"]
        result[i]["special"] = item["special"]
    load_fields = {"draw_id", "date", "numbers", "special"}
    if archive.dtype.names is None or not load_fields.issubset(archive.dtype.names):
        raise ValueError("unsupported base record fields")
    stats = {
        "schema": "lotto649.shadow.api_bridge.v1",
        "status": "SHADOW_ONLY_RECENT_API_UNDOCUMENTED",
        "official_monthly_source": str(official_archive),
        "official_monthly_sha256": sha256_file(official_archive),
        "api_endpoint": API_URL,
        "api_snapshots": source_evidence,
        "original_draws": len(archive),
        "staged_draws": len(additions),
        "first_added": additions[0],
        "last_added": additions[-1],
        "new_total": len(result),
        "shadow_source": str(out),
        "financial_fields_status": "ZERO_PLACEHOLDERS_ON_NEW_RECORDS_ONLY",
        "warnings": [
            "API is an undocumented official-site interface; its contract may change.",
            "The original official monthly archive is never overwritten.",
            "Newly staged sales/tickets/prize fields are unknown, NOT verified zero values.",
            "Manual examination of API and official period results is recommended.",
            "Local fetch timestamps are not independent pre-draw publication evidence.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out.open("xb") as handle:
            np.save(handle, result, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        verify_numbers, verify_dates, verify_ids = load_official_main_draws(out)
        if (len(verify_numbers) != len(result)
                or int(verify_dates[-1]) != additions[-1]["date"]
                or int(verify_ids[-1]) != additions[-1]["draw_id"]):
            raise ValueError("staged shadow source failed round-trip validation")
        stats["shadow_sha256"] = sha256_file(out)
        with manifest.open("x", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2, ensure_ascii=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        out.unlink(missing_ok=True)
        manifest.unlink(missing_ok=True)
        raise
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6.1 official-site API shadow bridge")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--month", required=True, help="YYYY-MM")
    fetch.add_argument("--out", required=True, type=Path)
    fetch.add_argument("--timeout", type=int, default=20)
    stage = commands.add_parser("build")
    stage.add_argument("--official-archive", default="data/normalized/lotto649_base.npy", type=Path)
    stage.add_argument("--api-snapshot", action="append", required=True, type=Path)
    stage.add_argument("--out", required=True, type=Path)
    stage.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "fetch":
        output = fetch_month(month=args.month, out=args.out, timeout=args.timeout)
        print(f"[PASS] Fetched {output['response_draw_count']} records from official-site API")
        print(f"[DONE] {output['file']} SHA256={output['file_sha256']}")
    else:
        result = build_shadow_source(official_archive=args.official_archive,
                                     api_snapshots=args.api_snapshot,
                                     out=args.out, manifest=args.manifest)
        print(f"[PASS] Original {result['original_draws']}, staged {result['staged_draws']}, total {result['new_total']}")
        print(f"[DONE] {result['shadow_source']} SHA256={result['shadow_sha256']}")
        print("[CAUTION] Staged source is shadow-only; not an official monthly archive")


if __name__ == "__main__":
    main()
