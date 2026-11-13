#!/usr/bin/env python3
"""Lotto649 independent USNO historical astronomy snapshot (standard library only).

Downloads US Naval Observatory yearly moon phases and Earth seasons as source JSON,
normalizes event instants to UTC and Asia/Taipei, produces manifest + CSV.
Does not parse or touch any existing Lottery history / CWA datasets.

Usage:
  python scripts/fetch_usno_astronomy.py --start-year 2004 --end-year 2026
  python scripts/fetch_usno_astronomy.py --start-year 2004 --end-year 2026 --offline
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE = "https://aa.usno.navy.mil/api"
UTC = timezone.utc
TAIPEI = timezone(timedelta(hours=8))
ENDPOINTS = {
    "moon_phases": ("moon/phases/year", "phasedata"),
    "seasons": ("seasons", "data"),
}
PHASE_NAMES = {
    "New Moon": "new_moon", "First Quarter": "first_quarter",
    "Full Moon": "full_moon", "Last Quarter": "last_quarter",
}
SEASON_NAMES = {"Equinox", "Solstice", "Perihelion", "Aphelion"}
FIELDS = ("event_at_utc", "event_at_taipei", "event_type", "event_name",
          "source_year", "source_url", "raw_sha256")
UA = "Lotto649-USNO-Research/1.0 (official yearly astronomy archival download)"


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def fetch(url: str, *, timeout: float, retries: int) -> bytes:
    last = None
    for attempt in range(retries + 1):
        try:
            request = Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    raise ValueError(f"HTTP {response.status}")
                raw = response.read(2_000_001)
                if len(raw) > 2_000_000:
                    raise ValueError("response unexpectedly larger than 2 MB")
                return raw
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            last = exc
            if attempt < retries:
                time.sleep(min(2 ** attempt, 5))
    raise RuntimeError(f"fetch failed: {last}")


def validate_json(raw: bytes, *, year: int, kind: str, source_url: str) -> list[dict]:
    try:
        obj = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("not valid UTF-8 USNO JSON") from exc
    if not isinstance(obj, dict) or "error" in obj or obj.get("year") != year:
        raise ValueError(f"USNO response error or wrong year: {str(obj)[:220]}")
    array_key = ENDPOINTS[kind][1]
    records = obj.get(array_key)
    if not isinstance(records, list) or not records:
        raise ValueError(f"USNO response missing nonempty {array_key}")
    if kind == "moon_phases" and "numphases" in obj and obj["numphases"] != len(records):
        raise ValueError("USNO moon phase count mismatch")
    if kind == "seasons" and (obj.get("dst", False) or obj.get("tz", 0) != 0):
        raise ValueError("USNO seasons response not in UTC")
    result: list[dict] = []
    seen = set()
    for rec in records:
        if not isinstance(rec, dict):
            raise ValueError("USNO event is not an object")
        name = rec.get("phase") if kind == "moon_phases" else rec.get("phenom")
        if kind == "moon_phases":
            if name not in PHASE_NAMES:
                raise ValueError(f"unexpected moon phase: {name!r}")
            event_name = PHASE_NAMES[name]
        else:
            if name not in SEASON_NAMES:
                raise ValueError(f"unexpected season event: {name!r}")
            event_name = str(name).lower()
        clock = rec.get("time", "")
        if not isinstance(clock, str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", clock):
            raise ValueError(f"unexpected event clock: {clock!r}")
        dt = datetime(int(rec["year"]), int(rec["month"]), int(rec["day"]),
                      int(clock[:2]), int(clock[3:]), tzinfo=UTC)
        if dt.year != year:
            raise ValueError(f"event outside its requested source year: {dt}")
        key = (dt, event_name)
        if key in seen:
            raise ValueError("duplicate event within one source response")
        seen.add(key)
        result.append({
            "event_at_utc": dt.isoformat(timespec="minutes"),
            "event_at_taipei": dt.astimezone(TAIPEI).isoformat(timespec="minutes"),
            "event_type": kind,
            "event_name": event_name,
            "source_year": year,
            "source_url": source_url,
            "raw_sha256": digest(raw),
        })
    if kind == "moon_phases" and not (48 <= len(result) <= 52):
        raise ValueError(f"abnormal number of moon phases ({len(result)})")
    if kind == "seasons":
        for name, expected_count in (("equinox", 2), ("solstice", 2),
                                     ("perihelion", 1), ("aphelion", 1)):
            if sum(x["event_name"] == name for x in result) != expected_count:
                raise ValueError(f"expected {expected_count} yearly {name} events")
    return result


def atomic_write_new(path: Path, data: bytes) -> str:
    """Store new source; never overwrite an existing different byte stream."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = path.read_bytes()
        if old != data:
            raise FileExistsError(f"existing file has different contents; no overwrite: {path}")
        return "cached"
    with path.open("xb") as f:
        f.write(data)
    return "downloaded"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2004)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--out-dir", type=Path, default=Path("data/astronomy/usno_official"))
    parser.add_argument("--timeout", type=float, default=25.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--offline", action="store_true", help="Reparse only saved yearly raw JSON; no network")
    args = parser.parse_args(argv)
    if not (1700 <= args.start_year <= args.end_year <= 2100):
        parser.error("USNO allows year 1700-2100")
    if args.retries < 0 or args.timeout <= 0:
        parser.error("invalid timeout or retries")

    root = args.out_dir.resolve()
    manifest_records, events = [], []
    failures = 0
    for year in range(args.start_year, args.end_year + 1):
        for kind, (endpoint, _) in ENDPOINTS.items():
            url = f"{BASE}/{endpoint}?year={year}"
            dest = root / "raw" / kind / f"{year}.json"
            try:
                if dest.exists():
                    raw = dest.read_bytes()
                    source_state = "cached"
                elif args.offline:
                    raise FileNotFoundError(f"offline source missing: {dest}")
                else:
                    raw = fetch(url, timeout=args.timeout, retries=args.retries)
                    # Validate before putting bytes in trusted source cache.
                    validate_json(raw, year=year, kind=kind, source_url=url)
                    source_state = atomic_write_new(dest, raw)
                parsed = validate_json(raw, year=year, kind=kind, source_url=url)
                events.extend(parsed)
                manifest_records.append({"year": year, "kind": kind, "status": "ok",
                    "acquisition": source_state, "url": url, "path": dest.relative_to(root).as_posix(),
                    "sha256": digest(raw), "size_bytes": len(raw), "event_count": len(parsed)})
                print(f"{year} {kind:<12} OK   {len(parsed):>2} events ({source_state})")
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                failures += 1
                manifest_records.append({"year": year, "kind": kind, "status": "failed",
                    "url": url, "error": str(exc)})
                print(f"{year} {kind:<12} FAIL {exc}", file=sys.stderr)

    events.sort(key=lambda x: (x["event_at_utc"], x["event_type"], x["event_name"]))
    if len({(e["event_at_utc"],e["event_type"],e["event_name"]) for e in events}) != len(events):
        raise RuntimeError("duplicate USNO event across years")
    root.mkdir(parents=True, exist_ok=True)
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(events)
    out_csv = root / f"usno_events_{args.start_year}_{args.end_year}.csv"
    # Atomic latest artifact replacement is only permitted in the isolated USNO namespace.
    csv_bytes = text.getvalue().encode("utf-8")
    out_csv.write_bytes(csv_bytes)
    summary = {
        "schema": "lotto649.usno.astronomy.v1", "institution": "United States Naval Observatory",
        "source_docs": "https://aa.usno.navy.mil/data/api.html",
        "timezone_input": "UTC", "timezone_output": "Asia/Taipei UTC+08:00",
        "years": [args.start_year, args.end_year], "events": len(events),
        "moon_phase_events": sum(e["event_type"] == "moon_phases" for e in events),
        "seasons_and_apsides_events": sum(e["event_type"] == "seasons" for e in events),
        "successful_sources": sum(x["status"] == "ok" for x in manifest_records),
        "failed_sources": failures, "events_csv_path": out_csv.name,
        "events_csv_sha256": digest(csv_bytes),
        "limitations": [
            "USNO seasons API provides the four equinoxes/solstices plus perihelion/aphelion: NOT all 24 Chinese solar terms.",
            "Moon-phase dates do NOT constitute an official Chinese lunar calendar or leap-month assignment.",
            "USNO timestamps are UTC and converted to Asia/Taipei; these are not verified lottery draw times.",
            "No predictive relationship between astronomy and lottery outcomes is established.",
        ],
        "source_records": manifest_records,
    }
    (root / f"manifest_{args.start_year}_{args.end_year}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nSUMMARY: {len(events)} events; "
          f"{summary['successful_sources']}/{len(manifest_records)} source JSON verified; "
          f"failed={failures}")
    print(f"CSV: {out_csv}")
    print(f"Manifest: {root / f'manifest_{args.start_year}_{args.end_year}.json'}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
