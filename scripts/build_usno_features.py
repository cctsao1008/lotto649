#!/usr/bin/env python3
"""Build a one-time USNO-only 2004-2026 date-level feature table.

Inputs are the existing ZIP from fetch_usno_astronomy.py and optionally the
existing Lotto649 data/csv source directory. Stdlib only; never edits inputs.
Provenance: official USNO timestamps, sampled at 12:00 Asia/Taipei on each date.
Missing 2003 boundary observations are left blank rather than extrapolated.
"""
from __future__ import annotations
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from zipfile import ZipFile

UTC8 = timezone(timedelta(hours=8))
START = date(2004, 1, 1)
END = date(2026, 12, 31)
FEATURE_COLUMNS = [
    "date", "reference_at_taipei", "weekday_iso", "day_of_year",
    "last_moon_phase", "last_moon_phase_at_taipei", "hours_since_last_moon_phase",
    "last_new_moon_at_taipei", "days_since_last_new_moon",
    "last_season_marker", "last_season_marker_at_taipei", "days_since_last_season_marker",
    "last_perihelion_at_taipei", "days_since_last_perihelion",
    "last_aphelion_at_taipei", "days_since_last_aphelion",
    "usno_provenance", "reference_clock_assumption",
]
DRAW_COLUMNS = ["date", "draw_id"] + FEATURE_COLUMNS[1:]
VALID_NAMES = {"new_moon", "first_quarter", "full_moon", "last_quarter",
               "equinox", "solstice", "perihelion", "aphelion"}


def digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def read_zip(path: Path):
    with ZipFile(path) as z:
        if z.testzip() is not None:
            raise ValueError("bad ZIP CRC")
        prefix = "usno_official/"
        info = json.loads(z.read(prefix + "manifest_2004_2026.json"))
        sources = info["source_records"]
        if len(sources) != 46 or info["failed_sources"] != 0:
            raise ValueError("expected exactly 46 successfully downloaded USNO JSON sources")
        hashes = {}
        for src in sources:
            rel = src["path"]
            if rel.startswith("/") or ".." in Path(rel).parts:
                raise ValueError("unsafe source path")
            raw = z.read(prefix + rel)
            if digest(raw) != src["sha256"] or len(raw) != src["size_bytes"]:
                raise ValueError(f"source hash/size mismatch: {rel}")
            payload = json.loads(raw)
            parsed = payload["phasedata"] if src["kind"] == "moon_phases" else payload["data"]
            if len(parsed) != src["event_count"]:
                raise ValueError(f"source event count mismatch: {rel}")
            hashes[(src["kind"], src["year"])] = (src["sha256"], src["url"])
        if len(hashes) != 46:
            raise ValueError("duplicate/missing source identity")
        raw = z.read(prefix + info["events_csv_path"])
        if digest(raw) != info["events_csv_sha256"]:
            raise ValueError("event CSV hash mismatch")
        rows = list(csv.DictReader(StringIO(raw.decode("utf-8-sig"))))
        if len(rows) != 1276 or len(rows) != info["events"]:
            raise ValueError("USNO row count mismatch")
        timestamps = []
        keys = set()
        for row in rows:
            t_utc = datetime.fromisoformat(row["event_at_utc"])
            t_local = datetime.fromisoformat(row["event_at_taipei"])
            if t_utc.utcoffset() != timedelta(0) or t_local.utcoffset() != timedelta(hours=8) or t_utc != t_local:
                raise ValueError("UTC to Taipei time mismatch")
            if row["event_name"] not in VALID_NAMES:
                raise ValueError("unknown USNO event name")
            if row["event_type"] not in ("moon_phases", "seasons"):
                raise ValueError("invalid USNO event type")
            ident = (row["event_type"], int(row["source_year"]))
            if ident not in hashes or hashes[ident] != (row["raw_sha256"], row["source_url"]):
                raise ValueError("event-to-raw provenance mismatch")
            key = (t_utc, row["event_name"], row["event_type"])
            if key in keys:
                raise ValueError("duplicate USNO event")
            keys.add(key)
            timestamps.append(t_utc)
        if timestamps != sorted(timestamps):
            raise ValueError("USNO CSV events must be sorted by UTC")
        return rows, info, digest(raw), digest(path.read_bytes())


def event_index(events: list[dict]):
    classes = {
        "moon": {"new_moon", "first_quarter", "full_moon", "last_quarter"},
        "new": {"new_moon"},
        "season": {"equinox", "solstice"},
        "perihelion": {"perihelion"},
        "aphelion": {"aphelion"},
    }
    indexes = {}
    for cls, allowed in classes.items():
        entries = sorted((datetime.fromisoformat(row["event_at_taipei"]), row["event_name"])
                         for row in events if row["event_name"] in allowed)
        indexes[cls] = ([x[0] for x in entries], entries)
    return indexes


def latest(index, t: datetime):
    times, entries = index
    loc = bisect_right(times, t) - 1
    return entries[loc] if loc >= 0 else None


def duration(dt: datetime, t: datetime, divisor: int) -> str:
    return f"{(t - dt).total_seconds()/divisor:.6f}"


def features_for_day(day: date, indexes) -> dict[str, str]:
    ref = datetime(day.year, day.month, day.day, 12, tzinfo=UTC8)
    moon, new, season, peri, aph = (latest(indexes[k], ref)
                                    for k in ("moon", "new", "season", "perihelion", "aphelion"))
    return {
        "date": day.isoformat(),
        "reference_at_taipei": ref.isoformat(timespec="minutes"),
        "weekday_iso": str(day.isoweekday()),
        "day_of_year": str(day.timetuple().tm_yday),
        "last_moon_phase": moon[1] if moon else "",
        "last_moon_phase_at_taipei": moon[0].isoformat(timespec="minutes") if moon else "",
        "hours_since_last_moon_phase": duration(moon[0], ref, 3600) if moon else "",
        "last_new_moon_at_taipei": new[0].isoformat(timespec="minutes") if new else "",
        "days_since_last_new_moon": duration(new[0], ref, 86400) if new else "",
        "last_season_marker": season[1] if season else "",
        "last_season_marker_at_taipei": season[0].isoformat(timespec="minutes") if season else "",
        "days_since_last_season_marker": duration(season[0], ref, 86400) if season else "",
        "last_perihelion_at_taipei": peri[0].isoformat(timespec="minutes") if peri else "",
        "days_since_last_perihelion": duration(peri[0], ref, 86400) if peri else "",
        "last_aphelion_at_taipei": aph[0].isoformat(timespec="minutes") if aph else "",
        "days_since_last_aphelion": duration(aph[0], ref, 86400) if aph else "",
        "usno_provenance": "USNO_API_OFFICIAL_EVENTS_VALIDATED_SHA256",
        "reference_clock_assumption": "LOCAL_12_NOON_NOT_VERIFIED_LOTTERY_DRAW_TIME",
    }


def dates(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def get_draws(root: Path, lookup: dict[str, dict]):
    """Read original annual CSV in place, never write/modify original data."""
    records, audit = [], []
    for year in range(2004, 2027):
        source = root / f"lotto649_{year}.csv"
        raw = source.read_bytes()
        text = raw.decode("utf-8-sig")
        parsed = list(csv.DictReader(StringIO(text)))
        if not parsed or list(parsed[0]) != ["date", "draw_id", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"]:
            raise ValueError(f"unexpected source CSV schema: {source}")
        audit.append({"year": year, "rows_in_file": len(parsed), "sha256": digest(raw)})
        for row in parsed:
            ds = row["date"].replace("/", "-")
            d = date.fromisoformat(ds)
            if d.year != year:
                raise ValueError(f"source year mismatch: {ds}")
            issue = int(row["draw_id"])
            if issue > 115000104:
                continue  # Freeze original v2.4 2498-draw cutoff.
            values = [int(row[f"n{i}"]) for i in range(1, 7)]
            bonus = int(row["bonus"])
            if values != sorted(set(values)) or not all(1 <= x <= 49 for x in values) or not 1 <= bonus <= 49 or bonus in values:
                raise ValueError(f"invalid original lotto balls: {issue}")
            if ds not in lookup:
                raise ValueError(f"missing USNO feature date: {ds}")
            records.append({"date": ds, "draw_id": str(issue), **{k: v for k, v in lookup[ds].items() if k != "date"}})
    records.sort(key=lambda x: (x["date"], int(x["draw_id"])))
    ids = [x["draw_id"] for x in records]
    if len(records) != 2498 or len(set(ids)) != 2498:
        raise ValueError(f"frozen draw baseline mismatch: count={len(records)}")
    if (records[0]["date"], ids[0], records[-1]["date"], ids[-1]) != ("2004-01-05", "93000001", "2026-11-10", "115000104"):
        raise ValueError("first/last draw does not match original v2.4 baseline")
    return records, audit


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return digest(path.read_bytes())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--usno-zip", type=Path, required=True, help="your downloaded usno_official.zip")
    p.add_argument("--csv-root", type=Path, help="existing Lotto649 data/csv; optional")
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    args = p.parse_args(argv)
    outdir = args.out_dir.resolve()
    filename = outdir / "v24_usno_daily_features_2004_2026.csv"
    drawsfile = outdir / "v24_usno_draw_context_2498.csv"
    reportfile = outdir / "v24_usno_data_audit_2004_2026.json"
    targets = [filename, reportfile] + ([drawsfile] if args.csv_root else [])
    if any(x.exists() for x in targets):
        p.error("output exists; refusing to overwrite preexisting research files: " + str([str(x) for x in targets if x.exists()]))
    events, manifest, raw_csv_sha, zip_sha = read_zip(args.usno_zip)
    index = event_index(events)
    daily = [features_for_day(d, index) for d in dates(START, END)]
    if len(daily) != 8401 or len({x["date"] for x in daily}) != 8401:
        raise ValueError("unexpected daily coverage")
    by_date = {x["date"]: x for x in daily}
    draws, draw_audit = get_draws(args.csv_root, by_date) if args.csv_root else ([], [])
    hash_daily = write_csv(filename, FEATURE_COLUMNS, daily)
    hash_draws = write_csv(drawsfile, DRAW_COLUMNS, draws) if args.csv_root else None
    missing = {k: sum(not r[k] for r in daily) for k in (
        "last_moon_phase", "last_new_moon_at_taipei", "last_season_marker",
        "last_perihelion_at_taipei", "last_aphelion_at_taipei")}
    draw_missing = {k: sum(not r[k] for r in draws) for k in missing} if draws else {}
    report = {
        "schema": "lotto649.usno.daily.v1", "source": "USNO official API ONLY",
        "source_zip_sha256": zip_sha, "source_usno_csv_sha256": raw_csv_sha,
        "source_original_json_count": len(manifest["source_records"]),
        "source_events": len(events), "source_year_range": [2004, 2026],
        "days": len(daily), "reference_clock": "12:00 Asia/Taipei assumption, not verified draw time",
        "daily_missing_beginning_of_2004": missing,
        "daily_output": str(filename), "daily_output_sha256": hash_daily,
        "frozen_draws_count": len(draws), "frozen_draws_missing": draw_missing,
        "draw_output": str(drawsfile) if draws else None, "draw_output_sha256": hash_draws,
        "lotto_source_csv_sha256": draw_audit,
        "notes": [
            "No CWA data, lunar-calendar date labels, or lunar leap month imputation.",
            "USNO provides 4 quarter-year markers (2 equinoxes, 2 solstices), NOT 24 solar terms.",
            "Entries requiring an event before Jan 2004 are blank; no 2003 time is fabricated.",
            "Only prior events at/before local noon are used; no future lottery outcomes or same-draw hexagrams.",
            "Date-level astronomy does not imply statistical predictability of fair Lotto649 draws."
        ],
    }
    reportfile.parent.mkdir(parents=True, exist_ok=True)
    with reportfile.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    print(f"PASS: USNO {len(events)} events / 46 verified originals / {len(daily)} dates")
    print(f"DAILY: {filename}")
    print(f"EARLY_2004_MISSING: {missing}")
    if draws:
        print(f"DRAWS: {len(draws)} historical draws / {drawsfile}")
        print(f"DRAW_MISSING: {draw_missing}")
    print(f"REPORT: {reportfile}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
