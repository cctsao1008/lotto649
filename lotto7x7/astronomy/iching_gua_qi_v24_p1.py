"""v2.4 P1: auditable USNO-only 12-state Gua Qi mapping for archived Lotto649.

This is an experimental astronomical encoding, NOT a traditional Chinese
lunar calendar, a precise 24-term calendar, divination, or a forecast model.
Inputs are immutable archives; all outputs use exclusive-create semantics.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
import csv
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path

SCHEMA = "lotto649.iching_guaqi.v2.4.p1"
RULE = "usno_quarter_anchor_centered_12state_half_up_v1"
TZ = timezone(timedelta(hours=8))
REFERENCE_TIME = "12:00:00+08:00"
EXPECTED_USNO_EVENTS_SHA256 = "f084a9248725cecbef257d23963b2660f2152e2bfce2f12054fd89337e33f6df"
EXPECTED_USNO_DRAWS_SHA256 = "230b7824d40a82197e0e25010e542dca17562fa6369d53c0520517a7e316d631"
EXPECTED_V24_P0_SHA256 = "d4ae60f416d4381317a1dd1fbbf225a0198a79b802cf883db78dce4f6fa6f3cd"
DRAW_COUNT = 2498

# Bottom-to-top line = bit 0..5, yang=1 and yin=0; never treat
# King Wen sequence numbers as these six-bit values.
GUA_NAMES = ("Fu", "Lin", "Tai", "Da_Zhuang", "Guai", "Qian",
             "Gou", "Dun", "Pi", "Guan", "Bo", "Kun")
GUA_CHINESE = ("復", "臨", "泰", "大壯", "夬", "乾",
               "姤", "遯", "否", "觀", "剝", "坤")
GUA_CODES = (1, 3, 7, 15, 31, 63, 62, 60, 56, 48, 32, 0)
# Solar anchor indices correspond to state *centers*, not starts.
ANCHORS = {"winter_solstice": 0, "spring_equinox": 3,
           "summer_solstice": 6, "autumn_equinox": 9}
SEASON_MONTHS = {(3, "equinox"): "spring_equinox",
                 (6, "solstice"): "summer_solstice",
                 (9, "equinox"): "autumn_equinox",
                 (12, "solstice"): "winter_solstice"}

FIELDS = (
    "date", "draw_id", "reference_at_taipei",
    "seasonal_state_12", "seasonal_hexagram_name", "seasonal_hexagram_chinese",
    "seasonal_hexagram_code", "seasonal_hexagram_binary_bottom_to_top",
    "seasonal_lower_trigram_code", "seasonal_upper_trigram_code", "seasonal_yang_count",
    "solar_phase_fraction", "solar_anchor_start", "solar_anchor_end",
    "solar_interval_start_taipei", "solar_interval_end_taipei",
    "moon_phase_fraction", "moon_interval_start_taipei", "moon_interval_end_taipei",
    "solar_missing", "moon_missing", "astronomy_known_without_outcome",
    "n1", "n2", "n3", "n4", "n5", "n6", "bonus",
    "same_draw_hexagram_code_OUTCOME_ONLY", "same_draw_hexagram_binary_OUTCOME_ONLY",
    "hamming_distance_OUTCOME_ONLY", "reference_clock_assumption",
    "source_provenance", "mapping_version"
)


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def checked_bytes(path: Path, expect: str) -> bytes:
    b = path.read_bytes()
    value = digest(b)
    if value != expect:
        raise ValueError(f"source SHA-256 mismatch: {path}; actual={value}; expected={expect}")
    return b


def parse_local(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None or dt.utcoffset() != timedelta(hours=8):
        raise ValueError("expected explicit +08:00 Taiwan timestamp: " + value)
    return dt


def read_csv_bytes(contents: bytes) -> list[dict[str, str]]:
    text = contents.decode("utf-8-sig")
    reader = csv.DictReader(text.splitlines())
    if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError("CSV empty or duplicate headers")
    rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError("CSV malformed rows")
    return rows


def verify_source_archive(events_file: Path, manifest_file: Path, raw_dir: Path) -> tuple[list[dict], dict, dict]:
    """Pin the CSV and the original 46 USNO responses; fail closed."""
    event_raw = checked_bytes(events_file, EXPECTED_USNO_EVENTS_SHA256)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if (manifest.get("events_csv_sha256") != EXPECTED_USNO_EVENTS_SHA256 or
            manifest.get("events") != 1276 or
            manifest.get("successful_sources") != 46 or
            manifest.get("failed_sources") != 0):
        raise ValueError("USNO manifest top-level invariant failed")
    originals = manifest.get("source_records", [])
    if len(originals) != 46:
        raise ValueError("expected exactly 46 source records")
    seen: set[tuple[int, str]] = set()
    source = {}
    for item in originals:
        year, kind = item.get("year"), item.get("kind")
        if (not isinstance(year, int) or not 2004 <= year <= 2026 or
                kind not in ("moon_phases", "seasons") or
                (year, kind) in seen or item.get("status") != "ok"):
            raise ValueError("untrusted USNO manifest record")
        seen.add((year, kind))
        rel = Path(item.get("path", ""))
        if rel.as_posix() != f"raw/{kind}/{year}.json":
            raise ValueError("unexpected source path in manifest")
        source_path = raw_dir / kind / f"{year}.json"
        b = checked_bytes(source_path, item["sha256"])
        if len(b) != item.get("size_bytes"):
            raise ValueError("USNO original size mismatch")
        source[(year, kind)] = item
    if len(seen) != 46:
        raise ValueError("missing year/kind raw USNO response")
    events = read_csv_bytes(event_raw)
    if len(events) != 1276:
        raise ValueError("expected 1,276 events")
    event_counter = Counter()
    prev_t = None
    for item in events:
        year = int(item["source_year"])
        kind = item["event_type"]
        ref = source.get((year, kind))
        if not ref or item["raw_sha256"] != ref["sha256"] or item["source_url"] != ref["url"]:
            raise ValueError("event source SHA / official URL mismatch")
        dt = parse_local(item["event_at_taipei"])
        if dt.astimezone(timezone.utc) != datetime.fromisoformat(item["event_at_utc"]):
            raise ValueError("event timezone mismatch")
        if prev_t is not None and dt <= prev_t:
            raise ValueError("USNO events must be strictly chronological")
        prev_t = dt
        event_counter[(year, kind)] += 1
    for key, ref in source.items():
        if event_counter[key] != ref.get("event_count"):
            raise ValueError(f"USNO event count mismatch for {key}")
    audit = {"usno_event_csv_sha256": digest(event_raw),
             "usno_manifest_sha256": digest(manifest_file.read_bytes()),
             "usno_original_json_files": len(source),
             "usno_original_json_sha256": {f"{y}/{k}": x["sha256"] for (y, k), x in sorted(source.items())}}
    return events, manifest, audit


def parse_anchors(events: list[dict]) -> tuple[list[tuple[datetime, str]], list[datetime]]:
    seasons: list[tuple[datetime, str]] = []
    moons: list[datetime] = []
    per_year: Counter = Counter()
    for e in events:
        dt = parse_local(e["event_at_taipei"])
        if e["event_type"] == "seasons" and e["event_name"] in ("solstice", "equinox"):
            name = SEASON_MONTHS.get((dt.month, e["event_name"]))
            if name is None or dt.year != int(e["source_year"]):
                raise ValueError("unrecognized or displaced equinox/solstice")
            seasons.append((dt, name))
            per_year[(int(e["source_year"]), name)] += 1
        elif e["event_type"] == "moon_phases" and e["event_name"] == "new_moon":
            moons.append(dt)
    for y in range(2004, 2027):
        if any(per_year[(y, s)] != 1 for s in ANCHORS):
            raise ValueError(f"missing or duplicate seasonal anchor for {y}")
    if len(seasons) != 92 or len(moons) != 284:
        raise ValueError("unexpected solar/moon counts")
    if any(ANCHORS[seasons[i+1][1]] != (ANCHORS[seasons[i][1]]+3)%12 for i in range(91)):
        raise ValueError("unexpected seasonal anchor ordering")
    if any(b <= a for a, b in zip(moons, moons[1:])):
        raise ValueError("new moon timestamps not increasing")
    return seasons, moons


def micros(span: timedelta) -> int:
    return (span.days * 86400 + span.seconds) * 1_000_000 + span.microseconds


def solar_state(when: datetime, anchors: list[tuple[datetime, str]]) -> dict[str, str]:
    times = [a[0] for a in anchors]
    idx = bisect_right(times, when) - 1
    if idx < 0 or idx + 1 >= len(anchors):
        return {}
    start, name = anchors[idx]
    end, end_name = anchors[idx+1]
    width = micros(end - start)
    elapsed = micros(when - start)
    if not 0 <= elapsed < width or ANCHORS[end_name] != (ANCHORS[name]+3)%12:
        raise ValueError("invalid solar interval")
    # A quadrant spans exactly three gua-state centers. Nearest-center
    # selection with half-up boundaries: index = floor(3*u+1/2).
    nearest = (6*elapsed + width)//(2*width)
    code_index = (ANCHORS[name] + nearest) % 12
    code = GUA_CODES[code_index]
    phase = ((ANCHORS[name] + 3*elapsed/width) % 12) / 12
    return {"seasonal_state_12": str(code_index),
            "seasonal_hexagram_name": GUA_NAMES[code_index],
            "seasonal_hexagram_chinese": GUA_CHINESE[code_index],
            "seasonal_hexagram_code": str(code),
            "seasonal_hexagram_binary_bottom_to_top": "".join(str((code>>j)&1) for j in range(6)),
            "seasonal_lower_trigram_code": str(code & 7),
            "seasonal_upper_trigram_code": str(code >> 3),
            "seasonal_yang_count": str(code.bit_count()),
            "solar_phase_fraction": f"{phase:.10f}",
            "solar_anchor_start": name,
            "solar_anchor_end": end_name,
            "solar_interval_start_taipei": start.isoformat(timespec="minutes"),
            "solar_interval_end_taipei": end.isoformat(timespec="minutes")}


def lunar_phase(when: datetime, new_moons: list[datetime]) -> dict[str, str]:
    idx = bisect_right(new_moons, when) - 1
    if idx < 0 or idx+1 >= len(new_moons):
        return {}
    start, end = new_moons[idx:idx+2]
    elapsed, total = micros(when-start), micros(end-start)
    if not 0 <= elapsed < total:
        raise ValueError("invalid new moon interval")
    return {"moon_phase_fraction": f"{elapsed/total:.10f}",
            "moon_interval_start_taipei": start.isoformat(timespec="minutes"),
            "moon_interval_end_taipei": end.isoformat(timespec="minutes")}


def six_ball_hexagram(balls: list[int]) -> int:
    if len(balls) != 6 or balls != sorted(set(balls)) or balls[0] < 1 or balls[-1] > 49:
        raise ValueError("invalid sorted six numbers")
    return sum((n & 1)<<i for i,n in enumerate(balls))


def load_join(draw_file: Path, postdraw_file: Path, postdraw_report: Path) -> tuple[list[dict], dict]:
    usno = read_csv_bytes(checked_bytes(draw_file, EXPECTED_USNO_DRAWS_SHA256))
    p0 = read_csv_bytes(checked_bytes(postdraw_file, EXPECTED_V24_P0_SHA256))
    report = json.loads(postdraw_report.read_text(encoding="utf-8"))
    if report.get("output_csv_sha256") != EXPECTED_V24_P0_SHA256 or report.get("rows") != DRAW_COUNT:
        raise ValueError("v2.4 P0 original frozen report mismatch")
    if len(usno)!=DRAW_COUNT or len(p0)!=DRAW_COUNT:
        raise ValueError("expected 2,498 frozen draws")
    seen=set()
    merged=[]
    for u,p in zip(usno,p0):
        if (u["date"],u["draw_id"]) != (p["date"],p["draw_id"]):
            raise ValueError("draw date/ID mismatch in original sealed snapshots")
        d_id = u["draw_id"]
        if d_id in seen:
            raise ValueError("duplicate draw_id")
        seen.add(d_id)
        draw_date = date.fromisoformat(u["date"])
        ref = parse_local(u["reference_at_taipei"])
        if ref.date() != draw_date or (ref.hour,ref.minute,ref.second)!=(12,0,0):
            raise ValueError("incorrect noon reference")
        nums=[int(p[f"n{i}"]) for i in range(1,7)]
        bonus=int(p["bonus"])
        if bonus in nums or not 1 <= bonus <= 49:
            raise ValueError("invalid bonus number")
        code=six_ball_hexagram(nums)
        if (str(code) != p["hexagram_code"] or
                "".join(str((code>>j)&1) for j in range(6)) != p["hexagram_binary_bottom_to_top"]):
            raise ValueError("legacy postdraw parity coding changed")
        merged.append({"date":u["date"], "draw_id":d_id,
                       "reference_at_taipei":u["reference_at_taipei"],
                       "numbers":p,"observed_code":code})
    if ((merged[0]["date"], merged[0]["draw_id"])!=("2004-01-05","93000001") or
            (merged[-1]["date"],merged[-1]["draw_id"])!=("2026-11-10","115000104")):
        raise ValueError("frozen history first/last anchor mismatch")
    if any(merged[i]["date"] > merged[i+1]["date"] for i in range(DRAW_COUNT-1)):
        raise ValueError("draw history not chronological")
    return merged, {"usno_draw_sha256":digest(draw_file.read_bytes()),
                    "v24_p0_draw_sha256":digest(postdraw_file.read_bytes()),
                    "v24_p0_report_sha256":digest(postdraw_report.read_bytes())}


def make_row(item: dict, seasons: list[tuple[datetime,str]], moons: list[datetime]) -> dict[str,str]:
    ref = parse_local(item["reference_at_taipei"])
    sky=solar_state(ref,seasons)
    moon=lunar_phase(ref,moons)
    observed=item["observed_code"]
    row={k:"" for k in FIELDS}
    row.update({"date":item["date"],"draw_id":item["draw_id"],
                "reference_at_taipei":ref.isoformat(timespec="minutes"),
                "solar_missing": "false" if sky else "true",
                "moon_missing": "false" if moon else "true",
                "astronomy_known_without_outcome": "true",
                "source_provenance": "USNO_46_ORIGINAL_JSON_SHA256_VERIFIED",
                "reference_clock_assumption": "LOCAL_12_NOON_NOT_VERIFIED_DRAW_TIME",
                "mapping_version": RULE,
                "same_draw_hexagram_code_OUTCOME_ONLY":str(observed),
                "same_draw_hexagram_binary_OUTCOME_ONLY": "".join(str((observed>>j)&1) for j in range(6))})
    row.update(sky)
    row.update(moon)
    p=item["numbers"]
    for k in ("n1","n2","n3","n4","n5","n6","bonus"):
        row[k]=p[k]
    if sky:
        row["hamming_distance_OUTCOME_ONLY"]=str((int(sky["seasonal_hexagram_code"])^observed).bit_count())
    return row


def generate(events_file:Path, manifest_file:Path, raw_dir:Path,
             draw_file:Path, postdraw_file:Path, postdraw_report:Path,
             out_dir:Path) -> dict:
    out_csv=out_dir / "v24_p1_gua_qi_draw_context_2498.csv"
    out_json=out_dir / "v24_p1_gua_qi_audit_2498.json"
    if out_csv.exists() or out_json.exists():
        raise FileExistsError("P1 artifacts exist: refusing to overwrite immutable outputs")
    events,_,source_audit=verify_source_archive(events_file,manifest_file,raw_dir)
    seasons,moons=parse_anchors(events)
    draws,join_audit=load_join(draw_file,postdraw_file,postdraw_report)
    rows=[make_row(x,seasons,moons) for x in draws]
    if len(rows)!=DRAW_COUNT or len({x['draw_id'] for x in rows})!=DRAW_COUNT:
        raise ValueError("draw output count/uniqueness failure")
    counts=Counter(x["seasonal_state_12"] for x in rows)
    by_year={}
    for y in range(2004,2027):
        group=[r for r in rows if r["date"].startswith(str(y))]
        by_year[str(y)]={"rows":len(group),
                         "solar_mapped":sum(r["solar_missing"]=="false" for r in group),
                         "solar_missing":sum(r["solar_missing"]=="true" for r in group),
                         "moon_mapped":sum(r["moon_missing"]=="false" for r in group),
                         "moon_missing":sum(r["moon_missing"]=="true" for r in group)}
    if sum(x['rows'] for x in by_year.values()) !=DRAW_COUNT:
        raise ValueError("per-year draw totals mismatch")
    report={"schema":SCHEMA,"rule_version":RULE,
            "reference_at":"12:00 Asia/Taipei (UTC+08:00) local-date assumption, not actual draw time",
            "season_mapping":"winter solstice center=Fu(0), spring equinox center=Da_Zhuang(3), summer solstice center=Gou(6), autumn equinox center=Guan(9); linear interpolation across adjacent USNO quarter events; nearest of 12 equally spaced state centers; exact half boundaries round forward",
            "season_names":list(GUA_NAMES),"season_chinese_names":list(GUA_CHINESE),
            "season_sixbit_codes_bottom_to_top":list(GUA_CODES),
            "moon_mapping":"time fraction from previous USNO astronomical new moon to next USNO new moon; NOT lunar date/month and NOT a new 64-state model",
            "source_provenance":"USNO ONLY; USNO 46 original JSON sources, 1276 events",
            "astronomy_fields_are_pre_outcome":True,
            "same_draw_labels_are_OUTCOME_ONLY":True,
            "frozen_draws":len(rows),
            "solar_mapped":sum(x['solar_missing']=='false' for x in rows),
            "solar_missing":sum(x['solar_missing']=='true' for x in rows),
            "moon_mapped":sum(x['moon_missing']=='false' for x in rows),
            "moon_missing":sum(x['moon_missing']=='true' for x in rows),
            "state_coverage":{str(k):counts[str(k)] for k in range(12)},
            "coverage_by_year":by_year,
            "missing_source_boundary_policy":"No 2003 winter solstice or pre-Jan-2004 new moon: missing; no fabricated event times",
            "limitations":["Twelve seasonal states are a versioned interpolation approximation, not traditional precise gua-qi dating or a complete 24-term calendar", "USNO astronomy is not a Chinese traditional leap-month calendar", "No evidence of predictability; astronomical variables may be correlated with calendar or draw schedule", "Same-draw outcome hexagram and Hamming distance are not prospective predictors"],
            "source_sha256":{**source_audit,**join_audit},
            "output_csv":"data/research/v24_iching/v24_p1_gua_qi_draw_context_2498.csv"}
    out_dir.mkdir(parents=True,exist_ok=True)
    with out_csv.open("x",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS,lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report["output_csv_sha256"]=digest(out_csv.read_bytes())
    with out_json.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(report,f,indent=2,sort_keys=True,ensure_ascii=False)
        f.write("\n")
    return report


def main(argv=None) -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events",type=Path,default=Path("data/astronomy/usno_official/usno_events_2004_2026.csv"))
    ap.add_argument("--manifest",type=Path,default=Path("data/astronomy/usno_official/manifest_2004_2026.json"))
    ap.add_argument("--raw-dir",type=Path,default=Path("data/astronomy/usno_official/raw"))
    ap.add_argument("--draw-context",type=Path,default=Path("data/research/v24_usno/v24_usno_draw_context_2498.csv"))
    ap.add_argument("--postdraw",type=Path,default=Path("data/research/v24/v24_astronomy_hexagram_2498.csv"))
    ap.add_argument("--postdraw-report",type=Path,default=Path("data/research/v24/v24_astronomy_hexagram_2498_report.json"))
    ap.add_argument("--out-dir",type=Path,default=Path("data/research/v24_iching"))
    args=ap.parse_args(argv)
    result=generate(args.events,args.manifest,args.raw_dir,args.draw_context,
                    args.postdraw,args.postdraw_report,args.out_dir)
    print(f"PASS: {result['frozen_draws']} source-validated draws, "
          f"solar={result['solar_mapped']}, moon={result['moon_mapped']}")
    print(f"CSV_SHA256: {result['output_csv_sha256']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
