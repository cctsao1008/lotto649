"""v2.4 P2: exact-time USNO solar8 x synodic lunar8 hexagram encoding.

This is an experimental 64-state representation, not traditional divination,
24 solar terms, measured lunar elongation, or a lottery forecasting system.
Immutable upstream inputs are validated and outputs are exclusive-create.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
import csv
from datetime import date, datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from . import iching_gua_qi_v24_p1 as p1

SCHEMA = "lotto649.iching_solar_lunar.v2.4.p2"
RULE = "usno_winter_anchor_solar8_newmoon_time8_gray_cyclic_v1"
P1_CSV_SHA256 = "53f35eb6fb0f6f5d6bc36e4b6a78aa63586263c924ba65940cc86c588c4fb1f1"
OUT_CSV = "v24_p2_solar_lunar_hexagram_draw_context_2498.csv"
OUT_AUDIT = "v24_p2_solar_lunar_hexagram_audit_2498.json"

# Integer trigram bit ordering is bottom-to-top, yin=0, yang=1.
# The order below is the 3-bit binary CODE, not Fu Xi or King Wen ordering.
TRIGRAMS = ("Kun", "Zhen", "Kan", "Dui", "Gen", "Li", "Xun", "Qian")
TRIGRAMS_ZH = ("坤", "震", "坎", "兌", "艮", "離", "巽", "乾")
FIELDS = (
    "date", "draw_id", "reference_at_taipei",
    "solar_phase_fraction", "solar_sector_8", "solar_gray_code", "solar_trigram_name",
    "solar_trigram_chinese", "solar_interval_start_taipei", "solar_interval_end_taipei",
    "lunar_phase_fraction", "lunar_sector_8", "lunar_gray_code", "lunar_trigram_name",
    "lunar_trigram_chinese", "lunar_interval_start_taipei", "lunar_interval_end_taipei",
    "astronomical_hexagram_code", "astronomical_hexagram_binary_bottom_to_top",
    "astronomical_lower_trigram_code", "astronomical_upper_trigram_code",
    "astronomical_yang_count", "solar_missing", "lunar_missing", "astronomy_pair_mapped",
    "missing_reason", "p1_seasonal_state_12", "p1_seasonal_hexagram_code",
    "n1", "n2", "n3", "n4", "n5", "n6", "bonus",
    "same_draw_hexagram_code_OUTCOME_ONLY", "hamming_distance_OUTCOME_ONLY",
    "astronomy_known_without_outcome", "reference_clock_assumption", "source_provenance",
    "mapping_version"
)


def gray3(index: int) -> int:
    if type(index) is not int or not 0 <= index < 8:
        raise ValueError("three-bit Gray sector must be 0..7")
    return index ^ (index >> 1)


def sector8(position: int, period: int) -> int:
    """Exact integer half-open binning: no floating-point threshold drift."""
    if type(position) is not int or type(period) is not int or period <= 0 or not 0 <= position < period:
        raise ValueError("invalid cyclical position/period")
    result = (8 * position) // period
    if not 0 <= result < 8:
        raise AssertionError("invalid sector")
    return result


def hexagram_from_sectors(solar_sector: int, lunar_sector: int) -> dict[str, str]:
    upper, lower = gray3(solar_sector), gray3(lunar_sector)
    code = (upper << 3) | lower
    return {
        "astronomical_hexagram_code": str(code),
        "astronomical_hexagram_binary_bottom_to_top": "".join(str((code >> i) & 1) for i in range(6)),
        "astronomical_lower_trigram_code": str(lower),
        "astronomical_upper_trigram_code": str(upper),
        "astronomical_yang_count": str(code.bit_count()),
    }


def solar_segment(when: datetime, anchors: list[tuple[datetime, str]]) -> dict[str, str]:
    """Interpolate a circular 8-sector solar year from 4 fixed quarter anchors."""
    times = [t for t, _ in anchors]
    idx = bisect_right(times, when) - 1
    if idx < 0 or idx + 1 >= len(anchors):
        return {}
    start, name = anchors[idx]
    end, end_name = anchors[idx + 1]
    if p1.ANCHORS[end_name] != (p1.ANCHORS[name] + 3) % 12:
        raise ValueError("misordered USNO solar anchor")
    total, elapsed = p1.micros(end - start), p1.micros(when - start)
    if total <= 0 or not 0 <= elapsed < total:
        raise ValueError("invalid solar interval")
    # P1 places solstice/equinox at cycle phase 0,1/4,1/2,3/4.
    # Express phase with integer numerator/denominator to avoid P1 10-digit rounding.
    phase_num = (p1.ANCHORS[name] * total + 3 * elapsed) % (12 * total)
    phase_den = 12 * total
    index = sector8(phase_num, phase_den)
    value = gray3(index)
    return {
        "solar_phase_fraction": f"{phase_num / phase_den:.10f}",
        "solar_sector_8": str(index),
        "solar_gray_code": str(value),
        "solar_trigram_name": TRIGRAMS[value],
        "solar_trigram_chinese": TRIGRAMS_ZH[value],
        "solar_interval_start_taipei": start.isoformat(timespec="minutes"),
        "solar_interval_end_taipei": end.isoformat(timespec="minutes"),
    }


def lunar_segment(when: datetime, new_moons: list[datetime]) -> dict[str, str]:
    """Time fraction between new moons; not a physical lunar elongation angle."""
    idx = bisect_right(new_moons, when) - 1
    if idx < 0 or idx + 1 >= len(new_moons):
        return {}
    start, end = new_moons[idx:idx + 2]
    total, elapsed = p1.micros(end - start), p1.micros(when - start)
    if total <= 0 or not 0 <= elapsed < total:
        raise ValueError("invalid synodic interval")
    index = sector8(elapsed, total)
    value = gray3(index)
    return {
        "lunar_phase_fraction": f"{elapsed / total:.10f}",
        "lunar_sector_8": str(index),
        "lunar_gray_code": str(value),
        "lunar_trigram_name": TRIGRAMS[value],
        "lunar_trigram_chinese": TRIGRAMS_ZH[value],
        "lunar_interval_start_taipei": start.isoformat(timespec="minutes"),
        "lunar_interval_end_taipei": end.isoformat(timespec="minutes"),
    }


def check_p1(p1_csv: Path, p1_audit: Path, draws: list[dict],
             events_audit: dict, anchors: list[tuple[datetime, str]],
             moons: list[datetime]) -> tuple[list[dict], dict]:
    source = p1.checked_bytes(p1_csv, P1_CSV_SHA256)
    report = json.loads(p1_audit.read_text(encoding="utf-8"))
    if (report.get("output_csv_sha256") != P1_CSV_SHA256 or
            report.get("rule_version") != p1.RULE or
            report.get("frozen_draws") != p1.DRAW_COUNT or
            report.get("solar_mapped") != 2476 or
            report.get("moon_mapped") != 2493 or
            report.get("source_sha256", {}).get("usno_event_csv_sha256") != p1.EXPECTED_USNO_EVENTS_SHA256 or
            report.get("source_sha256", {}).get("usno_original_json_files") != 46 or
            report.get("source_sha256", {}).get("v24_p0_draw_sha256") != p1.EXPECTED_V24_P0_SHA256):
        raise ValueError("P1 frozen audit incompatible with source archive")
    if report["source_sha256"]["usno_manifest_sha256"] != events_audit["usno_manifest_sha256"]:
        raise ValueError("P1 USNO manifest changed")
    previous = p1.read_csv_bytes(source)
    if len(previous) != len(draws):
        raise ValueError("P1 record count mismatch")
    for old, draw in zip(previous, draws):
        if (old["draw_id"], old["date"]) != (draw["draw_id"], draw["date"]):
            raise ValueError("P1 historical join mismatch")
        if old["reference_at_taipei"] != p1.parse_local(draw["reference_at_taipei"]).isoformat(timespec="minutes"):
            raise ValueError("P1 local-noon reference time mismatch")
        ref = p1.parse_local(draw["reference_at_taipei"])
        old_solar = p1.solar_state(ref, anchors)
        old_moon = p1.lunar_phase(ref, moons)
        if (old["solar_missing"] != ("false" if old_solar else "true") or
                old["moon_missing"] != ("false" if old_moon else "true") or
                old["seasonal_state_12"] != old_solar.get("seasonal_state_12", "") or
                old["solar_phase_fraction"] != old_solar.get("solar_phase_fraction", "") or
                old["moon_phase_fraction"] != old_moon.get("moon_phase_fraction", "") or
                old["same_draw_hexagram_code_OUTCOME_ONLY"] != str(draw["observed_code"])):
            raise ValueError("P1 frozen astronomical or outcome state mismatch")
        for k in ("n1", "n2", "n3", "n4", "n5", "n6", "bonus"):
            if old[k] != draw["numbers"][k]:
                raise ValueError("P1 numbers differ from source-audited P0")
    return previous, {"p1_context_sha256": p1.digest(source),
                      "p1_audit_sha256": p1.digest(p1_audit.read_bytes()),
                      "p1_rule_version": report["rule_version"]}


def make_row(draw: dict, prior: dict,
             anchors: list[tuple[datetime, str]], moons: list[datetime]) -> dict[str, str]:
    ref = p1.parse_local(draw["reference_at_taipei"])
    sun = solar_segment(ref, anchors)
    moon = lunar_segment(ref, moons)
    result = {key: "" for key in FIELDS}
    result.update({
        "date": draw["date"], "draw_id": draw["draw_id"],
        "reference_at_taipei": prior["reference_at_taipei"],
        "solar_missing": str(not sun).lower(), "lunar_missing": str(not moon).lower(),
        "astronomy_pair_mapped": str(bool(sun and moon)).lower(),
        "missing_reason": "+".join(k for k, available in (("solar_boundary_unavailable", sun),
                                                      ("lunar_boundary_unavailable", moon)) if not available),
        "p1_seasonal_state_12": prior["seasonal_state_12"],
        "p1_seasonal_hexagram_code": prior["seasonal_hexagram_code"],
        "same_draw_hexagram_code_OUTCOME_ONLY": str(draw["observed_code"]),
        "astronomy_known_without_outcome": "true",
        "reference_clock_assumption": "LOCAL_12_NOON_NOT_VERIFIED_DRAW_TIME",
        "source_provenance": "USNO_46_ORIGINAL_JSON_SHA256_VERIFIED_P1_CROSSCHECKED",
        "mapping_version": RULE,
    })
    result.update(sun)
    result.update(moon)
    for k in ("n1", "n2", "n3", "n4", "n5", "n6", "bonus"):
        result[k] = draw["numbers"][k]
    if sun and moon:
        result.update(hexagram_from_sectors(int(sun["solar_sector_8"]), int(moon["lunar_sector_8"])))
        result["hamming_distance_OUTCOME_ONLY"] = str((int(result["astronomical_hexagram_code"]) ^
                                                       draw["observed_code"]).bit_count())
    return result


def generate(*, events_file: Path, manifest_file: Path, raw_dir: Path,
             draw_file: Path, postdraw_file: Path, postdraw_report: Path,
             p1_file: Path, p1_report: Path, out_dir: Path) -> dict:
    out_csv = out_dir / OUT_CSV
    out_json = out_dir / OUT_AUDIT
    if out_csv.exists() or out_json.exists():
        raise FileExistsError("P2 immutable output already exists")
    events, _, source_audit = p1.verify_source_archive(events_file, manifest_file, raw_dir)
    anchors, moons = p1.parse_anchors(events)
    draws, join_audit = p1.load_join(draw_file, postdraw_file, postdraw_report)
    previous, p1_audit = check_p1(p1_file, p1_report, draws, source_audit, anchors, moons)
    rows = [make_row(draw, prior, anchors, moons) for draw, prior in zip(draws, previous)]
    if len(rows) != 2498 or len({r["draw_id"] for r in rows}) != 2498:
        raise ValueError("frozen draw identity mismatch")
    counts = Counter(int(r["astronomical_hexagram_code"]) for r in rows if r["astronomy_pair_mapped"] == "true")
    solar_counts = Counter(int(r["solar_sector_8"]) for r in rows if not r["solar_missing"] == "true")
    lunar_counts = Counter(int(r["lunar_sector_8"]) for r in rows if not r["lunar_missing"] == "true")
    coverage = {}
    for year in range(2004, 2027):
        by_year = [r for r in rows if r["date"].startswith(str(year))]
        coverage[str(year)] = {
            "rows": len(by_year),
            "pair_mapped": sum(r["astronomy_pair_mapped"] == "true" for r in by_year),
            "solar_missing": sum(r["solar_missing"] == "true" for r in by_year),
            "lunar_missing": sum(r["lunar_missing"] == "true" for r in by_year),
        }
    pair_mapped = sum(counts.values())
    assert pair_mapped + sum(r["astronomy_pair_mapped"] == "false" for r in rows) == 2498
    assert all(gray3(i) in range(8) for i in range(8))
    assert len({int(hexagram_from_sectors(s, l)["astronomical_hexagram_code"])
                for s in range(8) for l in range(8)}) == 64
    report = {
        "schema": SCHEMA, "mapping_version": RULE,
        "research_type": "DESCRIPTIVE_DATA_FOUNDATION_NO_PREDICTION_TEST",
        "astronomy_source": "USNO only; 46 SHA-256-verified original JSON files; 1276 events",
        "frozen_draws": len(rows), "pair_mapped": pair_mapped,
        "pair_missing": len(rows) - pair_mapped,
        "solar_mapped": sum(solar_counts.values()), "lunar_mapped": sum(lunar_counts.values()),
        "phase_coverage_by_year": coverage,
        "astronomical_hexagram_histogram": {str(i): counts[i] for i in range(64)},
        "solar_sector_histogram": {str(i): solar_counts[i] for i in range(8)},
        "lunar_sector_histogram": {str(i): lunar_counts[i] for i in range(8)},
        "mapping_description": {
            "solar": "P1 quarter-year linear interpolation; winter solstice phase=0, spring equinox=1/4, summer solstice=1/2, autumn equinox=3/4; exact integer bin=floor(8*phase)",
            "lunar": "Time fraction between adjacent astronomical USNO new moons; exact integer bin=floor(8*elapsed/interval); not actual lunar ecliptic phase angle",
            "gray_3bit": {str(i): gray3(i) for i in range(8)},
            "trigram_chinese_by_binary_code_0_to_7": list(TRIGRAMS_ZH),
            "upper_3bit_from": "solar sector Gray code", "lower_3bit_from": "lunar sector Gray code",
            "hexagram_code": "lower_gray | (upper_gray << 3)",
            "line_order": "bottom-to-top; bit0=bottom line; yin=0, yang=1",
            "sectors": "[k/8,(k+1)/8); exact boundary belongs to later bin",
            "reference_at": "12:00 Asia/Taipei local noon, NOT verified draw time",
            "missing_boundaries": "no 2003 prior solar/new moon event input; missing values remain blank",
        },
        "astronomical_fields_use_no_outcomes": True,
        "observed_parity_code_and_hamming_are_OUTCOME_ONLY": True,
        "source_sha256": {**source_audit, **join_audit, **p1_audit},
        "source_p1a_conclusion": "NO_EVIDENCE_UNCHANGED; P2 makes no statistical inference",
        "limitations": [
            "An astronomical hexagram is researcher-defined; not a traditional King Wen ordinal or a verified classical divination method",
            "Solar-8 and lunar-time-8 sectors may be nonuniformly populated across scheduled draw dates; 64 possible state combinations must not be assumed equally probable under the calendar sampling process",
            "Lunar time-phase interpolation is not true Sun-Moon elongation, illumination or Chinese lunar-calendar day",
            "Future USNO ephemeris timestamps are astronomically computable without winning numbers but may not represent a contemporaneous historical precommitment",
            "The same-draw observed parity-hexagram and Hamming distance are output labels only; using them to select the same draw would leak outcomes",
            "No statistical association or predictive advantage is demonstrated by this mapping",
        ],
        "output_csv": f"data/research/v24_iching_p2/{OUT_CSV}",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_csv.open("x", encoding="utf-8", newline="") as file:
        w = csv.DictWriter(file, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    report["output_csv_sha256"] = sha256(out_csv.read_bytes()).hexdigest()
    with out_json.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, sort_keys=True)
        file.write("\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=Path("data/astronomy/usno_official/usno_events_2004_2026.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/astronomy/usno_official/manifest_2004_2026.json"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/astronomy/usno_official/raw"))
    parser.add_argument("--draw-context", type=Path, default=Path("data/research/v24_usno/v24_usno_draw_context_2498.csv"))
    parser.add_argument("--postdraw", type=Path, default=Path("data/research/v24/v24_astronomy_hexagram_2498.csv"))
    parser.add_argument("--postdraw-report", type=Path, default=Path("data/research/v24/v24_astronomy_hexagram_2498_report.json"))
    parser.add_argument("--p1-context", type=Path, default=Path("data/research/v24_iching/v24_p1_gua_qi_draw_context_2498.csv"))
    parser.add_argument("--p1-report", type=Path, default=Path("data/research/v24_iching/v24_p1_gua_qi_audit_2498.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/research/v24_iching_p2"))
    args = parser.parse_args(argv)
    result = generate(events_file=args.events, manifest_file=args.manifest, raw_dir=args.raw_dir,
                      draw_file=args.draw_context, postdraw_file=args.postdraw,
                      postdraw_report=args.postdraw_report, p1_file=args.p1_context,
                      p1_report=args.p1_report, out_dir=args.out_dir)
    print(f"PASS: {result['frozen_draws']} verified draws; solar8+lunar8={result['pair_mapped']} "
          f"missing={result['pair_missing']}; unique astronomical hexagrams="
          f"{sum(v>0 for v in result['astronomical_hexagram_histogram'].values())}")
    print("CSV_SHA256:", result["output_csv_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
