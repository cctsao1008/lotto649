"""Cast a date-level hexagram and score candidates by matching its six lines."""
import csv
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from ..astronomy.iching_hexagram_association_v24_p2a import outcome_code
from ..astronomy.iching_solar_lunar_v24_p2 import hexagram_from_sectors, sector8

RULE = "cwa_solar_term_lunar_day_gray"
TAIPEI = timezone(timedelta(hours=8))
SOLAR_TERMS = (
    "winter_solstice", "minor_cold", "major_cold", "start_spring", "rain_water",
    "awakening_insects", "spring_equinox", "clear_and_bright", "grain_rain",
    "start_summer", "grain_full", "grain_in_ear", "summer_solstice",
    "minor_heat", "major_heat", "start_autumn", "end_of_heat", "white_dew",
    "autumn_equinox", "cold_dew", "frost_descent", "start_winter",
    "minor_snow", "major_snow",
)


def cast(target_date: date, calendar_dir: Path) -> dict:
    """Read the CWA-derived lunar day and solar term active at Taipei noon."""
    lunar = None
    target = target_date.isoformat()
    for path in sorted(calendar_dir.glob("cwa_pdf_lunar_daily_*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["date"] != target:
                    continue
                if lunar is not None:
                    raise ValueError(f"CWA calendar has duplicate rows for {target}")
                lunar = row
    if lunar is None:
        raise ValueError(f"CWA calendar has no lunar date for {target}")

    reference = datetime.combine(target_date, time(12), TAIPEI)
    instant, event = None, None
    for path in sorted(calendar_dir.glob("cwa_pdf_events_*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["event_kind"] != "solar_term":
                    continue
                observed = datetime.fromisoformat(row["event_at_local"])
                if observed.utcoffset() != TAIPEI.utcoffset(None):
                    raise ValueError("CWA solar term timestamps must have UTC+08:00 offset")
                if observed > reference:
                    continue
                if observed == instant and row["event_code"] != event["event_code"]:
                    raise ValueError(f"conflicting CWA solar terms at {instant}")
                if instant is None or observed > instant:
                    instant, event = observed, row
    if event is None or reference - instant > timedelta(days=17):
        raise ValueError(f"CWA calendar has no active solar term for {target}")

    day = int(lunar["lunar_day"])
    solar_sector = SOLAR_TERMS.index(event["event_code"]) // 3
    lunar_sector = sector8(day - 1, 30)
    lines = hexagram_from_sectors(solar_sector, lunar_sector)
    return {
        "rule": RULE,
        "code": int(lines["astronomical_hexagram_code"]),
        "lines_bottom_to_top": lines["astronomical_hexagram_binary_bottom_to_top"],
        "reference_at": reference.isoformat(),
        "solar_term": event["event_code"],
        "solar_term_at": instant.isoformat(),
        "lunar_year": int(lunar["lunar_year"]),
        "lunar_month": int(lunar["lunar_month"]),
        "lunar_day": day,
        "leap_month": lunar["is_leap_month"] == "1",
        "calendar_sources": sorted({event["source_pdf"], lunar["source_pdf"]}),
    }


def score(numbers: tuple[int, ...], code: int) -> tuple[int, int]:
    """The sorted ticket's odd numbers are yang and its even numbers are yin."""
    ticket_code = outcome_code(numbers)
    return ticket_code, 6 - (ticket_code ^ code).bit_count()
