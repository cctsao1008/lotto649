"""One-time, standard-library validation of the CWA PDF-derived 2025-2026 snapshot."""
from __future__ import annotations
import csv
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def read(path):
    with (ROOT / path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))

def main():
    days = read("data/astronomy/cwa_pdf_lunar_daily_2025_2026.csv")
    events = read("data/astronomy/cwa_pdf_events_2025_2026.csv")
    draws = read("data/research/v24_cwa/cwa_draw_context_2025_2026.csv")
    assert len(days) == 730 and len({r["date"] for r in days}) == 730
    assert len(events) == 149 and len({e["event_at_local"] for e in events}) == 149
    assert len(draws) == 222 and len({r["draw_id"] for r in draws}) == 222
    assert {e["event_kind"] for e in events} == {"solar_term", "moon_phase"}
    assert sum(e["event_kind"] == "solar_term" for e in events) == 49
    assert sum(e["event_kind"] == "moon_phase" for e in events) == 100
    assert len([d for d in draws if d["date"].startswith("2025-")]) == 118
    assert len([d for d in draws if d["date"].startswith("2026-")]) == 104
    day_map = {r["date"]: r for r in days}
    assert sorted(day_map) == [(date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(730)]
    anchors = {
       "2025-01-01": ("2024", "12", "2", "0"),
       "2025-07-24": ("2025", "6", "30", "0"),
       "2025-07-25": ("2025", "6", "1", "1"),
       "2026-02-17": ("2026", "1", "1", "0"),
       "2026-11-10": ("2026", "10", "2", "0"),
    }
    for day, expected in anchors.items():
        row = day_map[day]
        assert tuple(row[k] for k in ("lunar_year", "lunar_month", "lunar_day", "is_leap_month")) == expected
    event_by_kind = {kind: sorted(
        (datetime.fromisoformat(r["event_at_local"]), r["event_code"]) 
        for r in events if r["event_kind"] == kind
    ) for kind in ("solar_term", "moon_phase")}
    for row in draws:
        assert row["date"] in day_map
        drow = day_map[row["date"]]
        assert all(row[key] == drow[key] for key in
            ("lunar_year", "lunar_month", "lunar_day", "is_leap_month"))
        noon = datetime.fromisoformat(row["reference_time_local"])
        assert noon.isoformat() == row["date"] + "T12:00:00+08:00"
        for kind, field, time_field in (
            ("solar_term", "active_solar_term_at_noon", "solar_term_last_event_local"),
            ("moon_phase", "last_moon_phase_at_noon", "moon_phase_last_event_local")):
            preceding = [(t, name) for t, name in event_by_kind[kind] if t <= noon]
            assert preceding and preceding[-1] == (datetime.fromisoformat(row[time_field]), row[field])
        phase_dt = datetime.fromisoformat(row["moon_phase_last_event_local"])
        assert abs(float(row["moon_phase_elapsed_hours_at_noon"]) - (noon-phase_dt).total_seconds()/3600) < 0.001
    for year, n in ((2025, 118), (2026, 104)):
        original = read(f"data/csv/lotto649_{year}.csv")
        original = {r["draw_id"]: r["date"].replace("/", "-") for r in original}
        actual = [r for r in draws if r["date"].startswith(f"{year}-")]
        assert len(actual) == n
        assert all(original[r["draw_id"]] == r["date"] for r in actual)
    case = next(x for x in draws if x["draw_id"] == "115000104")
    assert (case["active_solar_term_at_noon"], case["last_moon_phase_at_noon"]) == ("start_winter", "new_moon")
    assert case["solar_term_last_event_local"] == "2026-11-07T17:52:00+08:00"
    assert case["moon_phase_last_event_local"] == "2026-11-09T15:02:00+08:00"
    print("CWA PDF-derived snapshot PASS: 730 days; 149 events; 222 source-gated draws.")
    print("Scope: 2025-2026 only; PDF transcription, not original CWA CSV.")

if __name__ == "__main__":
    main()
