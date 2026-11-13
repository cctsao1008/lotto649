"""Check new 2023–24 CWA PDF-derived data; use manually, not scheduled."""
from __future__ import annotations
import csv
from pathlib import Path
from datetime import datetime, date, timedelta
ROOT=Path(__file__).resolve().parent.parent
def read(path):
    with (ROOT/path).open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def main():
    lunar=read("data/astronomy/cwa_pdf_lunar_daily_2023_2024.csv")
    events=read("data/astronomy/cwa_pdf_events_2023_2024.csv")
    draws=read("data/research/v24_cwa/cwa_draw_context_2023_2024.csv")
    assert len(lunar)==731 and len({x["date"] for x in lunar})==731
    assert len(events)==149 and len({x["event_at_local"] for x in events})==149
    assert sum(e["event_kind"]=="solar_term" for e in events)==49
    assert sum(e["event_kind"]=="moon_phase" for e in events)==100
    assert len(draws)==234 and len({x["draw_id"] for x in draws})==234
    by_day={x["date"]:x for x in lunar}
    assert sorted(by_day)==[(date(2023,1,1)+timedelta(days=i)).isoformat() for i in range(731)]
    for d,values in {
        "2023-01-01":("2022","12","10","0"),
        "2023-03-22":("2023","2","1","1"),
        "2024-01-01":("2023","11","20","0"),
        "2024-02-29":("2024","1","20","0"),
        "2024-12-31":("2024","12","1","0"),
    }.items():
        assert tuple(by_day[d][k] for k in ("lunar_year","lunar_month","lunar_day","is_leap_month"))==values
    ordered={}
    for kind in ("solar_term","moon_phase"):
        ordered[kind]=sorted((datetime.fromisoformat(e["event_at_local"]),e["event_code"])
            for e in events if e["event_kind"]==kind)
    for year,expected in ((2023,116),(2024,118)):
        source=read(f"data/csv/lotto649_{year}.csv")
        original={r["draw_id"]:r["date"].replace("/","-") for r in source}
        subset=[r for r in draws if r["date"].startswith(str(year))]
        assert len(subset)==expected
        for row in subset:
            assert original[row["draw_id"]]==row["date"]
            assert all(row[k]==by_day[row["date"]][k] for k in
                ("lunar_year","lunar_month","lunar_day","is_leap_month"))
            noon=datetime.fromisoformat(row["reference_time_local"])
            assert noon.isoformat()==row["date"]+"T12:00:00+08:00"
            for kind,name,at in (
                ("solar_term","active_solar_term_at_noon","solar_term_last_event_local"),
                ("moon_phase","last_moon_phase_at_noon","moon_phase_last_event_local")):
                last=[(dt,ev) for dt,ev in ordered[kind] if dt<=noon][-1]
                assert last==(datetime.fromisoformat(row[at]),row[name])
            elapsed=(noon-datetime.fromisoformat(row["moon_phase_last_event_local"])).total_seconds()/3600
            assert abs(elapsed-float(row["moon_phase_elapsed_hours_at_noon"]))<0.001
    inventory=read("data/astronomy/cwa_pdf_historical_source_inventory_2004_2026.csv")
    assert len(inventory)==23
    assert [int(x["year"]) for x in inventory]==list(range(2004,2027))
    print("PASS: 731 calendar days, 149 events, 234 aligned draw dates; provenance PDF-derived only.")
if __name__=="__main__":main()
