# CWA Official PDF Calendar Research: 2025–2026 (One-Time Snapshot)

## Scope
This is a one-time **partial historical import**, not an automated data pipeline.
The source is the Central Weather Administration annual *Calendar Data Table*:
- https://www.cwa.gov.tw/Data/astronomy/2024cal.pdf (prior-event boundary only)
- https://www.cwa.gov.tw/Data/astronomy/2025cal.pdf
- https://www.cwa.gov.tw/Data/astronomy/2026cal.pdf
- Government dataset 157677: https://data.gov.tw/dataset/157677

The original CWA API CSV streams A-A0087-001/002/003 could not be ingested in this session.
**No file here claims to be an original downloaded CWA CSV**; timestamps are manually
transcribed from the official PDFs, with explicitly derived daily lunar dates.
Original PDF byte hashes were not obtainable, so this is publicly checkable by
the published PDF URL but is **not byte-level authenticated provenance**.

## Artifacts
- `data/astronomy/cwa_pdf_events_2025_2026.csv`: all 2025–2026 official PDF solar
  term transitions (48) and four principal lunar phase events (99), plus two
  boundary events from 2024 (149 records). Events use explicit UTC+08:00 times.
- `data/astronomy/cwa_pdf_lunar_daily_2025_2026.csv`: 730 Gregorian days, lunar
  year/month/day/leap labels *derived* from official PDF lunar month first-day
  boundaries and checked against the printed calendar. This is NOT a raw CWA
  A-A0087-001 daily export. In 2025, lunar month 6 is followed by leap month 6.
- `data/research/v24_cwa/cwa_draw_context_2025_2026.csv`: 222 date-aligned
  historical draws (2025:118, 2026:104 through #115000104), **no draw result
  numbers** duplicated and no future draws added. Source draw files are read-only.

## Clock-time caveat
The existing P0 snapshot uses 12:00 Asia/Taipei as a date-level reference;
so does the research draw join. This is **not a verified lottery draw time**.
`active_solar_term_at_noon` and `last_moon_phase_at_noon` refer to the latest
event at or before local noon; calendar date alone cannot determine whether a
draw was before/after a transition occurring later that same date.

## Audit anchors
- 2025-01-01: lunar 2024-12-02.
- 2025-07-24: lunar 2025-06-30; 2025-07-25: **leap 06-01**.
- 2026-02-17: lunar 2026-01-01.
- 2026-11-10: draw #115000104; lunar 2026-10-02; latest solar
  transition is `start_winter` on Nov 7, 17:52; latest new moon is Nov 9, 15:02.
- Observed dataset sizes: 730 calendar rows, 149 event rows (147 in 2025–26 +
  2 boundary records), 222 aligned historical draws.

## Constraints and next data acquisition
- **2004–2024 remains unimported for this CWA-derived lunar daily series**.
  Do not fill missing official coverage with opaque algorithms or claim complete
  2,498-draw CWA alignment.
- Obtain the three original CWA CSV exports (A-A0087-001/002/003), record
  SHA-256 and year coverage, and compare against the PDF-derived reference.
  For 2004–2024, find authenticated historical CWA archive sources first.
- The old `data/research/v24` 2,498-draw baseline and v2.3 prospective
  seals remain untouched. This import is a separate additive layer.
- No statistical relationship or prediction advantage is implied by lunar
  date, solar term, moon phase, or hexagram. New tests require time-split
  holdouts, appropriate schedule controls, and no same-draw target leakage.

## One-time audit
Execute manually, not on a schedule:
```powershell
python scripts/verify_cwa_pdf_calendar_2025_2026.py
```
