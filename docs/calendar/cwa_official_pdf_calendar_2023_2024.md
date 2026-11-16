# CWA PDF Calendar Extraction — 2023–2024 (Versioned Extension)

## Source and provenance

- Central Weather Administration official 2022 PDF (for 2023 prehistory): https://www.cwa.gov.tw/Data/astronomy/2022cal.pdf
- Central Weather Administration official 2023 annual PDF: https://www.cwa.gov.tw/Data/astronomy/2023cal.pdf
- Central Weather Administration official 2024 annual PDF: https://www.cwa.gov.tw/Data/astronomy/2024cal.pdf
- Government dataset https://data.gov.tw/dataset/157677 (A-A0087-001/002/003)

Solar-term and moon-phase timestamps were **manually transcribed** from the
published CWA year PDF; each event retains source URL and UTC+08:00 offset.
Daily lunar labels were **derived** using the official PDF's lunar new-month
boundaries; 2023 leap lunar month 2 is explicitly flagged. This is NOT a raw
CWA CSV download and PDF byte-level content hashes were not captured.
Treat the data as human-transcribed, publicly auditable research input.

## Files

- `data/astronomy/cwa_pdf_events_2023_2024.csv` (149 events: 24 x 2
  solar terms, 49+50 major moon-phase events, and two boundary events from 2022)
- `data/astronomy/cwa_pdf_lunar_daily_2023_2024.csv` (731 Gregorian days,
  inclusive 2023-01-01–2024-12-31).
- `data/research/v24_cwa/cwa_draw_context_2023_2024.csv` (234 historical draws:
  2023: 116 and 2024: 118, all date-aligned to the immutable source archive).
- `data/astronomy/cwa_pdf_historical_source_inventory_2004_2026.csv`
  records the verified public URL status and import progress for every year.
- `scripts/verify_cwa_pdf_calendar_2023_2024.py` supports a local manual audit.

## Scientific handling

The joined reference is **12:00 Asia/Taipei local time**, not a verified draw
time. A solar-term or lunar event that occurs later that same date is therefore
not yet the active event for the noon snapshot.

The 2,498-draw v2.4 P0 original, all source CSV/BIN and v2.3 sealed
prospective artifacts are unchanged. Calendar coverage from 2023–2026 is
now 456 unique historical draws, while 2004–2022 remain without corresponding
CWA-specific draw joins. Public annual PDFs from 2018–2022 are inventoried as
available but **NOT imported**. 2004–2017 direct PDF availability is not
verified; do not interpret missing evidence as proof of no archived official data.

No extra winning probability or causal astronomical/hexagram effect is
claimed. The proper next research step is a complete independently source-audited
calendar with time-blocked out-of-sample tests and a matched calendar/schedule
baseline; moon-phase and solar-term features must be fixed before a draw,
and same-draw hexagrams remain outcome labels, not candidate predictors.
