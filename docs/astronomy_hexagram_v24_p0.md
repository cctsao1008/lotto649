# Lotto649 v2.4 P0 — Historical Astronomy × Six-Line Hexagram Data Foundation

## Scope and data honesty

**Research question:** Can an auditable, date-aligned astronomical and six-line binary representation be prepared for all archived Taiwanese Lotto 6/49 draws, before any signal discovery or investment decision? **Yes, as a data-engineering project.** This version makes **NO** statistical association, prediction or payout claim, and does not modify v2.1–v2.3 prospective seals, previous reports, or any user's purchased bets.

The one-time historical snapshot freezes the first **2,498** archived draws ending **2026-11-10 / #115000104**, regardless of later repository appends. Original CSV/yearly files and BIN images/history/manifest are validated with the existing independent `draw_ingestion_m2.audit_source` plus BIN-regenerated CSV before using any rows. No forecast, recommendation or future-bet file is generated.

The data sources are deliberately separated:

| Data family | Source/algorithm | Availability in P0 | Interpretation |
| --- | --- | --- | --- |
| Historical 6+bonus, date, issue | Existing source-gated `data/csv` + BIN | Complete 2,498-draw audit required | Recorded historical outcome, not a prospective prediction |
| Calendar date / day-of-year / weekday / interval | ISO Gregorian date arithmetic | Complete | Date-known-before-draw |
| Six-line binary yin/yang | **Explicit experiment assumption:** 6 sorted main numbers, bottom to top; odd=yang (1), even=yin (0) | Complete | Post-draw outcome/label; do not use the same-draw code as a predictor |
| 64-state theoretical baseline | Exact integer dynamic programming across all C(49,6) combinations | Complete | Independent of Lotto draw history, not 64 equal probabilities |
| Solar longitude / lunar elongation / illumination / phase age | **PyEphem deterministic astronomical computation at 12:00 Asia/Taipei on the date** | Complete with `ephem>=4.2,<5` | **Calculated ephemeris**, *not CWA observations or assumed draw-time conditions* |
| Lunar calendar month/day/leap | **CWA A-A0087-001** *optional input* supplied as a separate original CSV | Blank until local source provided and validated | Source-format parsed, original raw bytes SHA recorded; cannot claim origin from CSV alone |
| Precise new/full/quarter-moon events | **CWA A-A0087-002** candidate dataset | Source inventoried only; **not imported** in P0 | Avoid confusing event time with daily moon illumination |
| Solar-term transition time | **CWA A-A0087-003** candidate dataset | Source inventoried only; **not imported** in P0 | `solar_term_noon_bin` is a longitude *state at noon*, not official transition timestamp |
| Weather station observations | [CWA historical ground observations](https://climate.cwa.gov.tw/GroundObservation) | Deferred: no station/time/location specified | Would require site and sampling definition, missing-value masks |

Official source inventory:

- [Taiwan government dataset 157677: CWA calendar, lunar phases, solar terms](https://data.gov.tw/dataset/157677). Dataset items: `A-A0087-001` (Gregorian/ROC/lunar date), `A-A0087-002` (moon phases), `A-A0087-003` (solar terms). The government metadata is updated annually. Actual historical coverage and full download availability **must be checked from the retrieved raw files, not assumed**.
- [CWA dataset `A-A0087-001` column contract](https://opendata.cwa.gov.tw/opendatadoc/Astronomy/A-A0087-001.pdf): `GregorianCalendar`, `RepublicOfChinaCalendar`, `AgriculturalCalendar`, `AgriculturalCalendarMonth`, `AgriculturalCalendarDay`, `Week`. Also accepts corresponding CWA Chinese headers and strict leap-month marker `閏`. Day-of-week/ROC-year crosschecks are fail-closed.
- [CWA open data membership/API access](https://opendata.cwa.gov.tw/devManual/description): a valid authorization code may be needed for bulk downloads. No fabricated keys or embedded personal secrets.
- [CWA historical ground station portal](https://climate.cwa.gov.tw/GroundObservation): station/day/time selection needed before ingestion.

### Hexagram coding — an experimental mapping, not traditional divination

For ascending main numbers `a_0 < ... < a_5`, let `b_i = a_i mod 2` and

`hexagram_code = sum_{i=0}^5(b_i * 2^i)`.

`b_0` is bottom yao, `b_5` is top yao, `lower_trigram_code = code & 7`, `upper_trigram_code = code >> 3`. `yang_count` is the popcount. `changed_lines_from_previous_draw` is the popcount of XOR with the *previous observed* code. This last field **depends on current-draw outcome**, hence is diagnostic/target-only, not a future-known covariate. `previous_hexagram_code` is the only corresponding historical state known before the next draw (with the first history row missing).

The theoretical mass of each of the 64 codes is **not** assumed to be 1/64: the ordered sample positions and 25 odd / 24 even labels impose constraints. Exact combinatorial recurrence iterates available labels `n=1..49` and positions `k=1..6`, updating `count[k, code | ((n%2) << (k-1))] += count[k-1, code]` in decreasing `k` to avoid reusing an available label. It yields:

- Exactly **13,983,816** total combinations and **64** nonzero patterns.
- Minimum class **134,596**, maximum **296,010**, a ~2.20× discrepancy. A uniform 1/64 chi-square null would be **incorrect**.
- Unit tests compare the DP against an independently brute-enumerated small `C(9,3)` universe.

No King-Wen hexagram names or auspicious/inauspicious interpretation is assigned in P0, because the adopted parity encoding is a study convention and ordinal numbering systems require a documented mapping.

### Astronomy reference time and provenance

We have the historical **date**, but no verified consistent draw clock time, device location/ambient weather, or historical on-site measurements. To avoid imputing a fake event time, calculate the astronomy at **12:00 Taiwan UTC+8**, explicitly recording `astronomy_reference_time_local`. Columns with provenance `PYEPHEM_CALCULATED_LOCAL_NOON_NOT_CWA_OBSERVED` are a modelled date-level environment only.

`solar_term_noon_bin` is the 15-degree solar ecliptic-longitude bin at local noon; it is *not* an official solar-term change date/time. `moon_illumination_percent`, `moon_sun_elongation_deg` and previous/next new-moon differences are calculated astronomical outputs; they must **not** be labeled meteorological measurements.

No `weather_temperature`, `air_pressure`, `moon_altitude_at_draw_time` or daily official lunar calendar fields are inferred from missing evidence.

## Data artifacts

`results/v24_astronomy_hexagram_2498.csv` contains one sorted-by-date row per historical draw with 40-odd columns; English field names, 6 main balls, actual bonus, immutable source IDs, true/false provenance markers and blank CWA fields if no verified CWA CSV is provided.

`results/v24_astronomy_hexagram_2498_report.json` includes source integrity audit; CSV SHA-256; coverage by year; exact 64-state theoretical/historical histogram; official CWA source footprint (`matched_draws`, per-year counts, raw SHA); and explicit research limitations. GitHub CI archives both outputs as an **ephemeral 30-day run artifact**; inputs/old reports remain unchanged.

One-time command, Windows PowerShell (`E:\lotto649` for the company PC):

```powershell
cd E:\lotto649
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install 'ephem>=4.2,<5'
& $py -m unittest discover -s tests -p 'test_astronomy_research_v24.py' -v
& $py -m lotto7x7.astronomy_research_v24 `
    --csv-root data/csv --bin-root data/bin `
    --out-csv results/v24_astronomy_hexagram_2498.csv `
    --out-json results/v24_astronomy_hexagram_2498_report.json
```

To join an **independently acquired** official CWA A-A0087-001 raw CSV in a *new, write-once* run, add `--cwa-calendar path/to/cwa-calendar-original.csv` and choose **new** output filenames. Keep original CSV and download metadata separate; a well-formatted CSV by itself does not prove publisher custody. Do not delete or overwrite the initial CWA-absent baseline artifact.

## Research follow-up: no signal-mining until data coverage is reviewed

A responsible next-stage analysis must (1) quantify CWA raw year coverage and any lunar leap-day ambiguity, (2) separate astronomic features known before drawing from labels computed from the winning draw, (3) control for the calendar's fixed Tuesday/Friday draw schedule and game regime changes, (4) compare exact conditional-IID null distributions rather than uniform 64-state assumptions, and (5) use held-out future outcomes for any predictive claim. These are planned *research questions*, **not evidence that cosmology or I Ching affects a fair draw**.

## Committed canonical research snapshot

Generated P0 artifacts are committed permanently under `data/research/v24/`, not only stored as temporary 30-day GitHub Actions artifacts. The 2,498-row, 37-column [CSV](../data/research/v24/v24_astronomy_hexagram_2498.csv) covers archived draws through 2026-11-10 / #115000104; its exact SHA-256 is `d4ae60f416d4381317a1dd1fbbf225a0198a79b802cf883db78dce4f6fa6f3cd`. The accompanying [audit report JSON](../data/research/v24/v24_astronomy_hexagram_2498_report.json) records matching `output_csv_sha256` and data provenance.

**Scope:** Astronomical features are **PyEphem-calculated local-noon values**, not official CWA observations. CWA lunar calendar fields are intentionally missing (`matched_draws=0`), pending retrieval and verification of original official CWA files. Preserve this exact snapshot; any later authenticated CWA augmentation should be separately versioned. Historical draw inputs and existing v2.3 seals have not been changed.
