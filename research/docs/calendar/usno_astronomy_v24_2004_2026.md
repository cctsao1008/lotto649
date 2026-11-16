# Lotto649 USNO astronomy historical archive (2004–2026)

## Scope

**USNO only**. This is a single archival import, not a recurring lottery
fetching pipeline. Data are from the US Naval Observatory API
(https://aa.usno.navy.mil/data/api.html). No CWA sources are used to
construct the USNO research features.

## Pre-verified user input

The user supplied an official USNO bundle with **46 annual JSON responses**
(23 years × {moon phases, seasons}), **1,276** normalized time-stamped events,
and an event table in UTC and Asia/Taipei. Validated local reference SHA-256:

- Original user-provided input ZIP:
  `14b52a690cec3c0264e7a542f2b37dfdfd197a158ed2737ad53c4fc7230a094f`
- Normalized event CSV:
  `f084a9248725cecbef257d23963b2660f2152e2bfce2f12054fd89337e33f6df`
- 8,401-row daily date-level feature CSV:
  `cd062ffd9d10fcd12df70835f039a7987df46e946ce041e203775d21edbde348`

The raw JSON responses are permanently committed individually. The
one-time GitHub workflow downloads them and checks the pinned event
CSV SHA-256; because normalized event rows embed each raw source SHA-256,
a byte-level change to a source JSON causes a digest mismatch and prevents
an incorrect archival commit.

The workflow ZIP is transient and its container-level hash can differ
from the user's original ZIP due to ZIP metadata; *raw JSON source bytes*
are verified against the pinned event digest.

## Committed artifacts (after one-time CI job succeeds)

- `research/artifacts/usno_official/raw/moon_phases/YYYY.json` (23 files)
- `research/artifacts/usno_official/raw/seasons/YYYY.json` (23 files)
- `research/artifacts/usno_official/manifest_2004_2026.json`
- `research/artifacts/usno_official/usno_events_2004_2026.csv`
- `research/artifacts/v24_usno/v24_usno_daily_features_2004_2026.csv` (8,401 days)
- `research/artifacts/v24_usno/v24_usno_draw_context_2498.csv` (2,498 archived draws)
- `research/artifacts/v24_usno/v24_usno_data_audit_2004_2026.json`
- `research/replays/fetch_usno_astronomy.py`, `research/replays/build_usno_features.py`

All existing Lotto649 original CSV/BIN data, v2.3 prospective seals,
v2.4 P0 and older astronomy data remain untouched. Date-level astronomical
feature calculations are conventionally sampled at **12:00 Taiwan time**,
not at a verified draw time. Some 2004 prior-event fields are intentionally
blank because the 2003 boundary events were not imported. USNO provides
four equinoxes/solstices, **not all 24 Chinese solar terms**; it does not
supply a traditional lunar calendar or leap-month designation.

No relationship between astronomical variables and lottery predictability
has been established.

## One-time archival mechanism

[The original one-time workflow](../../../.github/legacy-workflows/usno_archive_once_20261113.yml)
records how the source archive was created from the official USNO API.
It fails closed if USNO responds with data that do not match the submitted
source's pinned hash. If network access is blocked for GitHub runners, the
provided local scripts can rebuild identical research features from the
user's original, already verified `usno_official.zip`.

To reparse the committed source responses offline, copy the snapshot to a
local output directory first:

```bash
mkdir -p results/usno_official
cp -R research/artifacts/usno_official/raw results/usno_official/
python research/replays/fetch_usno_astronomy.py --offline
```
