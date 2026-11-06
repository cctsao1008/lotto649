# Lotto649 Phase 6.1 — Recent Official-Site API Bridge

Status: **implemented and committed; Windows network and end-to-end validation pending**.

## Reason for a separate live-data path

The Phase 6 shadow-sealing unit tests passed **64/64** on Windows. The study's audited monthly archive ends at **2026-10-30**, containing 2,183 draws.

The official Taiwan Lottery yearly-download page says it updates **on the fifth day of each month through the end of the preceding month**. Therefore the regular annual archive is not expected to include **2026-11-03** and **2026-11-06** immediately. Separate contemporaneous draw information is needed for a one-step prospective forecast.

Public draw listings consulted on November 7 describe the two intervening main-six draws (unverified against a directly fetched API response in this development environment):

- 2026-11-03, `115000102`, main 02 23 24 25 29 39, special 31.
- 2026-11-06, `115000103`, main 05 09 26 32 41 48, special 23.

The official site's **undocumented** read-only API is described by community integrations at:
`https://api.taiwanlottery.com/TLCAPIWeB/Lottery/Lotto649Result`

They describe a month query with parameters `period`, `month`, `pageNum`, `pageSize`, and JSON fields `content.lotto649Res[].period`, `lotteryDate`, and `drawNumberSize`. This is an internal site endpoint; availability and schema are **not guaranteed**. It must be validated on the user's own PC before drawing any inference. If unavailable or changed, **do not fabricate records**.

## Design

`lotto_live_data.py` adds two explicit commands:

- `fetch`: GET one month of the internal official-site API and preserve the exact response bytes to a uniquely named JSON file, with the hash printed separately. The downloaded response is never silently replaced.
- `build`: accept a local official structured archive and one or more saved month-response JSON files; strictly validate period, date, six main numbers, the special number, distinctness, and duplication; require no missing Tuesday/Friday dates between the archived last draw and latest new result; append to a **separate shadow-only structured NPY**. Generate a machine-readable evidence manifest.

The original `data/normalized/lotto649_base.npy` is never modified. The separate shadow NPY mirrors the existing NumPy dtype. When the API cannot supply sales, ticket counts or total prizes, those **non-model financial fields** are represented by **zero placeholders for newly appended rows only**; the manifest marks them explicitly as unknown, not genuine zero values. This composite must never be represented as an official audited archive or fed back into the historical normalizer.

The bridge supports **one-step probability predictions** because the Phase 6 model consumes only main-six numbers and chronology. If a gap or exceptional draw schedule is detected, it fails closed pending independent schedule review.

The API request's local timestamp is not trustworthy independent evidence of when a forecast seal was published. The seal still requires a third-party timestamping/publication verification process before any genuinely prospective claim.

## Windows execution

```powershell
cd D:\my-github\lotto649
git pull origin main

# Expected 73 tests: previous 64 plus nine bridge tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Inspect command-line interface
.\.venv\Scripts\python.exe .\lotto_live_data.py --help

# Stage an official-site API snapshot for November 2026
.\.venv\Scripts\python.exe .\lotto_live_data.py fetch `
    --month 2026-11 `
    --out data/shadow/api/lotto649_2026-11_initial.json

# Generate a distinct, shadow-only source with a manifest
.\.venv\Scripts\python.exe .\lotto_live_data.py build `
    --official-archive data/normalized/lotto649_base.npy `
    --api-snapshot data/shadow/api/lotto649_2026-11_initial.json `
    --out data/shadow/sources/lotto649_recent_20261107.npy `
    --manifest data/shadow/sources/lotto649_recent_20261107.manifest.json
```

If either expected API schema or recent-draw sequence is not valid, **stop and inspect the error**. Do not lower validation checks or rewrite the original archive merely to proceed.

Run only after separately confirming official upcoming draw date and a safe pre-draw cutoff:

```powershell
# Parameterized illustration, NOT a verified draw schedule.
$target = "<OFFICIALLY_VERIFIED_FUTURE_DRAW_DATE>"
$deadline = "<OPERATOR_SELECTED_SAFE_PRE_DRAW_TIME_WITH_OFFSET>"

.\.venv\Scripts\python.exe .\lotto_shadow.py prepare `
    --source data/shadow/sources/lotto649_recent_20261107.npy `
    --target-date $target --deadline $deadline
```

**Never use a target date for which an official draw is already known, even if the local archive is stale.** Once a forecast is prepared, verify it, publish it to an independently timestamped service before the draw, and retain the exact bytes and source manifest.

## Bridge verification gates

1. Nine new unit tests pass (API schema, duplicates, id/date continuity, gap rejection, file immutability, mock network fetch, and shadow financial placeholders).
2. The Windows official-site API request succeeds; main-six, special number, date and period agree with an independently checked official result for the last two known draws.
3. Output is separate from the original monthly dataset; all original 2,183 rows match exactly.
4. The composite includes **exactly two** new draws when the source cutoff remains 2026-10-30 and the fetched monthly snapshot ends on 2026-11-06.
5. No seal is created until completeness and external timestamping approach have been checked.

References: Taiwan Lottery annual archive page and FAQ for schedule; internal endpoint is a community-documented, unsupported site API. Source code comments, docstrings, and runtime output are English-only.
