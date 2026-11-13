# Lotto649 v2.4 P3 — Pre-Outcome 100-Draw USNO Hexagram Shadow Freeze (2026-11-14 Taipei)

## Engineering / preregistration result: PASS; prospective inference: PENDING

The P3 future-only shadow protocol was committed **before its first eligible draw**.
There have been **zero** authenticated future outcomes evaluated. No ticket
selection, wager, or claim of predictive improvement was generated.

- [Issue #37](https://github.com/cctsao1008/lotto649/issues/37).
- [Original research design and code commit `22c8d19`](https://github.com/cctsao1008/lotto649/commit/22c8d1999085fbd7c0d93da9de02d279dfba1d81).
- [Pure-standard-library import dependency fix `0e3b49f`](https://github.com/cctsao1008/lotto649/commit/0e3b49f033b778c99fead2b420dec32c1c5d8711): the initial CI import incorrectly reached NumPy through the unrelated v2.3 prize engine; revised module calls the original M2 source/audit functions and existing independent publisher-evidence verifier directly.
- **[Immutable pre-outcome USNO future source and sky seal commit `cbc7580`](https://github.com/cctsao1008/lotto649/commit/cbc7580027618870e7e051fae46282420bc94eca)**.
- GitHub commit server-reported committer time: **2026-11-13 17:47:01 UTC = 2026-11-14 01:47:01 Asia/Taipei**, three calendar days ahead of the first eligible 2026-11-17 event.
- [Final GitHub Actions #43476172128](https://github.com/cctsao1008/lotto649/actions/runs/43476172128): **SUCCESS**, 17/17 exact-math / adversarial tests PASS, source gate PASS, 100-draw pending check PASS.

## Frozen astronomical data and proof of custody

Original 2004–2026 USNO archive, pre-existing P0/P1/P1A/P2/P2A analyses, Lotto649 source CSV/BIN and all prior sealed future tickets were **not modified**.

New, standalone official 2027 source:
- [USNO 2027 moon phase original JSON](../data/astronomy/usno_future_2027/raw/moon_phases/2027.json): SHA-256 `cd7746fc328f88e6279d0869dfb8b16d50bf87a7cd457ac67b0732f50007c560`.
- [USNO 2027 seasons original JSON](../data/astronomy/usno_future_2027/raw/seasons/2027.json): SHA-256 `1a0c001621026c30979abb78c5036f43b9234b1000640db588a12cffbc0f6310`.
- [2027 manifest](../data/astronomy/usno_future_2027/manifest_2027_2027.json), SHA-256 `cf2b3905e3641ae353e162ad345e759351c8a1c8d94c7b5b1e6551ab0b805c5b`.
- 49 moon phase events + 6 seasonal/apside events = **55 official events**, **2/2 original JSON verified**, 0 failed. Normalized 2027 USNO CSV SHA-256 `d4f4b02f1a03160c3bf6cf948f1f67ead4290635d4b2c9511bd43bf4b5d4165d`.

**Pre-outcome future astronomical lookup:**
- [379-date solar/lunar sky table](../prospective/v24_p3_sky_calendar_20261114.csv): dates from **2026-11-17 through 2027-11-30**, SHA-256 **`55bb765698c6a8926626d909290d74013e7038db418a0a1ecad795cfa3179f8c`**.
- [100-draw precommitted seal](../prospective/v24_p3_usno_hexagram_static100_20261114_seal.json): SHA-256 **`ac1536662ae25077542e53e76d3c3a087b32a90f86a5eb738e20cc4ae8245cf8`**, original Git blob SHA-1 **`abf4963adf339d11f5bddd75b91ee5409abf82f1`**.
- The only pre-draw data used to create a sky code are USNO timestamps and the fixed **12:00 Asia/Taipei date-level reference** (not verified actual Lotto draw clock time). No outcome is in the precomputed future table.
- Example for first planned draw **2026-11-17/#115000106**: solar sector 7, lunar time sector 2, code **35** (upper Gen/艮, lower Dui/兌), purely a date-level experimental encoding, not a recommended ticket.

## Fixed one-primary-test definition

- Fair 6/49 sorted-parity hexagram distribution computed by exact DP over `C(49,6)=13,983,816`, **not uniform 1/64**.
- For sky code `s`, observed code `y`, let `d = popcount(s xor y)`.
- Pre-fixed alignment and anti-alignment alternative weights:
  - `w_plus(s,y)=5^(6-d)*4^d`
  - `w_minus(s,y)=5^d*4^(6-d)`
- `Z_±(s)=sum_y p0(y)*w_±(s,y)`, `L_±=w_±/Z_±`.
- **Exactly one primary statistic:** `E_n=(product L_plus + product L_minus)/2`, using exact rational arithmetic.
- For one fair-IID-null e-process, `E >= 20` is an **anytime MANUAL REVIEW ONLY** threshold, not automatic prediction, betting, or payout approval. It does not correct for all other separately conducted studies.
- First eligible `#115000106 / 2026-11-17`, **100 consecutive source-verified official draws**. No missing verified draw may be skipped. If a draw date is outside the precommitted sky lookup, the evaluator stops rather than interpolating new values opportunistically.

## Initial source-verified pending state (actual GitHub CI)

| Field | Value |
| --- | --- |
| Engineering CI | **SUCCESS** |
| Unit/adversarial tests | **17/17 PASS** |
| Observed eligible outcomes | **0** |
| Still pending | **100** |
| Exact initial e-process | **1/1** |
| Status | `PENDING_NO_PREDICTIVE_GO` |
| Actual wagers placed | **False** |
| Predictive superiority established | **False** |

Future observations require the existing repository's **original BIN/CSV/M2 archive consistency** and **two independent archived publisher-page evidence snapshots for each draw**, as well as the frozen 379-day sky table and SHA-256-verified source manifests. The model and seal must not change after future results become known. Append-only reports should use a *new* path under `results/` and verify the entire previous report prefix.

**Scope:** Issue #37 may close for **engineering / pre-draw sealing**, while the actual 100-draw experiment remains statistically **PENDING**, not completed. No automated draw ingestion or background watch was created.
