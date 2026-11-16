# Lotto649 v2.4 P3 — 100-Draw USNO Astronomical/Outcome Hexagram Prospective Shadow

## Scope

**Future-only, non-wager**, independent observational trial. Its target is not
six-number selection, prize-winning probability or divination. Previous
12-message hexagram (P1A) and solar-lunar 64-hexagram (P2A) post-hoc association
screens both found `NO_EVIDENCE`. Their historical records, code, and null results
remain immutable.

### Immutable first eligible outcome and horizon

- First eligible draw **#115000106, 2026-11-17 (Asia/Taipei)**, exactly matching
  the first target of the independently sealed v2.3 M0 static200 shadow.
- First **100 consecutive, independently authenticated official draws**, no
  skipping, reinsertion or retrospective case selection. The initial 2026-11-13
  bridge draw is **excluded**; archived past results through 2026-11-10 are not
  evaluated or used to fit the model. If the schedule changes, the actual
  verified draw date determines its previously committed sky state.
- A daily **2026-11-17 through 2027-11-30** table covers the intended horizon;
  an authentic draw outside this range **fails closed**, rather than silently
  computing a new convenient sky state. Neither alternative's parameters nor
  the astronomical six-bit mapping may be refit after the study begins.
- Historical USNO 2004–2026 raw JSON, derived files and manifest are unchanged.
  USNO **2027** original annual moon and season JSON are imported one time into
  `data/astronomy/usno_future_2027/` with their own manifest and CSV, SHA-256
  evidence, and verified before sky-table creation. No CWA data or forecast
  service is used.

## One fixed primary statistical hypothesis, exact fair 6/49 null

For sky code `s ∈ [0,63]`, outcome code `y ∈ [0,63]`, use the existing fixed
bottom-to-top yin/yang parity of the **six ordered main numbers**. Only the
astronomical code is pre-outcome known; outcome code and Hamming distance are
marked **OUTCOME_ONLY**. The special ball is verified with the real record but
is not part of this primary endpoint.

Let `d(s,y) = popcount(s XOR y)` (0..6). The **exact** probability of an outcome
code is `p0(y)=mass(y) / 13,983,816`, calculated by dynamic programming across
all legal **six sorted numbers**, not assumed to be uniformly 1/64.

Two fixed, non-fitted alternative distributions allow both resemblance and
anti-resemblance and are pre-committed **before the first target draw**:

- `w_plus(s,y) = 5^(6-d) 4^d` (more similarity)
- `w_minus(s,y) = 5^d 4^(6-d)` (more dissimilarity)
- `Z_plus/minus(s) = sum_y p0(y)*w_plus/minus(s,y)`
- `L_plus/minus(s,y) = w_plus/minus(s,y)/Z_plus/minus(s)`

The **single primary** likelihood-ratio e-process is an equal *mixture of
products*, not the product of separately averaged ratios:

`E_n = [product_t L_plus(s_t,y_t) + product_t L_minus(s_t,y_t)] / 2`.

Under fair conditional-IID draws and externally determined `s_t`, the two
alternatives each have conditional null mean LR = 1, so the mixture is a
nonnegative martingale with `E_0=1`. The code uses exact Python `Fraction` for
running E and integer arithmetic for the `E >= 20` comparison. With a single
fixed process, `20` is an anytime manual-review threshold (Ville upper bound
`1/20 = 0.05` under the ideal fair IID null). This **does not control familywise
error across other pre-existing prospective studies**, and does not establish
causality or a profitable forecast. All 100 draws must still be observed.

## Source and time-of-information controls

- Future sky is computed at **12:00 Asia/Taipei**, not a verified actual draw
  time, from the P2 solar8/new-moon8 exact integer temporal bins and 3-bit
  Gray codes. `upper=solar`, `lower=lunar`; this is an experimental mapping,
  **not classical I Ching divination or the Chinese lunar calendar**.
- `prospective/v24_p3_sky_calendar_20261114.csv` contains **only date and sky
  features**. The original seal
  `prospective/v24_p3_usno_hexagram_static100_20261114_seal.json` pins all
  source SHA-256, sky table SHA-256, exact parity masses, horizon, statistic,
  and 5:4 alternative weights. GitHub Commit and server timestamps are the
  pre-outcome attestation, not a locally editable claimed date field.
- To score a future draw, reuse the repository's existing M2 verified BIN/CSV
  source gate and two independently archived publisher-page sources *per draw*.
  There is no automatic result fetching, scheduling, purchasing, selection or
  fallback to unverified online numbers. No special ball leakage.
- Status remains `PENDING_NO_PREDICTIVE_GO` with `observed_count=0`,
  `pending_count=100` and `E=1` until genuine verified future draw evidence is
  independently ingested. Reports are write-once under `results/`, with
  append-only checks against the previous recorded verified report.

## Commands (PowerShell)

One-time GitHub workflow archives 2027 astronomy and freezes the sealed future
sky table; running `freeze` a second time refuses overwrite. Do **not** rerun
it to change a model after observing future draws.

```powershell
cd E:\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main
& $py -m unittest discover -s tests -p 'test_iching_prospective_v24_p3.py' -v

# Read-only pending or source-verified future evaluation, write new report:
& $py -m lotto7x7.iching_prospective_v24_p3 evaluate --out results/v24_p3_001.json
```

For subsequent checks use a distinct filename and `--previous
results/v24_p3_001.json`. A snapshot with 0 observations is *not* a successful
prediction or validation. If the original source manifest/BIN, independent
publisher evidence or USNO source hashes fail, **stop** rather than fabricate
rows.

## Engineering completion versus prospective inference

The issue's engineering can close when code, seal, future sky table, source
archival evidence, tests and initial 0/100 CI have all passed. The **statistical
study remains pending** until the 100th authenticated draw. Only crossing a
threshold may prompt manual audit; no automatic predictive GO is implemented.
