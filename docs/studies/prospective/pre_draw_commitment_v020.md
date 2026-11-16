# Lotto649 v0.20 — Pre-Draw Class Commitment / Post-Draw Audit

## Purpose

P0–P2 are retrospective, and v0.19 is **future-only**. The v0.20 addition
creates a timestampable **pre-draw forecast of the entire L1/L2/L3 categorical
probability vector** and independently reconciles that pre-draw vector to the
v0.19 event score after a genuine result is imported.

It does **not** select next-draw six-number tickets or claim improved winning
probabilities. Any particular legal six-number ticket still has fair chance
`1/13,983,816` if the drawing is fair.

## Data source: operational constraint

Taiwan Lottery's official source says Lotto649 is usually drawn **Tuesday and
Friday**, with draws from **20:30 to 21:00 Asia/Taipei**; draw numbers are
usually posted on its website by **21:30**. Its official annual download page
is updated **on the fifth of each month through the prior month**, so it is
**not** a dependable same-night feed.

- Official result page: https://www.taiwanlottery.com/lotto/result/lotto649/
- CTBC lottery-management results: https://lotto.ctbcbank.com/result_all.htm
- Official history downloads: https://www.taiwanlottery.com/lotto/history/result_download/
- Official FAQ: https://www.taiwanlottery.com/customer_service/faq/

**No unverified scraper is added.** A human or another separately verified
source must confirm the date, issue ID, six primary numbers and bonus before
editing the annual English-header CSV. The current historical dataset is
partly derived from an external third-party archive; SHA256 integrity is not
proof of its agreement with original official sources.

## Published initial preregistration

- Immutable baseline: `docs/prospective_freeze_20261108.json`.
- Pre-draw commitment: `docs/prospective_seals/115000104_2026-11-10.json`.
- Target: **115000104**, **2026-11-10**, standard Lotto649 Tuesday.
- This file is generated from the user's prior v0.19 `NO_FUTURE_DATA`
  report, and contains **zero** outcomes from that target draw.
- Its original SHA256 seal is accompanied by this Git repository's commit
  history. The local `created_at_utc` field is not a trusted public timestamp;
  the public Git commit's remote chronology requires separate confirmation.
- There are 6 L1, 11 L2 and 94 L3 category probabilities in the file.
- The fixed rare-class alternative is `P6_O1_D5` with fair probability
  `744 / 13,983,816` and prespecified alternative `10 × p_fair`.

## Generate a new prospective seal for the next unobserved draw

First run v0.19 **using all draws that already occurred**, confirming no
unscored intervening draws and saving a fresh JSON report. For the initial
2026-11-10 seal, the expected report is the uploaded zero-event v0.19 audit.

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main

# Only BEFORE 2026-11-10 20:30 Taipei time:
& $py -m lotto7x7 shape-prospective-seal `
    --freeze docs/prospective_freeze_20261108.json `
    --catalog results/v015_v016_p1_shape_catalog.npz `
    --universe results/v015_v016_p1_shape_catalog_universe.json `
    --report results/v019_prospective_first_check.json `
    --issue-id 115000104 `
    --issue-date 2026-11-10 `
    --out results/v020_115000104_pre_draw.json
```

The first seal is **already committed** in `docs/prospective_seals`; do not
regenerate or replace it. For later draws, run this command with the most
recent prior v0.19 report, next legal issue ID and date, **then commit/push
that newly generated JSON before the draw**:

```powershell
git add docs/prospective_seals/NEW_PRE_DRAW.json
git commit -m "Seal Lotto649 next draw class distribution before results"
git push origin main
```

A future sealed JSON stored only on a mutable local computer, without an
independently verifiable timestamp, is **not** decisive evidence of having
been created before results were known.

## Evaluate and audit after the official result is publicly available

1. Verify original result on the official Taiwan Lottery or CTBC site.
2. Append **real** data (not a sample) to `data/csv/lotto649_2026.csv`.
3. Run the frozen v0.19 scoring CLI with a new output file and `--previous`.
4. Run v0.20 audit against the original publicly committed pre-draw seal.

```powershell
& $py -m lotto7x7 shape-prospective-update `
    --freeze docs/prospective_freeze_20261108.json `
    --catalog results/v015_v016_p1_shape_catalog.npz `
    --universe results/v015_v016_p1_shape_catalog_universe.json `
    --csv data/csv `
    --previous results/v019_prospective_first_check.json `
    --out results/v019_after_115000104.json

& $py -m lotto7x7 shape-prospective-audit `
    --seal docs/prospective_seals/115000104_2026-11-10.json `
    --report results/v019_after_115000104.json `
    --out results/v020_115000104_audit.json

Start-Process .\results\v020_115000104_audit.html
```

The audit returns `AWAITING_DRAW` when the next event does not exist, and
`AUDIT_MATCH` only when the event ID/date, the *entire* sealed distribution,
all observed-class `q`, `p`, log gains and the rare-class increment match a
reconstructed pre-draw model. An invalid source, altered event or mismatched
model is rejected. An `AUDIT_MATCH` only verifies arithmetic/provenance
**within the provided reports**: the program deliberately reports
`source_authenticity_verified=false` and
`external_git_timestamp_verified=false`. Those require independent checks.

## Fail-closed behavior

- Rejects non-Tuesday/Friday dates (special announced drawing days require
  an explicitly reviewed exception).
- Rejects a target that is not exactly the next issue after the latest
  previously scored event, a date on/before the v0.19 freeze, a ROC-year
  inconsistency, or local creation at/after 20:30 on the scheduled draw day.
- Does not allow overwriting prior JSON or HTML artifacts.
- Tampering with the full snapshot invalidates its SHA256 seal. Re-sealing
  changed probabilities still fails post-draw recomputation.
- Refuses to accept historical data as an independent prospective signal.
- Does not download authoritative results, buy tickets, or run unattended.

## Validation

- **42/42 PASS** from v0.19 + v0.20 synthetic unit/CLI tests.
- Full real 2,497-draw P1 catalog, exact fair universe and user-supplied
  zero-event v0.19 report successfully generated initial 115000104 seal.
- Expected pre-draw result: no actual target draw, no model evidence beyond
  what was published in the earlier frozen research period.
