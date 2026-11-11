# M2E + M4 — Frozen Source Snapshot Verification and P5A Prospective Evaluation

**Purpose:** continue the previously verified M2/M3 workflow with source-evidence checks and a read-only P5A nine-draw evaluation path. The planned first P5A draw is **#115000105 / 2026-11-13**, so on **2026-11-11**, the proper current state is **0 observed / 9 pending**. No numbers or payouts are invented.

## M2E: archived publisher-page evidence

`lotto7x7.official_draw_evidence_m2` supports two independent sources that were observed publishing the #115000104 result:

- CTBC Bank (https://lotto.ctbcbank.com/result_all.htm): game-specific `大樂透`, period, ROC date, six ascending main numbers and special ball.
- Independent publisher (https://api.lottery.com.tw/bl): the visible `大樂透` results table with date, period, main six and special.

`--capture-latest` is an **explicit network operation** that fetches these two exact pinned URLs over validated HTTPS without redirecting, captures raw bytes (max 2 MB per page) under the ignored `data/private/lotto649-source/`, and records each source SHA-256 and capture time. It creates an index ONLY when both parsers yield exactly the same draw. If either page format changes or the providers disagree, the operation fails closed; partial raw captures may remain on disk, but no draw index is published. Repeated capture may fail with a non-overwrite error; use unique timestamps or preserve the already collected snapshot.

`--verify-draw <id>` runs offline, reads exactly two source records and corresponding SHA-addressed HTML files, validates original bytes, pinned provider URLs, dates and distinct special ball, and rejects mismatching result content. **Local hash integrity and parser consistency do not cryptographically authenticate web-page origin**: evidence still requires trustworthy TLS fetch and custody, and independent manual inspection where needed.

Published sources are allowed as *evidence* for a future draw, not as permission to add historical records. Existing original `data/bin` / `data/csv` files are never modified by these two commands.

## M4: preregistered P5A paired shadow evaluation

`lotto7x7.prospective_evaluation_m4` checks the original P5A seal **Git blob SHA1 546c7352fd0a98a42959b333e51498ef7c56cf9d** and its source anchors before grading, rejects changed ticket numbers, missing earlier draws, duplicated draws or modified dates, and requires each future result to satisfy **both**:

1. Present within the verified CSV / history BIN / image BIN / manifest source archive, after strict audit.
2. Exactly match both captured publisher-page snapshots for the same draw ID.

A case in which source snapshots exist but the active archive remains unmodified is **not accepted** for scoring by the normal CLI. Adding future results to the archive must be a **separate, explicitly reviewed and authorized transaction** with full rollback and source-page custody. This prevents simply declaring an unverified staged draw as real.

With no future rows, the CLI produces a reproducible report with **0/9 observed, 9 pending**. With one to eight verified future draws, the report shows per-arm prize-tier counts, best main hits, and the paired `any_award` difference **descriptively only**. The one-sided exact paired sign/McNemar-style p-value is computed **only after all nine draws**. Even then, this small pilot never issues a predictive GO decision. Realized profit and actual payout remain `null` for both unpaid shadow arms. Monetary reference requires separately authenticated published per-winner amount sources, and is not automatically included here.

## Windows PowerShell

```powershell
cd E:\lotto649
$py = '.\.venv\Scripts\python.exe'

# Run independent offline source snapshots and prospective evaluation tests
& $py -m unittest discover -s tests -p 'test_official_draw_evidence_m2.py' -v
& $py -m unittest discover -s tests -p 'test_prospective_evaluation_m4.py' -v

# Generate a report for the presently empty (0/9) prospective period
& $py -m lotto7x7.prospective_evaluation_m4 `
    --csv-root data/csv --bin-root data/bin `
    --evidence-root data/private/lotto649-source `
    --out results/p5a_m4_pending_20261111.json

# AFTER the official result has been published: explicit opt-in HTTPS snapshot capture
# & $py -m lotto7x7.official_draw_evidence_m2 --capture-latest

# Later, verify immutable saved evidence (does NOT append to the archive)
# & $py -m lotto7x7.official_draw_evidence_m2 --verify-draw 115000105

git status --short
```

All reports use create-only mode under Git-ignored `results/`. The source pages are private audit artifacts (not personal information, but intentionally excluded from the public source repository). No application automatically purchases lottery tickets, mutates the P5A/P3f seals, or changes the original user-paid ticket records.

## Verification gates

- [x] Pinned two-source URL and separately saved raw HTML SHA-256 verification.
- [x] Content-level two-source agreement; game-specific CTBC parser excludes adjacent 威力彩 and 今彩539 results.
- [x] Duplicate, malformed, missing bonus, mismatched source, tampered bytes and source-redirection rejection.
- [x] Strict sealed P5A SHA, nine fixed dates and unchanged strategy arms.
- [x] Paired primary outcome counting and nine-draw-only exact conditional sign test.
- [x] Source archive plus saved-page dual requirement for admitting a future draw.
- [x] Offline regression suites and real-source-gated zero-future-row CI smoke test.
- [ ] After #115000105, independently archive published source pages and **separately authorize** an atomic append. P5A must remain descriptive and no premature prospective predictive GO may be made.
