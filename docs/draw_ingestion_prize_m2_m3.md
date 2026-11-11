# M2 / M3 — Verified Draw Source Gate and Official Eight-Tier Grading

Tracking: [Issue #31](https://github.com/cctsao1008/lotto649/issues/31). This milestone verifies correctness and supplies read-only validation interfaces. It does **not** silently ingest future draws, rewrite historical BIN/CSV, grade the not-yet-published P5A future results, alter sealed tickets or place wagers.

## M2: Draw ingestion correctness and provenance levels

The project had 2,498 archived draws through 2026-11-10 / #115000104, comprising 23 annual flat CSVs, history BIN `L649HIS1`, 7x7 image BIN `L649IMG2`, and a source manifest. `lotto7x7.draw_ingestion_m2.audit_source()` independently replays the established BIN → annual CSV generator and insists the existing annual files match byte-for-byte (allowing Git's full LF→CRLF conversion only). It checks manifest/length/SHA, consecutive within-year issue IDs, increasing dates, six unique sorted main balls, and a distinct special ball.

**Provenance distinction:** The source manifest for the first 2,497 draws originates from a Biga third-party historical archive. The appended #115000104 originally cited independent CNA/H8 reports; the new M2 check additionally matches a separately published CTBC Bank draw result: #115000104, Tuesday 2026-11-10, main `11 23 30 32 40 41`, special `28`. This was reviewed on 2026-11-11. It is a *frozen manually checked independent anchor*, **not** a fresh live query by the CLI. It does **not** independently certify all 2,498 historical draws.

Primary operator's annual-download interface: https://www.taiwanlottery.com/lotto/history/result_download/ — published updates are monthly on the 5th for the previous month. Independently published CTBC result page: https://lotto.ctbcbank.com/result_all.htm. The operator's November data may not be available from its annual archive on 2026-11-11, so M2 deliberately does not claim an official cross-check for every draw.

The previously frozen 2,498-draw snapshot and P3f/P5A experiment cutoffs remain protected. If later verified draws are appended, M2 checks that the first 2,498 binary records are identical to the original pinned SHA256 after restoring the record-count header, and still verifies the single #115000104 independent published anchor.

### Untrusted staged future draw bundle

`validate_staged_batch` enforces a strict JSON contract, future-only chronological dates, consecutive issue numbers, six distinct main balls, one distinct special, base source manifest SHA256, two separately named publisher hosts and a self-consistent source declaration for each draw, including page SHA256 metadata. At least one claimed publisher kind must be `lottery_operator` or `licensed_bank`. The caller must externally verify the published page bytes and the claimed identity: **a syntactically valid URL, submitted SHA or publisher label is not itself authentication**.

The staging result explicitly says `PASS_STAGING_SCHEMA_ONLY_SOURCE_PAGES_NOT_FETCHED_OR_AUTHENTICATED`, `ready_to_mutate_archive: false`. M2 **does not contain an append/write command**. A later separately authorized append implementation requires a source review, a write-once original-page evidence archive and an atomic transaction with post-commit CSV/BIN/manifest verification. Fail-closed staging is a correctness gate, not automatic intake.

## M3: eight official prize tiers

Reuses the established and independently brute-force-tested `lotto7x7.official_prize_v026.prize_tier` predicate, and verifies it against independently constructed six-number tickets. The official game defines six main balls and one special ball not among the six; the special ball contributes to **second (5+special), fourth (4+special), sixth (3+special), and seventh (2+special)** only. The other prizes are first (6), third (5), fifth (4), and regular (3). One ticket can receive at most one of the eight mutually exclusive classes.

Published 2026-11-10 per-winner result reference (CTBC page, manually checked):

| Official tier | Match | Published per-winner TWD |
|---|---|---:|
| First | 6 | 0 (no winners announced) |
| Second | 5 + special | 0 (no winners announced) |
| Third | 5 | 56,494 |
| Fourth | 4 + special | 16,880 |
| Fifth | 4 | 2,000 |
| Sixth | 3 + special | 1,000 |
| Seventh | 2 + special | 400 |
| Regular | 3 | 400 |

**Critical monetary accounting rule:** Published amounts are an observed draw's *reference values*, not a hypothetical portfolio's claimable cash settlement. If a hypothetical ticket would have won a first/second prize in a draw with no actual winner, the published zero is not the hypothetical jackpot. First–fourth payouts depend on the prize pool and winning counts; even normally fixed fifth–regular prizes have exceptional adjustment provisions. Thus `published_per_winner_reference_twd` is `None` without independently reviewed source evidence, and `actual_settled_payout_twd` / `actual_realized_net_profit_twd` always remain `None` for research shadow bets.

Validated probabilities are exact for all `C(49,6) × 43 = 601,304,088` main-plus-special outcomes. Both four-disjoint P5A shadow portfolios have identical fair any-prize probability `0.122731458961909` per draw. The fact that their selected numbers differ does not change their marginal probability.

### PowerShell local verification

```powershell
cd E:\lotto649
$py = '.\.venv\Scripts\python.exe'

# 1. No-network regression tests
& $py -m unittest discover -s tests -p 'test_draw_ingestion_m2.py' -v
& $py -m unittest discover -s tests -p 'test_prize_engine_m3.py' -v

# 2. Read-only full historical archive and published-result anchor
& $py -m lotto7x7.draw_ingestion_m2 `
    --csv-root data/csv --bin-root data/bin `
    --out results/m2_source_audit_20261111.json

# 3. Eight-tier prize rules and published payout-reference fixture
& $py -m lotto7x7.prize_engine_m3 `
    --csv-root data/csv --bin-root data/bin `
    --out results/m3_prize_audit_20261111.json

# 4. Ensure no tracked source/seal files were changed
git status --short
```

Exclusive-create output under ignored `results/`; rerunning the same file path fails rather than overwriting evidence. The two normal commands do **not** make network calls or purchase tickets.

## Verification acceptance and limitations

- [x] Independently reconstruct all 23 annual CSVs from both authenticated-by-hash BIN files and compare against disk.
- [x] Source manifest SHA256, frozen original 2,498 prefix, chronological order and legal main/special rules verified.
- [x] One new documented, independently published 2026-11-10 result anchor verified; do not conflate that with the whole 2004–2026 archive being operator-certified.
- [x] Future candidate staging rejects invalid schema, conflicting publisher observations, stale manifests, gaps, duplicate IDs, illegal numbers and publisher host reuse; source itself is not authenticated by staging.
- [x] Eight exact official award predicates independently exercised with constructed tickets; non-awards and special-number edge cases covered.
- [x] Exact one-/four-ticket probability cross-check, P5A both arms equal fair odds.
- [x] Published payout table controls, lack-of-winners edge case and no fictional realized gains.
- [x] Automated GitHub CI source audits, unit tests and existing v0.26 / P3f fixture regressions.
- [ ] Future independently retrieved operator draw data may be staged and appended only in an additional authorized source-evidence workflow; no future draw can yet be graded as officially verified by these two modules alone.

**Decision:** M2 archive-integrity and single-source-anchor correctness PASS; M3 prize predicates and monetary evidence-separation PASS. Source authenticity of every historical row and any not-yet-published future draw are **not** claimed.
