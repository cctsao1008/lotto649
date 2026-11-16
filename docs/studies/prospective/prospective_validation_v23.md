# Lotto649 v2.3 — Precommitted Full-Universe M0 Prospective Shadow, 200 Draws

## Scientific question and distinction from previous research

The v2.1 study **proved a global mathematical optimum of a retrospective M0 objective** for the frozen 2,497-draw score reference. The v2.2 325-draw historical walk-forward was **exploratory predictive NO_GO** (B won on 47 vs 37 draws, paired p≈0.2954), and was not the exact v2.1 global solver. None of that validates the optimizer as a forecaster.

v2.3 is a **new prospectively sealed and shadow-only** test of the exact four-ticket v2.1 globally score-optimal portfolio against one **history-blind** four-ticket control, with *identical unconditional fair-IID official prize coverage*. It is not an online ticket order, does not change previous freezes, and does not learn/update/retune the tickets once sealed.

### Fixed pre-draw audit identity

- Immutable manifest: `prospective/v23_m0_global_static200_20261113_seal.json`.
- Canonical LF Git blob SHA-1: **`f02a94b37bc2287f89cfbc6dd7f6ab2daa4d16be`**. This value is pinned in the evaluator; changed JSON, changed schedule, altered tickets or unrecognized line endings fail closed.
- Source objective: v2.1 exact 6/49 full-universe enumeration, Git commit `80d403ca70a56e0d336696e50880c2ad2cb4e9ec`; report SHA-256 `9daf72152925a7a873418155b3f194bfbdce6e4cd37174ddc52e5d68ab811b86`.
- Optimized B used only the historical dataset through **2026-11-06, draw #115000103** and was certified at fixed quality tuple `(43549,195617,43549,54184)`. The archive at seal creation contains the later independently anchored **2026-11-10 draw #115000104**, but that outcome was **not** fed into B or used to select the control.
- Independent control A: `Random(20261113).shuffle(range(1,50))`, first four sorted six-ball blocks; deterministic seed set in the immutable manifest. Both arms consist of exactly four *pairwise disjoint within each arm* tickets, 24 distinct main-number labels; they can overlap **between arms**. Both have exactly the same official eight-tier award-event distribution under fair IID.
- First scored future event: **2026-11-17, #115000106**. Draw #115000105 on 2026-11-13 is explicitly **out of the study** and will never be scored, ensuring a clear pre-draw seal buffer. A GitHub commit/publication timestamp must precede the first scored draw before calling the study prospective.
- Horizon: **the first 200 consecutive authenticated official Lotto649 draws starting at #115000106**, including year rollover. This is an event-count stopping rule, not a calendar deadline; missing outcomes remain PENDING and cannot be treated as losses. Neither arm changes over the 200 draws.

## One primary outcome and fully specified monitoring rule

For each future official draw including six main balls and one distinct special ball, define `A_award` and `B_award` as the event that at least one of four tickets wins *any* official tier. Record the paired indicator in exactly one of `A_ONLY`, `B_ONLY`, `BOTH`, `NEITHER`. No outcome-based model selection is permitted.

The only primary directional hypothesis is that the **history-conditioned score-optimal B** has a higher any-prize rate than history-blind A. Both have the same theoretical per-draw probability:

`P(any official award from four pairwise-disjoint six-number tickets) = 0.1227314589619088...`

Therefore under a truly IID fair 6/49 draw, **regardless of between-arm overlap**, `P(B_ONLY | all information through prior draw) = P(A_ONLY | that information)`. Each new *discordance* is conditionally symmetric (B-only vs A-only). The fixed one-sided alternative parameter, chosen before first outcome, is `q=0.65` on B-only given discordance. The exact integer **anytime e-process** is

`E_t = 13^(number of B_ONLY) * 7^(number of A_ONLY) / 10^(number of discordant draws)`.

`E_0=1`; multiply by `1.3` after B-only, `0.7` after A-only, or `1` after BOTH/NEITHER. The sole threshold is **E≥20** (per-process, ideal fair-IID false-review probability ≤5% by Ville's inequality). The implementation compares the **exact integer numerator and denominator** and records every cumulative step. Under no circumstances does crossing E≥20 make an automated lottery-fairness, prediction, purchase or GO claim. It triggers **manual source and methodology review** while the original 200-draw observational schedule continues. No unseen null-distribution tuning, early termination, or alternative q selection is allowed.

After *exactly 200* authenticated prospective events, report the final paired counts, B-minus-A observed award-rate difference, one-sided exact conditional discordance p-value and terminal E. **Do not call a positive result corroborated unless source provenance, multiple previously explored strategies, effect stability and independent replication are reviewed.** The threshold is for this study only; it does not automatically correct familywise multiplicity across P3f/P5A, v2.2 and earlier research.

This one endpoint is deliberately more interpretable than simultaneously testing many derived geometry scores or prize tiers; all tier counts are **secondary descriptive metadata only**. A 200-draw experiment may have low power for small effects; no minimum effect-detection guarantee is claimed.

## Strict source verification and append-only evaluation

To make any **official observation** eligible, require both:

1. The canonical annual CSV, history BIN, image BIN and source manifest pass the existing `draw_ingestion_m2.audit_source`, including the immutable first-2,498-draw hashes; and
2. An independent source-evidence bundle is present for **each measured future draw**, with **two** individually SHA-256-verified and parseable archived publisher pages (CTBC publication plus the separate lottery.com.tw listing). The parsed date, draw ID, main numbers and special ball must agree with each other and the append-only archive. Merely copying a date into CSV is never sufficient to score the trial.

New events must be the contiguous official future draw prefix beginning #115000106, with strictly increasing dates, ROC issue IDs and valid year rollover; no missing or cherry-picked outcomes. The special is not a seventh main ball. Rows with incomplete evidence fail closed. An `--previous` report must be an **exact prefix** of the new report's event-plus-evidence records. Output is new-write-only under `results/`.

The evaluator does **not** fetch results or download source pages, auto-place bets, or use a future unconfirmed result. Source snapshots should first be captured by the already available provenance workflow and reviewed before archiving; locally staged snapshots are not cryptographic proof of the original publisher's identity. The two public studies P3f/P5A continue separately with their existing seals.

### Windows / PowerShell, current pending check

```powershell
cd D:\my-github\lotto649
python -m unittest discover -s tests -p 'test_prospective_validation_v23.py' -v

python -m lotto7x7.prospective_validation_v23 `
  --seal prospective/v23_m0_global_static200_20261113_seal.json `
  --csv-root data/csv --bin-root data/bin `
  --evidence-root data/private/lotto649-source `
  --out results/v23_prospective_check_001.json
```

Before the first prospective draw is **both completed and independently archived**, the only valid result is:

`status=PENDING_NO_PREDICTIVE_GO; observed_draw_count=0; pending_draw_count=200; E=1; terminal_one_sided_discordance_p=null`.

Later, after reviewing and archiving source evidence, use a fresh output name and include `--previous results/v23_prospective_check_001.json`. Never overwrite the manifest, older reports or previous sealed studies.

## Operational interpretation and closure criteria

- Engineering acceptance: exact pinned seal bytes, legally disjoint tickets, exact fair prize-coverage parity, initial 2,498-draw anchor, future chronological continuity, per-event two-page evidence, correct one-sided exact arithmetic, terminal gate and append-only output.
- Statistical acceptance: **PENDING** until 200 newly observed verified future events; E≥20 is **REVIEW_ONLY**. There is no automatic `PREDICTIVE_GO` state anywhere in this module.
- No retrospective reassignment or selective sample filtering; the original B is a **static full-universe score optimizer** rather than a time-varying recalibrated forecast.
- Under the fair IID null, all legal six-number tickets have identical next-draw probability `1/C(49,6)`. Improvement in historical repair score never directly implies improved next-draw or payout probability.

## First immutable publication and observed initial CI status

The original pre-target-draw seal and all selection/monitoring rules were **first published at Git commit [`c273cacedac76675222a7501dc351a292829f44a`](https://github.com/cctsao1008/lotto649/commit/c273cacedac76675222a7501dc351a292829f44a)**, GitHub committer timestamp **2026-11-13 03:02:14 UTC = 2026-11-13 11:02:14 Asia/Taipei**, four full calendar days before the first target **2026-11-17 draw #115000106**. The seal Git blob SHA-1 **`f02a94b37bc2287f89cfbc6dd7f6ab2daa4d16be`** is independently checked by the evaluator, tests and Actions `git rev-parse HEAD:...`. **Do not modify or replace its Git blob.**

[Initial preregistration GitHub Actions run **43356218299**](https://github.com/cctsao1008/lotto649/actions/runs/43356218299/job/131658774019) **PASS**:
- **12/12** adversarial/unit tests and exact prize-profile parity of the sealed history-conditioned portfolio versus fixed-seed history-blind control.
- Entire canonical original archive checked, including the authoritative first 2,498-draw BIN/history manifest.
- **0** source-verified eligible prospective targets, **200** still pending (no outcomes fabricated, scored, or backfilled).
- Anytime e-process **`E=1`**, terminal discordance p-value **`null`**, **`PENDING_NO_PREDICTIVE_GO`**, zero purchases.
- Initial read-only GitHub Actions artifact **`strategy-v23-presealed-pending-200` ID `14005583400`** (30-day retention), file `v23_prospective_ci_pending.json`.

The fixed code and plan were verified **before any eligible future outcome**. Future implementations may repair evaluator bugs under separate tracked commits without retrospectively modifying the freeze, changing the 200-draw population, or reclassifying observed outcomes. An evaluation without two saved publisher snapshots is ineligible for inclusion.
