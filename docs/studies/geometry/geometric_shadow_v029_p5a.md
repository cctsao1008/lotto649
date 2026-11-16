# v0.29 P5A — Nine-Draw Prospective Geometry Shadow Pilot

Tracking: https://github.com/cctsao1008/lotto649/issues/31

**No purchase. No M0/P3f mutation. All future numbers and dates must be sealed by Git commit before the first #115000105 draw on 2026-11-13.** Do not use a historical backtest or the source P3B optimization score as a prospective prediction claim.

## Experiment and history cutoff

- Experiment ID: `v029-p5a-geometry-shadow-static9-20261113`.
- New frozen artifact: `prospective/v029_p5a_static9_20261113_seal.json`.
- Historical training cutoff: 2,498 validated Lotto649 draws through #115000104 / 2026-11-10.
- P3B source: GitHub Actions run `42969777398`, artifact ID `13832926377`, SHA256 of the actual original P3B report JSON `9866e1c0294f86f1d078952146a5ad59e9f2053c2dc7be3e269de3e9ea480041`.
- Original P3f source-control Git blob SHA1: `b7dec534a44b1ec08d575a4f2e48e5c3580e76d1`.
- The full source fingerprints, fixed seed, pool size and covariance regularization are in the seal.

## Frozen arms

Both arms have exactly four legal six-number tickets, with 24 distinct numbers in each arm:

| Arm | Ticket A | Ticket B | Ticket C | Ticket D |
| --- | --- | --- | --- | --- |
| Geometry P3B | 07 10 12 19 30 34 | 11 26 27 31 40 46 | 06 20 21 24 33 49 | 05 08 17 22 32 37 |
| History-blind random control (original P3f) | 03 06 19 22 27 42 | 04 18 20 24 28 48 | 10 11 33 36 38 44 | 15 23 29 39 40 49 |

The geometry arm is copied from the **actual P3B CI JSON artifact**, not retroactively regenerated. The random arm is copied without alteration from the previous **public P3f frozen shadow control**. No user-purchased combinations are included.

## Nine future draws

The fixed drawing IDs/dates are inherited exactly from the existing P3f schedule, excluding the already elapsed #115000104:

| ID | Date |
| --- | --- |
| 115000105 | 2026-11-13 |
| 115000106 | 2026-11-17 |
| 115000107 | 2026-11-20 |
| 115000108 | 2026-11-24 |
| 115000109 | 2026-11-27 |
| 115000110 | 2026-12-01 |
| 115000111 | 2026-12-04 |
| 115000112 | 2026-12-08 |
| 115000113 | 2026-12-11 |

## Frozen outcomes, comparison and limitation

Primary outcome: for each draw, an indicator that *at least one* of the four tickets in an arm wins any of Taiwan Lotto649's officially defined award tiers (six main numbers and distinct special number).

Primary paired descriptive comparison: the geometry arm's number of any-award draws across all nine minus the random control's number of any-award draws. The directional hypothesis 'geometry greater than random' is recorded before observing #115000105 and must not be changed.

With four disjoint tickets per arm, both have identical fair marginal any-prize probability 0.122731458961909 per draw. Over nine draws, each has only about **1.1046** expected any-prize draws. Because both sets are judged on the *same* official draw, their prize events may be dependent. Nine observations yield very low statistical power; a difference does NOT justify a predictive GO decision, even if one arm appears ahead.

Secondary descriptive metrics: maximum number of main-ball hits among four tickets per draw, official prize tiers and observed payouts **only if verified payout amounts are available**. Do not invent monetary returns, and do not use hypothetical geometric repair gain as a lottery payoff.

Before the ninth result is verified, report only PENDING/descriptive outcomes. After the ninth result, official draw ingestion (M2) and prize-tier checks (M3) must be complete before evaluation. Outcomes must live in new append-only evaluation records, **never in the frozen manifest**. Future protocol changes require a new experiment ID and Git commit.

## Windows PowerShell

    cd E:\lotto649
    git -c http.version=HTTP/1.1 fetch origin main
    if ($LASTEXITCODE -ne 0) { throw 'Fetch failed' }
    git merge --ff-only FETCH_HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Fast-forward failed' }
    $py = '.\.venv\Scripts\python.exe'

    # Verify all original and new public research blob anchors.
    & $py -m lotto7x7.experiment_ledger verify

    # Validate future-only dates, both four-disjoint portfolios, original P3B/P3f evidence.
    & $py -m lotto7x7.geometry_shadow_v029

    # Run independent local shadow validation tests.
    & $py -m unittest discover -s tests -p 'test_geometry_shadow_v029_p5a.py' -v

## Acceptance

- [x] Static nine-draw future-only window and historical cutoff recorded.
- [x] Fixed source P3B candidate and original prior public history-blind control with immutable provenance.
- [x] Source report SHA256 and previous P3f Git blob SHA1 bound.
- [x] Shadow-only / no purchase; no real paid ticket records in public manifest.
- [x] M1 public registry extension with Git blob SHA1, plus future-draw and tamper negative tests.
- [ ] Post-draw M2 verified official results; M3 official prizes and append-only evaluation after all nine draws.

**Status: SHADOW_PRE_REGISTERED_NO_PURCHASE / NO FUTURE RESULTS EVALUATED.**
