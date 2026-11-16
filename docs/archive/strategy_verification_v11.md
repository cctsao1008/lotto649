# Lotto649 Strategy Verification v1.1 — Prize Independence + Frozen M0 Manifest Audit

## Purpose and acceptance scope

**Acceptance means correctly following a declared, frozen selection strategy—not matching a future draw.** This revision builds on the v1.0 independent P2B candidate/optimizer replay; it deliberately does not rewrite the P2B optimizer, paid tickets, original M0 four-ticket selection, P3f/P5A shadow seals, or any historical source.

Three distinct verification decisions are returned:

1. `STRATEGY_COMPLIANCE_PASS` (from v1.0): legal four-ticket P2B, matching candidate-pool SHA, data provenance, exact frozen rule/seed, 4-pass finite-pool replay, independent geometry/energy, and local one-swap search audit. **Not a global optimum certificate.**
2. `EXACT_PRIZE_COMPLIANCE_PASS` (new): independent mathematical verification of the exact special-ball-aware winning-line histogram and expected ticket tier counts. **Not prediction, payout EV, or validation against observed draws.**
3. `M0_FROZEN_MANIFEST_STRUCTURAL_PASS_OPTIMIZER_UNVERIFIED` (new): original immutable v0.28 M0 manifest identity, original cutoff, legal four-ticket portfolio, complete disjointness, independent 7x7 graph-component Shape partitions, Sum, high-precision integer-product Log bin, and declared positive block-score *signs*. **It does not independently recompute the original full historical block-score values or the 80,000-ticket finite-pool rank optimization.** Its scores remain recorded claims in the SHA-pinned manifest, not independently certified numerical results.

The top-level decision is `STRATEGY_V11_PASS_M0_OPTIMIZER_UNVERIFIED` **only** when all three components pass. One failure causes `STRATEGY_V11_FAIL` and a nonzero CLI exit status. `PASS` does not mean all original M0 selection parameters or statistical results were independently reproduced.

## Two mathematically distinct prize implementations

The production v0.30 `portfolio_prize_v030` groups all four ticket-membership patterns, chooses the special ball, and directly accumulates the 0..4 winning-line histogram using a joint dynamic program.

The independent v1.1 verifier instead:

1. Defines events `W_i` for each of four lines: `main_hits >= 3`, or `main_hits = 2 and special hit`.
2. Counts the **intersection** `cap_{i in S} W_i` separately for each nonempty subset `S` of the four tickets, by conditioning on the special ball and combining hypergeometric multiplicities for **only the membership groups of the subset**.
3. Let `b_k = sum_{|S|=k} count(cap_{i in S} W_i)` (and `b_0 = 49*C(48,6)`). Then exactly `b_k = sum_{j=k}^4 C(j,k) N_j`, where `N_j` counts outcomes with exactly `j` winning lines. Recover the 0..4 distribution with the exact integer binomial inverse:

   `N_j = sum_{k=j}^4 (-1)^(k-j) C(k,j) b_k`.

4. Independently re-derives **per-ticket** eight-tier expected counts via `C(6,h)*C(43,6-h)` six-main outcomes for `h` hits, with `6-h` choices for a special hit and `43-(6-h)` for no special hit. The formula is multiplied by four tickets, independent of ticket overlap. This separately cross-checks `sum(j*N_j) = sum_tier count(tier)`.
5. Verifies exact integer histogram, probabilities, total 601,304,088 (main,special) events, all tier rates, the nominal fixed lower-tier subtotal and fixed payout assumptions. A mutation of any claimed result must be rejected.

The two implementations use the same elementary fair-draw rules and hence are not independent evidence of physical draw fairness. However, they use distinct histogram derivations (direct four-way DP versus inclusion-exclusion/binomial inversion and separate single-ticket counting), testable against exhaustive small-universe enumeration.

## Frozen M0 verification boundaries

Source file `experiments/v028_m0_four_block_candidates_20261110.json` is pinned to its original **Git blob SHA1 `21dff1f975773ca919e6af9e762e93f8ffc3475c`**. On Windows, only byte-equivalent LF/CRLF checkout normalization is accepted.

The historical source cutoff inside the original manifest is **2026-11-06 / #115000103 / 2,497 draws**. This is intentionally distinct from the current 2,498-draw P2B source through **2026-11-10 / #115000104**. **Never recompute the 2026-11-06 M0 historical objective using the later draw and call it the same frozen experiment.**

M0's frozen four lines are checked against the recorded Sum and the 7x7 8-neighbor connected-component partition, and their Log Product bins are computed with independent 80-digit Decimal arithmetic and width 0.5. All four lines must be legal, have 24 different numbers, and be in four different Shape partitions. Original candidate block scores must remain positive in the immutable report; the validity of each actual score is a **separate, still-open** independent-score-recalculation task.

The expected-prize field of the M0 manifest is NOT changed; no existing prospective shadow experiment is reinterpreted as a new purchased ticket.

## CLI / CI

Runs against the current read-only local repo; `results/` writes are exclusive-create and intentionally ignored by Git:

```powershell
cd E:\lotto649
$py = '.\.venv\Scripts\python.exe'
& $py -m unittest discover -s tests -p 'test_strategy_verification_v11.py' -v
if ($LASTEXITCODE -ne 0) { throw 'Strategy v1.1 tests failed' }

# Use the previous frozen P2B report, or run the original v0.30 P2B CLI first.
& $py -m lotto7x7.strategy_verification_v11 `
  --selection-report results/strategy_v11_full_source.json `
  --m0-manifest experiments/v028_m0_four_block_candidates_20261110.json `
  --csv-root data/csv --bin-root data/bin `
  --out results/strategy_v11_independent_$((Get-Date).ToString('yyyyMMdd_HHmmss')).json
if ($LASTEXITCODE -ne 0) { throw 'Strategy v1.1 compliance failed' }
```

The dedicated GitHub workflow generates the P2B source report as part of CI rather than requiring a manually prepared report. It runs both the smoke and full 512-candidate/19-IID configurations and saves all original and independently verified JSON reports as artifacts. The CI also runs v1.0 regression/negative tests and v1.1 exhaustive combinatorial small-universe tests.

## Decision rules / future work

- `PASS` means **adherence to stated strategy contract**, mathematically independently checked within the stated scope. It is not a good/bad future prediction judgment and says nothing about jackpot advantage.
- P2B `ONE_SWAP_STATIONARY` remains finite-pool local search, not global optimality among all four-ticket portfolios.
- M0 still requires independently regenerating the exact frozen Shape/Sum/Log/Joint historical block scores and maximin candidate ranking to receive a full **optimizer-compliance** verdict.
- Keep historical scores and future/outcome observations separate. This change does not create, purchase or modify lottery tickets.
