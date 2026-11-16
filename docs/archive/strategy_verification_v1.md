# Lotto649 Strategy Verification Framework v1.0 — P2B four-ticket pilot

Tracking: [#32](https://github.com/cctsao1008/lotto649/issues/32) and the [research-evidence roadmap](research_issue_consolidation_20261112.md).

## Why

**A valid strategy implementation is not the same thing as a prediction of the next draw.** This framework makes strategy compliance a deterministic engineering acceptance gate, separate from statistical predictive validity and from the realized winnings of any ticket.

For version 1.0 we intentionally **reuse the existing frozen v0.30 P2B strategy** (four arms: random / coverage / structural repair / hybrid). No selected numbers, objective weights, existing reports, source archives, purchased tickets, M0, P3f or P5A are changed. The verifier accepts an existing P2B report and a source-audited archive, produces a new report, and never calls the P2B selector, scorer, or optimizer.

## Three separate verdicts

1. **Strategy compliance** (`STRATEGY_COMPLIANCE_PASS` or `FAIL`): same versioned strategy, exact configuration SHA, original data and pool fingerprint, legal tickets, correct four-arm deterministic path, independently recomputed counterfactual energies and controls.
2. **Optimization quality** (descriptive): stationary/improvable status under **one-ticket exchange within the fixed finite candidate pool**, not a global optimum or a promise of better draw outcomes. A bounded four-pass search might legally return an improvable result; that does not automatically mean the specified algorithm was incorrectly executed. The quality field reports the improvement separately.
3. **Outcome observation**: the verifier **does not** examine winning lottery outcomes, future results or predictive p-values. Its PASS is neither predictive GO nor proof of improved winning chances.

## Frozen verification boundary

From `lotto7x7.portfolio_benchmark_v030`:

- A fair candidate pool of `pool_size` distinct legal six-of-49 sets, drawn reproducibly under the P2B `SeedSequence(seed).spawn(2 * iid_reps + 1)[0]` stream. SHA256 is over the row-major little-endian int16 pool, in generated order.
- Exactly **four distinct** tickets per arm; the arm named `coverage` must cover **24 distinct numbers**; `hybrid` must cover at least `min_hybrid_unique` (default 22); `repair` has no extra overlap restriction except four distinct tickets.
- Same finite pool and four-ticket budget, random arm from the first random control subset, coverage arm from the seeded greedy disjoint search, repair/hybrid arms from their frozen random starts plus up to four deterministic coordinate-exchange passes.
- The score is the **joint** hypothetical four-append energy, never the sum of four individual ticket scores:

  `D_after(T) = ||R_history + sum_{ticket in T} z(ticket)||^2 / (N+4)`

  `D_before = ||R_history||^2 / N`

  `repair_gain(T) = D_before - D_after(T)`

- `z` uses the original sequential 7x7 three-feature geometry, exact pair-edge inclusion moments and correlation whitening with ridge `0.05`. The verifier **independently** computes every pair feature and covariance from the frozen 49-board mapping; it does **not** call the production P2B feature scorer.
- The verifier replays independent random controls and checks quantiles, sample count, coverage mean and repair-gain median.

**Fail-closed rules:** altered config, seed, geometry SHA, historical source fingerprint, candidate pool SHA, selected lines, coverage, energies, or random-control summary must result in FAIL; missing or tampered input cannot produce PASS. Reports are `results/` only and created exclusively without overwriting old reports.

**Independence caveat:** the verifier shares the source audit gate and the frozen geometric *definition* with the generator, and separately replays the frozen search algorithm. It is an independent implementation/checker, not a fully isolated, separately compiled implementation. Version 1.0 **does not** recalculate bonus-ball prize DP subreports; their exact probability engine has its own regression tests. Do not generalize v1.0's PASS to unimplemented v0.25/v0.28 or external strategies.

## Example — source-audited Windows PowerShell

```powershell
cd E:\lotto649
$py = '.\.venv\Scripts\python.exe'

# Source report may already exist. Use the existing P2B output when possible.
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
& $py -m lotto7x7.portfolio_benchmark_v030 `
    --pool-size 128 --random-portfolios 32 --min-hybrid-unique 22 `
    --iid-reps 3 --seed 649 `
    --out "results/p2b_strategy_source_$stamp.json"
if ($LASTEXITCODE -ne 0) { throw 'P2B source generation failed' }

& $py -m lotto7x7.strategy_verification_v1 `
    --selection-report "results/p2b_strategy_source_$stamp.json" `
    --csv-root data/csv --bin-root data/bin `
    --out "results/strategy_verified_$stamp.json"
if ($LASTEXITCODE -ne 0) { throw 'Independent strategy verification failed' }
```

Or to independently check the original 512-pool full P2B research report, specify its existing `results/v030_p2b_*.json` path directly. **Do not overwrite** original outputs.

## Acceptance and limits

- Unit tests include positive synthetic report, deliberately altered config/SHA, selected lines, objective values, source provenance, random control summary, illegal ticket, and missing fields. For any corruption the expected result is a clear **FAIL**, never a silent PASS.
- CI runs the P2B generator once on source-audited actual data and then the independent read-only verifier; JSON reports are uploaded as short-lived artifacts.
- Final PASS means the original documented finite-pool selection was executed reproducibly, not that the selected numbers are any more likely to win under an independent fair lottery.
- Global optimality across the approximately `C(13,983,816,4)` distinct four-line sets is **not** evaluated or claimed. A one-swap neighborhood finding is local evidence only.
- Future extensions could verify exact prize distributions independently and support additional strategies via separately versioned specifications. Those are outside v1.0.