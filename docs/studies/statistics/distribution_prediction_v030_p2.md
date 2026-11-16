# v0.30 P2A — Chronological Distribution-Compensation Prediction Audit

Tracking [Issue #32](https://github.com/cctsao1008/lotto649/issues/32).

## Frozen question

Does the **past-only**, P0-based distribution-repair score rank the next observed draw above a fair independent six-of-49 reference, rather than merely repairing the historical distribution if hypothetically appended?

This is a **retrospective exploratory test**. The 2,498 draws through 2026-11-10 have already been inspected by earlier research. Neither a low retrospective p-value nor an attractive ticket can establish independent prospective predictive skill.

## Definition and chronology

Reuse the v0.29 P3B three frozen pair-distance descriptors, exact fair 3-vector mean, SDs and ridge=0.05 whitening matrix `W` (without refitting). For legal draw `C`, define `u(C)=(phi(C)-mu)/sd`, `z(C)=u(C) @ W`. At chronological origin `t` with **exactly** `t` known draws, compute `U_t=sum_{i<t} u(Y_i)` and `R_t=U_t @ W`. Score any candidate `C` using:

`score_t(C) = - R_t dot z(C)`.

For each target `Y_t`, draw `K` independent **unconditional** fair 6/49 reference tickets `C_(t,j)` (no Sum/L3/LogBin filter). Evaluate the higher-is-better midrank percentile

`rank_t = (# {score(C_j) < score(Y_t)} + 0.5 * # {score(C_j) == score(Y_t)}) / K`.

Under fair IID, conditional on the past, `E[rank_t] = 1/2`, even when the score distribution has ties. `mean(rank_t)-1/2` is an **unscaled ranking effect**, not a probability, profit or exact six-match claim. The predeclared **primary endpoint** is its positive one-sided effect for the sequential mapping and combined 3-feature score.

The additional fixed comparison arms are three individual pair features (using unwhitened standardized residuals), the exact reverse-score sign, and frozen `snake`, `random:649`, `random:650` mapping sensitivities. They are correlated and **not** separate independent opportunities for significance. Null histories reproduce the same chronological origins, reference count, rank rules, features and arm search. A maximum absolute observed-null-SD-standardized deviation across the 8 arms is calibrated against the maximum of each IID path, to yield a **familywise** exploratory p-value. IID path SD estimation and small synthetic replicate counts are acknowledged limitations.

The RNG uses NumPy `SeedSequence(seed).spawn(2*R+1)`; one seed stream for observed references, paired streams for each IID history and its independent references. Every simulated drawing and reference ticket is an independent uniform six-subset of 1..49 without replacement **within** the ticket. The same ticket may recur across draws or references.

## Preset execution tiers

| Tier | `start-draw` | `step` | `references` | `iid-reps` | Purpose |
|---|---:|---:|---:|---:|---|
| Smoke CI | 500 | 10 | 16 | 3 | Integration only; no inference |
| Pilot default | 500 | 5 | 64 | 39 | Check pipeline and exploratory direction; p-grid 0.025 |
| Full independent-IID sensitivity | 500 | 1 | 256 | 199 | More stable whole-pipeline null; p-grid 0.005 |

Settings must be frozen *before any truly new draw is ingested* to qualify for a prospective test. The full sensitivity tier is optional and heavier; it cannot turn known historical results into a blind test.

## Windows PowerShell

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'

# Verify frozen P2A unit tests and existing v0.30 interfaces
& $py -m unittest discover -s tests -p 'test_distribution_prediction_v030_p2.py' -v
if ($LASTEXITCODE -ne 0) { throw 'P2A regression failed' }

# Run the predefined pilot without overwriting prior reports
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
& $py -m lotto7x7.distribution_prediction_v030 `
    --csv-root data/csv --bin-root data/bin `
    --start-draw 500 --step 5 --references 64 `
    --iid-reps 39 --seed 649 `
    --out "results/v030_p2a_pilot_$stamp.json"
if ($LASTEXITCODE -ne 0) { throw 'P2A pilot failed' }
git status --short
```

The CLI revalidates the manifest, 23 yearly CSV files and both history/image BINs; it fails closed before report creation on invalid provenance. Output must be within gitignored `results/` and is exclusive-create. JSON contains source fingerprints, configuration hash, per-origin actual rank and scores, full IID arm distributions, empirical primary and familywise p-values, and limitations. `configuration_sha256` is a canonical SHA256 of the algorithm parameters, **not** an independently authenticated timestamp or Git commit digest.

## Negative controls and gates

- Invalid/repeated numbers, invalid configurations, changed frozen geometry catalog, altered source files, and reused report paths must fail closed.
- Chronology is causal: no candidate reference or score at origin `t` can use actual `Y_t` to form the past residual. Reversed history is not an equivalent dataset.
- The P1 certificate `Sum=132, L3=76087329, LogBin=20` belongs only to the conditional structural study. Filtering the *unknown* next draw into that stratum would leak outcome information and is **prohibited** for P2A.
- The synthetic IID whole-pipeline upper-tail primary p-value and correlated-arm familywise p-value are exploratory. No post-hoc chosen arm is a new primary endpoint.
- For future independent confirmation, freeze code, theory, mapping, start/step, reference protocol and seed; register with a verifiable time and a source cutoff **before** the subsequent draw. Evaluate separately with a predefined sample size, alpha threshold, multiplicity policy and minimum effect. Until that happens, the decision is `EXPLORATORY_ONLY_NO_PROSPECTIVE_PREDICTIVE_GO`.
- Purchased ten-period Lotto649 tickets, P3f and P5A seals, old v0.28 M0, CSV/BIN/manifest sources, and all existing experimental JSONs remain unchanged.

## Next step

P2B four-ticket portfolio benchmarking is separate: ticket overlap/coverage and fair prize distributions cannot be inferred from a single-draw score percentile. Do not begin a monetary strategy integration without independently verified prospective evidence.
