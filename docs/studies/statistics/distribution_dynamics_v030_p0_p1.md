# v0.30 P0–P1: Distribution Dynamics, Martingale Compensation and Exact-Sum Conditional Monte Carlo

Tracking [Issue #32](https://github.com/cctsao1008/lotto649/issues/32). **Independent research track**; purchase bookkeeping and P5A shadow are unrelated, never changed. All retrospective results are exploratory.

## P0 — Analytical fair IID dynamics

Use the *previously fixed* v0.29 P3B three pair-distance features. Exact fair finite-population mean and full 3×3 covariance are obtained by pair-edge overlap identities. Standardize by exact SD and transform by the existing fixed ridge=0.05 symmetric inverse-square-root whitening W; the actual whitened covariance is C_z = W^T Corr W, **not an assumed identity matrix**.

Let z_i be the centered whitened feature from draw i, R_n = sum(z_i), tau = trace(C_z), and F_n the observed history through draw n. Under independent fair Lotto649 outcomes:

1. E[z_(n+1)|F_n] = 0.
2. E[-R_n dot z_(n+1)|F_n] = 0. This is the signed mathematical 'compensation' definition: no forced backward correction even if R_n looks large.
3. E[||R_(n+1)||² - ||R_n||² |F_n] = tau.
4. M_n = ||R_n||² - n*tau is a martingale; **the raw squared residual increases on average**, not necessarily repairs.
5. D_n=||R_n||²/n, so E[D_n]=tau and E[D_(n+1)-D_n|F_n]=(tau-D_n)/(n+1). The *conditional expectation* of normalized D moves toward tau (not zero). This denominator-induced relationship is a statistical identity, not evidence that physical balls compensate historical anomalies.
6. The squared norm of the **sample mean** ||R_n/n||² has expectation tau/n, hence its vanishing scale is a mathematical averaging effect.

Define past-conditioned standardized alignment for draws t>=2, a_t = -R_(t-1) dot z_t / sqrt(R_(t-1)^T C_z R_(t-1)). When R_prev is exactly zero, the undefined direction is omitted; no zero is counted as an observation. Under fair IID, the score has conditional mean zero and conditional variance one when defined. It need not be Gaussian or independent of past.

The **primary endpoint** is the maximum absolute cumulative alignment sum divided by sqrt(number of valid steps), across the predeclared prefix grid 100,250,500,1000,1500,2000,N. The **secondary** is maximum absolute (D_prefix - tau). Both are recalibrated against identical whole-history searches on 199 independent fair IID sequences, with empirical p=(1+null_max>=observed)/(R+1) and a predeclared two-endpoint Bonferroni adjustment. Nothing is declared predictive even if a small p occurs: features and modeling choices came from previously inspected historical research.

## P1 — Truly uniform Sum-conditional sampling, then strict v0.28 matching

Target the **already-known P3A certificate class**: exact Sum=132, frozen L3 edge-code from the original v0.15 classifier, original v0.28 integer Log Product bin=20. Note that v0.28 Joint(Sum,LogBin) is determined by these two coordinates, while nested L1/L2 are already determined by L3. This is conditioning on a ticket's *own* class, not forecasting its unknown future class.

Dynamic programming counts exact completions:

    F(lo,k,s) = F(lo+1,k,s) + F(lo+1,k-1,s-lo)

with F(lo,0,0)=1 and minimum/maximum feasibility pruning. **F(1,6,132)=144,587** legal tickets. Draw the next increasing number j with exact integer weight F(j+1,k-1,s-j), then recurse. Every sum-132 ticket receives the same probability **1/144,587**. Generate a fresh IID sample of 40,000 exact-sum tickets. Retain precisely those satisfying the original L3 edge class and exact integer LogBin 20; because the proposals are uniform on the Sum=132 population, rejection yields IID uniform samples on the fully conditioned stratum (duplicates can appear and must not be silently discarded).

Report the number of accepted proposals, unique accepted tickets, acceptance fraction, fair conditional mean/variance/quantiles for the **same three non-L3 pair-distance features**, and descriptive within-stratum ranks for the two preselected P3A ticket examples (03 10 19 23 31 46 versus 04 08 12 27 34 47). **Those ticket ranks are post-selected and NOT p-values**, because the stratum and examples were discovered during earlier exploration. A target with fewer than 25 accepted proposals fails closed.

Small-universe exhaustive combinatorial tests independently verify integer completion counts and near-uniform sampling; 6/49 validity and fixed-seed reproducibility are tested. This does not claim that a next draw will have a Sum of 132 or belong to this L3 class.

## PowerShell

```powershell
cd E:\lotto649
git -c http.version=HTTP/1.1 fetch origin main
if ($LASTEXITCODE -ne 0) { throw 'Fetch failed' }
git merge --ff-only FETCH_HEAD
if ($LASTEXITCODE -ne 0) { throw 'Fast-forward failed' }
$py = '.\.venv\Scripts\python.exe'

& $py -m unittest discover -s tests -p 'test_distribution_dynamics_v030_p0.py' -v
& $py -m unittest discover -s tests -p 'test_conditional_monte_carlo_v030_p1.py' -v

& $py -m lotto7x7.distribution_dynamics_v030 `
    --csv-root data/csv --bin-root data/bin `
    --iid-reps 199 --seed 649 `
    --out results/v030_p0_dynamics_20261111.json

& $py -m lotto7x7.conditional_monte_carlo_v030 `
    --csv-root data/csv --bin-root data/bin `
    --proposals 40000 --seed 649 `
    --out results/v030_p1_conditional_20261111.json

git status --short
```

CLI reports are exclusive-create under ignored `results/`, and the real history is fully revalidated read-only from the established CSV/BIN/manifest source gate. No P5A or purchased ticket is read or modified.

## Research citations and relation to established methods

- The conditional expectation given observed past is the core martingale concept. Under independence, centered innovations are a martingale difference sequence; see [Martingales (probability.app)](https://probability.app/laws/martingales) for a recent educational overview.
- Monte Carlo conditional/randomization tests must simulate from the **correct** finite conditional reference distribution rather than resample arbitrarily. See [Hu, Robinson & Wang (2012)](https://www.numdam.org/item/PS_2012__16__425_0/) on finite-population sampling and permutation tail behavior, and [Conditional Permutation Test, JRSS-B](https://academic.oup.com/jrsssb/article/82/1/175/7056014) for the importance of valid conditional sampling.
- These literature references motivate the design. The exact Lotto649 finite-49 formulas and proofs above derive from the project's stated combinatorial sampling model; they are not attributed to those papers.

## Acceptance and next research

- [x] P0 exact algebraic baseline separated from observed historical fluctuations.
- [x] Fair IID whole-pipeline maximum recalibration, multiple-endpoint correction and source gate.
- [x] P1 integer completion counts with exact small-universe enumeration and strict L3/Sum/LogBin filtering.
- [x] Distinguish conditional structural nonredundancy from an unconditionally predictable future advantage.
- [x] Offline regression tests, GitHub Actions and raw JSON research artifact outputs.
- [ ] P2: further **independent** conditional prediction/ablation tests with prespecified targets and wider research-selection adjustments; do not derive a ticket-ranking 'correction force' from P0 or P1 alone.

**Decision labels:** `MARTINGALE_NULL_BASELINE_NO_PREDICTIVE_GO` and `UNBIASED_CONDITIONAL_STRUCTURE_DESCRIPTIVE_NOT_PREDICTIVE`.
