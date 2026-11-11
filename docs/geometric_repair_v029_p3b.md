# Lotto649 v0.29 P3B — Counterfactual Structural Image Distribution Repair

Tracking: https://github.com/cctsao1008/lotto649/issues/31

**Research scope:** mathematically optimize the hypothetical effect of inserting one or four six-number tickets into the historical *geometric feature distribution*. This is NOT a claim about predictive probability, a prospective strategy, or improved lottery outcomes.

## Exact reference and objective

The P2B fixed three pairwise descriptors are used: sum of 15 pairwise squared Euclidean distances, number of pairs with squared distance 3..18, and number with squared distance >=25. The five-dimensional P2B design includes two graph-edge controls that are already encoded in L3. **Only the three nonidentical pair descriptors enter P3B**.

The exact fair mean and complete 3x3 covariance are derived analytically from 1,176 cell pairs. For each pair of pair-weight functions, two edges can be identical (2 included cells), share one vertex (3 included cells), or be disjoint (4 included cells). Use without-replacement inclusion probabilities p_k=(6)_k/(49)_k. All covariance diagonals are cross-checked against the P2B exact variance formulas.

The covariance is standardized to a correlation matrix R, then regularized as R_reg = R + 0.05 I, with fixed predeclared ridge. For each draw let z(C) = (phi(C)-mu) / sd times invsqrt(R_reg). Define:

    D_N = || sum_{i=1..N} z(C_i) ||^2 / N

and for a single hypothetical appended ticket:

    Gain_1(C) = D_N(history) - D_(N+1)(history + C)

and for exactly four pairwise-disjoint hypothetical tickets:

    Gain_4(C1,C2,C3,C4) = D_N(history) - D_(N+4)(history + C1 + C2 + C3 + C4).

The four-ticket gain must be evaluated JOINTLY; summing four single-ticket gains is not valid. The normalization accounts for sample-size changes and is used identically for controls. A gain can be negative. Even fair IID histories can be optimized to show positive counterfactual gains.

## Reproducible finite-pool comparison

Default candidate pool: 1,024 distinct legal tickets sampled uniformly from C(49,6), seed 649. All candidate tickets are six distinct integers 1..49. One ticket is selected by minimum updated distance; the four-ticket portfolio is subject to strictly disjoint ticket-number sets (24 distinct numbers).

Random baseline: 256 *randomized greedy* feasible four-disjoint portfolios from the identical pool, using an independently shuffled order for each. This is NOT uniform over all feasible portfolios. Heuristic optimization begins at the best randomly sampled baseline and runs up to four deterministic coordinate-exchange passes with disjointness checks. The selected result is a **finite-pool heuristic, not a certified global optimum**.

Report both raw four-ticket repair gain and repair excess over the median randomized baseline. Also report the pool SHA256, fixed seed, constraints, objective diagnostics, and research-only candidate ticket numbers under ignored results/. These are NOT purchase recommendations.

## Matched whole-pipeline fair IID Digital Twin

Default 99 synthetic histories, each containing the SAME number of independent fair 6/49 draws as the audited real archive. For each null history: generate a new candidate pool, 256 random feasible portfolios, and rerun the identical optimizer. Calibrate BOTH the optimized gain and its excess over median random gain with empirical upper-tail p=(1+number of null values >= observed)/(99+1). Apply two-endpoint Bonferroni correction.

This calibrates the *mathematical effect of an optimization procedure* on random data; it cannot establish next-draw predictability. P2A and P2B did not reject fair IID, and P3A only established algebraic nonidentity within existing v0.28 feature categories.

The P3A same-L3/Sum/LogBin certificate is carried into the report with separate single-append gains. It demonstrates different geometric objectives despite matching existing category keys; it is descriptive and does not justify a fifth predictive-weighted objective.

## Windows PowerShell

    cd E:\lotto649
    git -c http.version=HTTP/1.1 fetch origin main
    if ($LASTEXITCODE -ne 0) { throw 'GitHub fetch failed' }
    git merge --ff-only FETCH_HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Fast-forward failed' }
    $py = '.\.venv\Scripts\python.exe'
    & $py -m unittest discover -s tests -p 'test_geometry_repair_v029_p3b.py' -v
    & $py -m lotto7x7.geometry_repair_v029 --csv-root data/csv --bin-root data/bin --pool-size 1024 --random-portfolios 256 --iid-reps 99 --seed 649 --out results/v029_p3b_repair_20261111.json

Outputs are created only under ignored results/, using exclusive file creation. Existing reports are not overwritten. Audit historical CSV/BIN/date/manifest before selection; no actual tickets, seals or source data are mutated.

## Acceptance and boundaries

- [x] Exact fair 3x3 covariance, PSD and whitened-feature checks.
- [x] One-ticket and joint four-ticket counterfactual discrepancy gain.
- [x] Strict 24-distinct-number portfolio constraints.
- [x] Matched randomized greedy portfolio baselines from the same candidate pool.
- [x] Same entire optimization rerun under independent fair IID histories, with empirical p and two-endpoint correction.
- [x] Stable original v0.28 category certificate and explicit nonforecast labels.
- [x] Ten no-network unit tests and a real-source-gated CI integration report.
- [ ] Do not integrate image repair as a fifth M0 objective without separate evaluation of dependence, trading outcomes and independently frozen prospective evidence.

Decision: DESCRIPTIVE_COUNTERFACTUAL_ONLY_NO_PREDICTIVE_GO.
