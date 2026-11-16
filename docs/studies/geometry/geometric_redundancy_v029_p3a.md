# Lotto649 v0.29 P3A — Strict Conditional Structural Redundancy Audit

Tracking: https://github.com/cctsao1008/lotto649/issues/31

**Scope:** test algebraic nonredundancy to the v0.28 M0 categorical baseline, not forecast skill.
All analysis reads existing source data and reports to ignored results/. No paid tickets, shadow seal, history, or M0 manifests are rewritten.

## V0.28 M0 category baseline

The baseline class key is (L3 edge code, exact Sum, Log Product integer bin). The existing nested L1/L2 classes are determined by L3, while Joint(Sum,LogBin) is determined by Sum and LogBin. Exact product cutoffs are imported from original frozen width-0.5 grid via lotto7x7.log_product_v027._grid.

## Exact equal-baseline legal ticket certificate

| Numbers | Exact Sum | LogBin | Sum of 15 squared pair distances |
| --- | ---: | ---: | ---: |
| 03 10 19 23 31 46 | 132 | 20 | 172 |
| 04 08 12 27 34 47 | 132 | 20 | 258 |

Both tickets have the same L3 class, hence the same L1/L2 and graph-edge counts. This proves that longer-range pair geometry is not an algebraic function of the selected v0.28 categories. Their unbinned products differ, so this certificate makes no claim of nonidentity relative to raw exact product, full six-number identity, or all selection constraints.

## Descriptive sampling only

The report validates history and image BINs, all 23 annual CSVs, source manifest and chronology. It compares observed within-stratum pair-geometry variation in 2,498 historic draws and a separate reproducible 50,000-ticket uniform-IID sample. It counts baseline cells, multiply observed cells, and repeated cells with pair-distance feature variation.

The different sample sizes, sparse cells, and group collisions prohibit direct comparison of raw repeated-stratum fractions as a statistical significance test. **No conditional IID rejection, predictive p-value, ticket recommendation, or prospective evidence is claimed.** P2A/P2B found no significant anomaly; the result here does not overturn that.

## PowerShell

    cd E:\lotto649
    $py = ".\.venv\Scripts\python.exe"
    & $py -m unittest discover -s tests -p "test_geometry_redundancy_v029_p3a.py" -v
    & $py -m lotto7x7.geometry_redundancy_v029 --csv-root data/csv --bin-root data/bin --iid-sample 50000 --seed 649 --out results/v029_p3a_conditional_ablation_20261111.json

The output is create-only. Existing report files are never overwritten.

## Acceptance

- [x] Reuse original v0.28 L3, Sum, Log Product bin and derived Joint category.
- [x] Strict collision certificate with matching categorical baseline and different pair geometry.
- [x] Read-only source validation; separate historical and synthetic demonstration.
- [x] Ten synthetic regression tests, source-gated CI and reproducible artifact.
- [ ] P3B conditional repair versus fair candidate controls is a separate experiment, not automatically authorized by algebraic nonidentity.

**Decision: STRUCTURAL_NONIDENTITY_PROVEN_PREDICTIVE_NO_GO.**
