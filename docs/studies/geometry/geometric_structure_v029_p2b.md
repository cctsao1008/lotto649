# Lotto649 v0.29 P2B — Graph Topology & Pair-distance Convergence

Tracking: https://github.com/cctsao1008/lotto649/issues/31

This research is descriptive and outcome-blind until the historical audit step. It does not create tickets, change paid wagers, or claim predictive advantage.

## Frozen structure features

There are 15 unordered point pairs in each legal six-hot 7x7 bitmap. Three predeclared pair-distance features are:
- pair_squared_distance_sum: sum of squared Euclidean distances across all 15 pairs; exact mean 245.
- pair_midrange_3_to_18: number of pairs whose squared distance is 3 through 18 inclusive; exact mean 7.88265306122449.
- pair_far_25_plus: number of pairs whose squared distance is at least 25; exact mean 4.362244897959184.

Two graph-control features, cardinal_edges and diagonal_edges, are already completely determined by the existing L3 taxonomy and MUST NOT be treated as new independent objectives. Graph cycle rank is also determined by the L3 component count and total edge count through E - 6 + C.

Distinct longer-range pair distances can occur in tickets with the SAME L3 class; this is descriptive non-identity and does not imply conditional predictive information beyond original numbers, Sum, Log or Joint.

## Exact without-replacement null

For pair weight h(e) over all 1,176 pairs of 49 lattice positions, let T=sum h(e), Q=sum h(e)^2, S=sum h(e)h(f) for unordered pairs sharing one vertex, and D=(T^2-Q)/2-S for disjoint pairs.

Let p_k=(6)_k/(49)_k for k=2,3,4.

E[H] = p_2 T
Var(H) = p_2 Q + 2 p_3 S + 2 p_4 D - E[H]^2.

This finite-population formula is exact, independent of historical observations. Fixed bijective mappings preserve the unconditional fair distribution.

## Convergence / IID controls

Reuse P2A's strict source integrity gate for history BIN, image BIN, 23 annual CSVs, manifest SHA and chronology.

Fixed mappings: sequential, snake, random:649, random:650.
Fixed chronological prefixes: 100,250,500,1000,1500,2000,N.

Two whole-search endpoints are the maximum absolute standardized mean z across:
1. Three non-L3-determined pair-distance summaries.
2. Two L3-derived graph edge controls.

Each endpoint is calibrated against 199 independent full-length fair-IID synthetic histories, rerunning every mapping and prefix. Empirical p=(1+null_max>=observed_max)/(R+1), with Bonferroni correction across two P2B endpoints. This does not correct across earlier explored research strategies, including P2A.

## PowerShell

    cd E:\lotto649
    $py = ".\.venv\Scripts\python.exe"
    & $py -m unittest discover -s tests -p "test_geometry_structure_v029_p2b.py" -v
    & $py -m lotto7x7.geometry_structure_v029 --csv-root data/csv --bin-root data/bin --iid-reps 199 --seed 649 --out results/v029_p2b_structure_20261111.json

The new report uses exclusive creation within gitignored results/. No overwrite of existing research reports.

## Acceptance

- [x] Exact first and second finite-population moments for fixed pair features.
- [x] L3-derived graph controls explicitly separated from novel descriptive geometry.
- [x] Same L3 class can show different long-range geometry; this is not an advantage claim.
- [x] Seeded whole-pipeline fair-IID calibration and two-endpoint multiplicity correction.
- [x] Ten independent regression tests and source-gated CI.
- [ ] P3 conditional ablation against existing Shape, Sum, Log and Joint before any new repair selection criterion.

All findings are EXPLORATORY_DESCRIPTIVE_ONLY_NOT_PREDICTIVE; historical deficits do not force future IID draws to compensate.
