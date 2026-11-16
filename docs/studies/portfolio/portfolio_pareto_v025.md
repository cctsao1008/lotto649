# Issue #8 — Exact coverage and historical repair Pareto comparison

Use the frozen 2,497-draw prefix through 2026-11-06 (115000103), the exact
v0.16 shape fixture, the 163-feature population covariance, and ridge 0.25.
Each strategy gets the same 300 unique uniformly sampled tickets, seed 649,
and a cap of 1 + 3 × 4 × (300 − 4) portfolio evaluations. Uniform and
frozen v0.21 shape scoring use one portfolio by design. The original v0.21
coverage selector is included as a baseline with its existing search and
the same cap. All new searches start at the uniform portfolio and scan the
same candidate order for up to three full one-ticket swap passes. Four
distinct tickets are required. Number overlap is allowed.

The objectives are the exact main-ball union counts for 3+, 4+ and 5+ hits,
historical post-augmentation ridge-whitened squared distance to fair
theory, the number of distinct numbers, and the number of distinct L3
shape classes. The 6/6 count is always four and is reported separately.
Compensation-only minimizes the distance. Coverage-only maximizes the
v0.21 lexicographic coverage key (3+, then 4+, then 5+).
Hybrid first satisfies a coverage floor on each threshold, then minimizes
distance. Its required floors are 100%, 99.9% and 99% of the disjoint
four-ticket counts, rounded up to integer outcomes. While below a floor,
the search prioritizes unmet count deficits in threshold order. Exact
score ties are resolved by the stated diversity and ticket ordering.
No weighted objective or winner across policies is assumed.

Compute the nondominated set among all complete portfolios evaluated by
the new search and final baseline portfolios. Equal objective vectors
retain the lexicographically first portfolio and an equivalence count.
This frontier covers the visited set. The theoretical coverage-orbit
census ranges over all possible unlabeled four-ticket structures. This retrospective report measures
descriptive repair and exact mathematical coverage.

Use the membership-mask DP to report max hits and, for 3+, 4+, 5+ and 6/6,
how many tickets achieve the threshold in one draw. Cross-check the full
main-ball universe independently for a synthetic disjoint portfolio and
the uniform baseline.
Eight-tier official prizes include the separate special ball. Under
independence across draws, ten-draw event probability is 1 − (1 − p)^10
for a fixed portfolio. No new prospective trial is enrolled here.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 python -m lotto7x7.portfolio_pareto_v025
```

The candidate pool size and seed can be varied through the CLI. If the
bounded hybrid search does not attain the floors, its result is flagged
by its coverage status. The best floor-feasible portfolio across the
combined evaluated set is reported independently when one exists.

## Results — 2026-11-16

| Strategy | Main 3+ | Main 4+ | Official any award | Distance after four additions |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 7.18376% | 0.394342% | 11.69006% | 0.0572977864 |
| v0.21 coverage | 7.43786% | 0.394856% | 12.27315% | 0.0574082029 |
| v0.21 Shape Score | 7.26855% | 0.394599% | 11.85720% | 0.0629274905 |
| Compensation only | 7.15759% | 0.394342% | 11.61497% | 0.0561206394 |
| Coverage only | 7.43786% | 0.394856% | 12.27315% | 0.0566585204 |
| Hybrid | 7.43786% | 0.394856% | 12.27315% | 0.0566968310 |

All six portfolios have 1,036 main-ball outcomes with 5+ hits and four
6/6 outcomes. The complete 2,957-orbit structural census independently
finds maxima of 1,040,096, 55,216 and 1,036 main-ball outcomes for 3+, 4+
and 5+ respectively. One orbit attains all three maxima, the disjoint
structure. Individually the maxima are attained by 1, 16 and 972 orbits.
The full 13,983,816-draw intersection census agreed with the joint DP
on the synthetic disjoint and uniform portfolios, including the number
of tickets attaining each threshold.

The 6,781 visited complete portfolios yielded 22 distinct Pareto
objective vectors. Among visited portfolios meeting all hybrid floors,
the best historical distance is 0.0566585204, reached by the coverage-only
search. It improves on the specific hybrid search result at equal coverage.
The compensation-only output has 19 distinct numbers and gains historical
repair at the cost of lower coverage. The 100% 3+ threshold forces the
disjoint orbit, so the 4+ and 5+ floors are slack in this run.

The bounded search and common exact historical cutoff limit the
optimization claims. The complete structural maxima apply to theoretical
coverage under fair IID. They are independent of the finite-pool repair
ranking. See the [archived JSON](../../../data/research/v025_exact_covariance/v025_portfolio_pareto_20261116.json)
for all six strategies, official prize tiers, joint coverage and the frontier.
