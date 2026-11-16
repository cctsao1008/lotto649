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
