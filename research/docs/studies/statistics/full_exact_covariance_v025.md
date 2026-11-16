# v0.25 P0 — Exact Full 163-Feature Population Covariance

Tracking: Issue #8, https://github.com/cctsao1008/lotto649/issues/8.
Reference-only mathematics, never a forecast or silent replacement of the
legacy v0.25 Monte Carlo covariance optimizer.

## Feature registry and sample space

Enumerate ALL C(49,6) = 13,983,816 equally likely, sorted, six-number
combinations. Use the exact original sequential 7x7 raster and 8-neighbor
shape rule named shape8-graph-partition-cardinal-diagonal-v1.

Feature vector dimension 163:
- 49 number-presence indicators, numbers 1..49
- Sum, odd-number count and span (max minus min)
- L1 6 one-hot component classes; L2 11 one-hot component partitions;
  L3 94 one-hot partition/orthogonal/diagonal-edge classes

The model stores 163 exact integer sums S_i and 13,366 exact integer
upper-triangular second moments M_ij. Every population covariance is the
exact rational (U*M_ij - S_i*S_j)/U^2, where U=C(49,6).
No Monte Carlo variance or historical draw result is used.

## Four independent checks

1. Standalone C++17 sequentially enumerates all legal combinations.
   A scalar BFS, independent of the original NumPy Warshall classifier,
   emits 94 grouped L3 sufficient statistics: counts, total spans, 49
   number-indicator counts. Working memory is bounded independently of U.
2. Analytic hypergeometric identities give number indicator moments:
   same-number count C(48,5), pair count C(47,4). Sum and odd-count
   are exact linear combinations of the 49 number indicators.
3. Independent min/max combinatorics yield P(min=a,max=b) proportional
   to C(b-a-1,4), and exact span first, second and number cross-moments.
   E[span]=250/7, E[Sum]=150, Var(Sum)=1075.
4. L3 uniquely determines L2, which determines L1. Exact L3/number/span
   totals therefore define every Shape x Shape and numeric x Shape
   moment. All 94 individual L3 counts, their 11 L2 parents and their
   six L1 parents must match the original immutable v0.16 full-universe
   fixture (Git blob 7d301f16937d985ce5fb7cef590bb1b1a109b0b6).

The 163x163 covariance is necessarily singular. Number indicators sum
to six on every draw; Sum and odd-count are linear transforms of number
indicators; one-hot shape families have deterministic nesting.
Use separately justified rank reduction or ridge regularization in any
future optimizer. The legacy sample covariance and sealed portfolio
choices remain unchanged. Exact covariance does not create a predictive
edge or justify physical compensation of past lottery deviations.

## Reproduction

Run from repository root, with Python 3.12+ and GCC C++17:

    mkdir -p results
    g++ -O3 -std=c++17 -Wall -Wextra -Werror research/lotto7x7/portfolio/exact_covariance_v025.cpp -o results/v025_exact_census
    ./results/v025_exact_census > results/v025_exact_census.tsv
    python -m lotto7x7.exact_covariance_v025 --census-tsv results/v025_exact_census.tsv --fixture data/reference/shape_class_counts.json --out results/v025_full_163_exact_covariance.json
    python -m unittest discover -s tests -p test_exact_covariance_v025.py -v

The write-once GitHub Actions job commits an archive under
research/artifacts/v025_exact_covariance and uploads a reproducibility artifact
only after all tests and original v0.16 reference comparisons pass.

## Scope / next

Engineering: exact theoretical 163-feature covariance foundation only.
This DOES NOT complete #8 downstream global Pareto tradeoffs,
complete exact-vs-Monte-Carlo optimization benchmark, source-verified
out-of-sample testing or any predictive confirmation.
Decision label: EXACT_THEORETICAL_COVARIANCE_ONLY_NO_PREDICTIVE_CLAIM.
