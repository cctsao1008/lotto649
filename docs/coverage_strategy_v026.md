# v0.26 P1 — Coverage Strategy A: structural profiles

## Objective

Compare exact **per-draw / four-ticket probability of at least one official
award** while holding the number of distinct bets fixed at four. Alternative
reported objectives: `two-or-more-awards`, or an individual named prize tier.
No historical lottery data, learned weights, or stochastic forecast enter this
selection.

The eight-tier official-prize engine from v0.26 P0 enumerates exact
`C(49,6) * 43` equiprobable main+special outcomes by membership masks.
It produces the official prize award union, each tier union, and distribution
of 0..4 awarded tickets. **Different ticket tiers overlap in a draw.**

## Why classify overlap patterns instead of drawing many random labels?

Under a uniform 6/49 model, a permutation of labels 1..49 leaves probability
unchanged. Only the structure of the four six-number sets and all their
intersections matter. Therefore repeated relabelings of the same profile add
no information about the fair coverage objective.

This first implementation enumerates **15 explicit example profile families**:

- `disjoint`: all 24 labels different.
- `one_pair_1,3,5`: exactly one pair shares 1, 3 or 5 labels.
- `two_pairs_1,2,3,5`: two disjoint pairs each sharing k labels.
- `chain_1,2`: neighboring pairs in a chain share k labels.
- `star_1,2,3,4,5`: all four tickets share the same k labels.

For a fixed profile, `seed` only permutes labels. It does **not** influence
any theoretical prize probabilities. This search is over a deliberately
restricted list, **not all non-isomorphic configurations**, and the best
profile should never be described as a proven *global* optimum.

The immutable A–D ticket is always included as an external reference.
Expected total awarded-ticket count must remain constant for every
four-ticket configuration (linearity of expectation). The distribution of
*whether* any ticket wins and probability of two or more awards can vary.

## Run

```powershell
git switch feature/v025-deficit-compensation-foundation
git pull --ff-only

python -m lotto7x7 coverage-strategy `
  --objective any-prize `
  --seed 649 `
  --out results/v026_coverage_strategy_a.json

$r = Get-Content .\results\v026_coverage_strategy_a.json -Raw | ConvertFrom-Json
$r.reference
$r.best_profile
$r.best_objective_value
$r.difference_from_reference
$r.evaluated | Select-Object profile,objective_value,any_prize_probability,two_or_more_awards_probability,expected_awarded_tickets
```

Try a distinct objective **without modifying tickets or the primary
analysis**:

```powershell
python -m lotto7x7 coverage-strategy `
  --objective two-or-more-awards `
  --out results/v026_coverage_multi_awards.json
```

## Acceptance / scientific limits

- Legal, unique 4 x 6 bets in every template.
- Exact prize engine reused without sampling. Small-universe exhaustive
  engine regression tests from P0 remain active.
- Disjoint relabelings and original A–D produce exactly equal award event
  probabilities and expected ticket counts.
- Expected ticket count invariant across profile structures.
- Output retains per-tier and award-count distributions; selected objective
  is explicit and independent from any future draw.
- No claim of higher expected monetary payout under any fixed ticket count:
  award per-ticket marginals are fixed in fair IID, even though unions vary.
- The true global solution over all possible four-set overlap structures,
  explicit cost/utility preferences, Strategy B coupling, and walk-forward
  tests are **future work**. Avoid confusing an exploratory structural
  optimum with forecast skill.

Original paid A–D and its future-draw seal are not changed.
