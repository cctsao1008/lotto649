# v0.26 P1 — Complete four-ticket exact coverage certificate

## Computational result

At [GitHub Actions #42667043855](https://github.com/cctsao1008/lotto649/actions/runs/42667043855), a complete census of four-ticket overlap structures passed:

```text
61 passed in 20.47s
[ENUM] complete=True classes=2957
[MAX] reference=0.122731458962 best=0.122731458962 delta_outcomes=0
[PROOF] four-ticket disjoint optimum=True
[SHA256] 53058dbbd704fd225051f6ee4591401a54945d584c2e85116f3b8d6cd957b83d
[PASS] all four-ticket membership classes exact coverage certification
```

The result is a **complete computational combinatorial certificate** for
the specified model. It is not a symbolic or human-checkable one-line
theorem; its reliability depends on the enumerator and exact scoring
engine, which are accompanied by independent checks.

## Mathematical sample space and objective

Four distinct bets, each a six-element subset of labels 1..49. The
draw consists of an unordered six-element main subset and one further
special element outside the main subset:

```
N = C(49,6)*(49-6) = 601304088.
```

Maximize `P(at least one of the four tickets wins any of the eight official
prize tiers)` across all legal portfolios of four **different** tickets.

## Why a 15-dimensional structure is complete

For each of the 49 labels, define its four-bit vector of presence in the
four tickets. Let `c_m` be the number of labels with nonzero membership
mask `m ` in `1..15`. Each ticket contains exactly six labels:

```
  sum_{m: (m & (1 << i)) != 0} c_m = 6     i=0,1,2,3.
  c_m are nonnegative integers.
```

All labels *outside* the union have membership 0, and their count is
`49-sum(c_m)`. Because there are only 4*6 ticket memberships, this sum
is at most 24, so there is always enough room in the 49-label universe.

Conversely, for every feasible vector, assigning `c_m` unique labels
consecutively to that mask class yields four six-element sets. Thus the
recursion enumerating all nonnegative solutions to these four constraints
covers every legal ticket overlap structure. Relabeling the 49 balls
within their classes does not change uniform draw probabilities.

The recursion yields exactly **52,450** feasible vectors for *labeled*
ticket rows. A pair of identical tickets occurs if and only if no label
has different membership bits in their two ticket positions. Excluding
those leaves exactly **50,985** vectors for four distinct labeled bets.

Permuting four ticket positions (24 permutations, group S4) changes only
which row is called ticket A/B/C/D. Canonicalizing every vector by the
lexicographically minimal 15-component signature across all 24
permutations yields **2,957** distinct unlabeled structures. The
canonicalizer is tested against the earlier independent ticket-array
version and basic relabeling invariance. The full exact scoring routine
was previously checked against exhaustive nine-label outcome
enumeration and a single-ticket closed form.

## Score and conclusion

Each canonical structure is converted into four actual six-number
tickets and evaluated by `official_prize_coverage`: group labels by
membership pattern, choose six mains within groups using combinatorial
weights, then choose one of the remaining 43 special labels. The engine
uses integer counts until the final probability division. Each group
calculation conserves the exact 601,304,088 outcomes.

The global maximum `P(any official award)` across the **entire**
2,957-class census equals the value of the fixed purchased disjoint
A–D portfolio:

```
  Pmax = 0.1227314589619088
  max_improvement_over_disjoint = 0 / 601304088.
```

Therefore **pairwise-disjoint four-ticket portfolios attain the global
maximum of per-draw probability of at least one official award** under
the specified fair 6/49 model. The computation did not attempt to
prove the optimum is unique; `number_of_maximizing_structures`
is recorded in the complete output.

This does **not** imply that disjoint maximizes `P(W>=2)`, payout tail
probability, expected cash return, or probability of future outcomes
under a non-IID mechanism. In fact fixed-ticket-count `E[W]` is
invariant under fair IID, and all distinct four-ticket portfolios
have the same top-prize probability.

## Reproduction (full mode)

```powershell
git switch feature/v025-deficit-compensation-foundation
git pull --ff-only

python -m lotto7x7 coverage-exhaustive `
  --out results/v026_coverage_exhaustive.json

$r = Get-Content results/v026_coverage_exhaustive.json -Raw | ConvertFrom-Json
$r.completed
$r.raw_labeled_signatures
$r.valid_labeled_signatures
$r.unique_valid_unlabeled_structures
$r.global_disjoint_optimality_proven
$r.maximum_improvement_outcomes
$r.number_of_maximizing_structures
$r.score_sha256
```

Partial smoke testing is permitted with `--max-classes 12`, but
`completed` and `global_disjoint_optimality_proven` are then
necessarily `false`. A sampled or truncated run is never allowed
to certify an optimum.

## Further independent verification opportunities

- Build a second, standalone dynamic-programming counter for feasible
  degree vectors and an orbit-size/Burnside cross-check for the 2,957
  canonical classes.
- Independently verify the maximum by integer inclusion-exclusion over
  award events, or by an exchange/uncrossing inequality.
- Publish a machine-checkable max-score histogram certificate with
  checksums of canonical signatures and counts to reduce reliance on
  a single implementation.
- Keep historical HDDC and prospective prediction tests distinct.

The original paid four-ticket A–D and prospective freeze remain untouched.
