# v0.26 P1 — Disjoint-optimality research (proof status)

## Question

For four different six-number tickets of Taiwan Lotto 6/49, and a uniform
outcome of six main balls + one distinct special ball, is the probability
of at least one **official eight-tier** prize globally maximized by four
pairwise-disjoint tickets?

Do **not** assume the conjecture true merely because it was best among
15 representative profiles.

## Verified exact result for two tickets

For exactly two different tickets, relabeling invariance reduces all possible
pair structures to six cases: intersection size `t=0,1,2,3,4,5`.
(The `t=6` case is a prohibited duplicate.) The exact joint official-prize
engine exhaustively evaluates each class, giving:

| Shared labels | P(at least one official award) |
|---:|---:|
| 0 | 0.061724283504 |
| 1 | 0.060948968636 |
| 2 | 0.059083044185 |
| 3 | 0.056132866338 |
| 4 | 0.051551873035 |
| 5 | 0.043981959424 |

All counts are exact integers divided by `C(49,6)*43`, and the ordering
is strict. Because the structural classification `t=0..5` is exhaustive,
the **two-ticket global disjoint-optimality claim is established** within
the stated fair-6/49 eight-tier model. This is an exhaustive finite
combinatorial certificate, not a general probability theorem.

The two-ticket event union identity is:

```
P(W>=1) = E[W] - P(W=2)
```

where `W` is the number of awarded tickets. `E[W]` is invariant at a
fixed ticket count. The exact pairwise overlap table therefore certifies
that disjoint tickets minimize the chance that *both* tickets are
simultaneously awarded.

## Four-ticket exact experiments

- Original 15 intersecting templates: `disjoint` wins for `P(W>=1)`.
- Added deterministic 25 random mutation attempts; after canonicalization
  under all **24 ticket permutations** and all 1..49 label relabelings,
  the GitHub CI smoke evaluated **34 distinct overlap structures**.
- Best observed again `disjoint`, exact
  `P(W>=1)=0.1227314589619088`, equal to immutable purchased A–D.
- CI [run 42665490675](https://github.com/cctsao1008/lotto649/actions/runs/42665490675):
  **55 tests passed** and the real A–D and structure-search smokes passed.

However, ***this is still not a global proof for four tickets***. Four sets
have 15 nonzero membership-mask counts (and up to quadruple shared labels);
pairwise intersection sizes alone do not fully determine the award union.
A theorem for two tickets does not by itself prove the four-ticket claim:
triple/quadruple simultaneous wins enter inclusion-exclusion.

Exact general identity:

```
E[W] = P(W>=1) + P(W>=2) + P(W>=3) + P(W>=4)
```

For all-pairwise-disjoint six-number tickets, `W<=2` under these official
award tiers, because a winning ticket requires at least two drawn main
balls, and the seventh (special) ball can support no third awarded ticket
with all ticket numbers disjoint. Thus their union event is
`P(W>=1)=E[W]-P(W>=2)`. Allowing overlap changes all higher-tail
terms and requires a global argument, not just an individual-pair proof.

## Reproduce

```powershell
git switch feature/v025-deficit-compensation-foundation
git pull --ff-only
python -m lotto7x7 coverage-global-research `
  --seed 649 `
  --samples 200 `
  --out results/v026_coverage_global_research.json

$r = Get-Content .\results\v026_coverage_global_research.json -Raw | ConvertFrom-Json
$r.two_ticket_exact_structural_study
$r.four_ticket.best_tested
$r.four_ticket.improvement_over_reference
$r.four_ticket.distinct_structures_evaluated
$r.four_ticket.globally_optimal_proven
```

This does not modify the sealed A–D reference or employ past draw data.
Increasing samples only adds heuristic evidence; it **cannot** certify
four-ticket global optimality.

## Further research: proof or exhaustive certificate

- Find a valid *disjointification/exchange* lemma for the official award
  event union, explicitly handling the special ball and 3-/4-ticket
  intersections. If one exists, use it to establish the claim for 4 tickets.
- Alternatively, enumerate the full integer feasible set of nonzero
  membership counts `c_mask` for masks 1..15 with
  `sum(mask contains ticket_i) c_mask = 6` for each i,
  `c_mask >= 0`, total distinct numbers <=49, and no duplicate
  ticket rows; canonicalize up to the 24 permutations. Evaluate exact
  union counts per class. This is more than random label enumeration,
  but may require substantial computational optimization and an
  independently checked completeness proof.
- If either direction fails, publish a *counterexample* configuration
  with its exact integer prize-outcome counts, rather than preserve
  a preferred hypothesis.
- Keep expected prize amount and future prediction evidence separate:
  a structural coverage optimum is neither a profit guarantee nor a
  future-draw predictive advantage.

Reference: Taiwan Lottery official prize tier rules at
https://www.taiwanlottery.com/lotto/manual_match/lotto649/
and game prize table https://api.lottery.com.tw/getmoney/l649_getmoney.html.
