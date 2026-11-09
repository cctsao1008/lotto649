# v0.26 P0 — Official eight-tier exact prize coverage

## Scope

Taiwan Lotto 6/49: buy six different labels in 1..49; six *main*
numbers are drawn without replacement and then one distinct *special*
number. Official prize tier conditions are:

| Tier | Main matches | Special ball matches? |
|---|---:|---|
| First | 6 | irrelevant |
| Second | 5 | yes |
| Third | 5 | no |
| Fourth | 4 | yes |
| Fifth | 4 | no |
| Sixth | 3 | yes |
| Seventh | 2 | yes |
| Regular | 3 | no |

Official corroboration: CTBC prize allocation page
https://lotto.ctbcbank.com/result_all.htm
and game overview https://api.lottery.com.tw/getmoney/l649_getmoney.html .
Bonus never replaces a main-number hit. Each ticket can be awarded
at most one tier. For a multi-ticket portfolio, different tickets can
win distinct tiers in one draw, so tier-specific "any" probabilities
are **not additive**.

## Exact enumeration method

For each label 1..49, compute a bitmask describing which of the
portfolio's 1..4 unique tickets contain the label. Group numbers
with the same bitmask. Enumerate allocations of exactly six main
balls among mask classes; each allocation has multiplicity
`product(C(class_size, chosen_main_from_class))`. For each allocation,
enumerate the class containing the remaining special ball, weighted by
`class_size - chosen_main_from_class`. The exact equiprobable
outcome space has

```
C(49,6) * 43 = 601,304,088
```

outcomes. All computations are integer until probabilities are
formed at the end. No random sampling and no historical frequency
are involved. Returned values:

- Exact probability of at least **one officially awarded ticket**.
- Exact probability that at least one ticket wins a specified tier.
- Expected count of tickets winning each tier (different from tier
  union probability if awards overlap).
- Exact distribution of number of winning tickets (0..4),
  expected total winning tickets and outcome-mass checks.

This is a fair-outcome probability model. It does not include actual
payout amount or positive expected net return: first through fourth
awards are pool-dependent, and fixed prizes may be rule-dependent.

## Run against immutable purchased A–D

```powershell
git switch feature/v025-deficit-compensation-foundation
git pull --ff-only
python -m lotto7x7 official-prize-audit `
  --tickets 4,9,17,24,29,42 10,13,14,27,28,34 6,12,19,20,44,48 3,7,18,33,40,47 `
  --out results/v026_reference_official_prizes.json

$r = Get-Content results/v026_reference_official_prizes.json -Raw | ConvertFrom-Json
$r.any_prize_probability
$r.any_ticket_by_tier
$r.expected_awarded_tickets
$r.awarded_ticket_count_probabilities
```

Also supported: `python -m lotto7x7.official_prize_v026 ...`.
No historical files or future forecast seals are read or modified.

## Verification

- Single ticket: independently calculate each prize count by
  `C(6,k) C(43,6-k)` main draws, then special-in-ticket
  `6-k` or special-outside `37+k` remaining numbers.
- Small 9-label test universe: **enumerate every main six-set and
  every remaining special label** and compare exact integer totals,
  including overlapping four-ticket portfolios.
- Validate conservation of `C(N,6)(N-6)` sample outcomes and
  winning-ticket expectation identities.
- Original A–D head prize probability for four unique tickets must
  equal `4/C(49,6)`, consistent with previous main-only exact
  coverage.

## Limits and next steps

P0 computes exact prize-event probabilities but is **not** a general
Coverage-only optimizer (P1). Multi-objective constraints, official
prize expectation under variable payouts, and chronological
walk-forward are separate milestones. A higher probability of *any*
official prize does not itself imply better financial return. No
retrospective parameter search is evidence of a predictive advantage.
