# Lotto649 Strategy Verification v1.8 — New-Seed Candidate Pool Exact Certification

## Question

The v1.7 independently seeded candidate study produced feasible rank quality
`(43549,195617,43549,54184)`, which strictly exceeds the original fixed
13,704-source-sample optimum `(...,192715,...,52895)`. Does v1.7's bounded
six-restart greedy search actually attain the **full lexicographic optimum**
within its new **2,000-ticket screened pools**, or is there still a search gap?

## Frozen research protocol

- Source cutoff and provenance: 2,497 draws through 2026-11-06; v1.2 gate
  152/152 checks; exact 13,704-item frozen empirical scoring/ranking reference.
- Independent seed/screen selected from the already measured v1.7 arms,
  so this audit is **post-hoc search certification**, not a preregistered
  predictive experiment.
- Four exact v1.7 pool digests: `(649,80000,ranked)`,
  `(649,80000,shape_round_robin)`, `(20261111,160000,ranked)`,
  `(20261112,160000,ranked)`. Three distinct seeds and two policies.
- Rebuild all new candidate rows using **the existing v1.7 scorer** against
  frozen gains and 13,704 rank axes, and fail closed unless SHA-256 identity
  exactly matches the originally archived screen. Also re-score every
  original reference candidate and require exact rank-vector parity.
- Verify the original v1.7 bounded six-restart winner is legally feasible
  and repeats the exact original score. Any better exact result is surfaced
  as a stricter-quality **research counterexample**, not concealed or used
  to redefine the original candidate-pool objective.

## Lossless mathematical reduction

For four disjoint tickets from four different Shape partitions,

```
min(SumRank, LogRank, JointRank, ShapeRank)
   <= ShapeRank
   <= maximum of the sum of the four highest distinct Shape-partition caps.
```

If the fourth-highest cap is strictly greater than the fifth and the screened
pool contains a feasible portfolio attaining that exact upper bound, then
**every** first-objective optimum must choose (a) those exact four Shape classes
and (b) a ticket achieving the respective class's maximum Shape rank. The
reduced pool is *necessary and lossless for primary-optimal portfolios.*

The existing independent v1.5 C++17 exact-integer enumerator then checks
all combinations in the four retained classes, applying original 24-number
exclusivity and all other three primary rank constraints. It records
`cartesian_count`, `disjoint_four_portfolios`, `primary_feasible_portfolios`,
`best_total_rank`, `best_joint_rank`, and `tied_best_lexicographic_portfolios`.
No LP, MIP, rounding or optimizer numerical tolerance is used by the exact
enumerator. A Python integer scorer checks the returned actual four lines.
Small toy pools are also checked against an independent Python brute-force
oracle; tied fourth/fifth class cap and score/hash corruption fail closed.

## Acceptance / decision

- `STRATEGY_COMPLIANCE` iff source/rank/score/pool digests and all legal
  tickets and Shape constraints reproduce exactly.
- `PRIMARY_OPTIMALITY` iff a verified four-line witness attains a mathematically
  valid Shape-class primary upper bound.
- `FULL_LEXICOGRAPHIC_OPTIMALITY` iff the exact enumerator **fully completes**
  its search and returns the optimal quality tuple with a legal independent
  witness. If it exceeds v1.7's heuristic score, record the gap. There is no
  expectation or precondition that the v1.7 heuristic was globally optimal.
- The scope is each **specific frozen 2,000-screened pool**, not the 13k–54k
  new qualified samples, all possible six-number tickets, a prediction
  performance test, or an actual financial payout claim.

## Run

```powershell
python -m unittest discover -s tests -p 'test_strategy_verification_v18.py' -v
python -m lotto7x7.strategy_verification_v18 --out results/strategy_v18_new_seed_exact.json
```

No original history, old study, sealed report, or ticket selections are
modified. The output is an exclusively created independent JSON audit under
ignored `results/`.
