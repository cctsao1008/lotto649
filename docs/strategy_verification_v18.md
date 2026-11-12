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

## Observed CI results — 2026-11-12

**Exact integer certification complete:** [Final regression-locked GitHub Actions 43194686900](https://github.com/cctsao1008/lotto649/actions/runs/43194686900/job/131102483336) **PASS**. Seven out of seven test cases pass, including independent Python enumeration oracle and negative fail-closed gate tests. The final CI locks expected exact candidate counts and ties, not only source hashes. Archived Actions artifact **`strategy-v18-new-seed-exact` ID 13931933741** (30-day retention), JSON **`strategy_v18_new_seed_exact.json` SHA256 `6b004ff6b5cffd328d59d21d76026cad89ba5ba2efe70d33abaec193e6d16b43`**.

**Reference provenance:** original 2026-11-06 / 2,497-draw historical cutoff, exact preexisting v1.2 source replay **152/152 PASS**, four rank axes from the exact 13,704 qualified original reference (**`18000dcc2d84b08fbcd3acf98cc56c4047fb86ad953ea108d116d02593c9034b`**), all 13,704 baseline candidates independently rescored with exact rank parity; no recalibration or data mutation.

| Seed | Raw sample size | Screened policy | Lossless primary-optimum candidates | Four-class Cartesian portfolios | Legal 24-disjoint portfolios | Primary-optimal feasible portfolios | Tied full lexicographic maxima |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 649 | 80,000 | Ranked | 854 | 338,814,900 | 1,216,875 | 1,712 | 1 |
| 649 | 80,000 | Shape round-robin | 681 | 218,400,000 | 783,666 | 1,597 | 1 |
| 20261111 | 160,000 | Ranked | 997 | 869,103,900 | 3,297,345 | 31,577 | 27 |
| 20261112 | 160,000 | Ranked | 982 | 698,025,984 | 2,716,183 | 30,284 | 11 |

**All four exhaustive optima, independently recomputed with exact integers:**

```text
Quality tuple    [Maximin, Total Rank, Shape Rank, Joint Rank]
Greedy v1.7      [43549, 195617, 43549, 54184]
Exact v1.8       [43549, 195617, 43549, 54184]
Block rank sums  [48644, 49240, 54184, 43549]
```

- The v1.7 six-restart deterministic heuristic returned an **actual full lexicographic optimum** in **each of the four selected 2,000-screened pools**; its observed optimization gap is zero **for these pools**, not all 27 v1.7 arms.
- Optimal tickets are **not always unique**: the first two pools have a unique best four-ticket set, while the other pools have 27 and 11 tied sets. Ticket identity is not an independent higher-priority objective when rank quality ties.
- The new-seed feasible optimum **strictly improves** the original historical-sample v1.5 exact optimum `[43549,192715,43549,52895]` by Total **2,902** and Joint **1,289**, at equal primary maximin.
- This is independent proof of the best quality **within these four SHA-audited pools**, based on a provably lossless Shape partition reduction. It does not prove no higher quality exists in the full 13k–54k new-seed qualified candidates, across other seeds/screens, or all 13,983,816 six-number combinations.
- The verification only addresses **strategy compliance and optimization quality**, not empirical lottery predictability, odds or payout, and no source/seal file was rewritten.

**Decision:** `V18_FOUR_NEW_SEED_SCREENED_POOLS_EXACT_LEXICOGRAPHIC_CERTIFIED / GREEDY_EXECUTION_PASS / OPTIMIZATION_GAP_ZERO_IN_FOUR_AUDITED_POOLS`.

### Follow-up research opportunity (separate from this completed audit)

If expanding beyond the four selected screen pools, first analyze whether the frozen **rank scale saturates** for candidates outside the original 13,704 reference set: the reference-position rank `bisect_left` has a maximum of 13,704, which can create shared or clipped high scores across new samples. Candidate coverage, score saturation, and optimizer search completeness are separate phenomena. A new, versioned study should examine full qualified-sample optima and sensitivity to the fixed reference score axis, without modifying this completed finite-pool evidence.
