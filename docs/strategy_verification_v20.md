# Lotto649 Strategy Verification v2.0 — Coupled Feature + Disjoint Pair Bounds

## Question and scope

The v1.9 bound for the complete independent-seed, positive-qualified source was
**Total Rank <= 200,061**, versus a legal research-only four-ticket witness with
**Total Rank = 195,617**, a conservative 4,444 gap. Why was the upper loose?

The old bound maximized Sum, Log, Joint, Shape ranks **independently within each
Shape class**, without requiring those four maxima to occur on the *same* ticket,
and without requiring the tickets to be mutually disjoint.

v2.0 keeps the original M0 historical cutoff (2026-11-06; 2,497 draws),
original fixed **13,704-candidate percentile-rank axes**, and original M0
four-block balanced lexicographic objective. We do **not** change scoring,
exclusions, candidate-generation, source data, forecast seals or any investment.

### Bound 1: same-ticket coupled rank sum

For any top Shape class `p`, let `G_p` be the candidates in that class attaining
its best Shape rank `s_p`, and for candidate `t` define

`w(t) = rank_sum(t) + rank_log(t) + rank_joint(t) + rank_shape(t)`.

The published legal witness attains the maximum possible first-stage objective
`B = sum_p s_p = 43,549`. Provided the fourth-highest Shape-class maximum is
*strictly* greater than the fifth, any first-stage-optimal four-ticket portfolio
must select precisely one candidate from each of the four classes and each
selected candidate must lie in `G_p`. All these preconditions are checked.

Therefore, for **every** first-stage-optimal four-ticket portfolio,

`TotalRank <= U_coupled = sum_p max_{t in G_p} w(t)`.

This is an **exact integer upper bound** and explicitly uses the fact that the
Sum/Log/Joint/Shape rank entries belong to the same candidate ticket. It
may be tighter than `sum_k sum_p max_{t in G_p} rank_k(t)`.

### Bound 2: legally disjoint pair upper

When `U_coupled` does not equal the best verified legal witness, independently
compute, for every two of the four Shape groups,

`M(a,b) = max {w(t_a) + w(t_b): t_a in G_a, t_b in G_b, balls(t_a) ∩ balls(t_b) = ∅}`.

Every legal four-ticket portfolio is compatible under every partition of the
four Shape groups into two disjoint pairs. Thus

`TotalRank <= U_pairs = min(M(0,1)+M(2,3), M(0,2)+M(1,3), M(0,3)+M(1,2))`.

All three matchings are calculated via batched **integer bit-mask** comparisons,
without floating-point MIP tolerances. Cross-pair overlap may still make this
bound loose. It can **never** be below an independently validated feasible
four-ticket Total Rank.

### Full lexicographic result criterion

Compute `lower` from the original independently screened winner and from a
fresh one-ticket-exchange search over the **complete** qualified source. Compute
`upper = min(U_coupled, U_pairs)` when pair search is needed. `lower == upper`
certifies the **secondary Total Rank** optimum in the original qualified
source (conditional on the strict Shape partition reduction); otherwise report
an unresolved **integer** upper gap, not fictitious improved scores.

At the final Joint-rank tie-break, `JointRank <= sum_p max_{t in G_p} rank_joint(t)`
is separately calculated. To declare the **complete** lexicographic tuple
certified, the attained Total must equal its exact upper **and** the attained
Joint must equal its independent rigorous maximum. The original Shape score
is already fixed by the first-stage bound.

### Matrix and evidence

Three independent original v1.7 seeds `[649,20261111,20261112]`, each with
nested `[160000,320000]` raw unique samples. The raw source contains 27k–55k
qualified candidates, **not merely** the top 2,000 ranked candidates. The
original 152 source/replay checks and 13,704 original ticket exact rank parity
are required on each CI run, including the original historical SHA and
reference rank SHA. All numerical outputs are audited for integer feasibility.

Tests cover mathematical upper parity on tiny synthetic universes, tied
Shape maxima, non-primary witnesses, invalid tickets, a genuine coupled-bound
improvement, and cases where number overlap makes pairwise upper tighter.

**Decision levels:** `EXACT_FULL_LEXICOGRAPHIC_OPTIMUM_IN_SOURCE`,
`EXACT_TOTAL_BUT_JOINT_TIEBREAKER_UNRESOLVED`, or
`VALID_FULL_SOURCE_TOTAL_UPPER_GAP_UNRESOLVED`.

## Limits

This is mathematical strategy optimization research over sampled qualified
candidates, not evidence of non-IID lottery draws, next-draw predictability,
increased jackpot chances or payout. An exact finite source certificate is not
a global certificate over all `C(49,6)=13,983,816` ticket possibilities.

## Observed GitHub CI result — 2026-11-13

[Original validated CI run 43342243064](https://github.com/cctsao1008/lotto649/actions/runs/43342243064/job/131613072247) **PASS**, **10/10 adversarial and independent integer tests** and the complete six-case source-audited experiment. Original historical data audit and M0 replay checks **152/152 PASS**; reference rank axes `18000dcc2d84b08fbcd3acf98cc56c4047fb86ad953ea108d116d02593c9034b` and reference rescoring of all **13,704** candidates exactly match the original frozen history. Original source cutoff: **2,497** draws through **2026-11-06**.

Actions artifact `strategy-v20-full-qualified-bounds` (ID **14000076518**, 30-day retention); archived `strategy_v20_full_source_bounds.json` SHA256 **`fbc47c3b58b10fbcc998bcbaa131bd07f1a90deb16cb2a81631283881ac82046`**.

### Six original-source-qualified candidate sets

| Seed | Unique raw draws | All four-positive-qualified candidates | Primary-optimal mandatory Shape-group sizes | Coupled Total Upper | Legal Total | Exact Total Gap |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 649 | 160,000 | 27,256 | 4,447 / 1,266 / 805 / 388 | **195,617** | 195,617 | **0** |
| 649 | 320,000 | 54,651 | 9,019 / 2,419 / 1,603 / 782 | **195,617** | 195,617 | **0** |
| 20261111 | 160,000 | 27,185 | 4,555 / 1,177 / 823 / 402 | **195,617** | 195,617 | **0** |
| 20261111 | 320,000 | 54,333 | 9,099 / 2,343 / 1,641 / 787 | **195,617** | 195,617 | **0** |
| 20261112 | 160,000 | 27,053 | 4,431 / 1,175 / 773 / 422 | **195,617** | 195,617 | **0** |
| 20261112 | 320,000 | 54,179 | 8,963 / 2,371 / 1,577 / 802 | **195,617** | 195,617 | **0** |

**Core exact integer certificate in every case:**

| Mandatory Shape partition ID | Required Shape rank | Exact best *same-ticket* four-block Total |
| ---: | ---: | ---: |
| 70216 | 7,364 | 45,381 |
| 74304 | 9,553 | 47,570 |
| 102976 | 13,128 | 51,145 |
| 107008 | 13,504 | 51,521 |
| **Four-class sum** | **43,549** | **195,617** |

This exact **same-ticket** coupled rank upper is **4,444 lower** than v1.9's per-axis-independent envelope of **200,061**. Every source has an independently audited legal four-ticket set with block sums `[48,644,49,240,54,184,43,549]`. Thus its lexicographic `(minimum, total, shape, joint)` is **`[43,549,195,617,43,549,54,184]`**.

Since a legal witness attains the primary Shape-class upper **43,549**, the coupled Total upper **195,617**, and the classwise Joint tie-break upper **54,184**, **the entire four-stage frozen lexicographic objective is rigorously optimized over each of these six complete sampled qualified candidate sets**, not merely their screened 2,000 subsets. The proof relies on source-verified ticket ranks and exact integer inequalities rather than a numerical MIP status or a multi-billion-tuple brute-force search.

**Pairwise nonoverlap bound implementation tested but not needed on these sources:** It is calculated only when the coupled same-ticket upper remains above an audited legal witness. For all six samples the coupled upper already equals the valid lower, so `disjoint_pair_upper = null`. The witness's separately verified **24 distinct numbers** establishes feasibility; dropping overlap in the upper bound is harmless because the upper is already tight.

**Important scope distinctions:** We have a complete optimality certificate **inside each original independent-seed sampled qualified source** (27k–55k legal positive candidates), using **the original 13,704-point empirical rank scale**. We have **not** established the same upper bound over all `C(49,6)=13,983,816` six-number candidates nor under re-fitted scoring. None of this proves next-draw prediction, expected payout improvement or any statistical correction mechanism in a fair lottery.

**Decision:** `V20_FULL_SOURCE_BOUND_AUDIT_PASS / EXACT_FULL_LEXICOGRAPHIC_OPTIMUM_IN_SOURCE (6/6)`. The previous `4,444` conservative *source* bound gap is now **closed exactly to zero** under these locked assumptions.
