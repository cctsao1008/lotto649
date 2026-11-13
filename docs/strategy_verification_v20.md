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
