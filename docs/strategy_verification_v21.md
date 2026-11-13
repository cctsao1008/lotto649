# Strategy Verification v2.1 — Full 6/49 Universe Optimization

## Research question

Can the mathematically certified v2.0 four-block lexicographic optimum from six independently sampled qualified sources be extended to **the entire C(49,6) = 13,983,816 legal six-number universe**, without changing any historical source, percentile rank scale, positive-gain selection rule, Shape taxonomy, or search objective?

This is a finite combinatorial optimization study, **not a statistical test for future lottery predictability**. The performance objective is an internal retrospective deficit/shape score, not a return forecast.

## Frozen inputs and source gate

- Historical prefix: first **2,497** official draws through 2026-11-06 and draw #115000103, pinned SHA-256 `d865e1d241b607488e58a00b22fb595734f804595afd9d8850718808d6e8b7f8`.
- Run the original v1.2 independently recreated M0 input/score/search verification; **152** checks must pass.
- Four standardized positive-gain features: **Sum, Log Product, Joint Sum×Log, Shape taxonomy**. Use the frozen historical score functions and the unchanged ordered **13,704** reference gains for `bisect_left` rank positions, SHA-256 `18000dcc2d84b08fbcd3acf98cc56c4047fb86ad953ea108d116d02593c9034b`.
- Every ticket must have six distinct values in 1…49; selectable candidates must satisfy all four strictly positive original gain requirements and the unchanged original immutable selection-exclusion contract. **No new filtering objective, refitting, or paid-ticket analysis.**
- The four selected six-number tickets must have **24 distinct numbers** and **four different Shape partition codes**.
- Portfolio quality is exactly **lexicographic `(min(Sum,Log,Joint,Shape), Total, Shape, Joint)`** calculated from the original integer reference ranks.

## Exhaustive bounded-memory enumeration

The v2.1 scorer streams `itertools.combinations(range(1,50),6)` once, in fixed-size chunks, so no materialization of 14 million Python tickets is needed. The cheap Sum, Log and Joint eligibility checks precede vectorized eight-neighbor Shape graph classification. Scoring is independently recomputed using the frozen tables and cross-checked against **all original 13,704** ticket/Shape/rank records.

For every batch:

1. Count and digest **all** legal candidate six-tuples in canonical order, irrespective of eligibility.
2. Recompute Sum×Log fair-theory frequency counts, including rejected candidates. The final independent empirical stream must match the original exact full theoretical joint histogram SHA-256 `31136d04347ff7b84fa7b5d17e7d009068b784e46f43f1d49aa058bc7fcd422a`.
3. For qualifying candidates, calculate **all four gain/rank values** on the original fixed scale and group by Shape partition.
4. Retain only (a) each Shape group's maximum Shape rank, (b) the maximum **same-ticket coupled four-block total** among candidates achieving that group's Shape maximum, and (c) every six-number candidate tied at that coupled maximum. This is a **lossless bound representation** for the first two lexicographic objectives.

If the scan visits fewer than **13,983,816** combinations, has an extra item after the final combination, or fails the complete fair joint SHA, **no full-universe optimality certificate may be produced**. The test-only `allow_partial` flag returns explicitly noncertifying metadata.

## Exact upper-bound logic

Let `c_i` be the Shape-rank maximum within partition `i` over **all** qualifying six-tuples. A valid four-ticket portfolio has four distinct partitions, so

`Maximin <= ShapeTotal <= sum(top_four(c_i)) = P`.

**Only if** a legal witness reaches `P` and the fourth-largest Shape cap is strictly larger than the fifth can every primary-optimal portfolio be restricted to the four mandatory top partitions and tickets that individually achieve their Shape caps.

Within each mandatory partition let

`U_i = max {SumRank(t)+LogRank(t)+JointRank(t)+ShapeRank(t) : qualified(t), partition(t)=i, ShapeRank(t)=c_i}`.

For **every** legal primary-optimal portfolio,

`TotalRank <= sum_i U_i`.

The bound is exact integer arithmetic on **the same ticket's joint feature vector**; it cannot falsely mix incompatible per-axis maxima. Number overlap constraints can only lower the feasible maximum, so this is always an upper bound.

To certify a complete lexicographic optimum, a legal set of four disjoint tickets must **simultaneously attain** the primary bound, the coupled Total upper bound, and the classwise upper on Joint rank **conditional on attaining coupled Total**. If any condition is not met, the report provides the best feasible witness and conservative nonzero gap; it does **not** assert global optimality.

A bounded deterministic search among the exact per-partition coupled Total ties may find such a witness. A timeout/node-budget exhaustion does not imply no better portfolio exists.

## Reproducible CLI and CI

```powershell
python -m unittest discover -s tests -p 'test_strategy_verification_v21.py' -v

python -m lotto7x7.strategy_verification_v21 `
  --csv-root data/csv --bin-root data/bin `
  --batch-size 60000 `
  --out results/strategy_v21_full_universe.json
```

GitHub workflow: `.github/workflows/strategy_verification_v21.yml`. The report is **write-once**, stored under `results/`, and uploaded as a workflow artifact. Regeneration must never modify original CSV/bin inputs, historical M0/predictive seals, or previous reports.

## Three separate conclusions

| Verification level | Exact decision criterion |
| --- | --- |
| Source and scoring compliance | Original v1.2 152 checks + all 13,704 rank/Shape parity + complete fair joint histogram hash |
| Universe primary optimum | Legal witness reaches the strict top-four-partition Shape maximum |
| Universe full lexicographic optimum | Legal witness also reaches per-class **coupled same-ticket Total** and conditional Joint upper bounds |

**Scope limitation:** Even a proven finite-score global optimum over all 6/49 tickets cannot establish more likely future draws under fair IID rules; an internal deficit-repair objective remains descriptive/decision-rule-based rather than predictive unless validated prospectively.

## Observed verified CI outcome — 2026-11-13

The initial full-universe CI run [43349868754](https://github.com/cctsao1008/lotto649/actions/runs/43349868754/job/131638033244) **PASS**, 10/10 adversarial and small-oracle tests. Its exact exhaustive output provides the following *measured* findings; final regression-locked CI reference is recorded below once available.

| Gate | Observed count or value |
| --- | ---: |
| Exact legal six-tuples exhaustively scanned | **13,983,816** |
| Sum/Log/Joint strictly positive, before Shape | **3,262,114** |
| All four gain blocks positive, before original eligibility exclusions | **2,384,983** |
| Frozen positive-qualified candidates under original strategy | **2,384,976** |
| Distinct positive-qualified Shape partition classes | **5** |
| Fifth-best Shape partition rank cap | **6,624** |
| Highest four distinct Shape partition rank caps | **13,504 / 13,128 / 9,553 / 7,364** |
| Exact Shape primary upper | **43,549** |
| Coupled per-class Total maxima | **51,521 / 51,145 / 47,570 / 45,381** |
| Exact Total Rank upper | **195,617** |
| Exact conditional Joint upper | **54,184** |
| Independently validated 24-disjoint research witness | **[Maximin 43,549; Total 195,617; Shape 43,549; Joint 54,184]** |
| Final lexicographic optimization gap | **0** |

**Mathematical proof:** Because the fourth-best partition's Shape cap 7,364 strictly exceeds the fifth 6,624, any portfolio whose Maximin reaches 43,549 must use one cap-attaining candidate in each of those four classes. Full-universe streaming evaluated every eligible candidate in each class and identified the maximum **coupled same-ticket** four-block rank. The sum of those four maxima is exactly 195,617; a legal four-ticket portfolio with 24 distinct numbers attains this total, as well as the primary and conditional Joint caps. Thus no better full-lexicographic portfolio exists **within the frozen four-positive-qualified selection universe**.

The streaming code also independently recomputed the **full 13,983,816-combination theoretical Sum×Log histogram** and obtained the original expected SHA256 `31136d04347ff7b84fa7b5d17e7d009068b784e46f43f1d49aa058bc7fcd422a`, independently confirming complete coverage of the same mathematical 6/49 support.

The complete-enumeration SHA256 over canonical six raw bytes is `962b66c4670f65aa7a13d48967a8ca125e8ded2cec5187b6d7c7cbcd71891a21`; the qualified-score-stream SHA256 is `2f2d9fe0f3b837072a5336985524c7e78b962f4ae55717ab7f9ec2b1da7442c4`. The initial report JSON SHA256 is `b644db7497985db0bd0646b9eeed407250fb8127723e0c8033f6fc9be633c202`.

**Critical scope:** This is a complete optimization certificate **given the original frozen 2,497-draw scoring rule, 13,704-reference percentile positions, four-positive screening and Shape/24-distinct-number constraints**. No claim about independent fair-draw probabilities, prospective prediction, or maximizing expected lottery payout is supported. This is a final objective/implementation milestone, not evidence of a statistically exploitable mechanism.

## Final regression-locked verification — 2026-11-13

The [final GitHub Actions run 43350778709](https://github.com/cctsao1008/lotto649/actions/runs/43350778709/job/131641047715) **PASS** with 10/10 unit/adversarial tests, 152/152 frozen v1.2 source and scoring checks, complete exact enumeration of **13,983,816** legal six-number combinations, matching full theoretical joint histogram SHA, and an unchanged **exact full-lexicographic optimum of (43,549, 195,617, 43,549, 54,184)**. The regression lock includes both canonical raw/qualified enumeration SHAs, every mandatory Shape class/score cap, and the exact integer **zero** gap.

The final workflow artifact **`strategy-v21-full-universe-audit` ID 14003851881** (30-day retention) contains `strategy_v21_full_universe.json`, SHA256 **`9daf72152925a7a873418155b3f194bfbdce6e4cd37174ddc52e5d68ab811b86`**. The fully verified canonical six-byte stream SHA is `962b66c4670f65aa7a13d48967a8ca125e8ded2cec5187b6d7c7cbcd71891a21`, and the frozen-rule qualified stream SHA is `2f2d9fe0f3b837072a5336985524c7e78b962f4ae55717ab7f9ec2b1da7442c4`.

**Additional all-positive 6/49 universe sensitivity proof:** The original immutable M0 eligibility rule has 12 fixed contract exclusions, of which only **seven** pass all four positive-gain filters in the full universe. A separate read-only counterfactual scored all these fixed exceptions without exclusion. It independently checked their partition/Shape cap, coupled same-ticket total and conditional Joint values and verified that **none can increase the best full-universe class upper bound**. Hence the exact tuple is unchanged **even if the original historical exclusion list is disregarded**, while the original strategy policy and source files stay unchanged. This establishes that the result is not a historical-exclusion artifact.

**Final disposition:** `EXACT_FULL_UNIVERSE_LEXICOGRAPHIC_OPTIMUM`; **primary optimal / total optimal / conditional Joint optimal**, with a legal 24-disjoint-number witness. Finite-score combinatorial strategy verification under the historical frozen objective **is now complete**. Neither this result nor its numerical rank improvements constitute evidence that a fair lottery will correct prior frequency deficits or that future jackpot probability has increased.

**Next scientific question (separate version):** Does the frozen strategy's compensation/geometry rank have any **prospectively validated** association with held-out draw outcomes versus a fair-IID and randomized-ticket benchmark? Predeclare loss/decision metrics, respect chronological training cutoffs, check expected payouts and multiple-comparison uncertainty, and report NO_GO if any apparent edge is indistinguishable from chance. No further score-maximization claim is necessary on this fixed universe.
