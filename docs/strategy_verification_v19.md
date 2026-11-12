# M0 Strategy Verification v1.9 — Score Saturation & Candidate Coverage

## Research question

In v1.7/v1.8, independent candidate samples converged to the same score tuple `(43549, 195617, 43549, 54184)`. Is that stability due to the *original fixed reference-position rank representation*, or evidence that the screened sample reaches the best feasible discrete-score objective? We must distinguish **empirical rank aliasing**, **finite shape-category limits**, **candidate coverage**, and **search completeness**. This study **does not** test lottery predictability or change its scoring rules.

## Immutable evaluation contract

- Verify original source history at **2,497 draws through 2026-11-06** and original **152** v1.2 replay checks.
- Hold the four M0 score functions and the **13,704-candidate original empirical rank axes** fixed. Independently recompute all **13,704** reference candidates' raw gains and exact integer ranks before analyzing any new sample. Fail closed if even one differs.
- Regenerate the v1.7 **three independent Seeds (649, 20261111, 20261112)** and **80k / 160k / 320k** progressively nested raw candidate budgets per seed. Compare identical 2,000-candidate caps for **ranked, shape_round_robin, uniform_screen**. New samples are scored using the original frozen 2,497-draw history, not post-cutoff outcomes.
- Independently compute unrounded 4-block gains for all qualified new candidates, checking exact `bisect_left` rank parity and original Shape category. A raw score **strictly above** the frozen maximum receives the same integer position rank `13704` regardless of magnitude; an observed score **equal to** the reference max must be reported separately because `bisect_left` may place it *below* `13704` when the top reference score repeats.
- For each source sample and screened 2,000 pool report reference maximum ties, unique gain values, strict-above-maximum counts, ceiling counts, and gain-alias buckets (different raw gains that share an integer rank). Do not treat ordinary equal-feature ties as quantization loss.
- Compute an additional **mathematically valid class-separable bound** on the full raw qualified sample, without solving an enormous integer program. If the best legal witness reaches the Shape primary bound and Shape's 4th partition cap strictly beats the 5th, every primary-optimal portfolio must draw a Shape-cap candidate from each of those four partitions. The sum of the per-partition maxima is then a safe upper bound for each remaining block. If the legal witness simultaneously saturates **all** four block upper bounds, certify complete lexicographic optimum for **that qualified sample**. Otherwise report `VALID_UPPER_GAP_UNRESOLVED`, never an unjustified exact optimum.
- Both the original v1.7 frozen-rank strategy and the raw-score diagnostic are kept separate. Individual score improvements are *not* evidence of forecast advantage. Avoid any attention to prior purchases.

## Mathematical distinction

For block `b`, let `G_b(t)` be the fixed standardized gain of ticket `t`, and let `A_b` be the original sorted empirical gain vector with `N=13704` entries. The frozen rank is

`R_b(t) = bisect_left(A_b, G_b(t))`, in `[0, N]`.

When `G_b(t) > max(A_b)`, `R_b(t)=N` and further raw-gain differences are invisible in **that** rank objective. When `G_b(t)=max(A_b)` the rank is the first maximum's index, not necessarily `N-1` if the reference top has ties. Other kinds of ties come from the finite sums, product bins and discrete Shape partitions and are not necessarily artifacts of sample truncation.

Assume a set of four uniquely highest Shape partitions `P` with caps `c_p`, and a legal four-ticket witness reaches `U_shape = Σ_{p∈P} c_p` in the primary maximin objective. Any other primary-optimal four tickets must use exactly those Shape classes, attain their maxima, and satisfy all original ball-disjointness/score constraints. For each non-Shape block, a valid upper bound is `U_b=Σ_{p∈P} max_{t in class p, ShapeRank(t)=c_p} R_b(t)`. This relaxation drops ball-disjointness, so `U_b` may overstate what is jointly feasible. If a witness simultaneously meets each `U_b`, it is **exactly complete-lexicographic optimal** over all source candidates; otherwise only a bounded gap is established.

## Acceptance tests

1. Unit tests independently distinguish reference-top equality from strict clipping, count different gains mapped to one rank, fail closed on inconsistent scores, and check exact vs loose class upper bounds.
2. CI read-only full-history source gate, source-ranked-score parity (13,704/13,704), deterministic seed/size/policy audit and frozen raw sample identifiers.
3. The JSON and documentation must explicitly label measured score saturation, class upper bounds and any unresolved optimization gaps. Never infer that integer rank ceiling proves a true theoretical optimum, physical draw bias, or increased chances of winning.
4. No change to historical CSV/BIN, sealed M0/P3f/P5A, legacy outcomes, original 2,000-pool or previous v1.5–v1.8 reports.

## Reproduce

```powershell
python -m unittest discover -s tests -p test_strategy_verification_v19.py -v
python -m lotto7x7.strategy_verification_v19 --out results/strategy_v19_score_saturation.json
```

The result file is write-once; use another filename for a new independent run. GitHub CI stores the complete JSON as an artifact. Research findings must be based on observed CI output and recorded separately, not presumed by this specification.

## Measured v1.9 results — 2026-11-12

[GitHub Actions job 131154891153](https://github.com/cctsao1008/lotto649/actions/runs/43210005033/job/131154891153) **PASS**. Eight of eight regression tests, and 27/27 deterministic policy analyses after **152/152** original M0 independent source/replay checks. All original 13,704 reference candidates' scores, Shape classes and integer ranks were independently recalculated and matched. Artifact **`strategy-v19-frozen-rank-saturation`**, ID **13940408516** (30-day retention), `strategy_v19_score_saturation.json` SHA256 **`6e29e86480026b98db2c1b010184a1a187fdf7c58063339a5dd9d1d571e55b00`**.

### 1. Frozen reference resolution is low; this is mostly native feature discreteness

| Axis | Distinct raw standardized gain values among 13,704 frozen candidates | Candidates tied at the **maximum raw gain** | First integer rank of that maximum |
| --- | ---: | ---: | ---: |
| Sum | **115** | **432** | 13,272 |
| Log Product | **17** | **1,394** | 12,310 |
| Joint Sum×Log | **355** | **158** | 13,546 |
| Shape | **16** | **200** | 13,504 |

This confirms many different six-number tickets legitimately share the **same calculated feature gain** even before percentile ranking; the low feature resolution is **not** evidence of a bug. The highest reference gain is a tied score (particularly Log and Shape), and `bisect_left` properly maps all exact ties to its first index, not rank 13,704.

### 2. No observed strict beyond-reference maximum clipping

For all **nine** distinct Seed×raw-budget streams (80k, 160k, 320k for Seeds 649 / 20261111 / 20261112), **zero** of the new positive-qualified candidates in any of the four axes had a raw gain greater than that axis's original reference maximum, and **zero** had the maximum integer rank of 13,704. This observation rules out **above-maximum empirical-rank clipping** as the observed cause of the v1.7–v1.8 quality plateau **within these nine sampled batches**. It is **not** a proof for all 13,983,816 legal six-number combinations.

However, some **different** new raw scores map to the **same** existing reference rank, even below the maximum, when they fall between the same two reference score values. This is genuine rank quantization, distinct from the dominant exact-feature ties:

| Seed | Raw unique draws | Qualified candidates | Sum candidates in different-gain/same-rank buckets | Joint candidates in different-gain/same-rank buckets | Log / Shape different-gain rank aliases |
| --- | ---: | ---: | ---: | ---: | --- |
| 649 | 320,000 | 54,651 | 170 | 1,549 | 0 / 0 |
| 20261111 | 320,000 | 54,333 | 146 | 1,228 | 0 / 0 |
| 20261112 | 320,000 | 54,179 | 171 | 1,205 | 0 / 0 |

No raw-gain differences were collapsed by the *empirical ceiling itself* in these samples; the small rank-alias groups are due to a finite reference score grid. The report contains every per-axis value for all nine sample budgets and every 2,000-screened policy.

### 3. A valid new **full qualified-source** bound is still not tight

For the 160k and 320k samples of all three seeds, the ranked/Shape-round-robin screened portfolios attain the known primary Maximin 43,549 and the best feasible tuple `(43549, 195617, 43549, 54184)`. The mathematically safe **class-separable integer** envelope over *all* positive-qualified new candidates is:

```text
                     Sum      Log     Joint    Shape
Actual legal witness  48,644   49,240   54,184   43,549
Classwise upper        53,088   49,240   54,184   43,549
Remaining upper gap     4,444       0        0        0

Feasible Total rank  195,617
Safe Total upper     200,061
Uncertified gap <=     4,444
```

The primary Shape limit of 43,549 is attained. The maximum Log, Joint and Shape sums are also attained by a legal four-ticket witness. The conservative Sum upper 53,088, however, is **4,444 above** the witness's Sum 48,644. Since each block's independent maximum might require mutually incompatible candidate choices or overlapping ball numbers, the 4,444 difference is **only a relaxation gap**, not proof a better portfolio exists.

**Result:** `VALID_UPPER_GAP_UNRESOLVED` for the full newly qualified sources. This cannot supersede the **v1.8 exact proof** that the four specifically audited 2,000-ticket pools have the full lexicographic optimum 195,617. The bigger raw-qualified source populations were **not exhaustively enumerated**, and their full second-objective maximum remains unproven.

### 4. Engineering conclusion

- **Strategy Compliance:** PASS, including all original immutable historical/scoring checks and complete new-sample raw-gain-to-rank parity.
- **Score Representation:** Native discrete feature scores create massive legitimate ties; **strict > reference max clipping observed: none** in tested samples.
- **Rank Quantization:** Some previously unseen gains map to existing rank buckets (Sum and Joint), but this affected a relatively small fraction of scored candidates at 320k raw budget.
- **Candidate Coverage:** Independent-seed candidate generation still demonstrably improved the original 2,000-pool exact strategy objective by **2,902 Total Rank units**, a mathematical score-only improvement, not higher next-draw odds.
- **Optimization Certification:** v1.8 finite 2,000-pool exact optima retained. The full 13k–54k qualified samples have only valid 4,444-point **conservative upper gap**, **not** exact optimality.
- **Research status:** `V19_SCORE_SATURATION_DIAGNOSIS_PASS / FULL_NEW_SOURCE_SECONDARY_OPTIMUM_NOT_YET_CERTIFIED`.

**Next strongest research step:** combine the class-separable Sum upper with **all 24-ball disjointness and minimum Joint/Sum constraints**, to close the remaining full-source gap or produce a strict feasible improving witness. A second independent approach could compare exact continuous raw-gain ordering **only inside equal frozen-rank classes** as a *separate versioned diagnostic*, not a retrospective score change. No original data, source manifest, or sealed legacy experiment was modified.
