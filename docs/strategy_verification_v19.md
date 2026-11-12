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
