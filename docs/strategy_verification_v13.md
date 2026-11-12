# Lotto649 Strategy Verification v1.3 — M0 Bounded Optimization Gap & Decision Explanation

**Scope:** #32, completed frozen v0.28 M0 and the v1.2 independent verification, without reselecting, changing, or recommending any purchased ticket. This work evaluates **correct execution and optimization quality of a predetermined strategy**, not whether future lotteries must compensate historical distributions.

## Audited input and immutable evidence

The v1.2 module independently reproduced the original M0 prior-history data cutoff **2,497 draws, 2026-11-06/#115000103**, all **13,983,816** fair combinatorial outcomes, original exact Sum×Log joint and Shape L1/L2/L3 reference fingerprints, **80,000** sampled six-number candidates, **13,704** passing all four positive-gain constraints, final **2,000** candidate pool, and all three original heuristic portfolios. The new v1.3 module calls that v1.2 gate first and uses an **optional read-only export** of the *same verified 2,000 rows* (ticket, Shape partition code, integer rank of each of four blocks). `include_verified_pool=False` remains the v1.2 default and preserves its original output schema.

The original four blocks are: **Sum**, **Log Product**, **Joint Sum×Log Bin** and **Shape** (internally weighted L1=.25/L2=.35/L3=.40). Each ticket has a **rank vector** computed from the 13,704 qualified candidates, not probability estimates or independent likelihoods. The original balanced portfolio lexicographically maximizes `(min(rank_sums), sum(rank_sums), shape_rank_sum, joint_rank_sum)` in its finite seeded heuristic. `shape_first` and `joint_first` have different primary priorities and are reported separately.

## Formal optimality-gap certificate — only within the frozen pool

Let \(x_i\in\{0,1\}\) choose the original qualified candidate i from the verified pool of 2,000, \(r_{ik}\) be its **integer percentile rank** for block \(k\), and let \(z\) denote the four-block minimum. The integer maximin relaxation is

\[
\max z,\qquad \sum_i x_i=4,\quad \sum_{i:b\in C_i}x_i\le1 \ (b=1,\ldots,49),\quad
\sum_{i:\mathrm{part}(C_i)=p}x_i\le1,\quad
z\le \sum_i r_{ik}x_i\quad(k=1,\ldots,4).
\]

All candidate tickets already have strictly positive four-block gain, and old-ticket exclusions were applied by the verified v1.2 qualification pipeline. The 49 number constraints imply 24 distinct numbers; Shape partition constraints enforce four different classes. The **LP relaxation** permits \(0\le x_i\le1\), so its optimum is an upper bound on feasible integer maximin.

The implementation **does not simply trust floating-point LP optimality**. It takes LP-derived inequality dual weights \(\mu\ge0\) and equality weight \(\lambda\), rounds each to a fixed **1/100,000,000** rational denominator, and independently evaluates the following exact integer Lagrangian bound with a finite box \(0\le z\le Z\):

\[
U = \sum_{j\in \mathrm{number,partition}}\mu_j + 4\lambda
+ \sum_{i} \max\{0,\sum_{k}\mu_{k}^{\rm block}r_{ik}-\sum_{b\in C_i}\mu_b-\mu_{\mathrm{part}(i)}-\lambda\}
+ Z\max\{0,1-\sum_k\mu_k^{\rm block}\}.
\]

This is a valid upper bound **for any nonnegative inequality dual weights** and any real equality multiplier; therefore its integer implementation gives an **exact rational** bound even if an LP solver's floating-point solution is imperfect. Since the primary four-block rank total minimum is integer, \(\lfloor U\rfloor\) is a certified upper bound on the original **fixed-pool primary maximin objective**. Every nonzero integer dual weight and the exact rational numerator/denominator are included in the report so they can be independently checked. The primary feasible incumbent is the original **43,549** rank total. Thus a certified finite-pool gap is at most \(\lfloor U\rfloor-43,549\), never a claim about all possible lottery tickets.

An optional **time-bounded mixed-integer** search may find a stronger feasible lower bound or certify optimality, but must respect the exact same 24-disjoint-ball and 4-partition constraints. A stricter feasible primary rank total > 43,549 would **falsify** global optimality of the original finite-pool heuristic while leaving its correct algorithm execution and original frozen files intact. The optional search does not reproduce its lexicographic secondary objectives; only its primary maximin can be compared directly.

## Ticket-by-ticket decision explanation

For Primary, Shape-first and Joint-first, the report includes:

- Each ticket's original four integer rank contributions; original selection, four rank totals and active bottleneck(s).
- Every candidate that could replace **one different** ticket without breaking disjoint numbers or distinct Shape partitions, and the best feasible alternative.
- Whether any such one-ticket replacement strictly improves the appropriate **full lexicographic objective**. Local stationarity is weaker than a global integer-optimality certificate and is reported separately.

The selection narrative uses the versioned ranking contract rather than the next lottery outcome. Fair-IID chance and payout are not used as score weights. A future draw does not retroactively change compliance.

## Expected engineering states

- `M0_STRATEGY_EXPLANATION_AND_GAP_AUDIT_PASS` means the v1.2 original data and finite pool gate passed, the exact integer upper bound was formed, and original portfolio explanations were consistently checked.
- `EXACT_MAXIMIN_OPTIMUM_CERTIFIED_IN_FIXED_POOL` is reported only when the exact upper equals the original feasible lower; even then the *secondary lexicographic priority* is not implied.
- `ORIGINAL_HEURISTIC_STRICTLY_SUBOPTIMAL_IN_FIXED_POOL` is reported if an **independently verified integer feasible** portfolio within the same 2,000 candidate pool has strictly larger primary min rank.
- Otherwise `BOUNDED_GAP_UNRESOLVED_IN_FIXED_POOL` reports the exact worst-case difference and the primary 43,549 lower bound. It is **not** a failure of strategy compliance.

All results are **descriptive engineering audits**. No global-optimum claim across the full universe, no prospective predictive GO, no edits to the original M0 manifest, existing P3f/P5A seals, historical CSV/BIN, or paid tickets.

## Run

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m unittest discover -s tests -p 'test_strategy_verification_v13.py' -v
if ($LASTEXITCODE -ne 0) { throw 'v1.3 unit checks failed' }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
& $py -m lotto7x7.strategy_verification_v13 --mip-seconds 15 --out "results/strategy_v13_$stamp.json"
if ($LASTEXITCODE -ne 0) { throw 'v1.3 exact gap audit failed' }
```

CI executes the same exact original source audit, finite-pool reconstruction and integer upper certificate, then retains the report as a GitHub Actions artifact. The optional time-bounded MIP exploration is engineering support, not a frozen future-bet recommendation.

## First audited CI findings — original 2,000-candidate pool (2026-11-12)

Initial full-history [GitHub Actions run 43166416254](https://github.com/cctsao1008/lotto649/actions/runs/43166416254/job/131009225463) **PASS**, six original v1.3 tests plus source-verified 80,000-draw-candidate replay. Independently computed original **43,549** versus exact scaled-integer Lagrangian bound **43,549**, so the original **primary minimum-of-four-rank-sums** objective is globally optimal **within the original selected 2,000-candidate pool** (gap zero). A separate bounded mixed-integer solve also reached 43,549 (MIP optimum status), but its alternate tie-break rank vector need not be original M0's lexicographic optimum.

**A simpler, fully combinatorial independent proof** is also available and included in the revised audit report: with four distinct Shape partitions required, select at most one candidate from each partition. The original Primary ticket Shape ranks are **7,364**, **13,128**, **13,504**, and **9,553**. These are the highest Shape-rank contributions across the four highest permitted partition groups. Their sum is **43,549**. Since the minimum of four block sums is no greater than the Shape rank sum, no feasible selected four tickets in this pool can exceed 43,549. The original Primary sum vector **[47,199, 49,240, 49,829, 43,549]** attains this upper bound exactly.

This exact result resolves the **primary maximin gap only**. The original lexicographic secondary objectives—sum of all rank totals, Shape sum and Joint sum—have not yet received an independently certified optimality bound, though their execution and one-for-one local stationarity have been independently reproduced. It is also possible that a wider pool yields higher scores; the 80,000 sampled combinations were not an exhaustive search of every lottery ticket.
