# v0.26 P1 — Independent S4 Burnside / Partition-Lattice Cross-Check

Tracking: [Issue #8](https://github.com/cctsao1008/lotto649/issues/8), exact four-ticket coverage work in [original v0.26 certificate](coverage_exhaustive_certificate_v026.md).

## Question

The original exhaustive combinatorial certificate enumerates **52,450** labeled
nonnegative 15-mask structures, removes **1,465** with at least two identical
six-number tickets, and canonicalizes the remaining **50,985** labeled valid
structures under all 24 permutations of the ticket positions. It reports
**2,957** distinct unlabeled structures, scoring each against the official
six-main-plus-special eight-tier outcome model.

Can an **independent mathematical counting algorithm**, which neither
enumerates/canonicalizes those 52,450 structures nor uses the official
prize-scoring engine, reproduce the structural class counts?

## Method

For four ticket positions, label each nonempty membership pattern with
a mask \(m\in\{1,\ldots,15\}\) and integer count \(c_m\ge 0\), satisfying

\[
  \sum_{m:m\text{ has bit }i} c_m=6,\qquad i=0,1,2,3.
\]

The permutation group \(S_4\) of order 24 permutes ticket positions.
For a fixed permutation \(g\), each orbit of its action on the 15 masks
shares a common count \(q_O\); this reduces the fixed-point problem to
a small **four-dimensional nonnegative integer dynamic program**.
To require **four distinct tickets** and exclude structures where any
ticket rows coincide, use Möbius inversion over the **15 set partitions**
of four ticket positions:

\[
  F_{\rm distinct}(g)=
  \sum_{\pi\in\Pi_4}
    \left(\prod_{B\in\pi}(-1)^{|B|-1}(|B|-1)!\right)
    F(g,\pi),
\]

where \(F(g,\pi)\) is the number of degree-six count assignments fixed
by \(g\) in which all tickets in the same block of \(\pi\) are identical.
An orbit is forbidden under \(\pi\) if any mask in that orbit distinguishes
two ticket indices in the same block.

Finally apply **Burnside's lemma**:

\[
  \#\text{valid unlabeled structures}
  =\frac{1}{24}\sum_{g\in S_4}F_{\rm distinct}(g).
\]

The implementation evaluates every one of the 24 permutations and verifies
that permutations belonging to the same conjugacy class have equal
fixed-point counts.

## Exact fixed-point results

| Cycle type | Group multiplicity | Valid structures fixed per permutation | Contribution |
| --- | ---: | ---: | ---: |
| 1+1+1+1 | 1 | 50,985 | 50,985 |
| 2+1+1 | 6 | 2,467 | 14,802 |
| 2+2 | 3 | 1,309 | 3,927 |
| 3+1 | 8 | 114 | 912 |
| 4 | 6 | 57 | 342 |
| **Total** | **24** | — | **70,968** |

\[
  70,968/24=\boxed{2,957}.
\]

Also, the identity permutation with no ticket equality restrictions yields
exactly **52,450** feasible labeled structures; Möbius subtraction removes
**1,465** duplicate-ticket structures, leaving **50,985**.

These exact integer counts independently reproduce the original
classification. Crucially, the new algorithm has **no import of the
original exhaustive enumerator or canonicalizer**, no full signature
listing and **no prize-probability scoring**.

## Scientific scope

- **Confirmed by independent algorithm:** completeness of the original
  2,957 structural-class count and exclusion of identical-ticket cases.
- **Not independently confirmed here:** the original per-class
  official-prize scoring engine and its global maximum. That global optimum
  remains certified by the original exhaustive scoring computation and its
  separate micro-universe / single-ticket checks, not by this count alone.
- **Not claimed:** a forecast signal, higher probability of a specified
  six-number jackpot, better monetary return, or a reason to alter any
  precommitted tickets.

This cross-check reduces dependence on a single S4 canonicalization
implementation but does **not** complete all of Issue #8's remaining
multivariate covariance, Pareto-selection and prospective validation tasks.

## Reproducibility

From the repository root with Python 3.12, standard library only:

\`\`\`powershell
python -m unittest discover -s tests -p 'test_coverage_burnside_v026.py' -v
python -m lotto7x7.coverage_burnside_v026 --out results/v026_burnside_independent.json
\`\`\`

The output report is write-once under gitignored results/; no original
history CSV/BIN, signed source metadata, exact-prize catalog, paid
ticket records, or prospective seals are touched.
