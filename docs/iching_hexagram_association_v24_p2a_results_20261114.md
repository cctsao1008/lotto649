# Lotto649 v2.4 P2A — Solar/Lunar 64-Hexagram Association: Observed Result

**Status:** `EXPLORATORY_ASSOCIATION_NO_EVIDENCE` (engineering PASS; no predictive claim)

- Tracking issue: [#36](https://github.com/cctsao1008/lotto649/issues/36)
- Code/test/workflow commit: [`2ad56c1`](https://github.com/cctsao1008/lotto649/commit/2ad56c17539190afc3ff6dd539ac3a6cf2764798)
- Permanent data results: [`a4cd631`](https://github.com/cctsao1008/lotto649/commit/a4cd631074cedbc4013fdfbfed947549cc0cb0aa)
- [Successful GitHub Actions run](https://github.com/cctsao1008/lotto649/actions/runs/43469516373)
- [Full machine-readable result](../data/research/v24_iching_p2a/v24_p2a_hexagram_association_2498.json)
- [Descriptive 64-state table](../data/research/v24_iching_p2a/v24_p2a_hexagram_association_2498_states.csv)

## Frozen data and analysis

- Original P2 CSV: **2,498** historical draws through 2026-11-10 / #115000104, SHA-256 `a46f08269fc8021369f1a045116051e4f9591ba83801d5cdbe8c1a08873edf39`.
- **2,476** valid solar/lunar astronomical hexagrams; **22** missing early-2004 states are excluded without imputation.
- All **64** sky-code states were represented, between **32 and 57 draws per state**.
- Source astronomy: 46 archived USNO JSON responses, 1,276 events, date-level reference **12:00 Asia/Taipei**. Sky hexagram is a researcher-defined 8 solar-sector × 8 new-moon-time-sector Gray encoding, not traditional calendar divination.
- Outcome labels (six sorted-number parity, sum, odd counts, 7×7 L1 graph shape and Hamming distance) were independently recalculated from observed main numbers.
- **3,999** whole-outcome permutations within **112 calendar-year × weekday** strata; fixed random seed `240649`; plus-one empirical p-values; Bonferroni familywise correction across six global endpoints.

## Fixed six-endpoint results

| Endpoint | Permutation p | Bonferroni-adjusted p | Significant at familywise 0.05 |
| --- | ---: | ---: | --- |
| 64×64 astronomical-vs-outcome hexagram contingency | 0.05575 | 0.33450 | No |
| Mean six-bit XOR/Hamming distance, two-sided | 0.23400 | 1.00000 | No |
| Six-ball sum by 64 sky states | 0.36850 | 1.00000 | No |
| Odd-number count by 64 sky states | 0.68875 | 1.00000 | No |
| 7×7 L1 component-count mean by 64 sky states | 0.43450 | 1.00000 | No |
| 64×6 7×7 L1 component-count distribution | 0.13225 | 0.79350 | No |

The smallest unadjusted p is 0.05575; **after correction, 0.3345**. Thus no global endpoint supports a detectable association under this prespecified exploratory protocol. The sparse contingency matrix and historical sample size limit power. Non-significance is not a proof of independence, and it is not a forecast edge.

Additional measured diagnostics:

| Metric | Result |
| --- | ---: |
| Historical mean sum | 150.088045 |
| Historical mean odd count | 3.080775 |
| Historical mean 7×7 L1 components | 4.167205 |
| Historical mean sky-vs-outcome Hamming distance | 2.980210 |
| Exact fair six-number parity reference Hamming expectation for these sky states | 3.001624 |

**Important distinctions:**
1. The observed Hamming difference from a theoretical unconditional fair draw is a descriptive value. The P2A *inference* instead uses the exact year×weekday-conditional permutation null and a two-sided test.
2. The 64×64 categorical null was calibrated by permutation, **not asymptotic chi-square**, because most cells are sparse.
3. The exact fair sorted-parity hexagram mass over all `C(49,6)=13,983,816` tickets is **not uniform 1/64**.
4. Earlier P1A analyses used overlapping historical outcomes; P2A is retrospective/exploratory even though its six tests were frozen before this analysis.
5. Source lottery CSV/BIN, USNO original JSON, P0/P1/P1A/P2, previously bought tickets, and all prospective seals were **not modified**.

## Disposition

P2A association study is complete with the evidence label `EXPLORATORY_ASSOCIATION_NO_EVIDENCE`. No new recommended combination and no modification of existing bets. Any future claim of predictive performance needs an independently precommitted **future-only** protocol with a specific hypothesis and proper out-of-sample baseline; do not relabel this exploratory result as positive because one unadjusted p-value looked relatively close to 0.05.
