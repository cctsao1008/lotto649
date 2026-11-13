# Lotto649 v2.4 P1A — USNO Gua Qi historical association results (2026-11-13)

## Decision

**EXPLORATORY_ASSOCIATION_SCREEN_NO_EVIDENCE.** On the four frozen tests,
no seasonal Gua Qi association survives the four-family Bonferroni correction
at α=0.05. This is a retrospective observational null result, **not** a
proof of impossibility, a physical I Ching finding, or evidence of lottery
predictability.

## Reproducibility and scope

- Code: `lotto7x7/iching_gua_qi_stats_v24_p1a.py`; methodology:
  `docs/iching_gua_qi_v24_p1a_stats.md`.
- Research source: `data/research/v24_iching/v24_p1_gua_qi_draw_context_2498.csv`
  frozen SHA256 `53f35eb6fb0f6f5d6bc36e4b6a78aa63586263c924ba65940cc86c588c4fb1f1`.
- Input: 2,498 original historic draws through **2026-11-10**, 2,476 with
  USNO 12-state assignments; 22 early 2004 rows missing 2003 boundary.
- Research reports:
  `data/research/v24_iching/v24_p1a_gua_qi_association_2498.json` and
  `data/research/v24_iching/v24_p1a_gua_qi_association_2498_states.csv`;
  state CSV SHA256 `9577c8cf21511577658b724ac63e4f644836ea9aab09dc06e877481e4fe68fff`.
- GitHub Actions: [run 43406897791](https://github.com/cctsao1008/lotto649/actions/runs/43406897791) **PASS**;
  [source/code commit 128673a](https://github.com/cctsao1008/lotto649/commit/128673a9e82b8bfaf422be6b7525c34bf1a9e026);
  [archived outcome commit fcf501e](https://github.com/cctsao1008/lotto649/commit/fcf501e8c0542e7e0b89f4a723751272345dc82e).
- **3,999** deterministic permutations with `seed=649`, **112**
  `(year, ISO weekday)` strata, four predeclared global tests. Conditional
  label permutation preserves same-ticket feature correlations and
  year/weekday-specific observed outcomes. Bonferroni across 4 tests.
- Fair 6/49 exact baseline: sum mean 150, variance 1075; odd mean
  3.06122449, variance 1.3431903374; L1 component mean 4.165667369;
  shape category census `C1..C6` counts `[58620,669462,2775454,
  5093678,4166304,1220298]` in `C(49,6)=13,983,816` combinations.

## Four tests

| Predeclared global response | Observed statistic | Permutation p | Bonferroni p | Retrospective conclusion |
|---|---:|---:|---:|---|
| Sum mean heterogeneity | 11.824523 | 0.46325 | 1.000 | No evidence |
| Odd-number count mean heterogeneity | 14.821032 | 0.17900 | 0.716 | No evidence |
| Shape L1 component mean heterogeneity | 10.340654 | 0.55350 | 1.000 | No evidence |
| Shape L1 full 12×6 contingency | 45.073646 | 0.82075 | 1.000 | No evidence |

The four observed statistical effects are small: `η²(sum)=0.0047335`,
`η²(odd)=0.0059286`, `η²(L1 component)=0.0039169`;
`Cramer's V(L1 12×6)=0.06034`.

## Seasonal means and descriptive fair baselines

These are descriptive figures, **not 12 independent hypothesis tests**.

| Gua Qi | n | Mean Sum | Mean # Odd |
|---|---:|---:|---:|
| 復 | 187 | 148.791 | 3.059 |
| 臨 | 229 | 149.170 | 2.996 |
| 泰 | 256 | 148.121 | 3.160 |
| 大壯 | 195 | 153.087 | 3.097 |
| 夬 | 203 | 154.276 | 3.182 |
| 乾 | 203 | 149.039 | 3.108 |
| 姤 | 205 | 150.302 | 3.122 |
| 遯 | 205 | 152.468 | 3.249 |
| 否 | 206 | 152.583 | 2.922 |
| 觀 | 200 | 147.895 | 3.025 |
| 剝 | 196 | 146.974 | 2.974 |
| 坤 | 191 | 148.712 | 3.058 |

The apparent contrast between **夬 (mean sum 154.276)** and **剝 (146.974)**
is a retrospective maximum chosen from twelve categories, and cannot be used
as an unadjusted significance claim or ticket-selection advantage. The
correct predeclared Sum seasonal association has `p_adjusted=1.000`.

## Boundaries and next study

1. The twelve messages are a deterministic *research approximation* of the
   traditional Gua Qi ordering, not an authenticated historical 24-solar-term
   mapping, and no USNO events are treated as lunar dates or leap months.
2. Exact lottery theory and permutation-based association inference answer
   different questions. The latter conditions on 2,476 actual observed
   outcomes and is sensitive to the exchangeability assumption within each
   calendar-year/weekday block. It does not solve all technical bias, weekday,
   epoch or selection concerns.
3. No pre-registered prospective evidence of predictive gain was collected.
   A next P2 study may freeze a new **solar-8 × lunar-8 astronomical code**
   with a defined I Ching trigram convention and test it *without* optimizing
   the mapping against this historical sample. That is new research, not a
   reinterpretation of P1A as a success.
4. This study never modifies purchases, original 6+bonus draw records, P1
   astronomy state seals, USNO JSON, v2.3 pre-draw seals, or any frozen M0 work.