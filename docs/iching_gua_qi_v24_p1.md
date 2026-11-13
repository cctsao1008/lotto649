# Lotto649 v2.4 P1 — Twelve Gua Qi seasonal states from USNO astronomy

**Issue:** [#33](https://github.com/cctsao1008/lotto649/issues/33)  
**Status:** deterministic astronomical-to-I-Ching mapping and 2,498-draw historical audit; research encoding only, **not a prediction claim**.

## Exact data sources and provenance

* USNO API: `data/astronomy/usno_official/raw/` (46 original JSON files) + `usno_events_2004_2026.csv` (1,276 event rows); all 46 raw SHA-256 digests are validated against the archived USNO manifest.
* Pre-existing frozen date-only view: `data/research/v24_usno/v24_usno_draw_context_2498.csv`.
* Original post-draw outcome view: `data/research/v24/v24_astronomy_hexagram_2498.csv` and its report. Original source CSV and BIN are **not changed**. These original views have their respective SHA-256 values checked before generating output.
* All time instants are converted from USNO UTC to explicit UTC+08:00. **Reference clock is 12:00 local on the historical date**; the actual lottery draw time has not been verified. No local meteorological data or lunar-month labels are used.

## Twelve seasonal states: research assumption, not traditional precise calendrical calculation

The traditional **twelve waxing/waning hexagram names** (`復 臨 泰 大壯 夬 乾 姤 遯 否 觀 剝 坤`) are attached to an **experimental deterministic solar segmentation**:

1. Read successive USNO `equinox`/`solstice` event instants. Because USNO uses generic labels, use their month and event type to distinguish March spring equinox, June summer solstice, September autumn equinox, December winter solstice. Reject any unknown or inconsistent event.
2. Set **winter solstice = center of state 0 復**, **spring equinox = center of state 3 大壯**, **summer solstice = center of state 6 姤**, **autumn equinox = center of state 9 觀**. Each adjacent quarter pair spans precisely three nominal state centers, linearly *in elapsed time* between actual USNO timestamps.
3. For adjacent quarter anchors at `A` and `B`, let `u=(reference_at_taipei-A)/(B-A)` with `0<=u<1`. Let `k=floor(3*u + 0.5)`, where exactly half-state boundaries advance to the next state. The seasonal state index is `(anchor_index + k) mod 12`. Integer microsecond arithmetic is used for the boundary selection.
4. `solar_phase_fraction = ((anchor_index + 3*u) mod 12)/12`, measured from the winter-solstice **center**, not a Chinese lunar month or the start of a solar term.
5. For missing neighboring anchors, **do not extrapolate**: report an empty state and explicit missing flag. In particular, early 2004 cannot be bracketed because the USNO archive starts in 2004 and contains no December 2003 winter solstice. Later than December 2026 winter solstice also lacks the following 2027 spring equinox in this snapshot.

**Important:** This fixed-month-associated 12-state mapping only approximates the traditional 12 消息卦 sequence and does **not** assert historically exact 卦氣 boundaries. USNO contains **four** equinox/solstice events per year, not 24 Chinese solar terms. The interpolation assumes uniform progress **in time between quarter points**, not uniform solar longitude. Different mapping conventions require a distinct version and independent tests; no tuning against winning numbers is allowed.

### Six-line coding (unambiguous)

* Bit 0 is the **bottom** line; bit 5 is the **top** line. `1=陽（solid）`, `0=陰（broken）`.
* Named seasonal index → six-bit integer (`bottom-to-top`):

| Seasonal index | 卦 | Code | Yang lines |
| ---: | --- | ---: | ---: |
| 0 | 復 | 1 | 1 |
| 1 | 臨 | 3 | 2 |
| 2 | 泰 | 7 | 3 |
| 3 | 大壯 | 15 | 4 |
| 4 | 夬 | 31 | 5 |
| 5 | 乾 | 63 | 6 |
| 6 | 姤 | 62 | 5 |
| 7 | 遯 | 60 | 4 |
| 8 | 否 | 56 | 3 |
| 9 | 觀 | 48 | 2 |
| 10 | 剝 | 32 | 1 |
| 11 | 坤 | 0 | 0 |

The `code` is **not** a King Wen hexagram sequence number. The code from Lotto's original sorted six numbers (odd=1/even=0) remains in a clearly named `OUTCOME_ONLY` field. The same-draw Hamming distance between the historical outcome and seasonal code is *diagnostic only*, **never** a pre-draw input.

### USNO lunar phase fraction (non-hexagram P1 covariate)

Between consecutive *astronomical new moons* from USNO, calculate `moon_phase_fraction = elapsed_time / actual_lunation_time`. This is a continuous normalized astronomical phase in `[0,1)`, not a traditional agricultural/lunar date, leap month, and **not the experimental solar-8 × lunar-8 64-hexagram coding planned for P2**. Missing previous/next new moon events produce blank fractional values.

## Distinct information layers

| Fields | Source knowledge | Permitted use |
| --- | --- | --- |
| `date`, `draw_id`, `reference_at_taipei` | Frozen historical labels/date | Read-only identifiers |
| `seasonal_*`, `solar_phase_fraction`, `moon_phase_fraction` | Fixed USNO astronomical times + deterministic date conversion | Can be computed independent of winning numbers |
| `n1`–`n6`, `bonus`, `same_draw_hexagram_code_OUTCOME_ONLY`, `hamming_distance_OUTCOME_ONLY` | Historical winning outcomes | Targets/diagnostics **only**, not same-draw predictors |

The historical draw schedule, weekday and year could be correlated with astronomical phase; future scientific evaluation must use causal splits, matched fair-IID nulls, multiplicity control and a genuinely prospective seal. The P1 artifact reports **no predictive gain**.

## Reproduce on Windows (no extra dependencies)

```powershell
cd E:\lotto649
$py = '.\.venv\Scripts\python.exe'
& $py -m unittest discover -s tests -p 'test_iching_gua_qi_v24_p1.py' -v
& $py -m lotto7x7.iching_gua_qi_v24_p1 --out-dir results/v24_iching_p1_local
```

For local use, test ZIP-dependent cases skip if `_usno_official_test.zip` is not present, but the CLI always re-verifies all 46 USNO JSON files and all SHA-256 source gates. The dedicated one-time GitHub Actions job builds the transient ZIP itself and executes all 15 tests, and then creates the *versioned, immutable* `data/research/v24_iching/` CSV and audit JSON.

Do not write over pre-existing output files. The CLI uses `open('x')` for final artifacts. The one-time workflow only stages the two P1 artifacts; no original data, original research seal, or current ticket is rewritten.

## Next research stage (separate version)

P2 (solar 8 × lunar 8 → 64 astronomical states) requires predeclared trigram and Gray-code conventions; P3 tests distributions, Hamming distances and shapes with the **exact** non-uniform combinatorial baseline of lottery-derived hexagram codes. P1 alone does not infer a relation between celestial states and the random lottery outcome.
