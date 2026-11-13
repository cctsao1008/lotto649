# Lotto649 v2.4 P2 — USNO Solar × Lunar 64-State Astronomical Hexagram Foundation

**Scope:** experiment design and deterministic source-audited data generation, **not** an investigation that demonstrates relationships or forecast performance. Issue [#35](https://github.com/cctsao1008/lotto649/issues/35).

## Dependencies and non-negotiable source boundaries

- Use **USNO only**: `data/astronomy/usno_official/raw/{moon_phases,seasons}/YYYY.json` (46 original authenticated-by-SHA source responses); `usno_events_2004_2026.csv` (1,276 normalized events); its manifest; 23 years 2004–2026.
- Reuse locked v2.4 P1 solar/year interpolation and USNO new-moon intervals (do not reinterpret prior P1 results): `lotto7x7/iching_gua_qi_v24_p1.py`, `data/research/v24_iching/v24_p1_gua_qi_draw_context_2498.csv`.
- Frozen P1 output CSV SHA-256: `53f35eb6fb0f6f5d6bc36e4b6a78aa63586263c924ba65940cc86c588c4fb1f1`. P1A tests reported **NO EVIDENCE** for association; P2 does not revise those results.
- Read-only frozen historical outcomes: v2.4 P0 `data/research/v24/v24_astronomy_hexagram_2498.csv` and `data/research/v24_usno/v24_usno_draw_context_2498.csv`. All original `data/csv`, `data/bin`, previously purchased tickets, all prospective seals, and P0/P1/P1A research snapshots remain unchanged.

## Formal v1 mapping (pre-registered, explanatory)

For each archived draw date, use `t = date 12:00+08:00` **by convention**; the lottery's actual historical draw clock is not supplied or inferred. All fractions/bins use USNO event instants in Taiwan local time. Interpolations use exact integer microseconds, **not rounded CSV floats**.

### Solar annual phase

P1's quarter-year solar anchors are: winter solstice center = `0/12`; spring equinox = `3/12`; summer solstice = `6/12`; autumn equinox = `9/12`.

In the known interval `[anchor_i, anchor_(i+1))`, with `a` the P1 center index (0, 3, 6 or 9), `e = t-anchor_i` and `d = anchor_(i+1)-anchor_i` in microseconds, calculate the circular phase

`solar_fraction = ((a*d + 3*e) mod (12*d)) / (12*d)`.

Define `solar_sector_8 = floor(8*solar_fraction)`, integer 0..7. This is **linear interpolation of temporal quarter-event intervals**, not the precise physical solar ecliptic-longitude bin nor traditional 24 節氣 or authenticated 卦氣.

### Lunar synodic time phase

Find successive USNO astronomical new-moon times `new_moon_i <= t < new_moon_(i+1)` and define

`lunar_fraction = (t-new_moon_i)/(new_moon_(i+1)-new_moon_i)`;

`lunar_sector_8 = floor(8*lunar_fraction)`, integer 0..7.

**Important:** the observed full moon does not necessarily coincide with the exact temporal midpoint. This is a *time-normalized new-moon interval proxy*, not measured illumination, phase angle, astronomical waxing/waning transition, or a traditional Chinese lunar date/month.

### Yin/yang six-line coding

For each 0..7 sector `k`, use three-bit **reflected cyclic Gray code** `gray(k) = k XOR (k >> 1)`; the sequence is `0,1,3,2,6,7,5,4`, with one-bit differences at every adjacent transition *including wraparound 7→0*.

- **Upper** three lines (bits 3,4,5 from bottom to top) = Gray code of **solar** sector.
- **Lower** three lines (bits 0,1,2) = Gray code of **lunar** sector.
- Six-bit integer `astronomical_hexagram_code = gray(lunar_sector) | (gray(solar_sector) << 3)`.
- Yin = 0 (broken line), yang = 1 (solid line); `astronomical_hexagram_binary_bottom_to_top` lists bits 0..5 **bottom-to-top**.
- Three-bit trigram labels from the binary code `0..7`: `坤 震 坎 兌 艮 離 巽 乾` (Kun, Zhen, Kan, Dui, Gen, Li, Xun, Qian); **not** a declaration of King Wen hexagram ordinal 0..63.
- The 64 astronomical codes are algebraically reachable (Gray is bijective); their distribution on fixed historical draw dates **is not inherently uniform**. Do not use 1/64 as an assumed conditional fair-draw event probability.

All sector boundaries are half-open `[k/8,(k+1)/8)`; exact integer boundary is assigned to the **following** sector. P2 64-state transition may flip more than one line when solar and lunar transitions coincide; the Gray adjacency guarantee concerns one 8-sector cycle at a time.

## Missing historical boundaries

Data begin in 2004. Previous 2003 winter solstice and new-moon context are unavailable. Solar and lunar state fields are left blank when the necessary prior and next event timestamps cannot be established from these original USNO records. Do **not** synthesize 2003 data, invent draw-clock observations, or fill missing state from winning numbers. Each frozen draw remains in the output with `solar_missing`, `lunar_missing`, `astronomy_pair_mapped`, and a missing reason.

## Data leakage separation

`solar_sector_8`, `lunar_sector_8`, their Gray encodings and `astronomical_hexagram_code` depend *only* on the fixed date/time convention and USNO ephemeris. They use **no current-draw winning numbers**. The source USNO event table was collected retrospectively; a historical “pre-draw known” claim means astronomical calculability independently of drawing outcomes, **not a timestamped prediction issued at the historic time**.

The six actual numbers, special, `same_draw_hexagram_code_OUTCOME_ONLY`, and `hamming_distance_OUTCOME_ONLY` are historical **post-draw labels**; no same-draw forecast may use them as inputs. The P1 parity encoding remains unchanged (six sorted main balls, `number%2`, bottom to top).

## Output format

- `data/research/v24_iching_p2/v24_p2_solar_lunar_hexagram_draw_context_2498.csv`: 2,498 draws with 8-sector features and outcome columns explicitly separated.
- `data/research/v24_iching_p2/v24_p2_solar_lunar_hexagram_audit_2498.json`: mapping version, 64-code histogram, sector histograms, coverage by year, frozen input SHA-256, output SHA-256, limitations and no forecast claim.
- `lotto7x7/iching_solar_lunar_v24_p2.py`, `tests/test_iching_solar_lunar_v24_p2.py`: stdlib-only implementation and exact-boundary tests.
- `.github/workflows/iching_solar_lunar_v24_p2_once.yml`: one-time CI/integration archive, exact read-only regression on rerun; no recurring new-draw downloads.

Windows PowerShell from repo root:

```powershell
cd E:\lotto649
git pull --ff-only origin main
$py = '.\.venv\Scripts\python.exe'
& $py -m unittest discover -s tests -p 'test_iching_solar_lunar_v24_p2.py' -v
# For read-only re-generation, choose a NEW output directory; never overwrite archived P2.
& $py -m lotto7x7.iching_solar_lunar_v24_p2 --out-dir results/v24_p2_local_recheck
```

## P2 → P2A statistical phase (separate authorization/research task)

P2 itself **does not test correlation**. If testing 64 sky states against actual numbers, totals, odd count, P1 shape taxonomy, or six-line Hamming distance, predefine a small number of tests, compare against exact fair 6/49 baselines, account for the **unequal calendar sampling** of sky states, and perform *year × weekday constrained whole-pipeline IID/permutation* plus multiple-testing corrections. With 2,498 draws, individual 64 categories are sparse; never claim effects from post-selected extremes. Reuse P1A NO EVIDENCE as a fixed previous research conclusion, not as permission for uncontrolled incremental hypothesis searches.
