# Lotto7x7 v0.6 Phase 1 — Shape Engine and Candidate Pool

## Goal and non-goals

The research object is **the 7×7 six-active-pixel image**. First identify an
image shape; then generate its legal spatial placements; then build a fixed-size
pool of distinct six-number combinations. Finally, an optional **uniform random
1–5 candidate preview** shows what a small practical selection looks like.

Phase 1 does **not** implement the v0.6 multi-objective optimizer,
historical rolling backtest, or AI shape policy. The preview is not a purchase
recommendation and not evidence of above-chance forecasting. The old v0.2–v0.5.1
image models and historical data are untouched.

## Mathematical definitions

- A draw image is a binary matrix in `{0,1}^(7×7)` with **exactly six ones**.
- A shape is the unordered set of six `(row,column)` lattice coordinates.
- A **shape family** is the equivalence class of a shape under translations,
  D4 rotations (multiples of 90°), and reflections, with no toroidal wrapping.
  A canonical key selects the lexicographically smallest normalized point tuple
  across the distinct D4 orientations; normalization subtracts the minimum row
  and column. The key is stable across translations and all D4 transforms.
- An **orbit** is the set of legal, distinct placements of the canonical shape
  within the 7×7 board. `orbit_size()` counts them without materializing all
  placements. `enumerate_placements()` independently generates the complete
  orbit and verifies its cardinality. The family key is retained alongside
  a short SHA256-derived ID.
- The sequential board mapping is the only one used for Phase 1: 1..7 on row 1,
  8..14 on row 2, ..., 43..49 on row 7. Different board mappings affect shape
  membership; this is **not** the physical geometry of the lottery machine.

A 2×3 six-pixel rectangle has exactly 60 legal placements: 30 horizontal and
30 vertical. Six consecutive horizontal pixels have 28 legal placements:
14 horizontal and 14 vertical. Transformations cannot wrap past the edges.

For each valid `K`-combination pool with no repeated six-number set, the
exact-six chance under a fair independent 6/49 draw is simply

`K / C(49,6)`, with `C(49,6)=13,983,816`.

This is a *reference probability*, not a measured Lotto7x7 forecasting gain.
Historical shapes have different orbit sizes; drawing shape families uniformly
is *not* the same as drawing six-number combinations uniformly.

## Candidate pool construction (Phase 1)

1. Read the existing verified BIN v2 (`draw_id + 7x7 bitmap`), ascending draw
   order, **only through the supplied cutoff ID**. Build a per-family index with
   first/last observed ID and frequency. No future results are used.
2. Choose `--families 20` distinct historical shape families, without replacement.
   `--family-policy` explicitly chooses among:
   - `uniform-family` (default; equal chance of selecting each observed family),
   - `orbit-weighted` (proportional to each family's legal placement count),
   - `historical-frequency` (proportional to observed frequency; exploratory).
3. Independently permute each selected family's complete legal orbit with a
   seeded NumPy RNG, then round-robin one new placement per family until exactly
   `--size 100` **distinct** six-number combinations have been produced.
   If the chosen families cannot provide the requested size, reject the run.
4. Emit full canonical shape bitmaps, orbit sizes, transformation metadata,
   candidate six-number sets, their bitmaps and source family ID, input cutoff
   ID, input SHA256 and an ordered candidate-pool SHA256.
5. Optionally sample `--sample-count 1..5` **distinct** candidates uniformly
   from the resulting pool using an *independent* preview RNG; do not classify
   these previews as optimized bets. Larger sample counts are rejected.

`--seed`, input bytes, policy and parameters fully determine the result. IDs
and generated combinations are sorted before hashing/serialization where
applicable. No real-money transaction is implemented.

## Windows usage

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'

git pull --ff-only origin main
& $py -m unittest discover -s tests -v

& $py -m lotto7x7 shape-pool `
    --size 100 `
    --families 20 `
    --family-policy uniform-family `
    --seed 649 `
    --sample-count 5 `
    --out results/shape_pool_v06.json `
    --html-out results/shape_pool_v06.html

Invoke-Item .\results\shape_pool_v06.html
```

The HTML is standalone and offline: 20 canonical shape previews, 100 candidate
bitmaps with their numbers, and five (or fewer) **random** sample sets. The JSON
is the machine-readable input for later v0.6 selection/coverage experiments.
The CLI does not change the source BIN and does not train a model.

## Required checks and regression coverage

- Exactly six valid pixels per image, no out-of-range numbers or duplicates.
- Canonical shape invariant under every D4 transform and legal translation.
- Exact orbit enumeration equals computed orbit cardinality; no cross-family
  duplicate numbers in the generated pool.
- For known rectangles and straight-six shapes: 60 and 28 legal placements.
- Candidate count matches `--size`, uniqueness holds, and every candidate's
  bitmap/shape ID matches its source family.
- Identical historical data + parameters + random seed reproduce identical
  ordered candidate JSON and SHA256; changing input cutoff changes input hash.
- `--sample-count` restricted to 1..5, sampled without replacement; the
  displayed sample must be a subset of the exact 100 candidates.
- Fair independent exact-six coverage equals `K / 13,983,816`; do not compute
  selection success using the target draw during pool construction.
- Existing 124 v0.5.1 repository unit tests must remain green.

## Deferred: v0.6 Phase 2–4

**Phase 2 — Selection:** Random, Shape-Diverse and Coverage-Optimized must
all take the *same frozen pool* and *same N (1..5)*. The objective must not
consult an unknown target draw. Keep a global random-six-set baseline,
a random-100 pool baseline, and a shape-100 pool baseline.

**Phase 3 — Exact probability / multi-objective audit:** For each final `N`
maintain `P(max hits≥3)`, `P(max hits≥4)`, `P(max hits≥5)` and
`P(max hits=6)`, plus the entire best-hit distribution `0..6`. Use exact
combinatorial union calculations or a carefully validated disjoint-event
implementation when possible. IID Monte Carlo uses **independent**
calibration and evaluation draws, plus confidence intervals. Do not use
unseen draw outcomes to choose optimizer weights. The exact-six probability
is always `N/C(49,6)` for distinct six-set tickets under IID draws.

**Phase 4 — Blind tracking:** Strict walk-forward history truncation and
prospective timestamped candidate-pool snapshots, each including data cutoff,
parameters, random seed, commit SHA and pool SHA256. Existing historical data
have been repeatedly inspected; describe old-data replays as exploratory.

## Known limitations

- A historical shape family is not a physical mechanism affecting future draws.
- The default uniform-family policy is deliberately **not** fair-uniform over
  all 13,983,816 combinations; it is a controllable image-shape exploration.
- Historical-frequency and orbit-weighted policies are not predictions.
- A 100-candidate research pool must not be mistaken for 100 purchased tickets.
  Five preview candidates have the same complete 6/6 probability as any other
  five distinct precommitted candidates if the draw is fair/independent.
- Special bonus numbers and real prize payout are not part of Phase 1.
