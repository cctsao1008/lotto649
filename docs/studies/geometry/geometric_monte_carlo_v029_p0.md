# v0.29 P0 — Outcome-blind geometric feature catalog

Tracking: [Issue #31](https://github.com/cctsao1008/lotto649/issues/31). This is a **feature-definition** milestone only: no historical fit, probability edge, compensation score, Monte Carlo inference or ticket purchase is implemented here.

## Fixed discrete 7×7 model

The canonical map is sequential: number `n` occupies the zero-based cell `((n-1)//7, (n-1)%7)`. Exactly six **distinct** numbers from 1..49 must be supplied. The alternative `snake` layout and seeded `random` bijections are negative-control maps, not better predictor selections.

Predeclared regions use lattice-cell **center inclusion**, including boundary equality, with center at `(3,3)`. Discrete disk `r=3` contains **29 cells**, not the continuous circle area `9*pi`. A six-number draw is sampled **without replacement**.

Ten region masks are versioned:

| Region | Cells | Interpretation |
|---|---:|---|
| disk_r1 / disk_r2 / disk_r3 | 5 / 13 / 29 | inclusive squared Euclidean radius |
| annulus_r2_r3 | 16 | squared radii `4 < d2 <= 9` |
| diamond_r3 | 25 | Manhattan distance `<=3` |
| center_square_3 | 9 | centered 3×3 square |
| border | 24 | outer rows or columns |
| diagonal_union | 13 | both diagonals, center not double-counted |
| ellipse_horizontal / ellipse_vertical | 19 / 19 | fixed axis-aligned lattice ellipses |

The first eight regions are invariant under all eight D4 transformations. The two oriented ellipse occupancy counts may **swap** under a quarter-turn; their unordered pair is D4-invariant. An arbitrary random reassignment of number labels to pixels is **not** a D4 symmetry.

## Output feature schema

`lotto649.geometry.v0.29.p0.v1` provides:

- Canonical sorted ticket, chosen mapping ID, seven 7-character binary bitmap rows.
- Region occupancy counts.
- Exact 8-neighbor graph connected-components, sorted component partition, horizontal/vertical and diagonal edge counts, and graph cycle rank `E - V + C` (graph cycles, **not** cubical persistent homology).
- All 15 distinct pairwise squared Euclidean distances in a fixed-support histogram.
- D4-invariant sorted bounding-box spans and squared radial second moment around the board center.

Note that individual region occupancies are linear combinations of the original 49 pixel indicator variables. They **must not** be treated as independent new predictive information. Pairwise and higher-order descriptors will require conditional redundancy testing against v0.28 Shape/Sum/Log/Joint in later phases.

The catalog has SHA256 fingerprint **`5aa5f2820c3287fde1d548de8c90a46b1aed95ed0b7b0f8b9a84c2ea39c82f47`**, computed over the canonical JSON feature definitions excluding the fingerprint itself. Its schema and exact 49-cell masks are fixed for P1/P2 validation.

## Windows PowerShell

```powershell
cd E:\lotto649
$py = ".\.venv\Scripts\python.exe"

# Print fixed region definitions
& $py -m lotto7x7.geometry_v029 --catalog

# Inspect a known six-number example with the canonical mapping
& $py -m lotto7x7.geometry_v029 --numbers 4 9 17 24 29 42

# Run the same six-number example under the snake control mapping
& $py -m lotto7x7.geometry_v029 --numbers 4 9 17 24 29 42 --mapping snake

# Create a new, non-overwriting report under gitignored results
& $py -m lotto7x7.geometry_v029 --catalog --out results/v029_geometry_catalog.json

# Execute standalone no-network regression checks
& $py -m unittest discover -s tests -p "test_geometry_v029_p0.py" -v
```

A second attempt to use the same `--out` path fails rather than overwriting existing research output.

## P0 acceptance and next steps

- [x] Fixed region schema, coordinate boundaries and reproducible catalog SHA256.
- [x] Deterministic sequential, snake and seeded-random bijections.
- [x] Region occupancy, graph topology, pairwise distance and invariant shape descriptors.
- [x] Seven offline tests covering six-hot validity, region counts, D4 invariance, graph cycles, mapping bijections, and deterministic catalog.
- [ ] **P1:** exact hypergeometric region occupancy law, checked Monte Carlo sampling, and joint-feature reference distributions as warranted.
- [ ] **P2:** observed 2,498-draw history vs fair IID convergence, whole-pipeline Monte Carlo controls, mapping sensitivity, and multiple-comparison treatment.
- [ ] **P3:** evaluate geometric repair and whether it adds nonredundant information; separately gate predictive claims through prospective results.

Neither existing 2004–2026 draw files, the P3f seal, v0.28 M0 candidates, nor private paid-ticket records are changed by P0.
