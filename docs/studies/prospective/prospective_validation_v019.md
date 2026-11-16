# Lotto649 v0.19 — Frozen Prospective Validation

## Goal and interpretation

All P0–P2 decisions were made after inspecting the same 2,497 historic draws.
Their numerical backtests are **retrospective and exploratory**, not proof of
predictive advantage. v0.19 is a **future-only, append-checked research protocol**.
It scores only legally verified 6/49 main-number drawings **after** the
publicly frozen study cut-off. Do not edit the original frozen reference after
committing it. Under a fair drawing, all exact six-number tickets retain equal
probability `1/13,983,816`; a shape class being more prevalent does **not**
change these odds.

### Immutable reference (initial protocol)

* `docs/prospective_freeze_20261108.json` freezes the exact P1 data content,
  rule hash, 2,497 historic draws, the last historical draw ID `115000103`,
  the initial L1/L2/L3 class counts, the P2 rare-class hypothesis, the model
  hyperparameters and four monitoring tests.
* The seal includes the freeze creation UTC timestamp and its Taiwan local
  calendar date. For a draw to count as prospective, **both** `draw_id` must
  be greater than `115000103` **and** its authenticated Gregorian draw date
  must be strictly later than the freeze local date.
* This seal includes SHA-256 digests of the P1 catalog NPZ and universe JSON.
  The update command refuses to use changed inputs. The checksum is not a
  digital signature; preserve the committed Git history and original inputs.
* The P1 `shape8-graph-partition-cardinal-diagonal-v1` rules and exact
  universe masses stay unchanged. There is no fitting of KDE/GMM or discovery
  of a new rare pattern in the prospective phase.
* Only **main six numbers** enter the statistical model. Special numbers,
  payout information and discretionary betting decisions are not used.

## Four predetermined anytime-valid e-processes

For each one-step event `t` after the frozen source history, the L1/L2/L3
class predictors use the previously declared Dirichlet-shrinkage rule:

`q_t(k) = (n_previous(k) + 384 * p_fair(k)) / (N_previous + 384)`.

The category-specific likelihood-ratio increment is
`q_t(y_t)/p_fair(y_t)`, recorded on the log scale. Crucially **the observed
event is added only after recording its prediction**. Under IID fair 6/49,
this multiplicative process is a nonnegative mean-one martingale conditional
on the frozen historical state. Three processes correspond to L1, L2, L3;
these are **dependent** and must not be treated as independent discoveries.

The rare category `P6_O1_D5` was discovered retrospectively. The fourth
future-only process fixes `p0 = 744/13,983,816` and the simple one-sided
alternative `p1 = 10*p0` **before any new outcome**. Its likelihood ratio
multiplies by `p1/p0` for a hit or by `(1-p1)/(1-p0)` for a miss.

The **single** prespecified monitoring statistic is the equal-weight mixture

`E_t = (E_L1,t + E_L2,t + E_L3,t + E_rare,t) / 4`.

It is an anytime-valid e-process under the fair IID null even when inspecting
results repeatedly after each draw. `E_t >= 20` is the prespecified research
review boundary corresponding to familywise type-I error ≤0.05 through
Ville's inequality. The reported bound `min(1,1/E_t)` is an anytime-valid
upper-bound p-value under this *entire predeclared four-process mixture*;
it does not mean a drawing is unfair with a particular posterior probability.
A crossing requires external source verification and independent review;
it is not a winning-number prediction or proof of mechanical bias.

No simulated IID sequence needs to be repeatedly refit for this frozen
prospective e-process. The P2 retrospective IID results remain separate.

## Reading future CSV records

Use existing annual English-header CSV files. Minimal required columns:

```csv
date,draw_id,n1,n2,n3,n4,n5,n6,special
2026-11-10,115000104,1,8,16,22,24,30,49
```

**This row is only a schema example, not an actual lottery result.** Never
append example values to the real repository data. `date` or `draw_date` must
be a Gregorian `YYYY-MM-DD` date for future rows, with the ROC year in the
issue ID matching the Gregorian date. The code ignores `special` and extra
columns. Existing P1 historical CSVs can omit `date`, because only strictly
future draws are scored. File names and folder names can be in English.

CSV input may be **one or more files**, an annual CSV directory or a quoted
wildcard. New ID sequence must be continuous within a ROC calendar year.
Duplicate IDs, missing future dates, mismatched frozen historic values and
missing new issue IDs fail closed. A year rollover is allowed only when the
first issue in the new ROC year has sequence `000001`; sources should still
be checked for completeness at the year boundary. The tool reads the files
locally and does **not** independently authenticate the winning numbers.

## Windows / PowerShell

For this release, use the **already committed freeze**; do **not** run freeze
again just before scoring newly available draws, because that would move the
prospective starting date and invalidate the study's intended time boundary.

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main

& $py -m lotto7x7 shape-prospective-update `
  --freeze docs/prospective_freeze_20261108.json `
  --catalog results/v015_v016_p1_shape_catalog.npz `
  --universe results/v015_v016_p1_shape_catalog_universe.json `
  --csv data/csv `
  --out results/v019_prospective_first_check.json

Start-Process .\results\v019_prospective_first_check.html
```

When a **new** draw appears later, save its true, source-verified row in the
appropriate annual file and run again using a **new output filename**:

```powershell
& $py -m lotto7x7 shape-prospective-update `
  --freeze docs/prospective_freeze_20261108.json `
  --catalog results/v015_v016_p1_shape_catalog.npz `
  --universe results/v015_v016_p1_shape_catalog_universe.json `
  --csv data/csv `
  --previous results/v019_prospective_first_check.json `
  --out results/v019_prospective_second_check.json
```

`--previous` is an optional *append-only cross-check* of already published
rows. Reuse the previous report and a new output path at each observation.
If an old prospective row was edited/deleted, processing is rejected. The
new report re-evaluates the **entire** future sequence from the original
frozen counts, and verifies every prior event before writing its result.

If no new draw exists, all values must remain
`eligible_new_draws=0`, `mixture_e=1`, `anytime_p_upper_bound=1`, and
`decision=NO_FUTURE_DATA`.

Three outputs from each update: immutable JSON audit with event-level
pre-draw probabilities, CSV timeline (header only when no new draws), and
local offline HTML summary.

The optional `shape-prospective-freeze` CLI is for setting up a **new,
separately named and independently dated** future experiment, not resetting
this particular experiment. Its arguments are `--catalog`, `--universe`,
`--watchlist`, `--tau` and `--out`.

## Acceptance and risks

**Development:** 25 tests; full 2,497-row prior-draw consistency scan; proper
forecast normalization; no-data and synthetic prospective events; strict
calendar/ROC issue validation; previous-report append-only checks; checksum
verification; exact P1 fair priors; local CLI/HTML/CSV/JSON end-to-end tests.

**Research:** zero actual post-freeze draws are currently present, so no
prospective claim can be accepted or rejected. Never count P2's 1,997 draws
as future evidence. Record raw-source provenance, preserve original CSV and
all reports, avoid cherry-picked observation intervals and monitor only
these four frozen tests. Releasing any new model version requires an
independent, separately dated preregistration before evaluating it.
