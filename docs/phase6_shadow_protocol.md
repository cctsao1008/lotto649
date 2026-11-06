# Lotto649 Architecture v0.2 — Phase 6 Prospective Shadow Validation

Status: **seal/verify/score implementation committed; local Windows tests pending. No production forecast sealed.**

## Decision from Phase 5 (already observed)

The official main-number archive used in Phase 5 contains **2,183 historical draws**, 2007-01-02 through **2026-10-30**. The final 437 drawings were 2023-01-31 through 2026-10-30.

| Model | Validation log-score lift | Test log-score lift |
| --- | ---: | ---: |
| IID | 0 | 0 |
| Static | +0.001917 | -0.004212 |
| Lag | -0.000003 | +0.000046 |
| MLP | -0.002132 | -0.006617 |

Static was selected by validation and lost to IID on the chronological historical test portion. The MLP also lost to IID. Lag was essentially neutral and was not supported by stable evidence. Calendar-year inspection did not establish persistent advantage. All historical validation remains **exploratory**, as these dates have already been used for prior research.

**Phase 6 policy:** IID is the fixed default probability distribution. Static, Lag and Small MLP are scientific challengers only. Do not select a challenger by looking at the future test score, and do not place bets based on any of these forecasts.

## The implemented file format

`lotto_shadow.py` implements:

1. **prepare**: freeze the exact 49 logits for each of IID, Static, Lag and MLP into a self-contained JSON seal for **one explicitly specified future draw date**. Store the source archive SHA-256, source main-six/draw-ID prefix SHA-256, number of observed draws, last date/ID, repository Git HEAD, model hyperparameters, a local preparation timestamp, the operator-entered closing deadline, and a canonical JSON payload hash.
2. **verify**: check the SHA-256 envelope, declared model policy, exact 49 finite logits for all models, time ordering and schema. Optionally compare the byte-for-byte seal file to a specified Git commit.
3. **score**: after the target draw, compare the committed sealed bytes with the working file, require a later updated archive with the same original historical prefix, and require the **very first newly appended draw** to match the sealed date. Score the observed full six-number subset against the frozen 49-logit distributions. Produce a new score report without overwriting earlier files.

The seal is created with exclusive file creation (`open('x')`) and an existing seal is **never overwritten**. Each stored `forecasts[model].logits` array is sufficient to reproduce the exact six-number probability distribution under the reference decoder. The six-number set itself is not chosen or stored in advance. It does not generate `lotto_select.py`.

### Training protocol

- The Static and MLP challengers use the **existing Phase 5 architecture and hyperparameters**, including 60% training, 20% validation and validation-based early stopping. The held-out older historical tail is not used for parameter fitting.
- The Lag parameter is estimated only from the chronological training segment.
- At the forecast origin, all previously published historical draws may be used as the one-step causal *input context*.
- The current frozen forecast is **never recomputed from later observed data**; scoring uses the logits inside the seal only.
- The IID reference uses exactly 49 zero logits and its log probability is computed exactly as `-log(C(49,6))`.

These challenger models are **not** being promoted because of historical results. Their prospective comparison is a predeclared research diagnostic.

## Critical timing and source freshness safeguards

**This module does not know the official upcoming drawing schedule or when its draw is held.** The operator must verify these facts from an authoritative source *before* any real seal.

- `--target-date` must be a future or current Taipei calendar date, later than the last recorded draw.
- `--deadline` must be an **explicit timezone-aware ISO 8601 timestamp** on that same Taipei calendar date. The operator must verify that it is strictly *before* the actual drawing and choose a sufficient safety margin.
- `prepare` refuses a draw archive older than **7 calendar days** by default. This is a freshness guard, not proof that every recent official draw has been downloaded.
- `score` refuses gaps between the sealed cutoff and the target: the target must be the **first newly appended record**, not an arbitrary later drawing.
- Old records are compared via a canonical date/draw-ID/main-six prefix digest, so subsequent modification or backfilling of earlier records causes scoring to fail.

**The currently studied archive ends on 2026-10-30. It must be refreshed and verified for completeness before preparing any actual future forecast.** No actual target date/deadline should be guessed in documentation or automatically selected.

### A Git commit does not prove publication before the draw

An ordinary Git commit stores client-controlled commit timestamps. A local `prepared_at_utc` field is also client-controlled. Even verifying that a file matches `git show COMMIT:path` **does not prove the commit was published before the drawing**.

To establish external timing evidence, publish the sealed payload SHA-256 or exact committed file to a third-party platform **before** the operator-verified drawing deadline, and later verify its *server-generated* publication time and content against the seal. A separately verifiable RFC 3161 timestamp token is another approach. Protect against editing or backdated records. The initial implementation does **not** automatically verify such external evidence and therefore deliberately marks post-draw reports `CONTENT_COMMIT_VERIFIED_PRE_DRAW_PUBLICATION_UNVERIFIED`, even if the Git content matches.

Until independent publication timing is verified, results must not be reported as conclusively precommitted or blinded.

## Windows PowerShell: safe development verification

```powershell
cd D:\my-github\lotto649
git pull origin main
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe .\lotto_shadow.py --help
.\.venv\Scripts\python.exe .\lotto_shadow.py prepare --help
.\.venv\Scripts\python.exe .\lotto_shadow.py verify --help
.\.venv\Scripts\python.exe .\lotto_shadow.py score --help
```

The expected new suite is **64 tests** (55 verified previously + 9 Phase 6 tests). These are fixture-based and do not prepare an actual upcoming draw.

## Future operational workflow (only after official data and draw-time verification)

The following is a **parameterized template**, not an instruction to seal from the stale archive. Replace the placeholders only with a confirmed upcoming official draw date and a confirmed pre-draw close time.

```powershell
$target = "<OFFICIAL_UPCOMING_DATE_YYYY-MM-DD>"
$deadline = "<VERIFIED_PRE_DRAW_DEADLINE_WITH_OFFSET>"
$compactDate = $target.Replace("-", "")
$seal = "data/shadow/seals/forecast_$compactDate.json"

.\.venv\Scripts\python.exe .\lotto_shadow.py prepare --target-date $target --deadline $deadline

# Review the newly prepared content and its hash.
.\.venv\Scripts\python.exe .\lotto_shadow.py verify --seal $seal

# Publish the exact seal before the verified close time.
git add -- $seal
git commit -m "Seal prospective one-step probability forecast for $target"
git push origin main

$sealCommit = git rev-parse HEAD
.\.venv\Scripts\python.exe .\lotto_shadow.py verify --seal $seal --seal-commit $sealCommit

# Separately obtain independently checkable, trusted pre-draw publication evidence.
# Do not claim prospectivity until this evidence has been verified.
```

After the exact target draw is included as the FIRST newly appended draw in the official archive:

```powershell
.\.venv\Scripts\python.exe .\lotto_shadow.py score --seal $seal --seal-commit $sealCommit

# Preserve the score report after reviewing it.
git add -- "data/shadow/scores/score_$compactDate.json"
git commit -m "Record sealed one-step forecast score for $target"
git push origin main
```

If any guard fails, **do not bypass it merely to create a favorable result**. Investigate freshness, correct the data pipeline, and retain an explicit audit history.

## Scope and next gates

1. Pass all fixture-based tests on the Windows host.
2. Verify current official records and the authoritative next-draw schedule *outside* this source file before beginning prospective data collection.
3. Document a real external timestamp-verification mechanism before using the words `precommitted` or `prospective verified`.
4. Freeze the forecast policy, source version, and commit; keep all prospective trials, including unfavorable results.
5. Later consider one-/three-/twelve-month survival evaluation as a **separate** methodology. One-step log probability is not interchangeable with a three-month first-hit objective.

All code comments, docstrings and runtime messages are English-only; explanations to the user remain in Traditional Chinese. **`lotto_select.py` is still the last stage.**
