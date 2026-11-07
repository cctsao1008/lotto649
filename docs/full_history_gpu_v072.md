# Lotto7x7 v0.7.2 — Full-History GPU Scenario-Coverage Replay

## Purpose and exact scope

Replace the previous 12/100-target *functional sampling* with a chronological
replay across **every eligible historical target**, retaining a strictly
past-only training/shape-library cutoff. With 2,183 source images and the default
`--warmup 200`, the experiment evaluates **1,983 target draws**. The first 200
draws are solely historical context, not predictions; `--warmup 50` evaluates
2,133 targets if there are enough shape families at the earliest cutoff.

**The previous v0.7 `baseline-study` and v0.7.1 `robustness-sweep` remain
unchanged.** This is a separate accelerated method with a separate schema.
Its `scenario-coverage` selector is NOT the v0.6 exact-greedy selector and
cannot be compared to old results as though the selection policy was identical.

## Where GPU is useful, and where CPU is necessary

- **CPU, incrementally**: canonicalize each new six-pixel historical image
  once; add its family to a sorted, past-only family registry; sample each
  target's image-derived 100-combination pool with the exact same uniform-family
  and round-robin orbit policy as v0.6. This avoids rescanning the entire
  historical archive separately for every target.
- **CPU**: generate a uniform 100-combination random comparison pool and direct
  B0/B2 baseline tickets with the same reproducible v0.7 seed derivation.
- **GPU via CUDA PyTorch**: form an independent matrix of fair six-out-of-49
  scenarios. For each candidate pool, use a batched matrix multiplication to
  calculate all six-number overlaps at once. Select up to five distinct tickets
  greedily by sampled coverage at thresholds >=3, >=4, and >=5, with
  deterministic tie breaking. Scenarios are independent of the actual target
  result and are reused between B1 and S1 for a fair paired comparison.
- **CPU, exact**: recompute the *selected final portfolio* fair coverage via
  the existing membership-class combinatorial DP. The mathematically
  equivalent disjoint-portfolios anchor is cached. If a fully disjoint
  candidate selection exists and has strictly better exact coverage than the
  scenario-greedy portfolio, use that reference instead. This safeguard is
  not an exhaustive/global optimization guarantee.
- **GPU via CUDA PyTorch**: batch-evaluate precommitted tickets against every
  target's actual six numbers *after* all portfolios are constructed.

The original exact-greedy v0.6 search evaluates many CPU DPs per candidate,
which is the relevant bottleneck. Moving only the *post-hoc* score to CUDA
would not meaningfully accelerate full-history analysis. The new fast scenario
selector changes the optimization procedure explicitly to obtain useful GPU
work, rather than merely adding a `--cuda` flag to CPU-bound code.

This module does **not** train a CNN/GRU, and cannot create a lottery edge.

## Six portfolios with the SAME N and the SAME target draws

| Portfolio | Candidate source | Selection |
| --- | --- | --- |
| `B0-global-random/random` | Fair universe | Direct uniform random N |
| `B1-random-pool/random` | Uniform M=100 | Seeded random N |
| `B1-random-pool/scenario-coverage` | Uniform M=100 | CUDA fair-scenario coverage + exact audit |
| `S1-shape-pool/random` | Historical D4/translation M=100 | Seeded random N |
| `S1-shape-pool/scenario-coverage` | Historical D4/translation M=100 | Same CUDA scenario method + exact audit |
| `B2-disjoint/disjoint` | Fair universe | Random disjoint N |

Shape-diverse and the earlier exact-greedy selectors remain in v0.7/v0.7.1,
but are *not* silently substituted into this faster six-portfolio experiment.

## Windows / PowerShell: GPU prerequisites

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main
& $py -m unittest discover -s tests -v
& $py -m lotto7x7 gpu-check
```

Verify `CUDA available=True` and the GPU name. If `False`, the environment may
contain a CPU-only PyTorch wheel. Use the current official PyTorch website
**Start Locally** installation selector to choose the Windows / pip / Python /
CUDA build compatible with the NVIDIA driver. Do not assume that merely owning
an RTX 4060 causes `torch.cuda.is_available()` to become true.

### Smoke test: only the latest five eligible target draws

```powershell
& $py -m lotto7x7 full-history-gpu `
    --device cuda --warmup 200 --tail-targets 5 `
    --count 5 --size 100 --families 20 `
    --scenarios 8192 --seed 649 `
    --checkpoint results/full_history_v072_smoke_checkpoint.jsonl `
    --out results/full_history_v072_smoke.json
```

### Full historical evaluation: 1,983 of 2,183 draws

```powershell
& $py -m lotto7x7 full-history-gpu `
    --device cuda --warmup 200 `
    --count 5 --size 100 --families 20 `
    --scenarios 8192 --seed 649 `
    --checkpoint results/full_history_v072_checkpoint.jsonl `
    --out results/full_history_v072_all.json
```

The `.jsonl` checkpoint is append-only and created exclusively, so rerunning
without `--resume` refuses to overwrite it. To continue after an interruption,
repeat **exactly the same arguments**, additionally specifying `--resume`.
The checkpoint binds parameters and the entire source archive SHA256; a changed
historical source is rejected. The report is written atomically at completion.
Do not run two writers against the same checkpoint simultaneously.

To compare a wider fair scenario sample, run a **separate** experiment using
`--scenarios 16384` and a different checkpoint/output path. Different
candidate optimizers/scenario budgets must be recorded as distinct methods.

If CUDA is unavailable, `--device auto` or `--device cpu` supports correctness
checks, but CPU performance can be substantially slower, and no CUDA speedup
should be claimed.

## Statistical outputs and caveats

- The complete per-target betting record, independent cutoff ID, B1/S1 pool
  hashes, actual target result and **exact** selected-portfolio fair coverage.
- For each of six methods: number of targets, `max_hits` histogram 0–6,
  mean maximum hits, observed >=3 rate, fair expected >=3 rate and year slices.
- Direct S1 vs B1 paired contrasts for `random` and `scenario-coverage`:
  S1-only/B1-only counts, exact two-sided McNemar p-value, paired bootstrap CI,
  fair-coverage differences and excess difference beyond fair coverage.
- Only chronological targets are treated as distinct outcomes; the 8,192 fair
  *optimization scenarios* are **not** new historical targets and must not be
  counted as independent statistical evidence.
- All 2,183 historical images have been used in some form, but the first
  `warmup` draws are context-only. There is no valid way to retrospectively
  evaluate a first draw without prior history under this shape-library policy.
- The dataset was repeatedly inspected, so even the complete historical
  replay is exploratory. No unadjusted p-value is proof of a predictive edge.
  Prospective timestamped shadow predictions remain necessary.
- The approximate GPU portfolio selector and exact CPU coverage evaluator are
  explicitly separated in both the output schema and documentation.

## Validation plan

- Reuse all 176 prior tests; add full-history CUDA fallback, deterministic
  uniform pooling, disjoint-mathematics, CPU/GPU-compatible batched scoring,
  resume/checkpoint, no-target-leakage and CLI tests.
- On the real repository, compare the incremental S1 pool at several cutoffs
  against the original v0.6 `create_shape_pool`: both the ordered candidate
  sets and the candidate-pool SHA256 must match exactly.
- Run the 5-target GPU smoke test before committing to all eligible draws.
- Check 1,983 targets with cutoff less than the target ID, N=5 tickets per
  strategy, no duplicated tickets within a portfolio and constant 6/6 fair
  chance `5 / 13,983,816`.
