# Lotto649 v0.32 P3 — Prospective Study Readiness & Evidence Gate

Tracking: [Issue #32](https://github.com/cctsao1008/lotto649/issues/32); uses the [Research Knowledge System](research_knowledge_system.md), and does **not** replace the existing [M1 Experiment Ledger](experiment_ledger_m1.md).

## Why this is the next step

The previously executed v0.31 H2 chronological geometry ablation was negative: combined-minus-M0 incremental Brier gain **−0.0002140645**, matched 39-IID-path empirical p **0.575**. Repeatedly tuning that same inspected historical sample cannot make an independent forecast finding. Before new eligible draws begin on **2026-11-17**, the priority is auditability: all studies must retain their original frozen identities, correct schedule and different evidentiary strengths.

## Four immutable forward protocols

| Original study | Original first eligible draw | Horizon | Immutable Git blob SHA-1 (prefix) |
| --- | --- | ---: | --- |
| v0.27 P3f four-ticket shadow | 2026-11-10, #115000104 | 10 | b7dec534a44b |
| v0.29 P5A geometry shadow | 2026-11-13, #115000105 | 9 | 546c7352fd0a |
| v2.3 static M0 vs history-blind | 2026-11-17, #115000106 | 200 | f02a94b37bc2 |
| v2.4 P3 USNO hexagram | 2026-11-17, #115000106 | 100 | abf4963adf33 |

All four full original Git blob hashes are hard-pinned in the v0.32 verifier. The USNO shadow also requires an exact SHA-256 match for its original 379-day future sky table.

These are **four research protocols**, not four independent confirmations. P5A's 9 draws are exactly the last 9 draws of P3f, and the 100 USNO study draws are the initial 100 draws of the 200-draw v2.3 study if the planned schedules proceed. In particular, the studies re-use some of the same actual lottery draws. Their discoveries cannot be pooled by informal vote-counting or by choosing the best-looking p-value.

## Explicit archive-versus-evidence boundary

The standard-library tool \`lotto7x7.prospective_readiness_v032\` calls the existing M2 full source integrity audit, including 23 annual CSVs, history/image BIN and source manifest with the pinned 2,498-draw baseline. It then reads the already validated original four seals.

**A draw existing in the public CSV/BIN archive is NOT an authenticated prospective trial.** The original evaluators (not this tool) require two saved, independently obtained matching publisher-page evidence snapshots *for each draw* to admit it for scoring.

This preflight **never** inspects the private evidence directory and never evaluates any prizes, Hamming distance, odds, post-draw results, p-values or e-values. It reports only how many target slots are already archived, not source-qualified. It therefore never describes archive rows as \`verified prospective outcomes\`.

### Snapshot read from the canonical archive as of 2026-11-14

The existing main branch archive had **2,499** rows ending **2026-11-13, #115000105**, with historical original 2,498 prefix still pinned and preserved. Pending true source authentication and separate evaluator admission:

| Study | Source-archived target slots (NOT scored) | Target slots not yet archived |
| --- | ---: | ---: |
| P3f 10 draws | 2 | 8 |
| P5A 9 draws | 1 | 8 |
| v2.3 M0 200 draws | 0 | 200 |
| v2.4 P3 USNO 100 draws | 0 | 100 |

This is an **archival readiness snapshot** and not a claim that individual source pages have already been reviewed or are missing; the private evidence bundles are deliberately outside the scope.

## Statistical multiplicity across studies

Both v2.3 M0 and v2.4 USNO have individually committed **anytime E-process thresholds of E ≥ 20**, each with an *individual* fair-IID false-review bound ≤0.05 **if the process assumptions hold**. By the union bound, the probability of at least one of these two separate valid processes crossing during their respective horizons is at most **0.10**, even without independence. It is **not** automatically controlled at 0.05 across studies.

The older P3f and P5A fixed-horizon tests and other studies are **not** covered by that two-process bound. Crossing either threshold can lead only to **manual source/methodology review**; it is not proof of unfairness, success, nor authorization to bet.

This tool makes no changes to the original studies' multiplicity, stopping rules, windows, ticket choices or scientific statuses.

## Reproducibility

On the repository root, with Python 3.12; no NumPy/SciPy/AI dependencies:

\`\`\`powershell
python -m unittest discover -s tests -p 'test_prospective_readiness_v032.py' -v

python -m lotto7x7.prospective_readiness_v032 --expected-last-id 115000105 --expected-draws 2499 --out results/v032_preflight_20261114.json
\`\`\`

The \`--expected\` flags intentionally pin **this particular November 14 snapshot**. For future new draw archives, rerun using a **new report path** and omit the snapshot-only constraints. The module rejects all attempts to overwrite an existing report and cannot write outside the ignored \`results/\` directory.

The small [GitHub Actions workflow](../.github/workflows/prospective_readiness_v032.yml) runs adversarial tests and validates the authentic source archive; its JSON report is a read-only Actions artifact. It never fetches or scores new numbers. The original retrospective \`NO_EVIDENCE\`, prospective \`PENDING\` and frozen files are not changed.

## Research gate

**Engineering result:** a passing preflight means the current original seals and source archive are consistent with the stated protocol, *not* that any lottery advantage exists. The first #115000106 prospective event is **2026-11-17**. After independently sourced future data arrive, score them only through each study's own original evaluator with strict append-only evidence. Do not create alternative new hypotheses in response to early outcomes.
