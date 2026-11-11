# Lotto649 M1 — Experiment Ledger

M1 separates **public research identities** from **locally stored paid lottery tickets**. It records experiment boundaries but does **not** score outcomes, import new draws, claim increased winning probabilities, or create pre-draw proof retrospectively.

## Public read-only research anchors

`experiments/ledger_registry_v1.json` currently registers three read-only Git objects (and supports additional future research entries):

- P3f static 10-draw prospective shadow seal, original Git blob SHA1 `b7dec534a44b1ec08d575a4f2e48e5c3580e76d1`. The existing immutable seal retains its three original strategies, ticket numbers and schedule.
- v0.28 M0 multiobjective candidate manifest, Git blob SHA1 `21dff1f975773ca919e6af9e762e93f8ffc3475c`. It is explicitly `DESCRIPTIVE_CANDIDATE_ONLY_NOT_A_PAID_TICKET`; this file by itself is **not purchase evidence**.
- v0.29 P5A prospective geometry shadow seal for nine future draws from #115000105, Git blob SHA1 `546c7352fd0a98a42959b333e51498ef7c56cf9d`; two fixed **shadow-only, unpaid** strategy arms. The original P3f public shadow is preserved unchanged, and actual paid tickets remain private.

The verifier checks each artifact's original Git blob SHA1, allowing only a full Git-for-Windows CRLF checkout conversion. No public entry may contain a paid-ticket role. Source path escapes and modified artifacts are rejected.

## Private write-once paid-ticket registration

**Do not commit purchased-ticket data.** `.gitignore` excludes `data/private/`. Keep registration inputs under `data/private/ledger-input/` and the generated write-once records under `data/private/experiment-ledger/`.

A paid registration has schema `lotto649.private_paid_registration.v1`. It specifies exact six-number tickets, the draw/date schedule, user-reported purchase date, source classification, historical cutoff SHA256, required disjointness, ticket price and total cost.

Validation rejects invalid/unsorted/duplicate numbers, repeated or overlapping tickets when disjointness is required, schedule gaps, date/draw-ID mismatches, inconsistent price totals, unknown fields and embedded outcome data. Duplicate experiment IDs cannot be overwritten using the CLI.

Registration writes a canonical SHA256 receipt; preserve the printed receipt separately to make later alteration detectable. A locally recorded digest is **not** independent timestamp notarization.

Both paid records are labeled `RETROSPECTIVE_UNSEALED`, because this tool is being introduced on 2026-11-11, after their first scheduled drawing. This does **not** dispute the user's earlier purchase; it distinguishes the reported purchase date from an independent pre-draw seal. The P3f shadow seal retains its pre-draw Git evidence.

## Windows PowerShell

```powershell
cd E:\lotto649
$py = ".\.venv\Scripts\python.exe"

# Verify the two original public research artifacts
& $py -m lotto7x7.experiment_ledger verify

# Store paid registration input JSONs under ignored local private data
& $py -m lotto7x7.experiment_ledger register-paid `
    --input data/private/ledger-input/paid-original-4x10-20261109.json

& $py -m lotto7x7.experiment_ledger register-paid `
    --input data/private/ledger-input/paid-m0-4x5-20261110.json

# Verify frozen public references and all local private paid records
& $py -m lotto7x7.experiment_ledger verify

# Confirm that private records are excluded by Git
git check-ignore -v data/private/experiment-ledger/paid-m0-4x5-20261110.json
```

`verify` prints registration counts but intentionally does not expose all private ticket numbers. Do not include private paths in `git add -f`. If a registration is wrong, do not silently edit or overwrite it; retain the original, explain the correction and create an explicitly new experiment ID.

## Responsibility boundaries

- **M1:** experiment registration, immutable public source anchors, local private records and validations.
- **M2:** separately validated, source-audited draw ingestion and updates to historical BIN/CSV/manifest.
- **M3:** official prize-tier rules, hits, payout and costs; outcomes stored separately from registration.
- **Issue #31:** v0.29 geometric topology, Monte Carlo controls, Image Distribution Repair, and statistical learning.

M1 does not modify existing drawing results, purchased tickets, M0 research candidates or the P3f seal.
