# v0.27 — P3f preregistered prospective static shadow test

**This is public research with no wager or automatic ticket purchase.** Already purchased tickets are maintained separately and never copied into this public manifest.

## Immutable pre-first-draw seal

- First pre-draw Git commit: [ae15748a88e7aad334d68354b04338fa71d6bfdf](https://github.com/cctsao1008/lotto649/commit/ae15748a88e7aad334d68354b04338fa71d6bfdf), committer timestamp 2026-11-10 08:54:19 UTC (2026-11-10 16:54:19 Taiwan time).
- Git blob SHA1 for `v027_p3f_static10_20261110_seal.json`: `b7dec534a44b1ec08d575a4f2e48e5c3580e76d1`.
- Frozen historical prefix: 2026-11-06 / #115000103 / 2497 draws; SHA256 `d865e1d241b607488e58a00b22fb595734f804595afd9d8850718808d6e8b7f8`.
- Original P2 proof-of-work: [GitHub Actions run 42761581450](https://github.com/cctsao1008/lotto649/actions/runs/42761581450).
- Fixed future draw periods #115000104 through #115000113 inclusive, 2026-11-10 to 2026-12-11.

Three four-ticket portfolios are frozen STATIC for all 10 future draws: original P2 Primary, original P2 four-distinct-sums, and a history-blind seeded random control. All have 24 distinct numbers. No predictions may be revised after future results become available. This is NOT rolling selection and does not substitute for the 35-checkpoint retrospective P3e.

## Evaluation

New actual results must be independently verified and appended to `data/csv/lotto649_2026.csv` in the existing flat annual schema. Before grading, the evaluator validates the first 2497 historical draws against their frozen SHA256, validates the exact Git blob of the immutable preregistration and checks every future period ID, date, main six numbers, bonus and contiguous coverage. Missing future rows remain **PENDING**, never losses.

The fixed endpoint is the number of draws where at least one of the four legal disjoint tickets wins a prize among the eight official tiers. Exact fair IID per-draw odds are approximately 12.273145896191% for every valid four-disjoint set. Only after all ten planned outcomes are available, two one-sided exact Binomial(10,p) tests plus a Bonferroni correction for the two named P2 hypotheses become eligible. The random control is descriptive; no early inferential claim is published.

## Windows PowerShell

```powershell
cd E:\lotto649
git fetch origin
git switch feature/v027-prospective-p3f-static10
$py = ".\.venv\Scripts\python.exe"

& $py -m pytest -q tests/test_prospective_p3f_static10_v027.py

# Use new report filenames on every check. Never overwrite prior reports.
& $py -m lotto7x7.prospective_p3f_static10_v027 `
    --seal prospective/v027_p3f_static10_20261110_seal.json `
    --csv-dir data/csv `
    --out results/v027_p3f_shadow_check_01.json `
    --md-out results/v027_p3f_shadow_check_01.md
```

Before any future results have been added to the frozen repository, the correct output is `[P3F STATUS] 0/10 observed; 10 pending`. This is not a failure. Use `_02`, `_03`, etc. for later runs. Do not modify the original seal; Git commit and pinned blob SHA authenticate exactly the same fixed pre-draw choices.

**Scope:** 10 draws are underpowered; the P2 strategy was developed on earlier historical draws; correction covers only two hypotheses, not all previously explored model families. Report `PREDICTIVE_NO_GO / INCONCLUSIVE` without further prospective replication. This code does not verify the external CSV against the official lottery operator; do so independently before publishing results.
