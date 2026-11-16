# M2 Preview — Safe, Verified One-Draw Append Candidate

Tracking: https://github.com/cctsao1008/lotto649/issues/31

**Purpose:** prepare and self-audit a candidate archive for the next P5A draw without changing any original tracked CSV/BIN/manifest file. The user explicitly controls whether and when a reviewed candidate is later incorporated. This module contains **no command to apply, commit, or publish** a candidate archive.

## Preconditions and proof levels

1. The fixed original P5A shadow manifest is verified against the pre-draw Git blob SHA1; it remains immutable. The next accepted period must be the immediate consecutive entry in the frozen #115000105–#115000113 schedule, and the source archive must pass M2's full integrity gate.
2. The user separately runs M2E `--capture-latest` after the draw and retains the two separately published HTTPS pages with capture time, pinned URLs and SHA256. The raw page parsing must produce the *same* six main balls and special number for the same scheduled period.
3. Both saved-page capture timestamps must be on or after the advertised drawing date in Asia/Taipei; a page purportedly captured before the draw is rejected.
4. The M2E verified local snapshots remain evidence of publisher-page *content consistency*, not cryptographic proof that a publisher authenticated the result. The final archive must still be independently reviewed before any real source update.

## Preview construction

For one verified future draw, `lotto7x7.draw_append_preview_m2` reproduces:

- The original `<8sII` binary headers with count incremented by one;
- One new `<II7B` history record and one new `<I49B` image record;
- All previous binary data bytes unchanged, source-provenance metadata appended and old original-source snapshot retained;
- Updated source manifest record counts, last issue/date, year count, and both binary byte lengths/SHA256;
- All 23 canonical annual CSVs regenerated in a separate directory and fully re-verified;
- Copied raw source pages and source index beside the staged preview for audit.

Everything is written to a fresh, privately ignored location under `results/`. Construction takes place in a sibling temporary directory, passes the existing full M2 source audit including the original pinned 2,498-draw prefix, then is renamed to the requested new directory. No overwrite of an existing preview path is allowed.

**Critical:** the preview intentionally reports `ready_to_commit_to_main = false`; passing numerical and consistency tests is not independent official authentication, final review or authorization to modify the historical repository.

## PowerShell commands

```powershell
cd E:\lotto649
$py = '.\.venv\Scripts\python.exe'

# Pure offline tests; synthetic fake future numbers are used only in temporary test fixtures
& $py -m unittest discover -s tests -p 'test_draw_append_preview_m2.py' -v

# After #115000105 has really been drawn, and ONLY after saved source pages agree:
# & $py -m lotto7x7.official_draw_evidence_m2 --capture-latest
# & $py -m lotto7x7.official_draw_evidence_m2 --verify-draw 115000105
# & $py -m lotto7x7.draw_append_preview_m2 `
#     --draw-id 115000105 `
#     --csv-root data/csv --bin-root data/bin `
#     --evidence-root data/private/lotto649-source `
#     --out-dir results/m2_verified_preview_115000105

# Inspect the independent candidate archive and compare SHA/provenance to source.
# Do NOT copy preview binaries into active data/bin without a separately reviewed transaction.
git status --short
```

## CI and safety tests

- [x] Valid synthetic two-source HTML produces one standalone 2,499-draw preview verified by the existing BIN/image/CSV/manifest gate.
- [x] The original four source files remain bit-for-bit unchanged.
- [x] The preview can be read by M4 for one **synthetic** future draw without a final p-value or predictive claim.
- [x] Source pages dated earlier than the advertised draw are rejected.
- [x] Gap/rollback, missing sources, invalid target path and preview overwriting are rejected.
- [x] CI independently validates the real current archive as **0/9 future draws**, never introduces fake source outcomes to the real history.

**State on 2026-11-11:** still awaiting #115000105 scheduled for 2026-11-13. No source append authorized or performed.
