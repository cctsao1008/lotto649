# Lotto649 Research Knowledge System — Governance v1

> **Organize accumulated knowledge into a coherent, disciplined, reproducibly testable system.**
>
> **Let each experiment move our understanding closer to the true answer.**
>
> **Explore beyond current theories without lowering the standard of evidence.**

This is a governance framework for reproducible research, **not** a prediction strategy or a lottery purchase mechanism.

## Navigation

- [Current evidence index](research_evidence_index.md): original observed, negative and pending research states.
- [Machine-readable evidence registry](../research/evidence_registry_v1.json): stable IDs, original Git blob hashes and scientific decisions.
- [Historical research consolidation](research_issue_consolidation_20261112.md): preserved original 2026-11-12 knowledge snapshot.
- [New hypothesis checklist](research_hypothesis_template.md): new proposal and pre-registration requirements.
- [M1 Experiment Ledger](experiment_ledger_m1.md): original authoritative *separate* public seal identity and private paid-ticket record management.
- [Research navigation hub](RESEARCH_INDEX.md).
- [Issue #38](https://github.com/cctsao1008/lotto649/issues/38): governance workstream.

## Knowledge architecture

| Domain | Objects, assumptions and open questions |
| --- | --- |
| Mathematics and probability | Uniform sampling without replacement from 1–49; exact 6/49 combinatorics, distributions, parity patterns and moments |
| Statistical dynamics | IID, martingale convergence, conditional Monte Carlo, historical deficits, temporal association and multiple-testing correction |
| Geometry and topology | Frozen 7×7 mappings, D4 invariants, connected components, L1/L2/L3 graph classes and pairwise geometry |
| Astronomy | SHA-256-verified official USNO seasonal and lunar events, timestamps, provenance and date-level interpolation |
| I Ching encodings | Experimental twelve 消息卦 and solar8 × lunar8 Gray-coded 64-state astronomical hexagrams |
| Selection and portfolio | Historical discrepancy repair, exact coverage, four-ticket optimization and separate prize-event profiles |
| Evidence engineering | Data source manifests, CSV/BIN validation, publisher snapshots, pre-draw seals, CI, Git blobs and immutable research records |

**Knowledge dependency:** research question → falsifiable hypothesis → model/feature version → frozen dataset/cutoff → exact theory or null → predeclared protocol → results → scientific decision → next question.

The expected long-run convergence of empirical distributions is **not** a causal mechanism forcing future draws to compensate old deviations. Pairwise geometric features may add information relative to *aggregate marginal counts*, but do not add independent information beyond a complete draw history. A mathematically optimal retrospective repair score does not change the uniform jackpot probability of a particular six-number ticket. Astronomical state encoding does not prove causal effect or forecast value.

## Evidence stages

| Stage | Permitted inference | Excluded interpretation |
| --- | --- | --- |
| SOURCE_ARCHIVE | Source fingerprints and recorded provenance are checked | Not a lottery forecast |
| STRUCTURAL_THEOREM | An exact finite combinatorial or algebraic result is verified | Not future predictive evidence |
| DESCRIPTIVE | A defined dataset or counterfactual objective is measured | Not independent confirmation |
| EXPLORATORY_RETROSPECTIVE | Past-data hypothesis screen with explicit leakage and selection caveats | Not a new untouched future sample |
| PROSPECTIVE_PRESEALED | Original future outcome protocol and schedule are pinned in advance | Results remain **PENDING** |
| PROSPECTIVE_EVALUATED | Future observations under a pinned protocol and independently verified sources | Not automatically VALIDATED_GO |

**Scientific state and engineering state are independent.** Engineering PASS means the implementation or data audit ran; it never means the forecast is successful. NO_EVIDENCE is a conclusion limited to the tested endpoints, data and statistical power; it is not proof every conceivable signal is absent. A completed engineering Issue can be closed while the sealed prospective experiment stays PENDING. The initial registry deliberately prohibits automatic VALIDATED_GO; any such future conclusion requires source review, predeclared multiplicity, independent replication and a separate reviewed governance revision.

## Research lifecycle and evidence gates

1. **Question and null:** Define the falsifiable claim, fair-IID competing hypothesis and exact prediction target.
2. **Source boundary:** Identify authentic versus indirect publication records, dataset version, earliest/latest dates, missingness, cutoff, hashes and source limitations.
3. **Feature definition:** Freeze 7×7 mapping, bins, trigram bits, rounding, calendar reference times and all eligibility rules before confirmation data.
4. **Protocol:** Predeclare endpoints, alternatives, seeds, chronology, training/holdout partitions, stopping rule, power and multiplicity across variants.
5. **Execution:** Run independent source checks, unit/adversarial tests, correct without-replacement nulls, negative controls and whole-pipeline IID retraining when selection is adaptive.
6. **Reporting:** Record effect sizes, uncertainty, p-values or mathematical identity, test count, sample size, failures and the exact evidence stage.
7. **Status:** Do not promote a post-hoc near-significant finding, or source/CI PASS, into a scientific discovery.
8. **Knowledge update:** Append an evidence identity with original file hashes, implementation commit, CI run, decision, limitations and the next falsifiable question.

## Registry and M1 boundaries

The registry at \`research/evidence_registry_v1.json\` is an **index of original study snapshots**, not a live betting account or a dynamic reopening of old seals. Every original report/protocol/seal is pinned by Git blob SHA-1; source reports may also pin SHA-256. Neither digest alone proves a third-party trusted timestamp.

An entry's \`evidence_as_of\` is the original snapshot date, not a claim of continuously current results. New draw results belong to a **new append-only evaluation identity**, keeping the original PENDING seal and original historical NO_EVIDENCE report untouched. The old 2,497- and 2,498-draw cutoffs remain frozen even when the source archive gains later official draws. Governance does not rewrite the source data or cause unattended harvesting.

The existing \`experiments/ledger_registry_v1.json\` under M1 remains authoritative for public shadow identities and local private paid records; #38 does **not** duplicate M1 or publish purchased-ticket details.

## Validation

Pure Python standard library; no network access, expensive Monte Carlo replay or full-universe enumeration is needed:

\`\`\`powershell
python -m unittest discover -s tests -p 'test_research_governance.py' -v
python -m lotto7x7.research_governance verify
python -m lotto7x7.research_governance render
\`\`\`

The verifier checks stable IDs, safe paths, pinned original blobs, valid evidence categories, selected native P1A/P2A conclusion fields, initial pending seal states and a deterministic Markdown index. CI may finish a governance build, but may not promote a study to predictive GO.
