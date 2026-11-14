# Lotto649 — New Research Hypothesis Template

Copy this **structure** into a new Issue or versioned study plan before inspecting new confirmatory outcomes. It is not itself a sealed study.

## Identity & why this question matters

- Stable proposed experiment ID / parent issue / related previous work:
- Falsifiable claim and independently testable opposite or null:
- Whether the study is structural, descriptive, retrospective exploratory or truly future-only:
- Expected information gain even when the hypothesis fails:

## Source and modeling boundaries

- Original dataset/version, ownership and reliability limitations:
- Frozen source cutoff, number of records and SHA-256/Git blob references:
- Feature and mapping version (including date timezone, 7×7 map, bit ordering, zero bins and missing values):
- Training/validation/test chronology and forbidden outcome leakage:
- Exact fair probability model: six distinct integers chosen from 1..49 without replacement, or an expressly justified alternative:

## Before-outcome analytic protocol

- One main hypothesis/endpoint, including units and direction:
- Fixed candidate/model set, parameter values and seeds:
- Correct conditional/exchangeability or synthetic IID null, with full-pipeline re-selection if training/tuning occurs:
- Multiplicity across endpoints, variants and earlier inspected research:
- Sample size/power rationale, uncertainty treatment and planned decision threshold:
- Negative controls, adversarial tests and deterministic rerun command:
- Stop, review and terminal evaluation conditions (avoid outcome-driven truncation):

## Verification, decision and next work

- Artifact path and original Git blob SHA / pinned source report:
- Implementation and source-reproducibility CI evidence:
- Result (including effect size, uncertainty/p-value or exact identity, not only a PASS flag):
- Scientific decision label (may be NO_EVIDENCE or PENDING):
- Known limitations, source authenticity and out-of-sample qualifications:
- New question this result resolves or raises:
- **Non-goals:** no automatic betting, no editing earlier Seal, no promotion from retrospective exploration to prospective prediction.

A completed engineering task and an unfinished study can coexist: a closed Issue is not evidence of a statistically verified future result.
