# Lotto649 — Research Evidence Index

This index is generated from [`research/evidence_registry_v1.json`](../research/evidence_registry_v1.json).
It describes **recorded evidence at the listed snapshot dates**, not a live status feed or a proof of lottery predictability.
The Git blob hashes pin *original artifacts*, so later source append operations do not silently rewrite historical experiments.

| Experiment | Domain | Evidence tier | Scientific decision | Engineering | Quantity* | Source |
| --- | --- | --- | --- | --- | ---: | --- |
| fair-shape-exact-v016 | probability | STRUCTURAL_THEOREM | PROVEN_MATHEMATICALLY | PASS | 13983816 | [mathematical_proof](../docs/shape_taxonomy_complete_universe_p1.md) |
| geometry-incremental-h2-v031 | geometry | EXPLORATORY_RETROSPECTIVE | NO_EVIDENCE | PASS | 200 | [report](../docs/incremental_geometry_v031_h2_results_20261112.md) |
| iching-p1a-20261113 | iching | EXPLORATORY_RETROSPECTIVE | NO_EVIDENCE | PASS | 2476 | [report](../data/research/v24_iching/v24_p1a_gua_qi_association_2498.json) |
| iching-p2a-20261114 | iching | EXPLORATORY_RETROSPECTIVE | NO_EVIDENCE | PASS | 2476 | [report](../data/research/v24_iching_p2a/v24_p2a_hexagram_association_2498.json) |
| iching-p3-static100-20261114 | iching | PROSPECTIVE_PRESEALED | PENDING | PASS | 0 | [seal](../prospective/v24_p3_usno_hexagram_static100_20261114_seal.json) |
| m0-v23-static200-20261113 | portfolio | PROSPECTIVE_PRESEALED | PENDING | PASS | 0 | [seal](../prospective/v23_m0_global_static200_20261113_seal.json) |
| m0-walkforward-v22 | portfolio | EXPLORATORY_RETROSPECTIVE | NO_EVIDENCE | PASS | 325 | [report](../docs/strategy_validation_v22.md) |
| martingale-deficit-v030 | statistics | EXPLORATORY_RETROSPECTIVE | NO_EVIDENCE | PASS | 2498 | [report](../docs/distribution_dynamics_v030_p0_p1.md) |
| usno-source-2004-2026 | source_engineering | SOURCE_ARCHIVE | VERIFIED_SOURCE | PASS | 1276 | [source](../data/astronomy/usno_official/manifest_2004_2026.json) |

*Quantity is the original record unit: legal combinations for exact theory, astronomical events for USNO source, historical draws for retrospective studies and evaluated future draws for sealed trials.*

## Interpretation

- `PROVEN_MATHEMATICALLY` describes an exact structural or combinatorial result; it does **not** imply a forecast advantage.
- `NO_EVIDENCE` records the scope of a particular exploratory test; it is not a proof that all possible signals are absent.
- `PENDING` on a `PROSPECTIVE_PRESEALED` record means a fixed study exists **before its future outcomes are evaluated**. An issue can be closed for engineering while its statistical result remains pending.
- `PASS` means integrity/engineering checks succeeded. It never means `VALIDATED_GO` or a better chance of winning.
- Source/protocol/result documents retain their original versioned cutoffs; newer official draws must not be silently joined to those older experiments.

## How to extend

See [Research Knowledge System](research_knowledge_system.md) for domain taxonomy, lifecycle gates, status transitions and examples.
Follow [New Research Hypothesis Template](research_hypothesis_template.md).
Existing [M1 Experiment Ledger](experiment_ledger_m1.md) remains the authority for public seal identities and local private paid registrations.

## Validation

```powershell
python -m lotto7x7.research_governance verify
python -m unittest discover -s tests -p 'test_research_governance.py' -v
```
