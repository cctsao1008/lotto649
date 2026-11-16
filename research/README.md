# Lotto649 research

| Location | Contents |
| --- | --- |
| [lotto7x7/](lotto7x7/) | Versioned implementations by research domain |
| [experiments/](experiments/) | Experiment settings and [ledger registry](experiments/ledger_registry_v1.json) |
| [prospective/](prospective/) | Pre-draw seals, frozen settings, and the sky calendar |
| [artifacts/](artifacts/) | Registered reports and input snapshots, including the USNO archive |
| [docs/](docs/) | [Study index](docs/research_index.md), [evidence index](docs/research_evidence_index.md), and [early image studies](docs/archive/image_research.md) |
| [replays/](replays/), [tools/](tools/), [tests/](tests/) | Replay scripts, C++ tools, and domain regressions |

Run commands from the repository root. Historical CLI names are available with
`python -m lotto7x7 research --help`. [Replay instructions](../docs/architecture.md#ci-and-historical-replays)
describe the CI matrix and local execution. The [evidence registry](evidence_registry_v1.json)
pins the original bytes of published evidence, including documents with historical
paths and commands. The current replay IDs and scripts are listed in
[replays/cases.json](replays/cases.json) and [replays/](replays/).
Replays write local reports to `results/`.

```bash
python -m lotto7x7.experiment_ledger verify
python -m lotto7x7.research_governance verify
python -m unittest discover -s research/tests -p 'test_prospective_readiness_v032.py' -v
```
