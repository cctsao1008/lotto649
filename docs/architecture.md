# Repository architecture

The `lotto649` package runs a single strategy. `draw_archive.py` verifies the
historical draws, `statistics.py` scores the candidate pool, and
`hexagram.py` scores matches to the target date's hexagram.
`pipeline.py` forms statistical bands and applies hexagram ordering inside
each band before returning the requested number of combinations.

| Location | Responsibility |
| --- | --- |
| `lotto649/` | Main pipeline, command line, draw archive, and shared scoring primitives |
| `config/strategy.json` | Strategy parameters |
| `data/bin/`, `data/csv/` | Historical draw archive and yearly CSV exports |
| `data/astronomy/`, `data/reference/` | Calendar snapshots and exact shape reference |
| `research/lotto7x7/` | Versioned implementations grouped by research domain |
| `research/evidence_registry_v1.json` | Registry of research evidence |
| `research/experiments/` | Experiment registry, settings, and registered candidate artifacts |
| `research/prospective/` | Preserved pre-draw protocols and related artifacts |
| `research/artifacts/` | Research snapshots, USNO source data, and published results |
| `research/docs/`, `research/replays/` | Research notes and replay entry points |
| `research/tools/`, `research/tests/` | Research utilities and domain tests |
| `tests/strategy/` | Selection, archive integration, and CLI tests |
| `docs/operations/` | Data formats and maintenance |
| `docs/calendar/` | Calendar data provenance for the main strategy |
| `results/` | Ignored local output from selections and research replays |

Run the main pipeline with `python -m lotto649 select --help`. Historical
command names remain available with `python -m research.lotto7x7 research --help`.
Run the full test suite with `python -m pytest -q tests research/tests`.

## CI and historical replays

`research_regression.yml` runs domain regression groups. From its GitHub Actions
page, choose a `replay` ID from `research/replays/cases.json` to
repeat an archived end-to-end experiment, or choose `all` to run the full
matrix. Replays can take several minutes each and keep their former validation
commands and report assertions. Run one locally from the repository root:

```bash
export LEGACY_ARTIFACT_DIR="$(mktemp -d)"
bash research/replays/coverage_burnside_v026.sh
```

Original workflow definitions are retained under `research/legacy-workflows/`
as historical snapshots. The one-time publications and branch cleanup are
archived there with their original scope and Git history. The dated append
tests now run in the M2/M3 integrity workflow and the shared regression suite.
The multi-job v0.27 P3e replay retains its separate manual sharding and aggregation workflow.
The v0.25 foundation workflow remains independent for its exact covariance, joint coverage, and retrospective research checks.

## Frozen artifact paths

Registries reference the current paths and keep the original blob hashes.
Sealed documents and published evidence retain their original bytes, including
any embedded historical paths. The prospective readiness audit resolves the
sealed sky-table reference in the current layout and verifies its original
SHA-256. Use the commit and original path recorded by the study when
inspecting its initial publication in Git history.

## Register a parameter-only research experiment

Use `research/experiments/configs/<experiment_id>.json` with the hypothesis, supported
engine, historical prefix and all run parameters. Add a `RESEARCH_CONFIG`
entry with the same ID, path and `git hash-object <file>` to the M1 public
registry, then run:

```bash
python -m research.lotto7x7.experiment_ledger verify
python -m research.lotto7x7.experiment_runner v031-h2-geometry-ci --out results/v031_h2_replay.json
```

The example reuses the existing H2 engine. The runner verifies the registered
Git blob and both BIN prefixes against the source archive, evaluates only the
pinned history, and writes a new report under `results/`. Later append-only
draws leave the experiment input fixed. Each report records the configuration
hash and the input fingerprint. Parameter variations of a supported engine
use a new configuration and experiment ID. A new algorithm belongs in the
appropriate domain and can be exposed through the same runner.

## Historical I Ching artifacts

The P1 and P2 replay CSVs omit duplicate localized name columns. Romanized
hexagram and trigram names, codes, timestamps, and draw data remain available.
The P1A state table omits its duplicate localized name column. These changes
produce new file hashes. The replay engines verify the new input hashes.

The registered P1A and P2A JSON reports retain their original result and
source hashes. Replaying all 3,999 permutations with the English-label inputs
reproduces the original statistics and decisions. The original files are
available in [Git history](https://github.com/cctsao1008/lotto649/tree/345399c09b037cbdac67467bd908f18f7e4187ba/data/research).
