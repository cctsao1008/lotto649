# Repository layout and research reproducibility

| Location | Responsibility |
| --- | --- |
| `lotto7x7/selection/` | Main selection pipeline, statistical scoring, hexagrams and JSON CLI |
| `config/selection.json` | Selection parameters |
| `data/reference/` | Exact fair shape-class counts reused by selection and research |
| `lotto7x7/` | Shared bitmap, BIN, coverage and research infrastructure |
| `lotto7x7/geometry/` | Shape taxonomy, geometry and v0.31 incremental geometry |
| `lotto7x7/statistics/` | v0.30 distribution, inference and conditional Monte Carlo |
| `lotto7x7/portfolio/` | Coverage, prize probability and historical portfolio experiments |
| `lotto7x7/prospective/` | Versioned seals and evaluation protocols |
| `lotto7x7/astronomy/` | USNO and I Ching research |
| `lotto7x7/legacy/` | Historical CLI, image methods and frozen strategy-verification series |
| `docs/archive/` | Historical project overview and version notes |

Tests follow the same domains. Run all tests from the repository root with
`python -m pytest -q`, or use `python -m unittest discover -s tests -v` for
the unittest subset. Narrow discovery by the existing file name pattern
still works. The historical image training and diagnostics files retain
their top-level locations because v0.3 verifies their exact source hashes.

Run the main pipeline with `python -m lotto7x7 select --help`. For archived research commands, use `python -m lotto7x7 research --help`.

Existing Python imports and `python -m lotto7x7.<old_module>` invocations
resolve to the canonical implementation. New internal imports use the domain
path. Direct source file paths use the new layout.

## CI and historical replays

`regression_legacy.yml` runs domain regression groups. From its GitHub Actions
page, choose a `replay` ID from `scripts/legacy-regression/cases.json` to
repeat an archived end-to-end experiment, or choose `all` to run the full
matrix. Replays can take several minutes each and keep their former validation
commands and report assertions. Run one locally from the repository root:

```bash
export LEGACY_ARTIFACT_DIR="$(mktemp -d)"
bash scripts/legacy-regression/coverage_burnside_v026.sh
```

Original workflow definitions are retained under `.github/legacy-workflows/`
as historical snapshots. The one-time publications and branch cleanup are
archived there with their original scope and Git history. The dated append
tests now run in the M2/M3 integrity workflow and the shared regression suite.
The multi-job v0.27 P3e replay retains its separate manual sharding and aggregation workflow.
The v0.25 foundation workflow remains independent for its exact covariance, joint coverage, and retrospective research checks.

## Register a parameter-only research experiment

Use `experiments/configs/<experiment_id>.json` with the hypothesis, supported
engine, historical prefix and all run parameters. Add a `RESEARCH_CONFIG`
entry with the same ID, path and `git hash-object <file>` to the M1 public
registry, then run:

```bash
python -m lotto7x7.experiment_ledger verify
python -m lotto7x7.experiment_runner v031-h2-geometry-ci --out results/v031_h2_replay.json
```

The example reuses the existing H2 engine. The runner verifies the registered
Git blob and both BIN prefixes against the source archive, evaluates only the
pinned history, and writes a new report under `results/`. Later append-only
draws leave the experiment input fixed. Each report records the configuration
hash and the input fingerprint. Parameter variations of a supported engine
use a new configuration and experiment ID. A new algorithm belongs in the
appropriate domain and can be exposed through the same runner.
