# Lotto649

A research tool for ranking Lotto649 combinations. **Statistics rank the candidate pool, hexagram scores refine the order within each statistical band, and the pipeline returns the top N combinations.**

## Quick start

Run from the repository root with Python 3.10 or later. Selection requires NumPy.

```powershell
python -m pip install "numpy>=1.26,<3"
python -m lotto649 select --date 2026-11-17 --out results/selection_20261117.json
```

`--date` is the target draw date. Selection uses draws strictly before that date. Omit `--out` to print JSON to the terminal. The report records the history cutoff and fingerprint, the date's hexagram, and each candidate's statistical components, band, and hexagram score.

## Selection architecture

| Stage | Responsibility | Implementation |
| --- | --- | --- |
| Statistical ranking | Generate distinct candidates with a fixed seed. Score deficits in the exact 6/49 sum distribution, individual number frequencies, and 7×7 shape classes. | [statistics.py](lotto649/statistics.py) |
| Hexagram ordering | Build a hexagram from the target date's solar term and lunar day. Within each statistical band, rank candidates by the number of matching lines. | [hexagram.py](lotto649/hexagram.py) |
| Output | Preserve the order of the statistical bands and return the top N candidates. | [pipeline.py](lotto649/pipeline.py) |

[config/strategy.json](config/strategy.json) controls the strategy. The defaults use 2,000 candidates, return four, and give equal weight to the three statistical components. A band includes candidates within `epsilon` (default 0.01) of its highest remaining statistical score. Within a band, the pipeline sorts by hexagram score, statistical score, and then ascending number tuple. Each statistical component ranges from −1 to 1.

The calendar snapshots are derived from Taiwan's Central Weather Administration annual PDFs and cover **2023–2026**. The active solar term at noon in Taipei determines the upper trigram by grouping the 24 terms in threes from the winter solstice. The lunar day determines the lower trigram using eight sectors on a 30-day scale. Both trigrams use Gray coding. Candidate numbers are sorted from the bottom line to the top, with odd numbers representing yang and even numbers representing yin. Dates outside the calendar coverage return an error.

## Repository layout

| Path | Contents |
| --- | --- |
| `lotto649/` | Selection pipeline and shared ranking primitives |
| `config/strategy.json` | Strategy parameters |
| `data/` | Draw history, calendars, and exact reference counts |
| `research/` | Study code, experiments, sealed protocols, artifacts, replays, and research tests |
| `tests/strategy/` | Main pipeline tests |
| `docs/` | Architecture, data operations, and calendar sources |
| `scripts/` | Draw archive and calendar maintenance |
| `.github/workflows/` | Active GitHub Actions workflows |

## Data and research

- [Draw archive and source manifest](data/bin) provide the draw history. [Annual CSV exports](data/csv) are available for inspection.
- [Exact shape reference](data/reference/shape_class_counts.json) covers all 13,983,816 combinations. It complements the exact sum distribution used in statistical ranking.
- [Repository architecture](docs/architecture.md) describes the selection workflow and maintenance layout. The [research guide](research/README.md) links study documentation, evidence, and replay commands.

The statistical scores describe historical distribution deficits. The predictive performance of the combined strategy still needs out-of-sample validation.
