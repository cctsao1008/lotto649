# Lotto649 v0.22.2 — Unlimited Fast Historical Hit Stream

## Purpose

Run at maximum available CPU throughput until the user presses Ctrl+C. This
supersedes the *paced* behavior of v0.22.1 only when invoking the **new**
`historical-hit-stream` command; the old command stays intact.

## Command

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main

& $py -m lotto7x7 historical-hit-stream `
    --catalog results/v015_v016_p1_shape_catalog.npz `
    --universe results/v015_v016_p1_shape_catalog_universe.json `
    --target all `
    --strategy shape_score `
    --batch-size 10000 `
    --out results/v0222_stream_01.json
```

Stop with Ctrl+C. The console prints a final result containing `hits/guesses`,
percentage, average guesses per historic full-six match, and filenames.
`--seed` is omitted by default, so each run draws a new 128-bit seed from the
OS cryptographic entropy source; the exact seed is saved in JSON to reproduce
that run. The generated guesses use high-throughput NumPy PCG64 pseudorandom
sampling, not a fresh OS entropy call for each of the six numbers.

## Method and counting

- A guess is **one** legal 6-number combination sampled uniformly and
  independently from all `C(49,6)=13,983,816` possibilities. Different
  guesses are sampled **with replacement**, even inside one batch.
- `--batch-size` is a **processing chunk**, *not a guess limit*. The default
  is 10,000 for prompt Ctrl+C responsiveness. `--max-batches 0` means there is
  no total count or time limit; use `Ctrl+C` to stop.
- Under `--target all`, targets are the 2,497 earlier historical drawings.
  `hits` counts **every exact-six historical match event**, including
  rediscovery of the same historic six-set; `guesses` counts **all completed
  guesses**, including duplicates. The ratio is exactly `hits/guesses` and
  its percentage is `100 * hits / guesses`. A separate field counts distinct
  historical six-sets found.
- A batch's guesses are generated before checking targets. `shape_score`
  merely sorts that random batch by **frozen L3 historical class ratios**.
  `uniform` leaves the IID pool order intact; because both use the same set
  for each completed batch, each batch has the same number of matches in
  either order. A full completed batch's hit ratio is unaffected by sorting.
- Expected hit rate for a uniform draw and a fixed target archive with `M`
  distinct historical six-sets is `M / 13,983,816`; e.g. approximately
  `0.017855%` when `M=2497`. More attempts change the precision of the
  historical-retrieval estimate, not the chance of a future lottery ticket.
- The `all` scorer was trained on the same archive being searched. This is
  **in-sample historical retrieval**, not predictive validation. For an
  exploratory train/target separation, pass `--target holdout`.

## Console and files

The console displays the first 10 **new distinct** historic matches and then
only every 100th new distinct match; repeating already discovered hits still
increase the total counter. This prevents terminal flooding at high speed.
`--show-all-hits` prints every *new distinct* historic match if desired.

Three non-overwritten outputs are created:

- `.json`: settings (including OS-derived seed), per-guess and exact expected
  baseline hit rate, aggregate counters, `INTERRUPTED`/`COMPLETED` status.
  Periodic safe atomic checkpoints at least once per 10 seconds (configurable).
- `_hits.csv`: unique previously drawn six-sets discovered, with their first
  discovery position and matching historical draw IDs. The total hit count
  may exceed CSV row count because repeated hits are not duplicated there.
- `_rounds.csv`: completed batch stats, cumulative guesses and hits.

Counts are updated after **entire batches** have been analyzed. If the user
presses Ctrl+C while computing a batch, that incomplete batch is omitted from
the final numerator and denominator. Ctrl+C normally flushes and closes the
CSV files and atomically writes final JSON. A forced terminal kill or power
loss can lose progress since the last checkpoint.

The purpose is historical lookup throughput and measurement, not predicting
future winning numbers. At most one six-number ticket is one guess; no
purchase or bets are placed by the command.
