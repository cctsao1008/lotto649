# Lotto7x7 v0.6 Phase 2–3 — Ticket Selection and Coverage Audit

Phase 1 produces the historical-image-derived **candidate pool** (default M=100). Phase 2 selects **N=1..5 distinct tickets** from that frozen pool. Phase 3 independently audits their fair-draw combinatorial coverage. This is a research-only selection system, not evidence of above-chance prediction.

## Three commands

```powershell
cd D:\my-github\lotto649
$py = '.\.venv\Scripts\python.exe'
git pull --ff-only origin main
& $py -m unittest discover -s tests -v

& $py -m lotto7x7 shape-pool --size 100 --seed 649 --out results/shape_pool_v06.json
& $py -m lotto7x7 bet-select --pool results/shape_pool_v06.json --count 5 --strategy all --out results/bet_selection_v06.json
& $py -m lotto7x7 coverage-audit --bets results/bet_selection_v06.json --out results/coverage_audit_v06.json
```

Optional controls: `bet-select --strategy random|shape-diverse|coverage-optimized --seed 1658`; `coverage-audit --samples 50000 --seed 20261108`. `coverage-audit --draw 1,2,3,4,5,6` optionally scores a **known, already-revealed actual draw** after selection; never include this during selection.

## Selection

- **Random:** seeded uniform sampling without replacement from the same fixed pool.
- **Shape-Diverse:** greedy minimax D4/translation shape-distance selection with exact-number-overlap tie breakers. This treats 7×7 image similarity separately from common numbers. The pool's sequential spatial layout is artificial, not lottery physics.
- **Coverage-Optimized:** exact combinatorial greedy forward selection, followed by a deterministic single-swap improvement pass. The comparison objective is **lexicographic**: maximize the exact number of draws where the portfolio has at least 3 hits; break exact ties by at least 4, then at least 5. No tuned probability weights are used. The result is an *approximate combinatorial portfolio optimizer* (greedy/local search), **not** a guaranteed global maximum and not a prediction of the next draw.

The `all` mode generates three alternative N-ticket portfolios; it does **not** mean buying 3×N tickets. The seed influences the random portfolio and tie-breaking. The source draw cutoff, history hash and ordered pool SHA256 are carried to the report; the audit checks the frozen pool commitment and membership of each ticket.

## Exact portfolio probabilities

For a *fixed* portfolio of 1–5 distinct bets, the auditor computes the **exact** distribution of `max_hits=0..6` over all `C(49,6)=13,983,816` equally likely six-number outcomes. For each number from 1 to 49, construct an N-bit mask describing which tickets contain it. Group by identical masks. A dynamic program counts ways to choose `k` numbers from each class, weighted by `C(class_size,k)`, until exactly six are chosen. Count the maximum hits of any ticket. This computes the union for `P(max_hits≥3/4/5/6)` without enumerating 13.98 million outcomes and without Monte Carlo error.

Checks: all counts sum to 13,983,816; the single-ticket distribution is `C(6,h)C(43,6-h)`; and exact full-hit coverage for any distinct N-ticket portfolio is **N / 13,983,816**, regardless of strategy. The exact numerator for six hits is N, not the pool size M.

## The M=100 research-pool estimate

For a candidate pool larger than 5, the current audit computes *partial-hit* pool coverage via IID draws sampled uniformly over all 6/49 combinations. It marks these values `iid_uniform_monte_carlo` and provides successes, standard error and a 95% Wilson interval for the ≥3/4/5 thresholds. It reports the pool's exact 6/6 coverage separately as `M / 13,983,816` — a property of all M unique pool combinations, **not the practical N-ticket probability**.

The pool Monte Carlo sequence is independent of selection/optimization; no simulated or actual winning draw is fed into the optimizer. Use more `--samples` to improve precision of rare 5-hit pool estimates. The small N-ticket portfolios require no statistical confidence interval because their coverage is calculated exactly.

## Comparison and limitations

The auditor identifies observed Pareto-nondominated strategies by their exact ≥3/4/5 coverage. Coverage-Optimized can improve partial-hit union coverage for a fixed N, but it cannot improve the full 6/6 probability under a fair independent draw. Any historical retrospective analysis must use an earlier cutoff for each target (rolling-origin), and previously inspected draws are exploratory rather than a prospective validation. Baselines B0 (global random N), B1 (random M=100 followed by selections), and S1 (image-derived M=100) remain to be implemented as a separate rolling-origin evaluation; they are **not** claimed by these three CLI commands.