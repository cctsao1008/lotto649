# v0.30 P2B — Four-Ticket Portfolio Structural Benchmark

Issue: [#32](https://github.com/cctsao1008/lotto649/issues/32).

## Purpose and nonclaims

P2A found no evidence of a predictive improvement: the fixed chronological
pilot produced 400 targets, mean percentile 0.505215, one-sided fair-IID
p=0.325 and eight-arm familywise p=1.000. Therefore P2B compares **four-ticket
portfolio geometry**, not a forecast model and not new real-money tickets.

For four **different** legal six-number combinations and one six-of-49 draw,
**jackpot probability is identically** `4 / C(49,6) = 1 / 3,495,954`, irrespective
of their union size. Portfolio overlap changes the *distribution* of less-than-six
main-ball matches. The match events are dependent; assuming four independent
prize events is invalid.

## Frozen evaluation design

Reuse v0.29 P3B's sequential 7x7 pair geometry, exact fair three-feature
moments and ridge-whitened coordinates, frozen geometry fingerprint and fully
audited history. For prefix size `n`, define `R=sum(z(history_i))` and the
**joint counterfactual** four-append energy `D4=||R+sum(z(ticket_j))||²/(n+4)`.
Repair gain is `||R||²/n - D4`. Four separate single-append gains **must not**
be summed as a surrogate.

Generate one fixed-seed, uniformly sampled, **distinct** candidate pool of
`pool-size` legal six-of-49 tickets. The four arms share this same pool and
four-ticket budget:

1. `random`: the first independent uniform 4-subset of pool indices; no tuning.
2. `coverage`: randomized greedy construction of four pairwise-disjoint tickets
   attaining the maximum 24 distinct numbers; no repair-driven tie-breaking.
3. `repair`: best initial random portfolio by `D4`, then at most four greedy
   coordinate-exchange passes minimizing `D4` without an overlap floor.
4. `hybrid`: minimize `D4` with at least 22 unique numbers (configurable),
   starting from a disjoint portfolio and the best eligible random seed, then
   the same fixed four-pass local search. This is a **coverage floor**, not
   an empirically fitted scalar objective weight.

The finite-pool optimization is a heuristic, **not certified globally optimal**.
All four portfolios are legitimate research examples but **not bets**.

## Exact probability engine

For each portfolio, assign every number in 1..49 a 4-bit ticket-membership mask.
Group numbers by mask; if a group contains `m` numbers, taking `j` of them
has multiplicity `C(m,j)`. Multiply all such group factors by an integer dynamic
program tracking total six drawn balls and four ticket hit counts. Normalize
integer coefficients by `C(49,6) = 13,983,816`. Report the **exact**
probability of at least 1/2/3/4/5/6 main-ball matches in the best of four
lines. This fully models dependencies due to overlap. Test against brute-force
enumeration on a small universe and preserve exact six-hit event count=4.

**Limit:** This does **not** include the special ball, exact prize tiers,
variable payouts, lottery taxes or the monetary expected value. It must not
be called an exact prize-winning probability.

## Fair-IID reference

An unoptimized random portfolio is generated from the same pool and four-ticket
budget; summarize the empirical coverage and counterfactual-repair distribution
from fixed-seed independent pool subsets. In addition, simulate `iid-reps`
independent 2,498-draw whole histories, rerunning the **entire candidate-pool,
random-control and heuristic-selection procedure**. Calibrate only the descriptive
repair-gain endpoints against these IID full pipelines with +1 finite-MC
correction. The null simulation does not validate future performance or uncover
fair-IID predictability.

Default configuration: `pool-size=512`, `random-portfolios=256`,
`min-hybrid-unique=22`, `iid-reps=19`, `seed=649`. CI smoke: 128 candidates,
32 random portfolios, 3 full IID controls. Increasing IID paths improves
p-value resolution but does not convert retrospective studies to prospective
confirmation. Do not adjust arm definitions based on the reported pilot.

## Execute in the existing Windows company checkout

```powershell
cd E:\lotto649
$py = '.\.venv\Scripts\python.exe'

# Run P2B regression tests
& $py -m unittest discover -s tests -p 'test_portfolio_benchmark_v030_p2b.py' -v
if ($LASTEXITCODE -ne 0) { throw 'P2B tests failed' }

# Run the frozen P2B structural pilot
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
& $py -m lotto7x7.portfolio_benchmark_v030 `
    --csv-root data/csv --bin-root data/bin `
    --pool-size 512 --random-portfolios 256 `
    --min-hybrid-unique 22 --iid-reps 19 --seed 649 `
    --out "results/v030_p2b_portfolio_$stamp.json"
if ($LASTEXITCODE -ne 0) { throw 'P2B pilot failed' }
```

CLI output is exclusive-create under ignored `results/`, includes the source
manifest and history fingerprints, configuration SHA256, pool SHA256, legal
research-only ticket examples, coverage, repair gain, exact fair max-match
histograms/probabilities, IID matched-gain controls and stated limitations.

## Study boundaries

- Source history is read only. Do not touch existing purchased four lines, ten
  drawings, prior sealed P3f/P5A experiments, M0 candidates or source files.
- Complete official source validation remains a separate provenance concern;
  the repository data gate verifies internal consistency.
- Ticket selection uses all already inspected data. Results are **exploratory**.
- A larger model-prediction claim would require future outcomes under a
  separately frozen and timestamped preregistration, not choosing the most
  favorable retrospective strategy.
- Neither optimizing coverage nor decreasing historical distribution energy
  increases the conditional probability of the next fair independent draw.
