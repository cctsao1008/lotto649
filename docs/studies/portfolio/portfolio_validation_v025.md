# Issue #8 P5 — Future-only portfolio validation design

Status: design only. The archived walk-forward compared covariance
estimators under identical theoretical prize coverage, and reused the
historical v0.26 folds. The Pareto archive measures exact coverage and
descriptive distribution repair. Those studies supply the motivation
and technical controls for a future test.

## Design before enrollment

Select one research comparison and one direction before viewing any
eligible outcome. A candidate design pairs a disjoint four-ticket
historical-repair policy with a history-blind disjoint four-ticket
control. For both, use the same frozen pool and require the same disjoint coverage.
Pin a source archive cutoff, ordered history hash, code commit,
candidate-pool hash, scoring rule, ridge, search budget, and tie policy.
From the frozen cutoff, generate the two portfolios deterministically
and check that both pass full official prize-coverage parity. The same
four tickets per arm stay fixed for all draws in the test.

Before sealing, measure the exact probability of a discordant award
between the two selected portfolios under fair IID. Use the eight-ticket
joint event by enumerating each six-main draw once and counting legal
special balls in the two portfolios’ award sets using bit masks. Each marginal
award probability is 12.2731459% under fair IID. Their discordant
probability depends on the cross-arm intersections. Equal portfolios
produce zero discordance and are ineligible. Choose the fixed number
of eligible draws, with no interim stopping, for a stated alternative
and at least 80% power based on the calculated discordance probability.
Choose the effect size on the paired **award incidence difference** scale
before finalizing a horizon. No trial is enrolled until its exact
discordance calculation, practical horizon and power are recorded.

## Endpoint and decision

For each admitted draw, record whether each portfolio has at least one
official eight-tier prize. The primary effect is (repair-only awards
minus control-only awards) divided by the number of verified draws.
Conditional on the fixed number of discordant draws, fair IID implies
an exact Binomial(discordant draws, 1/2) sign test because the marginal
award probabilities are equal. Use one upper-tail test at alpha 0.05
and report wins, losses, ties and an exact interval for the sign rate
among discordances. Record absolute paired incidence difference alongside
that interval. No other threshold or arm is a confirmatory endpoint in
this design. Expand to more endpoints only under a separate prior
multiplicity plan.

Before the first scheduled draw, write a new research seal with a UTC
timestamp, chronological target schedule and endpoints, frozen input and
model hashes, both exact generated portfolios, power calculation and
analysis code. Keep append-only verified results through the sealed
fixed horizon. Require two independent publisher snapshots per admitted
draw, checked against each other and hashed with a draw ID, local date,
six mains and special ball. A mismatch blocks that draw and must be
resolved under the prespecified admission rule.

At the sealed horizon, a result with p <= 0.05 and a positive effect
qualifies for independent replication. Completed results failing the
criterion receive NO_GO for the registered hypothesis. Incomplete
data evidence, invalid coverage parity, or a prematurely stopped study
receive INCONCLUSIVE. A conclusion of predictive advantage requires a
separate replicated outcome and review.

## Horizon feasibility — 2026-11-16

The archived Pareto repair optimum meeting the coverage floors and the
history-blind v0.21 coverage output both have 24 distinct labels.
Enumeration over all 601,304,088 main-plus-special outcomes gives
59,980,697 repair-only outcomes and 59,980,697 control-only outcomes.
The exact fair discordance probability is 19.9502%. Independent single-arm
prize DP reproduces each marginal. The exact sign-test powers below average
over a binomial discordance count at that fair discordance rate.

| Absolute incidence gain | Power at 200 draws | Evaluated horizon at or above 80% power |
| --- | ---: | ---: |
| +1 percentage point | 7.0% | 12,800 |
| +2 percentage points | 12.3% | 3,200 |
| +3 percentage points | 20.0% | 1,600 |
| +5 percentage points | 42.1% | 800 |

These are grid checkpoints. The alternative assumes independent paired
outcomes and fixes discordance at the fair value. The planned effect size
and horizon remain unset. This design has no enrolled draws.

```bash
OPENBLAS_NUM_THREADS=1 python -m lotto7x7.portfolio_validation_v025
```

See the [fixed-horizon power artifact](../../../data/research/v025_exact_covariance/v025_portfolio_validation_power_20261116.json).
