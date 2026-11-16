# Lotto649 prepaid ticket audit (2026-11-09 purchase)

Status as of 2026-11-14 (Asia/Taipei). This is a purchased-ticket settlement ledger, **not** a prediction or a modification to the canonical historical dataset.

## Immutable ticket definition

- Official draw IDs: **115000104 through 115000113**, inclusive (10 draws).
- Ticket dates: 2026-11-10 through 2026-12-11.
- Four lines per draw, NT$50 per line; **prepaid cost NT$2,000**.
- **Ticket order must never be alphabetically re-sorted:**
  1. A: 04 09 17 24 29 42
  2. B: 10 13 14 27 28 34
  3. D: 03 07 18 33 40 47
  4. C: 06 12 19 20 44 48

## Draw reconciliation

| Draw ID | Date | Six main numbers (ascending) | Special | A main/special | B main/special | D main/special | C main/special | Prize this draw | Verification |
|---|---|---|---|---|---|---|---|---:|---|
| 115000104 | 2026-11-10 | 11 23 30 32 40 41 | 28 | 0/no | 0/yes | 1/no (40) | 0/no | NT$0 provisional | Numbers corroborated in previous review, **official historical record not directly verified**; do not mark official-confirmed |
| 115000105 | 2026-11-13 | 07 08 13 15 25 46 | 43 | 0/no | 1/no (13) | 1/no (07) | 0/no | NT$0 confirmed | Official CTBC result and prize table verified 2026-11-14 |
| 115000106–115000113 | Future scheduled draws | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Not yet drawn / not verified |

All four lines for #115000105 are **no prize** under the official prize categories; official payout per line NT$0. For #115000104, no line meets the ordinary prize thresholds on the reported numbers, but operator-source confirmation remains outstanding.

## Financial summary

- Progress: **2/10 draws**, 8 pending.
- Official-confirmed winnings: **NT$0** (draw #115000105).
- Provisional winnings from reviewed draw #115000104: **NT$0**, subject to official verification.
- Total provisional cumulative winnings: **NT$0**.
- Net against full prepaid ticket cost NT$2,000: **−NT$2,000 provisional** (includes eight undrawn prepaid draws; not realized loss from two draws alone).
- Cost allocated to the two drawn periods: NT$400.

## Verification and update policy

1. At each update, recheck every draw from #115000104 through the latest officially published draw; deduplicate strictly by draw ID.
2. For each draw record six main numbers, special number, per-line main matches and special hit, official prize category, and **official per-line payout**; use `pending` when unavailable.
3. Do not substitute news coverage for official confirmation. Do not infer official prize amounts for unverified draws.
4. Preserve A/B/D/C ticket order and original numbers. Do not recommend additional purchases.
5. Stop after official verification of #115000113 (scheduled 2026-12-11).
6. Keep this ticket ledger separate from source BIN/CSV and sealed research snapshots.

Official verification pages:
- https://lotto.ctbcbank.com/result_all.htm (official current results and prize allocation; #115000105 confirmed 2026-11-14)
- https://www.taiwanlottery.com/lotto/result/lotto649/ (official historical result lookup)
