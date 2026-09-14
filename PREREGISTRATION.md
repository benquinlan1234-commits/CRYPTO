# Pre-registration: funding-rate extremes (Task 1)

**Frozen:** 2026-09-14, before any multi-year funding data existed on disk.
**Status:** pre-data. The authoring session had no network route to any exchange
or to the archive, so nothing in this document could have been informed by
looking at the answer. Amendments after the first `03_` run must be recorded in
"Amendments" with a date and a reason.

**Data source:** `data.binance.vision`, futures/um monthly `fundingRate` and
`klines/1h`, from 2020-01. The `fapi.binance.com` API answers 451 from
Australia; the archive does not, and reaches further back regardless.

## 1. The unresolved question

Prior work on OKX tested four funding-extreme configurations. All four flipped
sign from exploration onward. Validation and holdout both showed extremes
*following through* rather than reverting — but those slices were 19 days each
and adjacent, because OKX caps funding history near 286 points (~95 days).

This is the one prior test where **data length, not the result, was the binding
constraint.** The purpose here is to redo it with three slices long enough to
mean something.

## 2. Hypothesis

At each funding settlement, rank the universe cross-sectionally by funding rate.

- **Crowded longs** = top X% by funding (longs paying shorts).
- **Crowded shorts** = bottom X% by funding.

Portfolio **P**: short the crowded longs, long the crowded shorts, equal weight,
dollar-neutral.

```
ret_P(t -> t+h) = mean(ret of bottom-X%) - mean(ret of top-X%)
```

- `ret_P > 0` ⇒ **reversion**: crowding is punished.
- `ret_P < 0` ⇒ **follow-through**: crowding persists. (What the OKX 19-day
  slices suggested.)

**The direction is not pre-specified.** Follow-through is as tradeable as
reversion — you flip the book. So the test is two-sided, and "stable sign" is
the bar rather than "positive sign". Two-sidedness is priced into the p-values.

## 3. Universe and sample

14 symbols, carried over unchanged so results stay comparable to the prior work:
BTC, ETH, SOL, XRP, DOGE, NEAR, UNI, SUI, ARB, BCH, WLD, LTC, AVAX, LINK.

The panel is **unbalanced** — SUI and WLD list in 2023, ARB in 2023. A
cross-sectional rank is meaningless on three names, and a 20% cutoff of 3 names
selects nothing. So:

- **`min_breadth = 8`.** A timestamp enters the sample only if ≥8 symbols have
  both a settled funding rate and a complete forward return. This is fixed now,
  not tuned later.
- The 60/20/20 calendar split is applied **to the breadth-qualified span**, not
  to the full history of BTC alone.

## 4. Timing, and why there is no lookahead

The funding rate settled at time `t` is public at `t`. It predicts the return
from `t` to `t+h`. Nothing is estimated from a formation window, so the
"signal takes N seconds to measure" objection does not apply — the rate is
published, not computed.

Returns are log close-to-close on the funding grid, taken from 1h klines
sampled at grid timestamps.

## 5. The funding-interval trap

Binance moved several alts from 8h to 4h funding. **A raw cross-sectional rank
that mixes 4h and 8h rates is corrupted** — a 4h symbol's per-interval rate is
mechanically about half an 8h symbol's for the same annualized carry, so the
4h names would sort to the bottom for a reason that has nothing to do with
crowding.

Mitigation, fixed in advance: the ranked quantity is always **8h carry** — the
funding actually paid over the same eight hours. The common grid is
00:00 / 08:00 / 16:00 UTC, and the raw rates settled inside each 8h bucket are
summed. For an 8h symbol the bucket holds one settlement and the sum is the rate
itself; for a 4h symbol it holds two, and their sum is the comparable quantity.
No per-rate rescaling is applied, because summing already produces the
like-for-like figure.

The archive carries `funding_interval_hours` in the file, so the interval is
**read, not inferred from settlement spacing**. It is used to check bucket
completeness: a bucket holding fewer settlements than the stated interval
implies is scaled up to a full 8h of carry, and one missing more than half its
settlements is dropped rather than extrapolated.

## 6. Cells tested — count stated up front

| cell | horizon | cutoff |
|---|---|---|
| `h8_q20`  | 8h  | top/bottom 20% |
| `h8_q10`  | 8h  | top/bottom 10% |
| `h24_q20` | 24h | top/bottom 20% |
| `h24_q10` | 24h | top/bottom 10% |

**4 cells.** Holm–Bonferroni across all 4. No cell is added later; if one is,
the count changes and every p-value is recomputed.

## 7. Inference

- Day-clustered (UTC) standard errors. **The effective sample is days, not
  observations** — the 8h grid gives 3 rows per day, and they are not
  independent.
- Reported alongside: a block bootstrap over whole days, 2000 resamples.
- Drift demeaning is applied to the per-symbol leg returns before forming `P`.
  The long-short construction is already cross-sectionally demeaned, so this is
  belt-and-braces; both the demeaned and raw figures get reported.

## 8. Cost bar

The portfolio trades two legs. Cost is charged on **realized** turnover — name
overlap between consecutive rebalances — not on an assumption of full rotation,
because funding extremes are persistent and assuming full rotation would
overstate costs.

```
cost per rebalance = fee_bps × (turnover_long + turnover_short)
```

with `fee_bps = 5` (taker, per side) and turnover counting both exits and
entries. Full rotation ⇒ 4.0 × 5bps = **20bps per rebalance**; that is the
worst case, and the realized figure will be lower.

Both gross and net-of-cost means are reported. **The net figure is the one that
decides.**

## 9. Pass criteria — all must hold

1. **Sign stability.** Identical, non-zero sign of mean `ret_P` in exploration,
   validation and holdout. A flip anywhere kills the cell outright, whatever
   the magnitudes.
2. **Exploration.** Holm-corrected p < 0.05 across the 4 cells, day-clustered.
3. **Validation.** Same sign, and p < 0.05 on the direction carried forward
   from exploration.
4. **Cost.** Net-of-cost mean keeps the same sign in every slice. A gross
   effect that costs eat is a negative result.
5. **Holdout.** Touched **once**, and only if 1–4 hold in exploration *and*
   validation. It confirms or kills; it never selects among cells.

## 10. What would make this a negative result

Stated now so it cannot be renegotiated later:

- Any sign flip across the three slices → dead, as the 4h reversion was
  (−17.8 → +54.6 → −19.2).
- Survives gross but not net of cost → dead.
- Only the 10% cutoffs survive while the 20% cutoffs flip → treated as dead,
  not as "the effect lives in the tails". That pattern is what thin-tail noise
  looks like, and the 10% cells hold roughly 1–2 names per side.
- Holdout disagrees → **reported as a failure.** The holdout is not re-run with
  a different parameter.

A negative result here is a real outcome: it would close out the one prior test
whose verdict was blocked by data length rather than by the data.

## Amendments

**2026-09-14 (pre-data, before any `03_` run on real data).** Data source changed
from the Binance REST API to the `data.binance.vision` archive, because the API
is geo-blocked (451) from Australia while the archive is not. Consequences:
history starts 2020-01 rather than at the API's retention limit, and
`funding_interval_hours` is read from the file instead of inferred from
settlement spacing — see §5. No hypothesis, cell, cutoff, horizon or pass
criterion was changed. Recorded here rather than edited in silently.
