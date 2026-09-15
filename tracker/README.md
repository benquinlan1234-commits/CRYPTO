# Tracker instrumentation

`paper_track_instrumentation.py` implements the four reporting changes asked for
on the funding-fade / OI-spike paper tracker.

**Provenance note.** The live tracker (`paper_track.py`, `strategy_bybit.py`,
`bybit_backtest.py`) lives in `C:\Users\Ben\reclaim-mbo` on the local machine and
is *not* present in this repo or in the container this was written in. So this
module is written against an interface rather than against the real file: every
function takes explicit inputs and returns a string, and the book normaliser
accepts whatever shape `paper_track.py` already holds. Copy it into `reclaim-mbo`
and wire it with the snippets below. It is staged here only because this is the
crypto repo the session can reach.

Nothing here touches the signal, sizing or universe. Reporting only.

```bash
python tracker/paper_track_instrumentation.py    # self-test, prints a sample report
```

## 1. Expectation

Replaces `FUNDING should track ~0.9 Sharpe (proven)`.

The old line was wrong twice over: `0.9` is the *top* of the full-history band
(0.6-0.9, point estimate 0.57), not its centre, and full history is the wrong
benchmark for a forward test running in 2026. Both bands are now printed, with
the per-year series underneath so the regime dependence stays visible, and the
line states explicitly that scoring is against the recent-regime band.

```
full history 2022-2026 : Sharpe 0.6-0.9 (point est 0.57)
recent regime 2024+    : Sharpe 1.3-1.5
by year                : 2022:-1.32 2023:-0.46 2024:+1.07 2025:+1.37 2026:+2.09
SCORED AGAINST         : 1.3-1.5 (recent regime)
```

`EVALUATION_SHARPE` (1.4, the midpoint of the recent band) is what the noise band
is centred on. Change that one constant to rescore.

## 2. Noise band

Cumulative return over `t` years has mean `S*sigma*t` and sd `sigma*sqrt(t)`.
The mean is linear in `t`, the noise goes as `sqrt(t)`, so short windows are
dominated by noise. At 28 days and 11% annualised vol:

| | |
|---|---|
| expected P&L (S=1.4) | +1.18% |
| 1sd range | **-1.87% to +4.23%** |
| 2sd range | -4.91% to +7.27% |
| same band if edge were 0 | -3.05% to +3.05% |
| noise / signal | 2.6x |
| SE of a Sharpe estimated here | +/-5.1 Sharpe units |

So a *flat-to-down* four weeks is entirely ordinary even if the edge is real at
1.4 — the 1sd lower bound is -1.87%, and the null band overlaps the live band
almost completely. There is no pass/fail to be had here.

The band only clears zero at 1sd after `1/S^2` years:

* **186 days (~6 months)** for 1sd separation at S=1.4
* **745 days (~2 years)** for 2sd separation

`assess()` returns `WITHIN_BAND` for `|z| <= 1` and flags only outside it, so
ordinary noise is silent. A 2sd breach prints an investigate line that points at
execution, funding capture and slippage first, ahead of any conclusion about
edge.

## 3. Signal overlap

`book_overlap()` splits shared names by direction, which the raw count does not:

* **same-side** — same name, same direction in both books. This is what erodes
  diversification: one doubled position wearing two hats.
* **opposite-side** — the signals disagree and the exposures partly cancel,
  which *adds* independence rather than removing it.

Today's book (THETA, 1000PEPE, CRV short and INJ long in both) is 4 same-side,
0 opposite-side.

`implied_corr_floor = (same_side - opposite_side) / sqrt(n_funding * n_oi)` is
the correlation the two P&L streams would show from shared positions alone, if
every name's return were independent with equal vol. On a 14-a-side book that is
**+0.29 against a measured 0.09** — the structural overlap is three times the
measured correlation, so the measured figure is understating the linkage and the
report flags it. (Substitute your real book sizes; the function reads them off
the books. It is a structural floor, not a realised-correlation estimate — it
ignores the common crypto factor, which the net-neutral construction mostly but
not entirely removes.)

Each run appends to a JSON log, so the running averages and the persistent-name
counts survive restarts. A corrupt log degrades to empty rather than taking the
tracker down.

## 4. pandas deprecation

`pd.Timestamp.utcnow()` is deprecated because it returned a *naive* timestamp
despite the name. `utc_now()` returns `pd.Timestamp.now("UTC")`, which is
tz-aware.

That change is not purely cosmetic: comparing the new tz-aware value against a
naive timestamp raises `TypeError`. Check every comparison and subtraction
against the replaced call — if the other side is naive, localise it.

```bash
grep -rn "Timestamp\.utcnow\|pd\.Timestamp\.utcnow" .
sed -i 's/pd\.Timestamp\.utcnow()/pd.Timestamp.now("UTC")/g' paper_track.py
```

## Wiring

```python
from paper_track_instrumentation import instrumentation_report, utc_now

START = pd.Timestamp("2026-08-18", tz="UTC")   # set to real go-live
elapsed_days = (utc_now() - START).total_seconds() / 86400

print(instrumentation_report(
    elapsed_days=elapsed_days,
    funding_book=funding_book,        # {sym: signed weight} / {"longs":[...]} / Series
    oi_book=oi_book,
    history_path="logs/overlap_history.json",
    actual_return=cum_return,         # fraction of capital, net; None if nothing closed
    capital=100_000,
    measured_corr=measured_corr,      # None until enough points to estimate
))
```

Pass `measured_corr=None` until there are enough return observations to estimate
it — two rebalances is not enough, and a correlation off two points will read as
+/-1 regardless of the truth.

The building blocks (`expectation_block`, `noise_band`, `noise_block`,
`book_overlap`, `OverlapHistory`, `overlap_block`) are usable individually if the
existing report layout should be kept.
