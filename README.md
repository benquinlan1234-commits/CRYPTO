# crypto-perp-research

Perp research separate from the NQ work. Nothing here touches `reclaim-mbo`.

## Status: task 1 is blocked on network, not on code

This repo was authored in a remote container whose egress policy is an
allowlist covering GitHub and package registries only. **Every** market-data
host is refused at the proxy with a 403 on CONNECT:

| host | result |
|---|---|
| `fapi.binance.com`, `api.binance.com` | 403 CONNECT — egress denied |
| `api.bybit.com` | 403 CONNECT — egress denied |
| `www.okx.com` | 403 CONNECT — egress denied |
| `data.binance.vision` | 403 CONNECT — egress denied |
| `api.coingecko.com`, `api.kraken.com`, `api.hyperliquid.xyz` | 403 CONNECT — egress denied |
| `api.github.com`, `pypi.org` | 200 |

OKX is on that list, and OKX worked from the earlier chat sandbox — so this is
the container's egress policy, **not** a geo-block, and it says nothing about
whether Binance is reachable from Australia. That question is still open and
only your machine can answer it.

So the probe is built rather than run. One command answers task 1:

```bash
pip install -r requirements.txt
python scripts/01_probe_access.py
```

It distinguishes `geo_blocked` (HTTP 451 / restricted-location 403 — the venue
refusing your jurisdiction) from `egress_denied` (your network refusing the
connection) so the answer is unambiguous, and prints the earliest funding
timestamp each venue actually serves plus what a 60/20/20 split would give.

Then:

```bash
python scripts/02_fetch_funding.py --venue binance          # cache funding + 1h klines
python scripts/03_funding_extreme_test.py --venue binance   # exploration + validation
python scripts/03_funding_extreme_test.py --venue binance --touch-holdout   # once, at the end
```

## What is validated, and what is not

The analysis harness is tested end to end against synthetic data with a known
planted truth — which is independent of having exchange access, so it is done:

```
python tests/test_protocol.py            # 8 checks
python tests/test_pipeline_synthetic.py  # 4 checks
```

Results worth knowing:

* **Day-clustering is load-bearing.** On day-correlated data with a true effect
  of exactly zero, a naive t-test rejects **50.7%** of the time; the clustered
  test rejects **6.7%** against a nominal 5%.
* **No lookahead in the grid alignment.** Under a true null across 12 seeds the
  exploration median p is 0.61 with signs split 6+/6−. (12 seeds is a coarse
  calibration — it rules out a gross leak, not a subtle one.)
* **A real effect is recovered.** A planted +30bps/8h reversion reads
  +46.6 / +54.8 / +45.2bps across the three slices; the ~1.6× is the long-short
  rank spread, so the magnitude is right, not just the sign.
* **Drift demeaning removes drift exactly**, and the recorded 4h reversion
  pattern (−17.8 → +54.6 → −19.2) correctly fails the sign-stability bar.

**Not validated:** anything about real market data. No real number has been
computed. Every figure above comes from synthetic data.

## Two alignment traps the pipeline handles

Both would have quietly corrupted the result:

1. **Kline timestamps.** A Binance bar stamped `open_time = t` covers
   `[t, t+1h)`, so its close is the price at `t+1h`. Using it as the price at
   `t` hands the test a free hour of the future. The price at `t` is the close
   of the bar *ending* at `t`. There is a test asserting this.
2. **Mixed funding intervals.** Binance moved several alts from 8h to 4h
   funding. A raw cross-sectional rank across mixed intervals sorts the 4h
   names systematically low for a purely mechanical reason. Rates are summed
   into 8h buckets so the ranked quantity is always 8h carry.

A third, subtler one: turnover is measured against the book opened `h` ago, not
the previous grid point. At a 24h horizon on an 8h grid those differ, and
getting it wrong charges full rotation every rebalance — 20bps instead of the
~10bps the persistence of funding extremes actually implies.

## Layout

```
PREREGISTRATION.md              frozen pre-data: hypothesis, 4 cells, pass criteria
src/cryptoresearch/exchanges.py Binance / Bybit / OKX clients + reachability probe
src/cryptoresearch/protocol.py  split, demeaning, clustered inference, Holm, sign bar
scripts/01_probe_access.py      task 1: reachability + history depth
scripts/02_fetch_funding.py     cached multi-year pull
scripts/03_funding_extreme_test.py  the test, holdout gated behind a flag
tests/                          harness + end-to-end validation
```

## Protocol

Carried over unchanged and encoded in `src/cryptoresearch/protocol.py` so no
single test can opt out: 60/20/20 chronological split on a shared calendar,
day/block-clustered inference, cross-sectional pooling, drift demeaning,
sign stability across three slices as the bar, Holm correction with the cell
count stated, and no signal that needs a formation window.

## Tasks 2 and 3

Not started. Task 2 (sub-hour order-book microstructure) needs tick or L2 data,
which is a different pull entirely — and the cost bar there is tight enough
that it should be settled before any modelling. Task 3 (cross-sectional
relative strength) runs on the same funding-grid panel this repo already
builds, so it is mostly a new `run_cell` and its own pre-registration.
