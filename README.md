# crypto-perp-research

Perp research, separate from the NQ work. Nothing here touches `reclaim-mbo`.

## Data source: the Vision archive, not the API

`fapi.binance.com` answers **451** from Australia. `data.binance.vision` does
not, goes back to **2020-01**, and needs no key — so the archive replaces API
access entirely.

**This container cannot reach the archive either.** Its egress policy is an
allowlist covering GitHub and package registries; `data.binance.vision:443`
returns `connect_rejected` at the proxy, same as every other market-data host
(OKX included, which is how we know it is the policy and not a geo-block). So
**the download runs on your machine**:

```bash
pip install -r requirements.txt
python scripts/02_download_vision.py --slim      # 14 symbols, 2020-01 -> now
python scripts/03_funding_extreme_test.py --venue binance
```

Resumable — a month already cached is skipped, so an interrupted run costs
nothing to restart. Months before a symbol listed return 404 and are counted,
not raised.

`--slim` keeps only `ts/close/symbol` for klines, which is all the test uses.
Expected cache size for 14 symbols over 6.7 years:

| | rows | size |
|---|---|---|
| funding | ~102k | 1.9 MB |
| klines `--slim` | ~818k | 15.4 MB |
| klines full OHLCV | ~818k | 48 MB |

**~17 MB for funding + slim klines**, so you can commit `data/cache/` and I'll
run the analysis here. (`.gitignore` currently excludes it — `git add -f
data/cache/*.parquet` to override.) Otherwise run `03_` locally and paste the
output.

## Run order

```
scripts/01_probe_access.py      venue reachability + history depth (API venues)
scripts/02_download_vision.py   archive -> local parquet cache      [needs network]
scripts/03_funding_extreme_test.py   the frozen test                 [offline]
```

The holdout is behind `--touch-holdout` so it cannot be burned by a re-run.

## What is validated

20 checks, all passing, none of which need archive access:

```bash
python tests/test_protocol.py           # 8  statistical harness
python tests/test_vision_format.py      # 6  archive format
python tests/test_pipeline_synthetic.py # 4  end-to-end, known planted truth
python tests/test_download_chain.py     # 2  archive ZIPs -> cache -> panel -> cell
```

Results worth knowing:

* **Day-clustering is load-bearing.** On day-correlated data with a true effect
  of exactly zero, a naive t-test rejects **50.7%** of the time; the clustered
  test rejects **6.7%** against a nominal 5%.
* **No lookahead.** Under a true null across 12 seeds, exploration median
  p = 0.61, signs split 6+/6−. (Coarse — rules out a gross leak, not a subtle one.)
* **A real effect is recovered.** A planted +30bps/8h reversion reads
  +46.6 / +54.8 / +45.2bps across the three slices; the ~1.6× is the long-short
  rank spread, so the magnitude is right, not just the sign.
* The recorded 4h reversion pattern (−17.8 → +54.6 → −19.2) correctly **fails**
  the sign-stability bar.

**No real market data has been touched.** Every number above is synthetic.

## Four traps handled, each with a test

1. **Bar timestamps.** A kline stamped `open_time = t` covers `[t, t+1h)`, so
   its close is the price at `t+1h`. Using it as the price at `t` hands the test
   a free hour of the future. The price at `t` is the close of the bar *ending*
   at `t`, and there is an assertion for it.
2. **Archive timestamp units drift.** Binance moved parts of the archive from
   millisecond to microsecond epochs during 2025. Reading a microsecond file as
   milliseconds places the data in **year 56971** — the unit is sniffed per file
   by magnitude.
3. **Archive headers drift.** Older monthly files have no header row, newer ones
   do. Parsing blind either eats the first observation or treats a header as
   data. Both dialects are detected and tested to parse identically.
4. **Mixed 4h/8h funding.** Ranking raw per-settlement rates sorts the 4h names
   systematically low for a purely mechanical reason. Rates are summed into 8h
   buckets so the ranked quantity is always 8h carry, and
   `funding_interval_hours` — present in the archive file — is used to verify
   bucket completeness rather than inferring the interval from spacing.

A fifth, subtler one: turnover is measured against the book opened `h` ago, not
the previous grid point. At a 24h horizon on an 8h grid those differ, and
getting it wrong charges full rotation every rebalance — 20bps instead of the
~10bps that the persistence of funding extremes actually implies.

## Protocol

Encoded in `src/cryptoresearch/protocol.py` so no single test can opt out:
60/20/20 chronological split on a **shared calendar** (row-count splitting would
give a late-listing alt different boundaries than BTC at the same instant),
day/block-clustered inference, cross-sectional pooling, drift demeaning, sign
stability across three slices as the bar, and Holm correction with the cell
count stated.

## Layout

```
PREREGISTRATION.md                    frozen pre-data: hypothesis, 4 cells, pass/kill criteria
src/cryptoresearch/binance_vision.py  archive URLs, ZIP/CSV parsing, unit + header sniffing
src/cryptoresearch/exchanges.py       Binance/Bybit/OKX API clients + reachability probe
src/cryptoresearch/protocol.py        split, demeaning, clustered inference, Holm, sign bar
scripts/                              probe / download / test
tests/                                20 checks
```

## Task 2 — order-book microstructure

Not started, and deliberately so: the execution-cost question comes first. At 1h
the median BTC move is only ~2× round-trip cost, and it gets worse below that,
so the thing to establish before any modelling is what a signal must clear.
`bookdepth_url()` and `parse_bookdepth()` are in place for when that is settled.

## Task 3 — cross-sectional relative strength

Not started. Runs on the same funding-grid panel `03_` already builds, so it is
a new cell definition plus its own pre-registration.
