"""Synthetic Binance-shaped caches with a known, planted truth.

Used to prove the pipeline recovers an effect that is really there and stays
quiet when it is not. Deliberately reproduces the awkward parts of the real
data: late listings, symbols that switch from 8h to 4h funding, and a market
factor that makes rows within a day correlated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SYMBOLS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "NEAR", "UNI",
           "SUI", "ARB", "BCH", "WLD", "LTC", "AVAX", "LINK"]
LATE = {"SUI": "2023-05-01", "WLD": "2023-07-01", "ARB": "2023-03-01"}
FOUR_HOUR = {"DOGE", "NEAR", "WLD"}          # switch to 4h funding mid-sample
SWITCH = pd.Timestamp("2023-06-01", tz="UTC")


def generate(outdir, start="2021-01-01", end="2026-09-01", effect_bps=0.0, seed=7):
    """effect_bps: mean 8h reversion, in bps, applied to the funding rank.

    Positive effect_bps means high funding is followed by underperformance
    (reversion), which the pipeline should surface as a positive `gross`.
    """
    rng = np.random.default_rng(seed)
    outdir = pd.io.common.stringify_path(outdir)
    import pathlib
    outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    hours = pd.date_range(start, end, freq="1h", tz="UTC")
    grid = pd.date_range(start, end, freq="8h", tz="UTC")

    # Shared market factor => intra-day correlation, so naive SEs would lie.
    mkt_h = rng.normal(0, 0.004, len(hours))

    # Funding signal per symbol on the 8h grid: persistent, mean-reverting.
    sig = {}
    for s in SYMBOLS:
        x = np.zeros(len(grid))
        for i in range(1, len(grid)):
            x[i] = 0.85 * x[i - 1] + rng.normal(0, 1)
        sig[s] = x * 0.0003 + rng.normal(0, 0.00005)     # ~3bp per 8h scale

    grid_pos = {t: i for i, t in enumerate(grid)}
    # Cross-sectional rank of the signal at each grid point drives the effect.
    sig_mat = np.column_stack([sig[s] for s in SYMBOLS])
    ranks = sig_mat.argsort(axis=1).argsort(axis=1)
    n = len(SYMBOLS)
    z = (ranks - (n - 1) / 2) / ((n - 1) / 2)            # +1 = highest funding

    for j, s in enumerate(SYMBOLS):
        listed = pd.Timestamp(LATE.get(s, start), tz="UTC")

        # ---- klines: hourly, with the planted effect spread over the next 8h
        idio = rng.normal(0, 0.006, len(hours))
        beta = 0.8 + 0.4 * rng.random()
        ret_h = beta * mkt_h + idio
        if effect_bps:
            # high funding (z=+1) -> negative forward return => reversion
            per_hour = -(effect_bps * 1e-4) / 8.0
            gi = np.searchsorted(grid, hours, side="right") - 1
            gi = np.clip(gi, 0, len(grid) - 1)
            ret_h = ret_h + per_hour * z[gi, j]

        px = 100 * np.exp(np.cumsum(ret_h))
        kl = pd.DataFrame({"ts": hours, "close": px, "symbol": s})
        kl["open"] = kl["close"].shift(1).fillna(100.0)
        kl["high"] = kl[["open", "close"]].max(axis=1) * 1.001
        kl["low"] = kl[["open", "close"]].min(axis=1) * 0.999
        kl["volume"] = rng.lognormal(10, 1, len(kl))
        kl = kl[kl["ts"] >= listed]
        kl.to_parquet(outdir / f"binance_klines_{s}.parquet", index=False)

        # ---- funding: 8h, switching to 4h mid-sample for a few symbols
        rows = []
        for i, t in enumerate(grid):
            if t < listed:
                continue
            r = sig[s][i]
            if s in FOUR_HOUR and t >= SWITCH:
                # same 8h carry, delivered as two 4h settlements
                rows.append({"ts": t - pd.Timedelta(hours=4), "rate": r * 0.5, "symbol": s})
                rows.append({"ts": t, "rate": r * 0.5, "symbol": s})
            else:
                rows.append({"ts": t, "rate": r, "symbol": s})
        pd.DataFrame(rows).to_parquet(outdir / f"binance_funding_{s}.parquet", index=False)

    return outdir
