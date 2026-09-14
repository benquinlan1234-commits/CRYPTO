#!/usr/bin/env python3
"""Download funding + 1h klines from data.binance.vision into the local cache.

Run this on a machine that can reach data.binance.vision:

    python scripts/02_download_vision.py                    # 14 symbols, 2020-01 -> now
    python scripts/02_download_vision.py --slim             # klines keep ts/close only
    python scripts/02_download_vision.py --symbols BTCUSDT  # one symbol

Resumable: a month already cached is skipped, so an interrupted run costs
nothing to restart. 404s are expected before a symbol lists and are counted,
not raised.
"""
from __future__ import annotations

import argparse, datetime as dt, os, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from cryptoresearch import binance_vision as bv  # noqa: E402
from cryptoresearch.exchanges import UNIVERSE  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = pathlib.Path(os.environ.get("CRYPTO_CACHE_DIR", ROOT / "data" / "cache"))
PARTS = CACHE / "_parts"


def load_month(session, kind: str, symbol: str, y: int, m: int, slim: bool):
    part = PARTS / f"{kind}_{symbol}_{y:04d}-{m:02d}.parquet"
    if part.exists():
        return pd.read_parquet(part), "cached"
    url = bv.funding_url(symbol, y, m) if kind == "funding" else bv.kline_url(symbol, "1h", y, m)
    try:
        blob = bv.fetch(url, session=session)
    except bv.NotInArchive:
        return None, "404"
    df = bv.parse_funding(blob, symbol) if kind == "funding" else bv.parse_klines(blob, symbol)
    if slim and kind == "klines":
        df = df[["ts", "close", "symbol"]]
    part.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(part, index=False)
    return df, "fetched"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(f"{b}USDT" for b in UNIVERSE))
    ap.add_argument("--start", default="2020-01")
    ap.add_argument("--end", default=dt.date.today().strftime("%Y-%m"))
    ap.add_argument("--slim", action="store_true",
                    help="klines keep only ts/close/symbol - all the test needs, ~5x smaller")
    ap.add_argument("--kinds", default="funding,klines")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    months = bv.month_range(args.start + "-01", args.end + "-01")
    session = requests.Session()
    session.headers.update({"User-Agent": "crypto-perp-research/0.1"})
    CACHE.mkdir(parents=True, exist_ok=True)

    print(f"{len(symbols)} symbols x {len(months)} months x {len(kinds)} kinds "
          f"= {len(symbols) * len(months) * len(kinds):,} files max")

    summary = []
    for symbol in symbols:
        base = symbol.replace("USDT", "")
        row = {"symbol": base}
        for kind in kinds:
            frames, n_404, n_new = [], 0, 0
            for (y, m) in months:
                try:
                    df, how = load_month(session, kind, symbol, y, m, args.slim)
                except Exception as exc:  # noqa: BLE001
                    print(f"  {symbol} {kind} {y}-{m:02d}: {type(exc).__name__}: {exc}")
                    continue
                if df is None:
                    n_404 += 1
                    continue
                n_new += int(how == "fetched")
                frames.append(df)
            if not frames:
                row[kind] = "none"
                print(f"{base:<6} {kind:<8} nothing in archive ({n_404} months absent)")
                continue
            full = pd.concat(frames, ignore_index=True).drop_duplicates("ts")
            full = full.sort_values("ts").reset_index(drop=True)
            # Cache files are keyed by base ("DOGE"), so the symbol column must
            # match - otherwise the panel join silently yields nothing.
            full["symbol"] = base
            full.to_parquet(CACHE / f"binance_{kind}_{base}.parquet", index=False)
            row[kind] = len(full)
            row[f"{kind}_from"] = full["ts"].min().date()
            row[f"{kind}_to"] = full["ts"].max().date()
            extra = ""
            if kind == "funding" and "interval_h" in full:
                ivs = sorted(pd.to_numeric(full["interval_h"], errors="coerce").dropna().unique())
                extra = f"  intervals={[int(i) for i in ivs]}h"
            print(f"{base:<6} {kind:<8} {len(full):>8,} rows  "
                  f"{full['ts'].min().date()} -> {full['ts'].max().date()}  "
                  f"(+{n_new} new, {n_404} absent){extra}")
        summary.append(row)

    s = pd.DataFrame(summary)
    print("\n" + s.to_string(index=False))
    f = s[s.get("funding").apply(lambda v: isinstance(v, int))] if "funding" in s else pd.DataFrame()
    if not f.empty:
        earliest, latest = f["funding_from"].min(), f["funding_to"].max()
        days = (pd.Timestamp(latest) - pd.Timestamp(earliest)).days
        print(f"\npanel span: {earliest} -> {latest}  ({days:,} days, {days / 365.25:.1f} years)")
        print(f"60/20/20  : {days * 0.6:,.0f}d / {days * 0.2:,.0f}d / {days * 0.2:,.0f}d"
              f"   (prior OKX slices: 19d each)")
    print(f"\ncached under {CACHE}")
    print("next: python scripts/03_funding_extreme_test.py --venue binance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
