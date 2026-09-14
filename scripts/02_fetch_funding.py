#!/usr/bin/env python3
"""Task 1, part B: pull multi-year funding history + 1h klines and cache them.

    python scripts/02_fetch_funding.py --venue binance

Caches one parquet per symbol per kind under data/cache/. Re-running only
fetches what is missing, so an interrupted pull resumes cheaply.
"""
from __future__ import annotations

import argparse, datetime as dt, os, pathlib, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402
from cryptoresearch.exchanges import UNIVERSE, VENUES, VenueError  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = pathlib.Path(os.environ.get("CRYPTO_CACHE_DIR", ROOT / "data" / "cache"))


def cache_path(venue: str, kind: str, base: str) -> pathlib.Path:
    return CACHE / f"{venue}_{kind}_{base}.parquet"


def fetch_one(venue, base: str, kind: str, start_ms: int, end_ms: int, force: bool) -> pd.DataFrame:
    path = cache_path(venue.name, kind, base)
    if path.exists() and not force:
        return pd.read_parquet(path)

    sym = venue.perp_symbol(base)
    if kind == "funding":
        rows = venue.funding_history(sym, start_ms, end_ms)
    else:
        rows = venue.klines(sym, "1h", start_ms, end_ms)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["symbol"] = base
    df = df.sort_values("ts").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", default="binance", choices=list(VENUES))
    ap.add_argument("--symbols", default=",".join(UNIVERSE))
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--force", action="store_true", help="ignore cache and refetch")
    ap.add_argument("--skip-klines", action="store_true")
    args = ap.parse_args()

    venue = VENUES[args.venue]()
    bases = [s.strip() for s in args.symbols.split(",") if s.strip()]
    start_ms = int(pd.Timestamp(args.start, tz="UTC").timestamp() * 1000)
    end_ms = int(time.time() * 1000)

    kinds = ["funding"] + ([] if args.skip_klines else ["klines"])
    summary = []
    for base in bases:
        row = {"symbol": base}
        for kind in kinds:
            try:
                df = fetch_one(venue, base, kind, start_ms, end_ms, args.force)
                if df.empty:
                    row[kind] = "empty"
                    print(f"{base:<6} {kind:<8} EMPTY")
                    continue
                row[kind] = len(df)
                row[f"{kind}_from"] = df["ts"].min().date()
                row[f"{kind}_to"] = df["ts"].max().date()
                print(f"{base:<6} {kind:<8} {len(df):>7,} rows  "
                      f"{df['ts'].min().date()} -> {df['ts'].max().date()}")
            except VenueError as exc:
                row[kind] = f"ERROR:{exc.status}"
                print(f"{base:<6} {kind:<8} FAILED  {exc.status}: {exc.detail[:100]}")
                if exc.status in ("egress_denied", "geo_blocked"):
                    print("\nVenue unreachable - stopping. Run scripts/01_probe_access.py first.")
                    return 1
        summary.append(row)

    s = pd.DataFrame(summary)
    print("\n" + s.to_string(index=False))

    fund = s[s["funding"].apply(lambda v: isinstance(v, int))]
    if not fund.empty and "funding_from" in fund:
        earliest, latest = fund["funding_from"].min(), fund["funding_to"].max()
        days = (pd.Timestamp(latest) - pd.Timestamp(earliest)).days
        print(f"\npanel span: {earliest} -> {latest}  ({days:,} days, {days/365.25:.1f} years)")
        print(f"60/20/20  : {days*0.6:,.0f}d / {days*0.2:,.0f}d / {days*0.2:,.0f}d")
        print("(prior OKX slices: 19d each)")
    print(f"\ncached under {CACHE}")
    print("next: python scripts/03_funding_extreme_test.py --venue " + args.venue)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
