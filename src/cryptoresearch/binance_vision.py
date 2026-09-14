"""Reader for the data.binance.vision public archive.

The archive replaces API access entirely for this project: it goes back to
2020-01 and is not geo-blocked, whereas fapi.binance.com answers 451.

Three things about the archive bite if you assume it matches the API:

* **Timestamp units are not constant.** Binance moved parts of the archive from
  millisecond to microsecond timestamps during 2025. A file-by-file unit sniff
  is the only safe way to read it; assuming ms silently places 2025+ data
  around the year 57000.
* **Headers are not constant.** Older monthly files have no header row, newer
  ones do. Parsing blind eats the first data row or treats a header as data.
* **Funding files carry their own interval.** ``funding_interval_hours`` is in
  the file, so the 4h/8h comparability problem is answered by the data rather
  than inferred from settlement spacing.
"""

from __future__ import annotations

import io
import time
import zipfile
from dataclasses import dataclass

import pandas as pd
import requests

BASE = "https://data.binance.vision/data/futures/um"

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
FUNDING_COLS = ["calc_time", "funding_interval_hours", "last_funding_rate"]
BOOKDEPTH_COLS = ["timestamp", "percentage", "depth", "notional"]


class NotInArchive(Exception):
    """404 - the symbol had not listed yet, or that day/month is absent."""


def funding_url(symbol: str, year: int, month: int) -> str:
    return f"{BASE}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{year:04d}-{month:02d}.zip"


def kline_url(symbol: str, interval: str, year: int, month: int) -> str:
    return (f"{BASE}/monthly/klines/{symbol}/{interval}/"
            f"{symbol}-{interval}-{year:04d}-{month:02d}.zip")


def bookdepth_url(symbol: str, day: pd.Timestamp) -> str:
    return (f"{BASE}/daily/bookDepth/{symbol}/"
            f"{symbol}-bookDepth-{day:%Y-%m-%d}.zip")


def detect_time_unit(v: float) -> str:
    """Sniff epoch units by magnitude.

    2020-01-01 is 1.58e9 s, 1.58e12 ms, 1.58e15 us. The bands are three orders
    of magnitude apart, so this is unambiguous for any plausible date.
    """
    v = abs(float(v))
    if v >= 1e17:
        return "ns"
    if v >= 1e14:
        return "us"
    if v >= 1e11:
        return "ms"
    return "s"


def to_utc(series: pd.Series) -> pd.Series:
    """Convert an epoch column to tz-aware UTC, sniffing the unit per file."""
    s = pd.to_numeric(series, errors="coerce")
    probe = s.dropna()
    if probe.empty:
        raise ValueError("no parseable timestamps in column")
    return pd.to_datetime(s, unit=detect_time_unit(probe.iloc[0]), utc=True)


def _read_zip_csv(blob: bytes, expected_cols: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError(f"no CSV inside archive (members: {z.namelist()})")
        raw = z.read(names[0])

    head = raw[:400].decode("utf-8", "replace").lstrip()
    first = head.split("\n", 1)[0]
    # A header row starts with a letter; a data row starts with a digit or sign.
    has_header = bool(first) and not (first[0].isdigit() or first[0] in "+-")

    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)
    if not has_header:
        df.columns = expected_cols[:len(df.columns)]
    else:
        df.columns = [str(c).strip() for c in df.columns]
    return df


def fetch(url: str, session: requests.Session | None = None,
          retries: int = 4, timeout: int = 120) -> bytes:
    s = session or requests.Session()
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = s.get(url, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            last = exc
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 404:
            raise NotInArchive(url)
        if r.status_code == 200:
            return r.content
        last = RuntimeError(f"HTTP {r.status_code} for {url}")
        time.sleep(2 ** attempt)
    raise RuntimeError(f"failed after {retries} attempts: {last}")


def parse_funding(blob: bytes, symbol: str) -> pd.DataFrame:
    """-> ts (UTC), rate, interval_h, symbol"""
    df = _read_zip_csv(blob, FUNDING_COLS)
    cols = {c.lower(): c for c in df.columns}
    ts_c = cols.get("calc_time", df.columns[0])
    iv_c = cols.get("funding_interval_hours")
    rt_c = cols.get("last_funding_rate", df.columns[-1])

    out = pd.DataFrame({
        "ts": to_utc(df[ts_c]),
        "rate": pd.to_numeric(df[rt_c], errors="coerce"),
    })
    # The archive states the interval directly - no need to infer it from spacing.
    out["interval_h"] = (pd.to_numeric(df[iv_c], errors="coerce")
                         if iv_c is not None else pd.NA)
    out["symbol"] = symbol
    return out.dropna(subset=["ts", "rate"]).sort_values("ts").reset_index(drop=True)


def parse_klines(blob: bytes, symbol: str) -> pd.DataFrame:
    """-> ts (UTC bar OPEN time), open, high, low, close, volume, symbol

    ts stays the bar's open time. The pipeline is responsible for knowing that
    such a bar closes at ts+interval; see build_panel.
    """
    df = _read_zip_csv(blob, KLINE_COLS)
    cols = {str(c).lower(): c for c in df.columns}
    out = pd.DataFrame({"ts": to_utc(df[cols.get("open_time", df.columns[0])])})
    for f in ("open", "high", "low", "close", "volume"):
        out[f] = pd.to_numeric(df[cols[f]] if f in cols else df.iloc[:, KLINE_COLS.index(f)],
                               errors="coerce")
    out["symbol"] = symbol
    return out.dropna(subset=["ts", "close"]).sort_values("ts").reset_index(drop=True)


def parse_bookdepth(blob: bytes, symbol: str) -> pd.DataFrame:
    """-> ts (UTC), percentage, depth, notional, symbol"""
    df = _read_zip_csv(blob, BOOKDEPTH_COLS)
    cols = {str(c).lower(): c for c in df.columns}
    out = pd.DataFrame({
        "ts": to_utc(df[cols.get("timestamp", df.columns[0])]),
        "percentage": pd.to_numeric(df[cols.get("percentage", df.columns[1])], errors="coerce"),
        "depth": pd.to_numeric(df[cols.get("depth", df.columns[2])], errors="coerce"),
        "notional": pd.to_numeric(df[cols.get("notional", df.columns[3])], errors="coerce"),
    })
    out["symbol"] = symbol
    return out.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)


def month_range(start: str, end: str) -> list[tuple[int, int]]:
    idx = pd.date_range(pd.Timestamp(start).replace(day=1),
                        pd.Timestamp(end).replace(day=1), freq="MS")
    return [(d.year, d.month) for d in idx]
