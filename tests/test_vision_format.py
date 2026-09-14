"""Archive-format tests for data.binance.vision.

Built against real ZIP bytes in both archive dialects, because the two format
drifts here are silent: a microsecond file read as milliseconds lands in the
year 57000, and a header row read as data eats the first observation.
"""
from __future__ import annotations

import io, pathlib, sys, zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from cryptoresearch import binance_vision as bv


def _zip(csv_text: str, name: str = "x.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(name, csv_text)
    return buf.getvalue()


def test_unit_sniff_covers_the_2025_microsecond_switch():
    assert bv.detect_time_unit(1_577_836_800) == "s"
    assert bv.detect_time_unit(1_577_836_800_000) == "ms"
    assert bv.detect_time_unit(1_735_689_600_000_000) == "us"
    # The failure this prevents: us read as ms lands ~55,000 years out.
    wrong = pd.to_datetime(1_735_689_600_000_000, unit="ms", utc=True)
    right = pd.to_datetime(1_735_689_600_000_000, unit="us", utc=True)
    assert right.year == 2025 and wrong.year > 50000
    print(f"  unit sniff: us-as-ms would give year {wrong.year}, sniffed correctly as 2025")


def test_funding_old_dialect_no_header_milliseconds():
    blob = _zip(
        "1577836800000,8,0.00010000\n"
        "1577865600000,8,-0.00005000\n"
        "1577894400000,8,0.00012500\n"
    )
    df = bv.parse_funding(blob, "BTCUSDT")
    assert len(df) == 3, "a data row was eaten as a header"
    assert df["ts"].iloc[0] == pd.Timestamp("2020-01-01 00:00:00", tz="UTC")
    assert np.isclose(df["rate"].iloc[1], -0.00005)
    assert (df["interval_h"] == 8).all()
    print(f"  old dialect: 3 rows, first {df['ts'].iloc[0]}, intervals 8h")


def test_funding_new_dialect_header_microseconds_four_hour():
    blob = _zip(
        "calc_time,funding_interval_hours,last_funding_rate\n"
        "1735689600000000,4,0.00002500\n"
        "1735704000000000,4,0.00003000\n"
    )
    df = bv.parse_funding(blob, "DOGEUSDT")
    assert len(df) == 2, "header row was parsed as data"
    assert df["ts"].iloc[0] == pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    assert df["ts"].iloc[0].year == 2025, "microsecond timestamps misread"
    assert (df["interval_h"] == 4).all(), "4h interval not carried through from the file"
    print(f"  new dialect: {df['ts'].iloc[0]} .. {df['ts'].iloc[1]}, intervals 4h")


def test_klines_both_dialects_agree():
    rows_ms = "\n".join(
        f"{1577836800000 + i * 3600000},100.0,101.0,99.0,{100 + i}.0,5.0,"
        f"{1577836800000 + (i + 1) * 3600000 - 1},500.0,10,2.0,200.0,0"
        for i in range(4))
    old = bv.parse_klines(_zip(rows_ms), "BTCUSDT")

    rows_us = "\n".join(
        f"{1577836800000000 + i * 3600000000},100.0,101.0,99.0,{100 + i}.0,5.0,"
        f"{1577836800000000 + (i + 1) * 3600000000 - 1},500.0,10,2.0,200.0,0"
        for i in range(4))
    new = bv.parse_klines(_zip(",".join(bv.KLINE_COLS) + "\n" + rows_us), "BTCUSDT")

    assert len(old) == len(new) == 4
    assert (old["ts"].to_numpy() == new["ts"].to_numpy()).all(), \
        "ms and us dialects of the same data disagree"
    assert list(old["close"]) == [100.0, 101.0, 102.0, 103.0]
    print("  klines: ms/no-header and us/header dialects parse identically")


def test_bookdepth_parses():
    blob = _zip("timestamp,percentage,depth,notional\n"
                "1700000000000,1,12.5,1200000.0\n"
                "1700000000000,-1,11.0,1050000.0\n")
    df = bv.parse_bookdepth(blob, "BTCUSDT")
    assert len(df) == 2 and set(df["percentage"]) == {1, -1}
    print(f"  bookDepth: {len(df)} rows, percentages {sorted(set(df['percentage']))}")


def test_urls_match_the_documented_layout():
    assert bv.funding_url("BTCUSDT", 2024, 1) == (
        "https://data.binance.vision/data/futures/um/monthly/fundingRate/"
        "BTCUSDT/BTCUSDT-fundingRate-2024-01.zip")
    assert bv.kline_url("ETHUSDT", "1h", 2023, 12) == (
        "https://data.binance.vision/data/futures/um/monthly/klines/"
        "ETHUSDT/1h/ETHUSDT-1h-2023-12.zip")
    assert bv.bookdepth_url("BTCUSDT", pd.Timestamp("2025-03-09")) == (
        "https://data.binance.vision/data/futures/um/daily/bookDepth/"
        "BTCUSDT/BTCUSDT-bookDepth-2025-03-09.zip")
    assert len(bv.month_range("2020-01-01", "2020-04-01")) == 4
    print("  urls: funding / klines / bookDepth layouts match")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\nall {len(fns)} archive-format checks passed")
