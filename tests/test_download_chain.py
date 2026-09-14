"""End-to-end: archive ZIPs -> downloader -> cache -> panel -> test.

Network is stubbed, everything else is the real code path, so this exercises
the chain that will run on real data without needing archive access.
"""
from __future__ import annotations

import importlib.util, io, pathlib, sys, tempfile, zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from cryptoresearch import binance_vision as bv

ROOT = pathlib.Path(__file__).resolve().parents[1]
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
        "NEARUSDT", "UNIUSDT", "BCHUSDT", "LTCUSDT", "LINKUSDT"]
LISTS = {"SOLUSDT": "2021-03"}          # one late lister
FOUR_H = {"DOGEUSDT"}                    # one symbol on 4h funding from 2023-06
START, END = "2021-01", "2023-12"


def _zip(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("d.csv", text)
    return buf.getvalue()


_PX_CACHE: dict[str, pd.Series] = {}


def _price_series(sym: str) -> pd.Series:
    """One continuous random walk per symbol for the whole span.

    Generating each month independently would restart the price every month and
    inject a jump at every month boundary - an artifact of the stub that shows up
    downstream as a spurious result.
    """
    if sym not in _PX_CACHE:
        idx = pd.date_range("2020-01-01", "2026-01-01", freq="1h", inclusive="left", tz="UTC")
        rng = np.random.default_rng(abs(hash(sym)) % 2**31)
        _PX_CACHE[sym] = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.005, len(idx)))), idx)
    return _PX_CACHE[sym]


def _fake_fetch(url: str, session=None, **kw) -> bytes:
    """Serve archive-shaped ZIPs, switching dialect in 2025 like the real thing."""
    name = url.rsplit("/", 1)[-1]
    sym = name.split("-")[0]
    ym = name.replace(".zip", "").rsplit("-", 2)[-2:]
    y, m = int(ym[0]), int(ym[1])
    if sym in LISTS and (y, m) < tuple(int(x) for x in LISTS[sym].split("-")):
        raise bv.NotInArchive(url)

    micro = y >= 2025                     # archive switched dialect mid-2025
    mul = 1_000_000 if micro else 1_000
    start = pd.Timestamp(f"{y}-{m:02d}-01", tz="UTC")
    stop = (start + pd.offsets.MonthBegin(1))

    if "fundingRate" in url:
        four = sym in FOUR_H and start >= pd.Timestamp("2023-06-01", tz="UTC")
        freq, iv = ("4h", 4) if four else ("8h", 8)
        idx = pd.date_range(start, stop, freq=freq, inclusive="left", tz="UTC")
        rng = np.random.default_rng(abs(hash((sym, y, m))) % 2**31)
        rates = rng.normal(0, 0.0002, len(idx)) / (2 if four else 1)
        body = "\n".join(f"{int(t.timestamp())*mul},{iv},{r:.8f}" for t, r in zip(idx, rates))
        if micro:
            body = "calc_time,funding_interval_hours,last_funding_rate\n" + body
        return _zip(body)

    idx = pd.date_range(start, stop, freq="1h", inclusive="left", tz="UTC")
    px = _price_series(sym).reindex(idx).ffill().bfill().to_numpy()
    body = "\n".join(
        f"{int(t.timestamp())*mul},{p:.4f},{p*1.01:.4f},{p*0.99:.4f},{p:.4f},1.0,"
        f"{int(t.timestamp())*mul + 3600*mul - 1},100.0,5,0.5,50.0,0"
        for t, p in zip(idx, px))
    if micro:
        body = ",".join(bv.KLINE_COLS) + "\n" + body
    return _zip(body)


def _mod(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_download_then_analyse():
    with tempfile.TemporaryDirectory() as td:
        cache = pathlib.Path(td) / "cache"
        dl = _mod("02_download_vision.py", "dl")
        dl.CACHE, dl.PARTS = cache, cache / "_parts"
        dl.bv.fetch = _fake_fetch

        sys.argv = ["x", "--symbols", ",".join(SYMS), "--start", START, "--end", END, "--slim"]
        assert dl.main() == 0

        files = sorted(p.name for p in cache.glob("binance_*.parquet"))
        assert len(files) == len(SYMS) * 2, f"expected {len(SYMS)*2} cache files, got {len(files)}"

        # Late lister must be absent before it listed, present after.
        sol = pd.read_parquet(cache / "binance_funding_SOL.parquet")
        assert sol["ts"].min() >= pd.Timestamp("2021-03-01", tz="UTC")
        # 4h switch must be reflected in the archive-provided interval column.
        doge = pd.read_parquet(cache / "binance_funding_DOGE.parquet")
        ivs = set(pd.to_numeric(doge["interval_h"]).dropna().unique())
        assert ivs == {4, 8}, f"expected both intervals from the archive, got {ivs}"
        print(f"  downloaded {len(files)} cache files; DOGE intervals from file: {sorted(ivs)}")

        # ---- now the real analysis path on that cache
        fx = _mod("03_funding_extreme_test.py", "fx")
        fx.CACHE = cache
        bases = [s.replace("USDT", "") for s in SYMS]
        panel = fx.build_panel(fx.load("binance", "funding", bases),
                               fx.load("binance", "klines", bases))

        # The 4h symbol must not be systematically ranked low after its switch.
        d = panel[panel.symbol == "DOGE"]
        pre = d[d.ts < "2023-06-01"]["signal"].std()
        post = d[d.ts > "2023-07-01"]["signal"].std()
        assert 0.6 < post / pre < 1.6, \
            f"8h-equivalent carry not preserved across the 4h switch (ratio {post/pre:.2f})"
        assert (d[d.ts > "2023-07-01"]["n_settlements"] == 2).mode()[0]

        breadth = panel.dropna(subset=["signal", "fwd_8h"]).groupby("ts").size()
        assert breadth.max() >= fx.MIN_BREADTH
        res = fx.run_cell(panel[panel["ts"].isin(breadth[breadth >= fx.MIN_BREADTH].index)],
                          8, 0.20, 5.0)
        r = fx.summarise(res, n_boot=0)
        print(f"  panel {panel['ts'].min().date()} -> {panel['ts'].max().date()}, "
              f"{len(panel):,} rows, max breadth {breadth.max()}")
        # Funding and prices in the stub are independent, so this is a plumbing
        # check, not a finding: it confirms the cell runs and clusters by day.
        print(f"  full-sample cell h8_q20 (independent stub series): "
              f"{r['mean_dm_bps']:+.1f}bps p={r['p']:.3f} "
              f"turnover={r['mean_turnover']:.2f} over {r['n_blocks']:,} day-blocks")
        assert r["n_blocks"] > 300


def test_resume_is_free():
    """A second pass must hit the part cache and issue no new fetches."""
    with tempfile.TemporaryDirectory() as td:
        cache = pathlib.Path(td) / "c"
        dl = _mod("02_download_vision.py", "dl2")
        dl.CACHE, dl.PARTS = cache, cache / "_parts"
        calls = {"n": 0}

        def counting(url, session=None, **kw):
            calls["n"] += 1
            return _fake_fetch(url, session, **kw)

        dl.bv.fetch = counting
        sys.argv = ["x", "--symbols", "BTCUSDT", "--start", "2021-01", "--end", "2021-06", "--slim"]
        dl.main()
        first = calls["n"]
        dl.main()
        second = calls["n"] - first
        print(f"  first pass fetched {first} files; resume fetched {second}")
        assert first == 12 and second == 0, "resume re-downloaded"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\nall {len(fns)} download-chain checks passed")
