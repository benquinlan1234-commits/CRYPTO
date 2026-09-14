"""End-to-end pipeline validation against synthetic data with a known truth.

Two questions, both of which must be answered before any real-data run is
worth believing:

1. Under a true null, does the pipeline stay quiet? If there were lookahead
   anywhere in the grid/kline alignment it would show up here as exploration
   p-values skewed small and a rejection rate far above 5%.
2. With a planted effect, does it recover the right sign and rough magnitude?
"""
from __future__ import annotations

import pathlib, sys, tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from make_synthetic import generate

from cryptoresearch.protocol import chronological_split

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import importlib.util


def _load_test_module():
    spec = importlib.util.spec_from_file_location(
        "fx", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "03_funding_extreme_test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FX = _load_test_module()


def _panel(cache: pathlib.Path):
    FX.CACHE = cache
    syms = [p.name.split("_")[-1].replace(".parquet", "")
            for p in cache.glob("binance_funding_*.parquet")]
    panel = FX.build_panel(FX.load("binance", "funding", syms),
                           FX.load("binance", "klines", syms))
    breadth = panel.dropna(subset=["signal", "fwd_8h"]).groupby("ts").size()
    panel = panel[panel["ts"].isin(breadth[breadth >= FX.MIN_BREADTH].index)].copy()
    labels, _ = chronological_split(panel["ts"])
    panel["slice"] = labels
    return panel


def test_null_is_calibrated():
    """No planted effect => exploration rejections near nominal, not above."""
    rejects, ps, signs = 0, [], []
    trials = 12
    with tempfile.TemporaryDirectory() as td:
        for seed in range(trials):
            cache = generate(pathlib.Path(td) / f"s{seed}", effect_bps=0.0, seed=100 + seed)
            panel = _panel(cache)
            sub = panel[panel["slice"] == "exploration"]
            res = FX.run_cell(sub, 8, 0.20, 5.0)
            r = FX.summarise(res, n_boot=0)
            ps.append(r["p"]); signs.append(np.sign(r["mean_dm_bps"]))
            rejects += int(r["p"] < 0.05)

    rate = rejects / trials
    print(f"  null: {rejects}/{trials} exploration rejections at 5% "
          f"(median p={np.median(ps):.3f}, signs {signs.count(1.0)}+/{signs.count(-1.0)}-)")
    assert rate <= 0.25, f"null rejecting {rate:.0%} - suggests lookahead or a bad SE"
    assert np.median(ps) > 0.20, f"median null p={np.median(ps):.3f} is too small"


def test_planted_effect_is_recovered_with_the_right_sign():
    """A planted 30bps/8h reversion must show up positive in every slice."""
    with tempfile.TemporaryDirectory() as td:
        cache = generate(pathlib.Path(td) / "eff", effect_bps=30.0, seed=11)
        panel = _panel(cache)
        means = {}
        for nm in ("exploration", "validation", "holdout"):
            res = FX.run_cell(panel[panel["slice"] == nm], 8, 0.20, 5.0)
            r = FX.summarise(res, n_boot=0)
            means[nm] = r["mean_dm_bps"]
            print(f"  planted +30bps -> {nm:<12} {r['mean_dm_bps']:+7.1f}bps  "
                  f"t={r['t']:+.2f} p={r['p']:.2e} net={r['mean_net_bps']:+.1f}")
        assert all(v > 0 for v in means.values()), f"sign not recovered: {means}"
        assert means["exploration"] > 5, "magnitude far too small"


def test_no_lookahead_in_price_alignment():
    """Price at t must be the close of the bar ENDING at t, never the bar opening at t."""
    with tempfile.TemporaryDirectory() as td:
        cache = generate(pathlib.Path(td) / "a", effect_bps=0.0, seed=3)
        FX.CACHE = cache
        syms = ["BTC", "ETH"]
        kl = FX.load("binance", "klines", syms)
        panel = FX.build_panel(FX.load("binance", "funding", syms), kl)
        btc_k = kl[kl.symbol == "BTC"].set_index("ts")["close"]
        row = panel[(panel.symbol == "BTC") & panel["px"].notna()].iloc[50]
        t = row["ts"]
        assert np.isclose(row["px"], btc_k.loc[t - pd.Timedelta(hours=1)]), \
            "px is not the close of the bar ending at t"
        assert not np.isclose(row["px"], btc_k.loc[t]), \
            "px equals the close of the bar OPENING at t - that is one hour of lookahead"
        print("  alignment: px(t) = close of bar ending at t, confirmed")


def test_four_hour_funding_symbols_are_made_comparable():
    """A symbol on 4h funding must not sort systematically low."""
    with tempfile.TemporaryDirectory() as td:
        cache = generate(pathlib.Path(td) / "b", effect_bps=0.0, seed=5)
        FX.CACHE = cache
        syms = ["BTC", "DOGE"]   # DOGE switches to 4h at 2023-06-01
        panel = FX.build_panel(FX.load("binance", "funding", syms),
                               FX.load("binance", "klines", syms))
        d = panel[panel.symbol == "DOGE"]
        before = d[d.ts < "2023-06-01"]["n_settlements"].mode()[0]
        after = d[d.ts > "2023-07-01"]["n_settlements"].mode()[0]
        assert before == 1 and after == 2, f"expected 1 then 2 settlements, got {before},{after}"
        # Bucket-summing must preserve the 8h carry across the interval change.
        sd_before = d[d.ts < "2023-06-01"]["signal"].std()
        sd_after = d[d.ts > "2023-07-01"]["signal"].std()
        ratio = sd_after / sd_before
        print(f"  4h switch: settlements {before}->{after}, signal scale ratio {ratio:.2f}")
        assert 0.7 < ratio < 1.4, \
            f"8h-equivalent carry not preserved across the funding-interval change (ratio {ratio:.2f})"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\nall {len(fns)} pipeline checks passed")
