#!/usr/bin/env python3
"""Task 1, part C: the funding-extreme test, exactly as pre-registered.

    python scripts/03_funding_extreme_test.py --venue binance                 # expl + val
    python scripts/03_funding_extreme_test.py --venue binance --touch-holdout # once, at the end

The holdout is gated behind an explicit flag so it cannot be burned by a
reflexive re-run. See PREREGISTRATION.md; nothing here is tunable from the
command line except the venue and the fee.
"""
from __future__ import annotations

import argparse, datetime as dt, json, math, os, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cryptoresearch.exchanges import UNIVERSE  # noqa: E402
from cryptoresearch.protocol import (  # noqa: E402
    chronological_split, cluster_test, demean_drift, holm, sign_stable,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = pathlib.Path(os.environ.get("CRYPTO_CACHE_DIR", ROOT / "data" / "cache"))

# Pre-registered. 4 cells. Changing this changes the correction.
CELLS = [("h8_q20", 8, 0.20), ("h8_q10", 8, 0.10),
         ("h24_q20", 24, 0.20), ("h24_q10", 24, 0.10)]
MIN_BREADTH = 8
GRID_H = 8
BPS = 1e-4


# ------------------------------------------------------------------ loading
def load(venue: str, kind: str, symbols: list[str]) -> pd.DataFrame:
    frames = []
    for s in symbols:
        p = CACHE / f"{venue}_{kind}_{s}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        raise SystemExit(f"no cached {kind} under {CACHE} - run scripts/02_fetch_funding.py first")
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values(["symbol", "ts"]).reset_index(drop=True)


def build_panel(funding: pd.DataFrame, klines: pd.DataFrame) -> pd.DataFrame:
    """One row per (8h grid timestamp, symbol) with the signal and forward returns.

    Two alignment rules do the real work here:

    * A kline stamped ``open_time = t`` covers [t, t+1h), so its close is the
      price at t+1h, NOT at t. Using it as the price at t would hand the test a
      free hour of the future. The price at t is therefore the close of the bar
      whose open time is t-1h.
    * The signal at t is the funding actually settled in (t-8h, t] - all of it
      public by t. Summing the raw rates inside the bucket makes 4h-funding and
      8h-funding symbols directly comparable, which a raw per-settlement rank
      would not be.
    """
    # --- price at t = close of the bar ending at t
    k = klines.copy()
    k["bar_end"] = k["ts"] + pd.Timedelta(hours=1)
    px = (k.set_index(["symbol", "bar_end"])["close"].sort_index())

    # --- 8h grid
    grid = pd.date_range(
        funding["ts"].min().ceil(f"{GRID_H}h"),
        funding["ts"].max().floor(f"{GRID_H}h"),
        freq=f"{GRID_H}h", tz="UTC",
    )

    # --- signal: funding settled in (t-8h, t], summed per bucket
    f = funding.copy()
    # bucket label = the grid point at or after the settlement, i.e. ceil
    f["grid"] = f["ts"].dt.ceil(f"{GRID_H}h")
    sig = (f.groupby(["symbol", "grid"], as_index=False)
             .agg(signal=("rate", "sum"), n_settlements=("rate", "size")))
    sig = sig.rename(columns={"grid": "ts"})
    sig = sig[sig["ts"].isin(grid)]

    # --- forward returns off the grid
    frames = []
    for sym, g in sig.groupby("symbol", sort=False):
        s = g.sort_values("ts").copy()
        try:
            p_sym = px.loc[sym]
        except KeyError:
            continue
        p_now = p_sym.reindex(s["ts"]).to_numpy()
        s["px"] = p_now
        for _, h, _ in CELLS:
            col = f"fwd_{h}h"
            if col in s.columns:
                continue
            p_fwd = p_sym.reindex(s["ts"] + pd.Timedelta(hours=h)).to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                s[col] = np.log(p_fwd / p_now)
        frames.append(s)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    return panel.sort_values(["ts", "symbol"]).reset_index(drop=True)


# ------------------------------------------------------------------ the test
def run_cell(panel: pd.DataFrame, h: int, q: float, fee_bps: float) -> pd.DataFrame:
    """Form the long-short book at every grid point. Returns one row per rebalance."""
    col = f"fwd_{h}h"
    d = panel.dropna(subset=["signal", col]).copy()

    # Drift demeaning before anything directional is claimed.
    d["ret_dm"] = demean_drift(d, col, mode="slice")

    out, books = [], {}
    step = pd.Timedelta(hours=h)

    for ts, g in d.groupby("ts", sort=True):
        n = len(g)
        if n < MIN_BREADTH:
            continue
        k = max(1, int(round(n * q)))
        if 2 * k > n:
            continue
        g = g.sort_values("signal", ascending=False)
        top, bottom = g.iloc[:k], g.iloc[-k:]          # crowded longs / crowded shorts

        gross = bottom[col].mean() - top[col].mean()
        gross_dm = bottom["ret_dm"].mean() - top["ret_dm"].mean()

        # Turnover is against the book this one replaces - the cohort opened h
        # ago, not the previous grid point. At a 24h horizon on an 8h grid those
        # are three grid points apart, and comparing to the neighbour would
        # charge full rotation on every single rebalance.
        cur = {"long": set(bottom["symbol"]), "short": set(top["symbol"])}
        prev = books.get(ts - step)
        if prev and prev["long"]:
            tno = sum(len(cur[s] ^ prev[s]) / max(len(prev[s]), 1) for s in ("long", "short"))
        else:
            tno = 4.0                                   # cold start: pay full rotation
        cost = fee_bps * BPS * tno

        out.append({"ts": ts, "breadth": n, "k": k,
                    "gross": gross, "gross_dm": gross_dm,
                    "turnover": tno, "cost": cost, "net": gross_dm - cost})
        books[ts] = cur

    res = pd.DataFrame(out)
    if res.empty:
        return res
    # Block clustering: blocks at least twice the return-overlap length.
    block_days = max(1, math.ceil(2 * h / 24))
    origin = res["ts"].min().normalize()
    res["block"] = ((res["ts"] - origin).dt.total_seconds() // (block_days * 86400)).astype(int)
    return res


def summarise(res: pd.DataFrame, n_boot: int) -> dict | None:
    if res.empty or res["block"].nunique() < 2:
        return None
    g = cluster_test(res["gross_dm"], res["block"], n_boot=n_boot)
    n = cluster_test(res["net"], res["block"], n_boot=0)
    return {
        "n_rebalances": int(len(res)),
        "n_blocks": int(g.n_clusters),
        "mean_gross_bps": float(res["gross"].mean() / BPS),
        "mean_dm_bps": float(g.mean / BPS),
        "t": float(g.t), "p": float(g.p),
        "ci_bps": [float(g.ci_low / BPS), float(g.ci_high / BPS)],
        "boot_ci_bps": [float(g.boot_ci_low / BPS), float(g.boot_ci_high / BPS)]
        if g.boot_ci_low is not None else None,
        "mean_turnover": float(res["turnover"].mean()),
        "mean_cost_bps": float(res["cost"].mean() / BPS),
        "mean_net_bps": float(n.mean / BPS),
        "net_p": float(n.p),
        "mean_breadth": float(res["breadth"].mean()),
        "start": str(res["ts"].min().date()), "end": str(res["ts"].max().date()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", default="binance")
    ap.add_argument("--symbols", default=",".join(UNIVERSE))
    ap.add_argument("--fee-bps", type=float, default=5.0, help="taker fee per side")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--touch-holdout", action="store_true",
                    help="PRE-REGISTERED: only after exploration and validation both pass")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    panel = build_panel(load(args.venue, "funding", symbols),
                        load(args.venue, "klines", symbols))

    # Sample = breadth-qualified timestamps; the split applies to that span.
    breadth = panel.dropna(subset=["signal", "fwd_8h"]).groupby("ts").size()
    ok_ts = breadth[breadth >= MIN_BREADTH].index
    if len(ok_ts) == 0:
        raise SystemExit(f"no timestamp reaches min_breadth={MIN_BREADTH}")
    panel = panel[panel["ts"].isin(ok_ts)].copy()
    labels, bounds = chronological_split(panel["ts"])
    panel["slice"] = labels

    print(f"venue={args.venue}  symbols={len(symbols)}  fee={args.fee_bps}bps/side")
    print(f"breadth-qualified span: {panel['ts'].min().date()} -> {panel['ts'].max().date()} "
          f"({(panel['ts'].max() - panel['ts'].min()).days:,} days)")
    for nm, b in bounds.items():
        d = (pd.Timestamp(b["end"]) - pd.Timestamp(b["start"])).days
        print(f"  {nm:<12} {b['start'][:10]} -> {b['end'][:10]}  ({d:,} days)")

    slices = ["exploration", "validation"] + (["holdout"] if args.touch_holdout else [])
    if not args.touch_holdout:
        print("\nholdout NOT touched (pass --touch-holdout once, only if expl+val pass)")

    results: dict[str, dict] = {}
    for key, h, q in CELLS:
        results[key] = {}
        for nm in slices:
            sub = panel[panel["slice"] == nm]
            res = run_cell(sub, h, q, args.fee_bps)
            results[key][nm] = summarise(res, args.n_boot)

    # ---- report
    for key, _, _ in [(c[0], c[1], c[2]) for c in CELLS]:
        print(f"\n=== {key} " + "=" * (58 - len(key)))
        print(f"  {'slice':<13}{'gross':>9}{'demeaned':>10}{'t':>7}{'p':>9}"
              f"{'turnover':>10}{'cost':>8}{'NET':>9}{'blocks':>8}")
        for nm in slices:
            r = results[key].get(nm)
            if not r:
                print(f"  {nm:<13}{'insufficient data':>40}")
                continue
            print(f"  {nm:<13}{r['mean_gross_bps']:>+9.1f}{r['mean_dm_bps']:>+10.1f}"
                  f"{r['t']:>7.2f}{r['p']:>9.3f}{r['mean_turnover']:>10.2f}"
                  f"{r['mean_cost_bps']:>8.1f}{r['mean_net_bps']:>+9.1f}{r['n_blocks']:>8,}")

    # ---- pre-registered gates
    print("\n" + "=" * 70)
    expl_p = {k: (results[k].get("exploration") or {}).get("p") for k in results}
    corrected = holm(expl_p)
    print(f"Holm across {len(CELLS)} pre-registered cells (exploration p-values):")
    for k, v in corrected.items():
        thr = f"{v['threshold']:.4f}" if v["threshold"] is not None else "n/a"
        print(f"  {k:<10} p={v['p'] if v['p'] is not None else float('nan'):.4f}  "
              f"thresh={thr}  {'REJECT H0' if v['rejected'] else 'not significant'}")

    print("\nsign stability (the bar):")
    verdicts = {}
    for k in results:
        means = {nm: (results[k].get(nm) or {}).get("mean_dm_bps") for nm in slices}
        stable = sign_stable(means) if all(v is not None for v in means.values()) else False
        nets = {nm: (results[k].get(nm) or {}).get("mean_net_bps") for nm in slices}
        net_stable = sign_stable(nets) if all(v is not None for v in nets.values()) else False
        shown = "  ".join(f"{nm[:4]}={means[nm]:+.1f}" if means[nm] is not None else f"{nm[:4]}=NA"
                          for nm in slices)
        passes = bool(stable and net_stable and corrected.get(k, {}).get("rejected"))
        verdicts[k] = {"sign_stable": stable, "net_sign_stable": net_stable,
                       "holm_rejected": bool(corrected.get(k, {}).get("rejected")),
                       "pass": passes}
        flag = "PASS" if passes else ("flips" if not stable else
                                      ("costs eat it" if not net_stable else "not significant"))
        print(f"  {k:<10} {shown:<44} {flag}")

    if not args.touch_holdout:
        eligible = [k for k, v in verdicts.items() if v["pass"]]
        print(f"\nholdout-eligible cells: {eligible if eligible else 'NONE'}")
        if not eligible:
            print("Pre-registered outcome: NEGATIVE. Do not touch the holdout.")

    rep = ROOT / "reports"
    rep.mkdir(exist_ok=True)
    payload = {"run_at": dt.datetime.now(dt.timezone.utc).isoformat(),
               "venue": args.venue, "fee_bps": args.fee_bps,
               "min_breadth": MIN_BREADTH, "n_cells": len(CELLS),
               "slices_touched": slices, "boundaries": bounds,
               "results": results, "holm": corrected, "verdicts": verdicts}
    name = "funding_extreme_holdout.json" if args.touch_holdout else "funding_extreme.json"
    (rep / name).write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {rep / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
