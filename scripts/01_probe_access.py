#!/usr/bin/env python3
"""Task 1, part A: which venues are reachable, and how deep is their funding history?

Run this from a machine with unrestricted egress (i.e. yours, in Australia):

    python scripts/01_probe_access.py

It reports, per venue: reachable or not, the failure class if not, the earliest
funding timestamp actually served, the funding interval, and per-symbol listing
dates for the full universe. Writes reports/access_probe.json.
"""
from __future__ import annotations

import argparse, datetime as dt, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from cryptoresearch.exchanges import UNIVERSE, VENUES, DAY_MS  # noqa: E402

EXPLAIN = {
    "ok": "reachable, funding history served",
    "geo_blocked": "venue refused this IP's jurisdiction (HTTP 451 / restricted-location 403)",
    "egress_denied": "the local network or proxy refused the connection - not the venue's doing",
    "http_error": "venue answered but rejected the request",
    "network_error": "could not establish a connection",
}


def fmt(ms: int | None) -> str:
    if ms is None:
        return "-"
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%Y-%m-%d")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venues", default="binance,bybit,okx")
    ap.add_argument("--symbols", default=",".join(UNIVERSE),
                    help="universe to check listing dates for")
    ap.add_argument("--quick", action="store_true",
                    help="only probe BTC/ETH, skip the per-symbol listing sweep")
    args = ap.parse_args()

    bases = ["BTC", "ETH"] if args.quick else [s.strip() for s in args.symbols.split(",") if s.strip()]
    out = {"run_at": dt.datetime.now(dt.timezone.utc).isoformat(), "venues": {}}

    for name in [v.strip() for v in args.venues.split(",") if v.strip()]:
        if name not in VENUES:
            print(f"unknown venue {name!r}, skipping")
            continue
        print(f"\n=== {name} " + "=" * (60 - len(name)))
        venue = VENUES[name]()
        res = venue.probe(bases=bases)
        out["venues"][name] = {
            k: v for k, v in res.__dict__.items()
        } | {"history_days": res.history_days}

        print(f"  status          : {res.status}  ({EXPLAIN.get(res.status, '')})")
        if res.detail:
            print(f"  detail          : {res.detail}")
        if not res.reachable:
            continue

        print(f"  funding earliest: {fmt(res.earliest_funding_ms)}")
        print(f"  funding latest  : {fmt(res.latest_funding_ms)}")
        days = res.history_days or 0
        print(f"  depth           : {days:,.0f} days  ({days / 365.25:.1f} years)")
        print(f"  interval        : {res.funding_interval_h:g}h" if res.funding_interval_h else "")
        print(f"  points per call : {res.max_points_per_call}")
        if venue.funding_total_cap:
            print(f"  TOTAL CAP       : {venue.funding_total_cap} points "
                  f"(~{venue.funding_total_cap * (res.funding_interval_h or 8) / 24:.0f} days) "
                  "- this is the constraint that made the prior funding test inconclusive")

        # Three-slice feasibility: the question the prior OKX work could not answer.
        if days:
            per_slice = days * 0.20
            verdict = "ADEQUATE" if per_slice >= 180 else ("THIN" if per_slice >= 60 else "TOO SHORT")
            print(f"  -> 60/20/20 gives {days * 0.6:,.0f}d / {per_slice:,.0f}d / {per_slice:,.0f}d  [{verdict}]")
            print(f"     (prior OKX slices were 19d each; the bar is real, independent slices)")
            out["venues"][name]["slice_days"] = [days * 0.6, per_slice, per_slice]
            out["venues"][name]["slice_verdict"] = verdict

        if not args.quick and res.per_symbol:
            print(f"\n  {'symbol':<10}{'venue symbol':<18}{'funding from':<14}{'days'}")
            now_ms = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)
            for b in bases:
                info = res.per_symbol.get(b, {})
                if "error" in info:
                    print(f"  {b:<10}{info.get('symbol', '-'):<18}{'ERROR: ' + info['error']}")
                    continue
                e = info.get("earliest_ms")
                d = (now_ms - e) / DAY_MS if e else 0
                print(f"  {b:<10}{info.get('symbol', '-'):<18}{fmt(e):<14}{d:,.0f}")

    rep = pathlib.Path(__file__).resolve().parents[1] / "reports"
    rep.mkdir(exist_ok=True)
    path = rep / "access_probe.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {path}")

    ok = [n for n, v in out["venues"].items() if v.get("status") == "ok"]
    print("\n" + "=" * 66)
    if not ok:
        print("NO VENUE REACHABLE - nothing downstream can run.")
        return 1
    print(f"reachable: {', '.join(ok)}")
    deep = max(ok, key=lambda n: out["venues"][n].get("history_days") or 0)
    print(f"deepest history: {deep} "
          f"({(out['venues'][deep].get('history_days') or 0) / 365.25:.1f} years) -> use --venue {deep}")
    print("next: python scripts/02_fetch_funding.py --venue " + deep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
