"""Harness validation on synthetic data with a known ground truth.

These are the checks that decide whether a result from this codebase can be
believed at all, so they run against data whose true effect is known by
construction.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from cryptoresearch.protocol import (
    chronological_split, demean_drift, cross_sectional_demean,
    cluster_test, holm, sign_stable,
)

RNG = np.random.default_rng(20260914)


def _panel(n_days=400, n_sym=14, day_sd=1.0, obs_sd=1.0, drift=None):
    """Panel where each day carries a shared shock: rows are NOT independent."""
    rows = []
    day_shock = RNG.normal(0, day_sd, n_days)
    for d in range(n_days):
        day = pd.Timestamp("2021-01-01", tz="UTC") + pd.Timedelta(days=d)
        for s in range(n_sym):
            for h in (0, 8, 16):
                mu = 0.0 if drift is None else drift[s]
                rows.append({
                    "ts": day + pd.Timedelta(hours=h),
                    "symbol": f"S{s}",
                    "day": day.date(),
                    "ret": mu + day_shock[d] + RNG.normal(0, obs_sd),
                })
    return pd.DataFrame(rows)


def test_split_is_calendar_shared_across_symbols():
    """Unbalanced panel: a late lister must still share BTC's slice boundaries."""
    df = _panel(n_days=300, n_sym=3)
    df = df[~((df.symbol == "S2") & (df.ts < "2021-06-01"))]  # S2 lists late
    labels, bounds = chronological_split(df["ts"])
    df = df.assign(slice=labels)

    for nm in ("exploration", "validation", "holdout"):
        sub = df[df["slice"] == nm]
        spans = sub.groupby("symbol")["ts"].agg(["min", "max"])
        # Every symbol present in a slice sits inside that slice's window.
        assert spans["min"].min() >= pd.Timestamp(bounds[nm]["start"])
        assert spans["max"].max() <= pd.Timestamp(bounds[nm]["end"])
    assert df["slice"].notna().all()
    # 60/20/20 by calendar, so exploration holds roughly 3x the validation days.
    frac = df.groupby("slice")["ts"].nunique()
    assert frac["exploration"] > frac["validation"] * 2
    print("  split: shared calendar boundaries hold on an unbalanced panel")


def test_cluster_test_controls_size_where_naive_test_does_not():
    """The headline check. Day-correlated noise, true mean zero.

    A naive t-test treats each row as independent and should reject far more
    than 5% of the time. The clustered test should land near nominal.
    """
    naive_rej = clust_rej = 0
    trials = 300
    for _ in range(trials):
        df = _panel(n_days=120, n_sym=6, day_sd=1.0, obs_sd=1.0)
        y = df["ret"].to_numpy()
        # naive: pretend every row is an independent observation
        t_naive = y.mean() / (y.std(ddof=1) / np.sqrt(y.size))
        if abs(t_naive) > stats.t.ppf(0.975, y.size - 1):
            naive_rej += 1
        r = cluster_test(y, df["day"])
        if r.p < 0.05:
            clust_rej += 1

    naive_rate, clust_rate = naive_rej / trials, clust_rej / trials
    print(f"  size under day-clustering: naive={naive_rate:.1%}  clustered={clust_rate:.1%}  (nominal 5%)")
    assert naive_rate > 0.30, f"naive test should blow up, got {naive_rate:.1%}"
    assert clust_rate < 0.10, f"clustered test should be near nominal, got {clust_rate:.1%}"


def test_cluster_test_still_finds_a_real_effect():
    """Power check: a genuine effect must survive the conservative SEs."""
    df = _panel(n_days=400, n_sym=6, day_sd=0.5, obs_sd=1.0)
    df["ret"] += 0.25  # true mean 0.25
    r = cluster_test(df["ret"], df["day"], n_boot=500)
    print(f"  power: mean={r.mean:.3f} t={r.t:.2f} p={r.p:.2e} "
          f"CI=({r.ci_low:.3f},{r.ci_high:.3f}) boot=({r.boot_ci_low:.3f},{r.boot_ci_high:.3f})")
    assert r.p < 0.01 and r.ci_low > 0
    assert r.n_clusters == 400, "effective sample must be days, not rows"
    assert r.n_obs == 400 * 6 * 3
    # Bootstrap and analytic CIs should broadly agree.
    assert abs(r.boot_ci_low - r.ci_low) < 0.05


def test_demeaning_kills_drift_manufactured_momentum():
    """Symbols with real drift must not read as an effect once demeaned."""
    drift = RNG.normal(-0.30, 0.05, 8)  # the CI-clean negative alt drift
    df = _panel(n_days=300, n_sym=8, day_sd=0.2, obs_sd=1.0, drift=drift)
    labels, _ = chronological_split(df["ts"])
    df = df.assign(slice=labels)

    raw = cluster_test(df["ret"], df["day"])
    df["ret_dm"] = demean_drift(df, "ret")
    dm = cluster_test(df["ret_dm"], df["day"])
    print(f"  drift: raw mean={raw.mean:+.3f} (p={raw.p:.1e}) -> demeaned mean={dm.mean:+.3f} (p={dm.p:.2f})")
    assert raw.p < 0.01 and raw.mean < 0, "raw panel should show the fake effect"
    assert abs(dm.mean) < 1e-9, "per-slice demeaning must remove it exactly"


def test_expanding_demean_is_causal():
    """mode='expanding' may only use strictly prior observations."""
    df = _panel(n_days=50, n_sym=2, day_sd=0.0, obs_sd=1.0).sort_values("ts").reset_index(drop=True)
    out = demean_drift(df, "ret", mode="expanding")
    first_per_symbol = df.groupby("symbol").head(1).index
    assert out.loc[first_per_symbol].isna().all(), "first obs has no prior mean"
    # Recompute one value by hand.
    s0 = df[df.symbol == "S0"].reset_index(drop=True)
    manual = s0.loc[5, "ret"] - s0.loc[:4, "ret"].mean()
    got = out.loc[df[df.symbol == "S0"].index[5]]
    assert abs(manual - got) < 1e-12
    print("  expanding demean: uses only strictly prior observations")


def test_cross_sectional_demean_removes_the_market():
    df = _panel(n_days=60, n_sym=5, day_sd=2.0, obs_sd=0.5)
    df["rel"] = cross_sectional_demean(df, "ret")
    per_ts = df.groupby("ts")["rel"].mean().abs().max()
    assert per_ts < 1e-12, "each timestamp must net to zero"
    print("  cross-sectional demean: market move removed at every timestamp")


def test_holm_corrects_for_the_stated_cell_count():
    pv = {"8h_20pct": 0.004, "8h_10pct": 0.02, "24h_20pct": 0.03, "24h_10pct": 0.9}
    out = holm(pv)
    assert all(v["n_cells"] == 4 for v in out.values())
    assert out["8h_20pct"]["threshold"] == 0.05 / 4
    assert out["8h_20pct"]["rejected"] is True
    # 0.02 > 0.05/3 = 0.0167, so it fails and everything after it fails too.
    assert out["8h_10pct"]["rejected"] is False
    assert out["24h_20pct"]["rejected"] is False
    print("  holm: 4 cells, only p=0.004 survives")


def test_sign_stability_is_the_bar():
    assert sign_stable({"exploration": -17.8, "validation": -12.0, "holdout": -19.2})
    # The real 4h reversion result: flips twice, must fail.
    assert not sign_stable({"exploration": -17.8, "validation": +54.6, "holdout": -19.2})
    assert not sign_stable({"exploration": 1.0, "validation": 1.0, "holdout": np.nan})
    print("  sign stability: the -17.8/+54.6/-19.2 flip correctly fails")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\nall {len(fns)} harness checks passed")
