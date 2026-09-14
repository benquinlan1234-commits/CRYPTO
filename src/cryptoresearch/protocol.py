"""The research protocol, as code.

Every rule the project treats as non-negotiable lives here so that no
individual test can quietly opt out of one:

* 60/20/20 chronological split on a shared calendar
* day/block-clustered inference - the effective sample is sessions
* drift demeaning before any momentum or reversion statement
* sign stability across all three slices as the pass bar
* explicit multiple-testing correction with a stated cell count
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_FRACS = (0.60, 0.20, 0.20)
SLICE_NAMES = ("exploration", "validation", "holdout")


# ---------------------------------------------------------------- splitting
def chronological_split(
    ts: pd.Series,
    fracs: Sequence[float] = DEFAULT_FRACS,
    names: Sequence[str] = SLICE_NAMES,
) -> tuple[pd.Series, dict]:
    """Label each row's slice by *calendar* position, not row count.

    Splitting on row count would give each symbol its own boundaries in an
    unbalanced panel, so a late-listing alt would land in a different slice
    than BTC at the same instant. Cross-sectional pooling needs one shared
    calendar, so the cut points come from the time axis.
    """
    if abs(sum(fracs) - 1.0) > 1e-9:
        raise ValueError(f"fracs must sum to 1, got {sum(fracs)}")
    if len(fracs) != len(names):
        raise ValueError("fracs and names must be the same length")

    ts = pd.to_datetime(ts, utc=True)
    lo, hi = ts.min(), ts.max()
    span = (hi - lo).total_seconds()
    if not span > 0:
        raise ValueError("time span is zero; cannot split")

    edges, acc = [lo], 0.0
    for f in fracs[:-1]:
        acc += f
        edges.append(lo + pd.Timedelta(seconds=span * acc))
    edges.append(hi)

    labels = pd.Series(pd.NA, index=ts.index, dtype="object")
    for i, nm in enumerate(names):
        left, right = edges[i], edges[i + 1]
        sel = (ts >= left) & (ts <= right if i == len(names) - 1 else ts < right)
        labels[sel] = nm

    boundaries = {
        nm: {"start": str(edges[i]), "end": str(edges[i + 1])}
        for i, nm in enumerate(names)
    }
    return labels, boundaries


# ------------------------------------------------------------- demeaning
def demean_drift(
    df: pd.DataFrame,
    value_col: str,
    symbol_col: str = "symbol",
    slice_col: str = "slice",
    mode: str = "slice",
    ts_col: str = "ts",
) -> pd.Series:
    """Remove per-symbol drift so it cannot masquerade as momentum.

    mode="slice"     - subtract each symbol's mean within each slice. Uses the
                       whole slice, so it answers "is there an effect beyond
                       this symbol's drift", not "could I have traded it".
    mode="expanding" - subtract each symbol's mean of all *strictly prior*
                       observations. Strictly causal, noisy early in the sample.
    """
    if mode == "slice":
        return df[value_col] - df.groupby([symbol_col, slice_col])[value_col].transform("mean")
    if mode == "expanding":
        out = df.sort_values(ts_col).groupby(symbol_col)[value_col]
        prior_mean = out.transform(lambda s: s.shift(1).expanding().mean())
        return (df[value_col] - prior_mean.reindex(df.index)).astype(float)
    raise ValueError(f"unknown mode {mode!r}")


def cross_sectional_demean(df: pd.DataFrame, value_col: str, ts_col: str = "ts") -> pd.Series:
    """Remove the market move at each timestamp, leaving relative performance."""
    return df[value_col] - df.groupby(ts_col)[value_col].transform("mean")


# ------------------------------------------------------------- inference
@dataclass
class ClusterTest:
    n_obs: int
    n_clusters: int
    mean: float
    se: float
    t: float
    p: float
    ci_low: float
    ci_high: float
    boot_ci_low: float | None = None
    boot_ci_high: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def cluster_test(
    y: Sequence[float],
    clusters: Sequence,
    n_boot: int = 0,
    seed: int = 0,
    alpha: float = 0.05,
) -> ClusterTest:
    """One-sample test of mean(y)=0 with one-way cluster-robust standard errors.

    This is the intercept-only CRVE: observations inside a cluster may be
    arbitrarily correlated, so the effective sample size is the number of
    clusters (here, trading days), not the number of rows. Degrees of freedom
    are G-1 for the same reason.
    """
    y = np.asarray(y, dtype=float)
    c = np.asarray(clusters)
    keep = np.isfinite(y)
    y, c = y[keep], c[keep]

    n = y.size
    if n == 0:
        raise ValueError("no finite observations")
    uniq = pd.unique(c)
    g = len(uniq)
    if g < 2:
        raise ValueError(f"need >=2 clusters, got {g}")

    mean = float(y.mean())
    resid = y - mean
    # meat = sum_g (sum_{i in g} e_i)^2
    sums = pd.Series(resid).groupby(pd.Series(c)).sum().to_numpy()
    meat = float((sums ** 2).sum())
    # Standard finite-sample correction for CRVE with K=1 parameter.
    corr = (g / (g - 1)) * ((n - 1) / max(n - 1, 1))
    var = corr * meat / (n ** 2)
    se = float(np.sqrt(var))
    df = g - 1

    if se > 0:
        t = mean / se
        p = float(2 * stats.t.sf(abs(t), df))
        crit = float(stats.t.ppf(1 - alpha / 2, df))
    else:
        t, p, crit = np.nan, np.nan, np.nan

    out = ClusterTest(
        n_obs=n, n_clusters=g, mean=mean, se=se, t=float(t), p=p,
        ci_low=mean - crit * se, ci_high=mean + crit * se,
    )

    if n_boot:
        lo, hi = block_bootstrap_ci(y, c, n_boot=n_boot, seed=seed, alpha=alpha)
        out.boot_ci_low, out.boot_ci_high = lo, hi
    return out


def block_bootstrap_ci(
    y: Sequence[float],
    clusters: Sequence,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI for the mean, resampling whole clusters with replacement."""
    y = np.asarray(y, dtype=float)
    c = np.asarray(clusters)
    keep = np.isfinite(y)
    y, c = y[keep], c[keep]

    order = np.argsort(c, kind="stable")
    y_sorted, c_sorted = y[order], c[order]
    _, starts = np.unique(c_sorted, return_index=True)
    blocks = np.split(y_sorted, starts[1:])

    rng = np.random.default_rng(seed)
    g = len(blocks)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, g, g)
        means[b] = np.concatenate([blocks[i] for i in pick]).mean()
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


# -------------------------------------------------- multiple testing / bar
def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni over an explicitly named set of cells."""
    items = [(k, v) for k, v in pvals.items() if v is not None and np.isfinite(v)]
    m = len(items)
    items.sort(key=lambda kv: kv[1])
    out, still_rejecting = {}, True
    for i, (k, p) in enumerate(items):
        thresh = alpha / (m - i)
        rejected = bool(still_rejecting and p <= thresh)
        if not rejected:
            still_rejecting = False
        out[k] = {"p": p, "threshold": thresh, "rejected": rejected, "n_cells": m}
    for k, v in pvals.items():
        if k not in out:
            out[k] = {"p": v, "threshold": None, "rejected": False, "n_cells": m}
    return out


def sign_stable(means: dict[str, float]) -> bool:
    """The bar: identical, non-zero sign in every slice tested."""
    vals = [v for v in means.values() if v is not None and np.isfinite(v)]
    if len(vals) < len(means) or not vals:
        return False
    signs = {np.sign(v) for v in vals}
    return len(signs) == 1 and 0 not in signs
