"""Instrumentation for the funding-fade / OI-spike paper tracker.

Drop-in, side-effect-free helpers for `paper_track.py`. Nothing here touches the
signal, the sizing or the universe -- it only changes what gets *reported*, so
the forward test stays clean.

Four things:

1. `expectation_block()`  -- report both the full-history and recent-regime
   Sharpe bands, and state which one the forward test is actually being
   evaluated against.
2. `noise_band()`         -- the +/-1sd P&L range for the elapsed period, so an
   ordinary negative month is not misread as a breakdown.
3. `book_overlap()`       -- shared-name count between the funding and OI books,
   split same-side vs opposite-side, with a running average persisted to disk.
4. `utc_now()`            -- `pd.Timestamp.now("UTC")`, replacing the deprecated
   `pd.Timestamp.utcnow()`.

Run `python paper_track_instrumentation.py` for a self-test that reproduces the
numbers quoted in the wiring notes.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

# --------------------------------------------------------------------------
# 1. Expectation -- measured figures, Bybit funding, K=14 / H=3, net of costs.
#
# These are the *Bybit* numbers. The Binance-funding figure (Sharpe 1.14) is
# deliberately absent: it was a venue artefact and is not a benchmark.
# --------------------------------------------------------------------------

ANN_VOL = 0.11  # annualised vol of the live book at 0.5x leverage

FULL_HISTORY_SHARPE = 0.57          # K=14 / H=3 point estimate, 2022-2026
FULL_HISTORY_BAND = (0.6, 0.9)      # honest range across the K x H grid
RECENT_SHARPE = 1.4                 # 2024-onward, midpoint of the band below
RECENT_BAND = (1.3, 1.5)            # honest range across the K x H grid, 2024+

BY_YEAR = {                         # Bybit funding, K=14 / H=3
    2022: -1.32,
    2023: -0.46,
    2024: +1.07,
    2025: +1.37,
    2026: +2.09,
}

#: The band the forward test is scored against. The live regime is 2026, so the
#: relevant prior is the recent-regime figure, not the full-history one.
EVALUATION_SHARPE = RECENT_SHARPE
EVALUATION_BAND = RECENT_BAND


def utc_now() -> pd.Timestamp:
    """Timezone-aware current time.

    Replaces `pd.Timestamp.utcnow()`, which pandas deprecated because it
    returned a *naive* timestamp despite the name. This one is tz-aware, so
    comparisons against tz-aware index values no longer raise.
    """
    return pd.Timestamp.now("UTC")


def expectation_block() -> str:
    """The expectation line, reporting both regimes and naming the live one."""
    lo_f, hi_f = FULL_HISTORY_BAND
    lo_r, hi_r = RECENT_BAND
    years = " ".join(f"{y}:{s:+.2f}" for y, s in sorted(BY_YEAR.items()))
    return (
        "EXPECTATION (Bybit funding, K=14 H=3, net of costs)\n"
        f"  full history 2022-2026 : Sharpe {lo_f:.1f}-{hi_f:.1f} "
        f"(point est {FULL_HISTORY_SHARPE:.2f})\n"
        f"  recent regime 2024+    : Sharpe {lo_r:.1f}-{hi_r:.1f}\n"
        f"  by year                : {years}\n"
        f"  SCORED AGAINST         : {lo_r:.1f}-{hi_r:.1f} (recent regime) -- "
        "the live regime is 2026, so full history is context, not the benchmark.\n"
        "  Full history spans two losing years (2022, 2023). Scoring a 2026\n"
        "  forward test against it would set the bar in the wrong place; the\n"
        "  full-history band is reported so the regime dependence stays visible."
    )


# --------------------------------------------------------------------------
# 2. Noise band
# --------------------------------------------------------------------------

DAYS_PER_YEAR = 365.0


@dataclass(frozen=True)
class NoiseBand:
    """Expected cumulative-return range for an elapsed window.

    All return figures are fractions of capital (0.0125 == +1.25%).
    """

    elapsed_days: float
    years: float
    sharpe: float
    ann_vol: float
    expected: float          # sharpe * ann_vol * years
    sd: float                # ann_vol * sqrt(years)
    lo_1sd: float
    hi_1sd: float
    lo_2sd: float
    hi_2sd: float
    null_lo_1sd: float       # same band centred on Sharpe 0
    null_hi_1sd: float
    days_to_1sd_separation: float
    days_to_2sd_separation: float
    sharpe_standard_error: float

    def assess(self, actual: float) -> tuple[str, float]:
        """Classify a realised cumulative return against the band.

        Returns (status, z) where z is in standard deviations from *expected*.
        Status is WITHIN_BAND unless |z| > 1, so ordinary noise does not flag.
        """
        if self.sd <= 0:
            return "NO_BAND", 0.0
        z = (actual - self.expected) / self.sd
        if abs(z) <= 1.0:
            return "WITHIN_BAND", z
        if abs(z) <= 2.0:
            return ("ABOVE_1SD" if z > 0 else "BELOW_1SD"), z
        return ("ABOVE_2SD" if z > 0 else "BELOW_2SD"), z


def noise_band(
    elapsed_days: float,
    sharpe: float = EVALUATION_SHARPE,
    ann_vol: float = ANN_VOL,
) -> NoiseBand:
    """Expected cumulative-return range after `elapsed_days` of live trading.

    Under the usual iid-return model, cumulative return over t years has
    mean `S * sigma * t` and standard deviation `sigma * sqrt(t)`. The mean
    grows linearly in t while the noise grows as sqrt(t), which is why a short
    window says nothing: at four weeks the sd term dominates the mean term by
    roughly an order of magnitude.

    The 1sd lower bound clears zero only once `t > 1/S**2`; that crossing is
    reported as `days_to_1sd_separation`.
    """
    if elapsed_days < 0:
        raise ValueError("elapsed_days must be non-negative")
    years = elapsed_days / DAYS_PER_YEAR
    expected = sharpe * ann_vol * years
    sd = ann_vol * math.sqrt(years)

    # Standard error of an annualised Sharpe estimated over `years` of data
    # (Lo 2002): sqrt((1 + S^2/2) / n_years).
    se = (
        math.sqrt((1.0 + 0.5 * sharpe**2) / years) if years > 0 else float("inf")
    )

    sep1 = DAYS_PER_YEAR / sharpe**2 if sharpe else float("inf")
    sep2 = 4.0 * DAYS_PER_YEAR / sharpe**2 if sharpe else float("inf")

    return NoiseBand(
        elapsed_days=elapsed_days,
        years=years,
        sharpe=sharpe,
        ann_vol=ann_vol,
        expected=expected,
        sd=sd,
        lo_1sd=expected - sd,
        hi_1sd=expected + sd,
        lo_2sd=expected - 2 * sd,
        hi_2sd=expected + 2 * sd,
        null_lo_1sd=-sd,
        null_hi_1sd=sd,
        days_to_1sd_separation=sep1,
        days_to_2sd_separation=sep2,
        sharpe_standard_error=se,
    )


def noise_block(
    band: NoiseBand,
    actual_return: float | None = None,
    capital: float | None = None,
) -> str:
    """Render the noise band, in % and optionally in currency."""

    def pct(x: float) -> str:
        return f"{100 * x:+.2f}%"

    def cash(x: float) -> str:
        return "" if capital is None else f" ({capital * x:+,.0f})"

    weeks = band.elapsed_days / 7.0
    out = [
        f"NOISE BAND ({band.elapsed_days:.0f} days = {weeks:.1f} weeks elapsed, "
        f"{100 * band.ann_vol:.0f}% ann vol, Sharpe {band.sharpe:.1f} prior)",
        f"  expected P&L    : {pct(band.expected)}{cash(band.expected)}",
        f"  1sd range       : {pct(band.lo_1sd)} to {pct(band.hi_1sd)}"
        + (
            ""
            if capital is None
            else f" ({capital * band.lo_1sd:+,.0f} to {capital * band.hi_1sd:+,.0f})"
        ),
        f"  2sd range       : {pct(band.lo_2sd)} to {pct(band.hi_2sd)}",
        f"  if edge were 0  : {pct(band.null_lo_1sd)} to {pct(band.null_hi_1sd)} "
        "(1sd) -- overlaps the band above almost entirely",
        f"  period sd       : {100 * band.sd:.2f}% vs expected "
        f"{100 * band.expected:.2f}% -- noise exceeds signal by "
        f"{band.sd / band.expected:.1f}x"
        if band.expected > 0
        else "  period sd       : n/a",
        f"  Sharpe est. SE  : +/-{band.sharpe_standard_error:.1f} Sharpe units "
        "over this window -- a realised Sharpe here is uninformative",
        f"  1sd separation from zero at {band.days_to_1sd_separation:.0f} days "
        f"({band.days_to_1sd_separation / 30.4:.1f} months)",
        f"  2sd separation from zero at {band.days_to_2sd_separation:.0f} days "
        f"({band.days_to_2sd_separation / 365:.1f} years)",
    ]

    if actual_return is None:
        out.append("  actual          : (no closed P&L yet)")
        return "\n".join(out)

    status, z = band.assess(actual_return)
    out.append(
        f"  actual          : {pct(actual_return)}{cash(actual_return)} "
        f"(z = {z:+.2f})"
    )
    if status == "WITHIN_BAND":
        out.append(
            "  -> WITHIN BAND. No signal either way. Do not read this as "
            "confirmation or breakdown."
        )
    elif status in ("BELOW_1SD", "ABOVE_1SD"):
        out.append(
            f"  -> FLAG {status}: outside 1sd but inside 2sd. Worth noting, "
            "not yet evidence."
        )
    elif status in ("BELOW_2SD", "ABOVE_2SD"):
        out.append(
            f"  -> FLAG {status}: outside 2sd. Investigate -- check execution, "
            "funding capture and slippage before concluding anything about edge."
        )
    return "\n".join(out)


# --------------------------------------------------------------------------
# 3. Book overlap
# --------------------------------------------------------------------------


def as_side_map(book) -> dict[str, int]:
    """Normalise a book into {symbol: +1 long / -1 short}.

    Accepts whatever `paper_track.py` already holds:
      * {"BTCUSDT": -1, ...} or {"BTCUSDT": -0.0625, ...}  (signed weights)
      * {"BTCUSDT": "short", ...}
      * {"longs": [...], "shorts": [...]}
      * pandas Series of signed weights
      * an object with `.longs` / `.shorts` attributes
    Zero / NaN weights are dropped, so a flat name never counts as overlap.
    """
    if book is None:
        return {}

    if isinstance(book, pd.Series):
        book = book.to_dict()

    if not isinstance(book, Mapping):
        longs = getattr(book, "longs", None)
        shorts = getattr(book, "shorts", None)
        if longs is None and shorts is None:
            raise TypeError(f"cannot interpret book of type {type(book)!r}")
        book = {"longs": longs or [], "shorts": shorts or []}

    lowered = {str(k).lower() for k in book}
    if lowered & {"longs", "shorts", "long", "short"}:
        out: dict[str, int] = {}
        for key, sign in (("long", +1), ("short", -1)):
            names = book.get(f"{key}s") or book.get(key) or []
            if isinstance(names, Mapping):
                names = list(names)
            for sym in names:
                out[str(sym).upper()] = sign
        return out

    out = {}
    for sym, val in book.items():
        if isinstance(val, str):
            v = val.strip().lower()
            if v in ("long", "buy", "l", "+1"):
                out[str(sym).upper()] = +1
            elif v in ("short", "sell", "s", "-1"):
                out[str(sym).upper()] = -1
            continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            continue
        if f != f or f == 0.0:  # NaN or flat
            continue
        out[str(sym).upper()] = 1 if f > 0 else -1
    return out


@dataclass(frozen=True)
class OverlapStats:
    n_funding: int
    n_oi: int
    same_side: int
    opposite_side: int
    shared_names: int
    same_side_symbols: list[str]
    opposite_side_symbols: list[str]
    overlap_fraction: float        # shared names / sqrt(n_a * n_b)
    implied_corr_floor: float      # (same - opposite) / sqrt(n_a * n_b)


def book_overlap(funding_book, oi_book) -> OverlapStats:
    """Shared names between the two books, split by direction.

    Same-side overlap is what erodes diversification: the same name in the same
    direction in both books is one doubled position wearing two hats.
    Opposite-side overlap is the reverse -- the signals disagree and the
    exposures partly cancel, which *adds* independence rather than removing it.

    `implied_corr_floor` is `(same - opposite) / sqrt(n_a * n_b)`: the
    correlation the two P&L streams would show from shared positions alone if
    every name's return were independent with equal vol. It is a structural
    floor, not an estimate of the realised correlation -- it ignores the common
    crypto factor, which the net-neutral construction mostly but not entirely
    removes. Compare it against the measured correlation: if the floor is the
    larger number, the measured figure is understating how linked the books are.
    """
    a = as_side_map(funding_book)
    b = as_side_map(oi_book)

    shared = sorted(set(a) & set(b))
    same = [s for s in shared if a[s] == b[s]]
    opp = [s for s in shared if a[s] != b[s]]

    n_a, n_b = len(a), len(b)
    denom = math.sqrt(n_a * n_b) if n_a and n_b else 0.0

    return OverlapStats(
        n_funding=n_a,
        n_oi=n_b,
        same_side=len(same),
        opposite_side=len(opp),
        shared_names=len(shared),
        same_side_symbols=same,
        opposite_side_symbols=opp,
        overlap_fraction=(len(shared) / denom) if denom else 0.0,
        implied_corr_floor=((len(same) - len(opp)) / denom) if denom else 0.0,
    )


@dataclass
class OverlapHistory:
    """Running overlap log, persisted as JSON so averages survive restarts."""

    path: Path
    runs: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "OverlapHistory":
        p = Path(path)
        runs: list[dict] = []
        if p.exists():
            try:
                loaded = json.loads(p.read_text())
                if isinstance(loaded, list):
                    runs = [r for r in loaded if isinstance(r, dict)]
            except (json.JSONDecodeError, OSError):
                runs = []  # corrupt log must not take the tracker down
        return cls(path=p, runs=runs)

    def record(self, stats: OverlapStats, ts: pd.Timestamp | None = None) -> None:
        self.runs.append(
            {
                "ts": (ts or utc_now()).isoformat(),
                "n_funding": stats.n_funding,
                "n_oi": stats.n_oi,
                "same_side": stats.same_side,
                "opposite_side": stats.opposite_side,
                "shared_names": stats.shared_names,
                "same_side_symbols": stats.same_side_symbols,
                "opposite_side_symbols": stats.opposite_side_symbols,
                "overlap_fraction": stats.overlap_fraction,
                "implied_corr_floor": stats.implied_corr_floor,
            }
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.runs, indent=2))

    def _mean(self, key: str) -> float:
        vals = [r[key] for r in self.runs if key in r]
        return sum(vals) / len(vals) if vals else 0.0

    def persistent_names(self, min_runs: int = 2) -> list[tuple[str, int]]:
        """Names that have been same-side in both books across several runs."""
        counts: dict[str, int] = {}
        for r in self.runs:
            for sym in r.get("same_side_symbols", []):
                counts[sym] = counts.get(sym, 0) + 1
        return sorted(
            ((s, c) for s, c in counts.items() if c >= min_runs),
            key=lambda kv: (-kv[1], kv[0]),
        )


def overlap_block(
    stats: OverlapStats,
    history: OverlapHistory,
    measured_corr: float | None = None,
) -> str:
    """Render this run's overlap plus the running averages."""
    n = len(history.runs)
    out = [
        f"SIGNAL OVERLAP (funding book {stats.n_funding} names, "
        f"OI book {stats.n_oi} names)",
        f"  same-side     : {stats.same_side}"
        + (f"  {', '.join(stats.same_side_symbols)}" if stats.same_side_symbols else ""),
        f"  opposite-side : {stats.opposite_side}"
        + (
            f"  {', '.join(stats.opposite_side_symbols)}"
            if stats.opposite_side_symbols
            else ""
        ),
        f"  shared names  : {stats.shared_names} "
        f"(overlap fraction {stats.overlap_fraction:.2f})",
        f"  implied corr floor from shared positions: "
        f"{stats.implied_corr_floor:+.2f}",
    ]
    if n:
        out += [
            f"  running avg over {n} run(s): same-side "
            f"{history._mean('same_side'):.1f}, opposite-side "
            f"{history._mean('opposite_side'):.1f}, shared "
            f"{history._mean('shared_names'):.1f}, corr floor "
            f"{history._mean('implied_corr_floor'):+.2f}",
        ]
        persistent = history.persistent_names()
        if persistent:
            names = ", ".join(f"{s}x{c}" for s, c in persistent[:10])
            out.append(f"  persistent same-side names: {names}")

    if measured_corr is not None:
        out.append(f"  measured P&L correlation: {measured_corr:+.2f}")
        if stats.implied_corr_floor > measured_corr + 0.05:
            out.append(
                "  -> FLAG: structural overlap exceeds measured correlation. "
                "The measured figure understates the linkage; treat the two "
                "books as less diversifying than it implies."
            )
        else:
            out.append(
                "  -> measured correlation is consistent with the shared "
                "positions; no hidden linkage indicated."
            )
    return "\n".join(out)


# --------------------------------------------------------------------------
# One call that renders all of it
# --------------------------------------------------------------------------


def instrumentation_report(
    elapsed_days: float,
    funding_book,
    oi_book,
    history_path: str | Path,
    actual_return: float | None = None,
    capital: float | None = None,
    measured_corr: float | None = None,
    sharpe: float = EVALUATION_SHARPE,
    ann_vol: float = ANN_VOL,
    persist: bool = True,
) -> str:
    band = noise_band(elapsed_days, sharpe=sharpe, ann_vol=ann_vol)
    stats = book_overlap(funding_book, oi_book)
    history = OverlapHistory.load(history_path)
    if persist:
        history.record(stats)
        history.save()

    return "\n\n".join(
        [
            expectation_block(),
            noise_block(band, actual_return=actual_return, capital=capital),
            overlap_block(stats, history, measured_corr=measured_corr),
        ]
    )


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

def _selftest() -> None:
    import tempfile

    # -- utc_now is tz-aware (the whole point of the deprecation) --------
    now = utc_now()
    assert now.tzinfo is not None, "utc_now must be tz-aware"

    # -- noise band, 28 days at Sharpe 1.4 / 11% vol --------------------
    b = noise_band(28, sharpe=1.4, ann_vol=0.11)
    assert abs(b.sd - 0.11 * math.sqrt(28 / 365)) < 1e-12
    assert b.lo_1sd < 0 < b.hi_1sd, "a normal month must be able to be negative"
    assert b.assess(b.expected)[0] == "WITHIN_BAND"
    assert b.assess(b.expected - 1.5 * b.sd)[0] == "BELOW_1SD"
    assert b.assess(b.expected - 3 * b.sd)[0] == "BELOW_2SD"
    assert b.assess(0.0)[0] == "WITHIN_BAND", "zero P&L at 4 weeks is not a flag"

    # -- book normalisation across input shapes -------------------------
    weights = {"THETAUSDT": -0.06, "INJUSDT": +0.06, "BTCUSDT": 0.0}
    sides = {"THETAUSDT": "short", "INJUSDT": "long"}
    split = {"shorts": ["THETAUSDT"], "longs": ["INJUSDT"]}
    base = {"THETAUSDT": -1, "INJUSDT": +1}
    assert as_side_map(weights) == base, as_side_map(weights)
    assert as_side_map(sides) == base
    assert as_side_map(split) == base
    assert as_side_map(pd.Series(weights)) == base
    assert as_side_map(None) == {}

    # -- today's book ---------------------------------------------------
    funding = {
        "THETAUSDT": -1, "1000PEPEUSDT": -1, "CRVUSDT": -1, "ENAUSDT": -1,
        "ORDIUSDT": -1, "TIAUSDT": -1, "WIFUSDT": -1,
        "INJUSDT": +1, "ATOMUSDT": +1, "DOTUSDT": +1, "FILUSDT": +1,
        "LTCUSDT": +1, "ETCUSDT": +1, "XLMUSDT": +1,
    }
    oi = {
        "THETAUSDT": -1, "1000PEPEUSDT": -1, "CRVUSDT": -1, "SEIUSDT": -1,
        "JUPUSDT": -1, "ARBUSDT": -1, "OPUSDT": -1,
        "INJUSDT": +1, "AVAXUSDT": +1, "NEARUSDT": +1, "AAVEUSDT": +1,
        "SUIUSDT": +1, "APTUSDT": +1, "RUNEUSDT": +1,
    }
    st = book_overlap(funding, oi)
    assert st.same_side == 4, st.same_side
    assert sorted(st.same_side_symbols) == [
        "1000PEPEUSDT", "CRVUSDT", "INJUSDT", "THETAUSDT",
    ]
    assert st.opposite_side == 0
    assert abs(st.implied_corr_floor - 4 / 14) < 1e-9

    # opposite-side must subtract, not add
    flipped = dict(oi, INJUSDT=-1)
    st2 = book_overlap(funding, flipped)
    assert st2.same_side == 3 and st2.opposite_side == 1
    assert st2.implied_corr_floor < st.implied_corr_floor

    # -- history persists and averages ----------------------------------
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "overlap_history.json"
        h = OverlapHistory.load(p)
        h.record(st); h.record(st2); h.save()
        h2 = OverlapHistory.load(p)
        assert len(h2.runs) == 2
        assert abs(h2._mean("same_side") - 3.5) < 1e-9
        assert dict(h2.persistent_names())["THETAUSDT"] == 2

        # corrupt log must degrade, not raise
        p.write_text("{ not json")
        assert OverlapHistory.load(p).runs == []

        print(
            instrumentation_report(
                elapsed_days=28,
                funding_book=funding,
                oi_book=oi,
                history_path=Path(d) / "report_history.json",
                actual_return=-0.004,
                capital=100_000,
                measured_corr=0.09,
            )
        )

    print("\nall self-tests passed")


if __name__ == "__main__":
    _selftest()
