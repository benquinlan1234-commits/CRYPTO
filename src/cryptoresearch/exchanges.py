"""Perp funding-rate and OHLCV clients for Binance, Bybit and OKX.

Every client exposes the same two calls:

    funding_history(symbol, start_ms, end_ms) -> list[dict(ts, rate)]
    klines(symbol, interval, start_ms, end_ms) -> list[dict(ts, open, high, low, close, volume)]

plus ``probe()``, which reports reachability and how far back the venue will
actually serve funding data. ``probe`` never raises: it classifies the failure
so the caller can tell a geo-block apart from a network problem apart from an
egress-policy denial.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import requests

# Universe carried over from the prior OKX work so results stay comparable.
UNIVERSE = [
    "BTC", "ETH", "SOL", "XRP", "DOGE", "NEAR",
    "UNI", "SUI", "ARB", "BCH", "WLD", "LTC", "AVAX", "LINK",
]

DAY_MS = 86_400_000


@dataclass
class ProbeResult:
    venue: str
    reachable: bool
    status: str                      # ok | geo_blocked | egress_denied | http_error | network_error
    detail: str = ""
    http_code: int | None = None
    earliest_funding_ms: int | None = None
    latest_funding_ms: int | None = None
    funding_points: int | None = None
    funding_interval_h: float | None = None
    max_points_per_call: int | None = None
    per_symbol: dict[str, Any] = field(default_factory=dict)

    @property
    def history_days(self) -> float | None:
        if self.earliest_funding_ms is None or self.latest_funding_ms is None:
            return None
        return (self.latest_funding_ms - self.earliest_funding_ms) / DAY_MS


class VenueError(RuntimeError):
    def __init__(self, status: str, detail: str, http_code: int | None = None):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail
        self.http_code = http_code


class BaseVenue:
    name = "base"
    base_url = ""
    # Documented cap on funding points returned per request.
    funding_page_limit = 100
    # Venues that hard-cap total retrievable funding history, regardless of paging.
    funding_total_cap: int | None = None

    def __init__(self, timeout: int = 20, max_retries: int = 3, pause: float = 0.25):
        self.timeout = timeout
        self.max_retries = max_retries
        self.pause = pause
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "crypto-perp-research/0.1"})

    # -- transport ---------------------------------------------------------
    def _get(self, path: str, params: dict | None = None) -> Any:
        url = self.base_url + path
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
            except requests.exceptions.ProxyError as exc:
                # An egress proxy refusing CONNECT is a policy decision, not a
                # transient fault. Retrying is pointless and the README says so.
                raise VenueError("egress_denied", f"proxy refused CONNECT to {url}: {exc}") from exc
            except requests.exceptions.RequestException as exc:
                last = exc
                time.sleep(2 ** attempt)
                continue

            if r.status_code == 200:
                time.sleep(self.pause)
                return r.json()
            if r.status_code == 451:
                raise VenueError("geo_blocked", _trim(r.text), r.status_code)
            if r.status_code == 403 and _looks_geo(r.text):
                raise VenueError("geo_blocked", _trim(r.text), r.status_code)
            if r.status_code in (418, 429):
                # Rate limited: back off hard, this one is worth retrying.
                time.sleep(5 * (attempt + 1))
                last = VenueError("http_error", f"rate limited {r.status_code}", r.status_code)
                continue
            raise VenueError("http_error", f"HTTP {r.status_code}: {_trim(r.text)}", r.status_code)

        raise VenueError("network_error", f"{type(last).__name__}: {last}")

    # -- interface ---------------------------------------------------------
    def perp_symbol(self, base: str) -> str:
        raise NotImplementedError

    def funding_history(self, symbol: str, start_ms: int, end_ms: int) -> list[dict]:
        raise NotImplementedError

    def klines(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> list[dict]:
        raise NotImplementedError

    # -- probe -------------------------------------------------------------
    def probe(self, bases: list[str] | None = None) -> ProbeResult:
        """Reachability + real history depth. Never raises."""
        bases = bases or ["BTC", "ETH"]
        res = ProbeResult(venue=self.name, reachable=False, status="unknown",
                          max_points_per_call=self.funding_page_limit)
        try:
            now = int(time.time() * 1000)
            # Anchor at 2017 so the venue hands back the oldest page it has.
            first = self.funding_history(self.perp_symbol(bases[0]), 1_483_228_800_000, now)
            if not first:
                res.status = "http_error"
                res.detail = "venue returned no funding rows for the earliest window"
                return res
            latest = self.funding_history(self.perp_symbol(bases[0]), now - 30 * DAY_MS, now)

            res.reachable = True
            res.status = "ok"
            res.earliest_funding_ms = first[0]["ts"]
            res.latest_funding_ms = latest[-1]["ts"] if latest else first[-1]["ts"]
            res.funding_points = len(first)
            if len(first) > 1:
                gaps = [first[i + 1]["ts"] - first[i]["ts"] for i in range(len(first) - 1)]
                gaps.sort()
                res.funding_interval_h = gaps[len(gaps) // 2] / 3_600_000

            for b in bases:
                try:
                    rows = self.funding_history(self.perp_symbol(b), 1_483_228_800_000, now)
                    res.per_symbol[b] = {
                        "symbol": self.perp_symbol(b),
                        "earliest_ms": rows[0]["ts"] if rows else None,
                        "rows_in_first_page": len(rows),
                    }
                except VenueError as exc:
                    res.per_symbol[b] = {"symbol": self.perp_symbol(b), "error": exc.status}
        except VenueError as exc:
            res.status = exc.status
            res.detail = exc.detail
            res.http_code = exc.http_code
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            res.status = "network_error"
            res.detail = f"{type(exc).__name__}: {exc}"
        return res


class Binance(BaseVenue):
    """USDⓈ-M perpetual futures. Deepest funding history of the three."""

    name = "binance"
    base_url = "https://fapi.binance.com"
    funding_page_limit = 1000

    def perp_symbol(self, base: str) -> str:
        return f"{base}USDT"

    def funding_history(self, symbol: str, start_ms: int, end_ms: int) -> list[dict]:
        out: list[dict] = []
        cursor = start_ms
        while cursor < end_ms:
            rows = self._get("/fapi/v1/fundingRate", {
                "symbol": symbol, "startTime": cursor,
                "endTime": end_ms, "limit": self.funding_page_limit,
            })
            if not rows:
                break
            out.extend({"ts": int(r["fundingTime"]), "rate": float(r["fundingRate"])} for r in rows)
            nxt = int(rows[-1]["fundingTime"]) + 1
            if nxt <= cursor:
                break
            cursor = nxt
            if len(rows) < self.funding_page_limit:
                break
        return _dedupe(out)

    def klines(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> list[dict]:
        out: list[dict] = []
        cursor = start_ms
        while cursor < end_ms:
            rows = self._get("/fapi/v1/klines", {
                "symbol": symbol, "interval": interval,
                "startTime": cursor, "endTime": end_ms, "limit": 1500,
            })
            if not rows:
                break
            out.extend({
                "ts": int(r[0]), "open": float(r[1]), "high": float(r[2]),
                "low": float(r[3]), "close": float(r[4]), "volume": float(r[5]),
            } for r in rows)
            nxt = int(rows[-1][0]) + 1
            if nxt <= cursor:
                break
            cursor = nxt
            if len(rows) < 1500:
                break
        return _dedupe(out)


class Bybit(BaseVenue):
    """Linear perpetuals. Returns newest-first, so paging walks backwards."""

    name = "bybit"
    base_url = "https://api.bybit.com"
    funding_page_limit = 200

    def perp_symbol(self, base: str) -> str:
        return f"{base}USDT"

    def _unwrap(self, payload: dict) -> list[dict]:
        if payload.get("retCode") not in (0, None):
            raise VenueError("http_error", f"retCode={payload.get('retCode')} {payload.get('retMsg')}")
        return payload.get("result", {}).get("list", []) or []

    def funding_history(self, symbol: str, start_ms: int, end_ms: int) -> list[dict]:
        out: list[dict] = []
        cursor = end_ms
        while cursor > start_ms:
            rows = self._unwrap(self._get("/v5/market/funding/history", {
                "category": "linear", "symbol": symbol,
                "startTime": start_ms, "endTime": cursor,
                "limit": self.funding_page_limit,
            }))
            if not rows:
                break
            out.extend({"ts": int(r["fundingRateTimestamp"]),
                        "rate": float(r["fundingRate"])} for r in rows)
            oldest = min(int(r["fundingRateTimestamp"]) for r in rows)
            if oldest >= cursor:
                break
            cursor = oldest - 1
            if len(rows) < self.funding_page_limit:
                break
        return _dedupe(out)

    def klines(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> list[dict]:
        iv = {"1h": "60", "4h": "240", "1d": "D", "1m": "1", "5m": "5"}.get(interval, interval)
        out: list[dict] = []
        cursor = end_ms
        while cursor > start_ms:
            rows = self._unwrap(self._get("/v5/market/kline", {
                "category": "linear", "symbol": symbol, "interval": iv,
                "start": start_ms, "end": cursor, "limit": 1000,
            }))
            if not rows:
                break
            out.extend({"ts": int(r[0]), "open": float(r[1]), "high": float(r[2]),
                        "low": float(r[3]), "close": float(r[4]), "volume": float(r[5])}
                       for r in rows)
            oldest = min(int(r[0]) for r in rows)
            if oldest >= cursor:
                break
            cursor = oldest - 1
            if len(rows) < 1000:
                break
        return _dedupe(out)


class OKX(BaseVenue):
    """Swap perpetuals. Funding history is hard-capped near 286 points (~95 days)."""

    name = "okx"
    base_url = "https://www.okx.com"
    funding_page_limit = 100
    funding_total_cap = 286

    def perp_symbol(self, base: str) -> str:
        return f"{base}-USDT-SWAP"

    def _unwrap(self, payload: dict) -> list[dict]:
        if payload.get("code") not in ("0", 0, None):
            raise VenueError("http_error", f"code={payload.get('code')} {payload.get('msg')}")
        return payload.get("data", []) or []

    def funding_history(self, symbol: str, start_ms: int, end_ms: int) -> list[dict]:
        out: list[dict] = []
        cursor = end_ms
        while cursor > start_ms:
            rows = self._unwrap(self._get("/api/v5/public/funding-rate-history", {
                "instId": symbol, "after": cursor, "limit": self.funding_page_limit,
            }))
            if not rows:
                break
            out.extend({"ts": int(r["fundingTime"]), "rate": float(r["fundingRate"])} for r in rows)
            oldest = min(int(r["fundingTime"]) for r in rows)
            if oldest >= cursor:
                break
            cursor = oldest
            if len(rows) < self.funding_page_limit:
                break
            if self.funding_total_cap and len(out) >= self.funding_total_cap:
                break
        return _dedupe(out)

    def klines(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> list[dict]:
        iv = {"1h": "1H", "4h": "4H", "1d": "1D", "1m": "1m", "5m": "5m"}.get(interval, interval)
        out: list[dict] = []
        cursor = end_ms
        while cursor > start_ms:
            rows = self._unwrap(self._get("/api/v5/market/history-candles", {
                "instId": symbol, "bar": iv, "after": cursor, "limit": 100,
            }))
            if not rows:
                break
            out.extend({"ts": int(r[0]), "open": float(r[1]), "high": float(r[2]),
                        "low": float(r[3]), "close": float(r[4]), "volume": float(r[5])}
                       for r in rows)
            oldest = min(int(r[0]) for r in rows)
            if oldest >= cursor:
                break
            cursor = oldest
            if len(rows) < 100:
                break
        return _dedupe(out)


VENUES = {"binance": Binance, "bybit": Bybit, "okx": OKX}


def _dedupe(rows: list[dict]) -> list[dict]:
    seen: dict[int, dict] = {}
    for r in rows:
        seen[r["ts"]] = r
    return [seen[k] for k in sorted(seen)]


def _looks_geo(text: str) -> bool:
    low = (text or "").lower()
    return any(s in low for s in
               ("restricted", "eligibility", "not available in", "service unavailable from"))


def _trim(text: str, n: int = 200) -> str:
    return " ".join((text or "").split())[:n]
