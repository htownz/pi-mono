"""Read-only Kalshi market data client.

Kalshi's market/event endpoints under /trade-api/v2 are reachable without
authentication. This client is intentionally read-only — no trading, no
account access (live trading lives behind the guarded executor in
`execution.py`). Pagination follows Kalshi's `cursor` convention.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Iterator, Optional

import httpx

from weather_trader.models import KalshiEvent, KalshiMarket

DEFAULT_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
DEFAULT_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
USER_AGENT = "weather-trader/0.1 (https://github.com/htownz/pi-mono)"


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value).astimezone(timezone.utc)
    except ValueError:
        return None


def _dollars_to_cents(s) -> Optional[int]:
    """Kalshi 2026 schema: prices arrive as dollar strings, e.g. "0.0900" -> 9."""
    if s is None or s == "":
        return None
    try:
        return int(round(float(s) * 100))
    except (TypeError, ValueError):
        return None


def _price_field(d: dict, dollars_key: str, cents_key: str) -> Optional[int]:
    if dollars_key in d and d[dollars_key] is not None:
        return _dollars_to_cents(d[dollars_key])
    val = d.get(cents_key)
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _int_field(d: dict, fp_key: str, int_key: str, default: int = 0) -> int:
    if fp_key in d and d[fp_key] is not None:
        try:
            return int(float(d[fp_key]))
        except (TypeError, ValueError):
            return default
    val = d.get(int_key)
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def market_from_dict(d: dict) -> KalshiMarket:
    return KalshiMarket(
        ticker=d["ticker"],
        event_ticker=d.get("event_ticker", ""),
        title=d.get("title", ""),
        yes_sub_title=d.get("yes_sub_title", ""),
        status=d.get("status", ""),
        close_time=_parse_dt(d.get("close_time")),
        yes_bid=_price_field(d, "yes_bid_dollars", "yes_bid"),
        yes_ask=_price_field(d, "yes_ask_dollars", "yes_ask"),
        no_bid=_price_field(d, "no_bid_dollars", "no_bid"),
        no_ask=_price_field(d, "no_ask_dollars", "no_ask"),
        last_price=_price_field(d, "last_price_dollars", "last_price"),
        volume=_int_field(d, "volume_fp", "volume"),
        open_interest=_int_field(d, "open_interest_fp", "open_interest"),
        raw=d,
    )


class KalshiClient:
    """Thin synchronous client over the public Kalshi REST API.

    Paces requests (default 200ms) and retries HTTP 429 with honored
    Retry-After, since unauthenticated calls are rate-limited aggressively.
    """

    DEFAULT_PACE_SECONDS = 0.2
    DEFAULT_BACKOFFS = (2.0, 5.0, 15.0, 30.0)

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        client: Optional[httpx.Client] = None,
        pace_seconds: float = DEFAULT_PACE_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )
        self.pace_seconds = pace_seconds
        self._last_request_at: float = 0.0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "KalshiClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        url = f"{self.base_url}{path}"
        for backoff in (*self.DEFAULT_BACKOFFS, None):
            now = time.monotonic()
            elapsed = now - self._last_request_at
            if elapsed < self.pace_seconds:
                time.sleep(self.pace_seconds - elapsed)
            self._last_request_at = time.monotonic()

            resp = self._client.get(url, params=params)
            if resp.status_code == 429 and backoff is not None:
                retry_after = resp.headers.get("Retry-After")
                try:
                    sleep_s = float(retry_after) if retry_after else backoff
                except ValueError:
                    sleep_s = backoff
                time.sleep(max(sleep_s, backoff))
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError("Kalshi 429 retry budget exhausted")

    def iter_markets(
        self,
        event_ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
        status: str = "open",
        limit: int = 200,
    ) -> Iterator[KalshiMarket]:
        cursor: Optional[str] = None
        while True:
            params: dict = {"limit": limit, "status": status}
            if event_ticker:
                params["event_ticker"] = event_ticker
            if series_ticker:
                params["series_ticker"] = series_ticker
            if cursor:
                params["cursor"] = cursor
            data = self._get("/markets", params=params)
            for m in data.get("markets", []):
                yield market_from_dict(m)
            cursor = data.get("cursor") or None
            if not cursor:
                return

    def get_market(self, ticker: str) -> KalshiMarket:
        data = self._get(f"/markets/{ticker}")
        return market_from_dict(data["market"])


# -- Temperature series discovery -------------------------------------------------

# Kalshi's temperature series naming is inconsistent across cities, so we map
# each known series ticker -> (metric, city_slug) explicitly. Metric is a plain
# string here to keep this module free of a models import cycle.
TEMPERATURE_SERIES: dict[str, tuple[str, str]] = {
    # Houston
    "KXHIGHHOU": ("high", "HOUSTON"), "KXHOUHIGH": ("high", "HOUSTON"),
    "KXHIGHTHOU": ("high", "HOUSTON"), "KXLOWTHOU": ("low", "HOUSTON"),
    "KXHIGHHOUSTON": ("high", "HOUSTON"), "KXLOWHOUSTON": ("low", "HOUSTON"),
    # NYC
    "HIGHNY": ("high", "NYC"), "KXHIGHNY": ("high", "NYC"),
    "KXLOWNY": ("low", "NYC"), "KXLOWNYC": ("low", "NYC"),
    "KXLOWTNYC": ("low", "NYC"), "KXHIGHNYC": ("high", "NYC"),
    # Chicago
    "HIGHCHI": ("high", "CHICAGO"), "KXHIGHCHI": ("high", "CHICAGO"),
    "KXLOWCHI": ("low", "CHICAGO"), "KXLOWTCHI": ("low", "CHICAGO"),
    "KXHIGHCHICAGO": ("high", "CHICAGO"),
    # Miami
    "HIGHMIA": ("high", "MIAMI"), "KXHIGHMIA": ("high", "MIAMI"),
    "KXLOWMIA": ("low", "MIAMI"), "KXLOWTMIA": ("low", "MIAMI"),
    "KXHIGHMIAMI": ("high", "MIAMI"),
    # Austin
    "HIGHAUS": ("high", "AUSTIN"), "KXHIGHAUS": ("high", "AUSTIN"),
    "KXLOWAUS": ("low", "AUSTIN"), "KXLOWTAUS": ("low", "AUSTIN"),
    "KXHIGHAUSTIN": ("high", "AUSTIN"),
    # Denver
    "KXHIGHDEN": ("high", "DENVER"), "KXDENHIGH": ("high", "DENVER"),
    "KXHIGHTDEN": ("high", "DENVER"), "KXLOWDEN": ("low", "DENVER"),
    "KXLOWTDEN": ("low", "DENVER"),
    # Los Angeles
    "KXHIGHLAX": ("high", "LA"), "KXLOWLAX": ("low", "LA"),
    "KXLOWTLAX": ("low", "LA"), "KXHIGHLA": ("high", "LA"),
    # Boston
    "KXHIGHTBOS": ("high", "BOSTON"), "KXLOWTBOS": ("low", "BOSTON"),
    # Las Vegas
    "KXHIGHTLV": ("high", "LASVEGAS"), "KXLOWTLV": ("low", "LASVEGAS"),
    # Phoenix
    "KXHIGHTPHX": ("high", "PHOENIX"), "KXLOWTPHX": ("low", "PHOENIX"),
    # New Orleans
    "KXHIGHTNOLA": ("high", "NEWORLEANS"), "KXLOWTNOLA": ("low", "NEWORLEANS"),
    # Atlanta
    "KXHIGHTATL": ("high", "ATLANTA"), "KXLOWTATL": ("low", "ATLANTA"),
    # Oklahoma City
    "KXHIGHTOKC": ("high", "OKCITY"), "KXLOWTOKC": ("low", "OKCITY"),
    # Seattle
    "KXHIGHTSEA": ("high", "SEATTLE"), "KXLOWTSEA": ("low", "SEATTLE"),
    # Dallas
    "KXHIGHTDAL": ("high", "DALLAS"), "KXLOWTDAL": ("low", "DALLAS"),
    # San Francisco
    "KXHIGHTSFO": ("high", "SF"), "KXLOWTSFO": ("low", "SF"),
    # San Antonio
    "KXHIGHTSATX": ("high", "SANANTONIO"), "KXLOWTSATX": ("low", "SANANTONIO"),
    # Philadelphia
    "KXPHILHIGH": ("high", "PHILLY"), "KXHIGHPHIL": ("high", "PHILLY"),
    "KXLOWPHIL": ("low", "PHILLY"), "KXLOWTPHIL": ("low", "PHILLY"),
    # Minneapolis
    "KXHIGHTMIN": ("high", "MINNEAPOLIS"), "KXLOWTMIN": ("low", "MINNEAPOLIS"),
    # DC
    "KXHIGHTDC": ("high", "DC"), "KXLOWTDC": ("low", "DC"),
}


def derive_series_from_event_ticker(event_ticker: str) -> Optional[str]:
    """`KXHIGHHOU-26MAY28` -> `KXHIGHHOU`. None if the shape is unexpected."""
    if not event_ticker or "-" not in event_ticker:
        return None
    return event_ticker.split("-", 1)[0].upper()


def iter_temperature_events(client: KalshiClient, status: str = "open") -> Iterator[KalshiEvent]:
    """Yield all currently-open Kalshi temperature events, grouped from /markets.

    Uses /markets (not /events) because /markets returns the full schema with
    live prices. Queries each known series exactly; groups by event_ticker.
    """
    seen: dict[str, KalshiEvent] = {}
    order: list[str] = []
    for series_ticker in TEMPERATURE_SERIES:
        try:
            for market in client.iter_markets(series_ticker=series_ticker, status=status):
                et = market.event_ticker
                if not et:
                    continue
                if et not in seen:
                    seen[et] = KalshiEvent(
                        event_ticker=et, series_ticker=series_ticker,
                        title="", sub_title="", markets=[],
                    )
                    order.append(et)
                seen[et].markets.append(market)
        except httpx.HTTPStatusError:
            # Retired series 404; skip and continue.
            continue
    for et in order:
        yield seen[et]
