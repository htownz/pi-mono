"""Public Kalshi market data client.

Kalshi's market/event/orderbook endpoints under /trade-api/v2 are accessible
without authentication; we only need an outbound HTTPS connection. The client
is intentionally read-only — no trading, no account access.

Pagination follows Kalshi's `cursor` convention: the response includes a
non-empty `cursor` string when more pages exist.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator, Optional

import httpx

from kalshi_scout.models import KalshiEvent, KalshiMarket

DEFAULT_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
DEFAULT_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
USER_AGENT = "kalshi-scout/0.3 (https://github.com/htownz/pi-mono)"


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    # Kalshi emits ISO-8601 with trailing 'Z'
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value).astimezone(timezone.utc)
    except ValueError:
        return None


def _market_from_dict(d: dict) -> KalshiMarket:
    return KalshiMarket(
        ticker=d["ticker"],
        event_ticker=d.get("event_ticker", ""),
        title=d.get("title", ""),
        yes_sub_title=d.get("yes_sub_title", ""),
        status=d.get("status", ""),
        close_time=_parse_dt(d.get("close_time")),
        yes_bid=d.get("yes_bid"),
        yes_ask=d.get("yes_ask"),
        no_bid=d.get("no_bid"),
        no_ask=d.get("no_ask"),
        last_price=d.get("last_price"),
        volume=int(d.get("volume") or 0),
        open_interest=int(d.get("open_interest") or 0),
        raw=d,
    )


def _event_from_dict(d: dict) -> KalshiEvent:
    return KalshiEvent(
        event_ticker=d["event_ticker"],
        series_ticker=d.get("series_ticker", ""),
        title=d.get("title", ""),
        sub_title=d.get("sub_title", ""),
        markets=[_market_from_dict(m) for m in d.get("markets") or []],
        raw=d,
    )


class KalshiClient:
    """Thin synchronous client over the public Kalshi REST API."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "KalshiClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        resp = self._client.get(f"{self.base_url}{path}", params=params)
        resp.raise_for_status()
        return resp.json()

    # -- Series / events / markets -------------------------------------------------

    def iter_events(
        self,
        series_ticker: Optional[str] = None,
        status: str = "open",
        with_nested_markets: bool = True,
        limit: int = 200,
    ) -> Iterator[KalshiEvent]:
        cursor: Optional[str] = None
        while True:
            params: dict = {
                "limit": limit,
                "status": status,
                "with_nested_markets": str(with_nested_markets).lower(),
            }
            if series_ticker:
                params["series_ticker"] = series_ticker
            if cursor:
                params["cursor"] = cursor
            data = self._get("/events", params=params)
            for e in data.get("events", []):
                yield _event_from_dict(e)
            cursor = data.get("cursor") or None
            if not cursor:
                return

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
                yield _market_from_dict(m)
            cursor = data.get("cursor") or None
            if not cursor:
                return

    def get_market(self, ticker: str) -> KalshiMarket:
        data = self._get(f"/markets/{ticker}")
        return _market_from_dict(data["market"])

    def get_orderbook(self, ticker: str, depth: int = 32) -> dict:
        """Returns the raw orderbook dict.

        Kalshi reports both `yes` and `no` arrays of [price_cents, contracts].
        Yes bid at X corresponds to No ask at 100-X by definition, so the
        caller can derive tradable prices for either side from either book.
        """
        return self._get(f"/markets/{ticker}/orderbook", params={"depth": depth})


# -- Temperature series discovery -------------------------------------------------

TEMPERATURE_SERIES_PREFIXES = (
    "KXHIGH",  # daily high temp series, e.g. KXHIGHHOUSTON
    "KXLOW",   # daily low temp series, e.g. KXLOWHOUSTON
    "KXTEMP",  # generic temperature variants seen on some cities
)


def is_temperature_series(series_ticker: str) -> bool:
    s = series_ticker.upper()
    return any(s.startswith(p) for p in TEMPERATURE_SERIES_PREFIXES)


def iter_temperature_events(client: KalshiClient, status: str = "open") -> Iterator[KalshiEvent]:
    """Yield all currently-open Kalshi temperature events across known prefixes.

    We pull /events with each known series prefix. If a city's series ticker
    doesn't match our known prefixes, the universe scanner will miss it; the
    parser layer is the safety net (it ignores titles it can't interpret).
    """
    seen: set[str] = set()
    for prefix in TEMPERATURE_SERIES_PREFIXES:
        # Kalshi accepts series_ticker as a prefix match in /events
        for event in client.iter_events(series_ticker=prefix, status=status):
            if event.event_ticker in seen:
                continue
            seen.add(event.event_ticker)
            yield event
