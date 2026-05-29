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


def _dollars_to_cents(s) -> Optional[int]:
    """Kalshi 2026 schema: prices come as strings in dollars, e.g. "0.0900".

    Returns the integer cent equivalent (9), or None when the field is
    missing or unparseable.
    """
    if s is None or s == "":
        return None
    try:
        return int(round(float(s) * 100))
    except (TypeError, ValueError):
        return None


def _price_field(d: dict, dollars_key: str, cents_key: str) -> Optional[int]:
    """Read a price from either the new _dollars key or the legacy cents key."""
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
    """Read an integer that might come as fractional-float string or int."""
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


def _market_from_dict(d: dict) -> KalshiMarket:
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

# Kalshi's temperature series naming is inconsistent across cities. There is no
# single prefix that catches them all — some use `KXHIGH<CITY>`, some `HIGH<CITY>`
# (no KX), some `KXHIGHT<CITY>` (extra T for "Temperature"), and some put the
# city first like `KX<CITY>HIGH`. City codes are 2-4 letters (HOU/NY/NYC/LAX).
#
# So we hardcode a series-ticker -> (metric_str, city_slug) map. Verified
# against the live /series endpoint (the temperature-relevant entries within
# the "Climate and Weather" category as of 2026-05). Adding a new city just
# means appending an entry here and a Station to stations.py.
#
# Metric is stored as a plain string here ("high"/"low") to keep this module
# free of the models.Metric enum import (models.py would create a cycle).

TEMPERATURE_SERIES: dict[str, tuple[str, str]] = {
    # Houston (HOU)
    "KXHIGHHOU":    ("high", "HOUSTON"),
    "KXHOUHIGH":    ("high", "HOUSTON"),
    "KXHIGHTHOU":   ("high", "HOUSTON"),
    "KXLOWTHOU":    ("low", "HOUSTON"),
    # Houston explicit full-city form (kept so existing test fixtures still parse).
    "KXHIGHHOUSTON": ("high", "HOUSTON"),
    "KXLOWHOUSTON":  ("low", "HOUSTON"),

    # NYC
    "HIGHNY":       ("high", "NYC"),
    "KXHIGHNY":     ("high", "NYC"),
    "KXLOWNY":      ("low", "NYC"),
    "KXLOWNYC":     ("low", "NYC"),
    "KXLOWTNYC":    ("low", "NYC"),
    "KXHIGHNYC":    ("high", "NYC"),

    # Chicago
    "HIGHCHI":      ("high", "CHICAGO"),
    "KXHIGHCHI":    ("high", "CHICAGO"),
    "KXLOWCHI":     ("low", "CHICAGO"),
    "KXLOWTCHI":    ("low", "CHICAGO"),
    "KXHIGHCHICAGO": ("high", "CHICAGO"),

    # Miami
    "HIGHMIA":      ("high", "MIAMI"),
    "KXHIGHMIA":    ("high", "MIAMI"),
    "KXLOWMIA":     ("low", "MIAMI"),
    "KXLOWTMIA":    ("low", "MIAMI"),
    "KXHIGHMIAMI":  ("high", "MIAMI"),

    # Austin
    "HIGHAUS":      ("high", "AUSTIN"),
    "KXHIGHAUS":    ("high", "AUSTIN"),
    "KXLOWAUS":     ("low", "AUSTIN"),
    "KXLOWTAUS":    ("low", "AUSTIN"),
    "KXHIGHAUSTIN": ("high", "AUSTIN"),

    # Denver
    "KXHIGHDEN":    ("high", "DENVER"),
    "KXDENHIGH":    ("high", "DENVER"),
    "KXHIGHTDEN":   ("high", "DENVER"),
    "KXLOWDEN":     ("low", "DENVER"),
    "KXLOWTDEN":    ("low", "DENVER"),

    # Los Angeles
    "KXHIGHLAX":    ("high", "LA"),
    "KXLOWLAX":     ("low", "LA"),
    "KXLOWTLAX":    ("low", "LA"),
    "KXHIGHLA":     ("high", "LA"),

    # Boston
    "KXHIGHTBOS":   ("high", "BOSTON"),
    "KXLOWTBOS":    ("low", "BOSTON"),

    # Las Vegas
    "KXHIGHTLV":    ("high", "LASVEGAS"),
    "KXLOWTLV":     ("low", "LASVEGAS"),

    # Phoenix
    "KXHIGHTPHX":   ("high", "PHOENIX"),
    "KXLOWTPHX":    ("low", "PHOENIX"),

    # New Orleans
    "KXHIGHTNOLA":  ("high", "NEWORLEANS"),
    "KXLOWTNOLA":   ("low", "NEWORLEANS"),

    # Atlanta
    "KXHIGHTATL":   ("high", "ATLANTA"),
    "KXLOWTATL":    ("low", "ATLANTA"),

    # Oklahoma City
    "KXHIGHTOKC":   ("high", "OKCITY"),
    "KXLOWTOKC":    ("low", "OKCITY"),

    # Seattle
    "KXHIGHTSEA":   ("high", "SEATTLE"),
    "KXLOWTSEA":    ("low", "SEATTLE"),

    # Dallas
    "KXHIGHTDAL":   ("high", "DALLAS"),
    "KXLOWTDAL":    ("low", "DALLAS"),

    # San Francisco
    "KXHIGHTSFO":   ("high", "SF"),
    "KXLOWTSFO":    ("low", "SF"),

    # San Antonio
    "KXHIGHTSATX":  ("high", "SANANTONIO"),
    "KXLOWTSATX":   ("low", "SANANTONIO"),

    # Philadelphia
    "KXPHILHIGH":   ("high", "PHILLY"),
    "KXHIGHPHIL":   ("high", "PHILLY"),
    "KXLOWPHIL":    ("low", "PHILLY"),
    "KXLOWTPHIL":   ("low", "PHILLY"),

    # Minneapolis
    "KXHIGHTMIN":   ("high", "MINNEAPOLIS"),
    "KXLOWTMIN":    ("low", "MINNEAPOLIS"),

    # DC
    "KXHIGHTDC":    ("high", "DC"),
    "KXLOWTDC":     ("low", "DC"),
}


def derive_series_from_event_ticker(event_ticker: str) -> Optional[str]:
    """Given an event ticker like `KXHIGHHOU-26MAY28`, return the series
    portion `KXHIGHHOU`. Returns None if the shape is unexpected."""
    if not event_ticker or "-" not in event_ticker:
        return None
    return event_ticker.split("-", 1)[0].upper()


def iter_all_open_events(
    client: KalshiClient,
    min_brackets: int = 2,
) -> Iterator[KalshiEvent]:
    """Yield every open Kalshi event with `min_brackets` or more markets,
    regardless of category.

    For arbitrage scanning across the full exchange. Single-market events
    are skipped because Yes+No on one contract always sums to 100 by
    construction (no cross-bracket arb possible).

    Pages through /markets without a series filter. With Kalshi's ~thousands
    of open markets this is ~15-25 pages = a 30-second pass.
    """
    seen: dict[str, KalshiEvent] = {}
    order: list[str] = []
    for market in client.iter_markets(status="open"):
        event_ticker = market.event_ticker
        if not event_ticker:
            continue
        if event_ticker not in seen:
            seen[event_ticker] = KalshiEvent(
                event_ticker=event_ticker,
                series_ticker="",
                title="",
                sub_title="",
                markets=[],
            )
            order.append(event_ticker)
        seen[event_ticker].markets.append(market)
    for event_ticker in order:
        event = seen[event_ticker]
        if len(event.markets) >= min_brackets:
            yield event


def iter_temperature_events(client: KalshiClient, status: str = "open") -> Iterator[KalshiEvent]:
    """Yield all currently-open Kalshi temperature events grouped from /markets.

    We use /markets — not /events with nested markets — because Kalshi's
    /events response returns simplified market objects without live prices
    or `rules_primary`. The /markets endpoint returns the full market
    schema (yes_ask_dollars, yes_bid_dollars, volume_fp, rules_primary, etc.).

    Groups results by event_ticker. Series query is exact-match per
    TEMPERATURE_SERIES entry.
    """
    seen_events: dict[str, KalshiEvent] = {}
    order: list[str] = []
    for series_ticker in TEMPERATURE_SERIES:
        try:
            for market in client.iter_markets(series_ticker=series_ticker, status=status):
                event_ticker = market.event_ticker
                if not event_ticker:
                    continue
                if event_ticker not in seen_events:
                    seen_events[event_ticker] = KalshiEvent(
                        event_ticker=event_ticker,
                        series_ticker=series_ticker,
                        title="",
                        sub_title="",
                        markets=[],
                    )
                    order.append(event_ticker)
                seen_events[event_ticker].markets.append(market)
        except httpx.HTTPStatusError:
            # Some historical series may have been retired; skip and continue.
            continue
    for event_ticker in order:
        yield seen_events[event_ticker]
