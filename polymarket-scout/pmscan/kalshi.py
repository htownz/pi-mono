"""Read-only Kalshi market-data bridge for cross-venue parity. DETECTION ONLY.

Kalshi's `/trade-api/v2/markets` endpoints are public (no auth), so this pulls live quotes
with the standard library only — pmscan never imports `kalshi_scout` (which carries httpx /
pydantic / signing deps). It mirrors `kalshi_scout.kalshi`'s parsing: prices arrive either as
new-schema `*_dollars` strings ("0.0900") or legacy integer cents; we normalize to dollars and
hand back the venue-agnostic `VenueQuote` the parity detector consumes.

No trading, no account access, no order placement — just GET /markets/{ticker}.
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from typing import Optional

from .parity import VenueQuote

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
UA = "pmscan (read-only cross-venue parity)"   # no version — avoids drift from __version__

# A non-tradable market still reports its last prices; quoting one (e.g. a dated game ticker
# after the event settled) would manufacture a cross-venue "lock" that can't be entered. Skip.
NON_TRADABLE_STATUSES = {"closed", "settled", "finalized", "determined", "expired",
                         "inactive", "unopened"}


def _dollar_price(raw: dict, dollars_key: str, cents_key: str) -> Optional[float]:
    """Price in dollars (0..1) from either the new `*_dollars` string or legacy cents int."""
    v = raw.get(dollars_key)
    if v not in (None, ""):
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    c = raw.get(cents_key)
    if c is None:
        return None
    try:
        return int(c) / 100.0
    except (TypeError, ValueError):
        return None


def market_to_quote(raw: dict) -> VenueQuote | None:
    """Build a venue-agnostic VenueQuote (prices in dollars) from a raw Kalshi market dict.

    Returns None for a tickerless or non-tradable (closed/settled/...) market, so stale prices
    on a finished market never become a phantom, unenterable cross-venue lock.
    """
    ticker = raw.get("ticker")
    if not ticker:
        return None
    if (raw.get("status") or "").lower() in NON_TRADABLE_STATUSES:
        return None
    return VenueQuote(
        venue="kalshi",
        market_key=ticker,
        label=raw.get("title") or raw.get("yes_sub_title") or ticker,
        yes_bid=_dollar_price(raw, "yes_bid_dollars", "yes_bid"),
        yes_ask=_dollar_price(raw, "yes_ask_dollars", "yes_ask"),
        no_bid=_dollar_price(raw, "no_bid_dollars", "no_bid"),
        no_ask=_dollar_price(raw, "no_ask_dollars", "no_ask"),
        # Top-of-book sizes aren't in the market object; would need /orderbook (follow-up).
    )


class KalshiQuotes:
    """Thin stdlib client over public Kalshi market data, with pacing + 429 backoff."""

    PACE_SECONDS = 0.2          # ~5 req/s; Kalshi rate-limits unauth calls aggressively
    BACKOFFS = (2.0, 5.0, 15.0, 30.0)

    def __init__(self, base_url: str = KALSHI_BASE, pace_seconds: float = PACE_SECONDS):
        self.base_url = base_url.rstrip("/")
        self.pace_seconds = pace_seconds
        self._last_request_at = 0.0

    def _get(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        for backoff in (*self.BACKOFFS, None):
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self.pace_seconds:
                time.sleep(self.pace_seconds - elapsed)
            self._last_request_at = time.monotonic()
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                if e.code == 429 and backoff is not None:
                    retry_after = e.headers.get("Retry-After") if e.headers else None
                    try:
                        sleep_s = float(retry_after) if retry_after else backoff
                    except ValueError:
                        sleep_s = backoff
                    time.sleep(max(sleep_s, backoff))
                    continue
                raise
        raise RuntimeError(f"Kalshi 429 retry budget exhausted: {url}")

    def get_quote(self, ticker: str) -> VenueQuote | None:
        """Live two-sided quote for one Kalshi ticker, or None if it doesn't resolve."""
        data = self._get(f"/markets/{ticker}")
        market = data.get("market") if isinstance(data, dict) else None
        return market_to_quote(market) if market else None

    def get_quotes(self, tickers: list[str]) -> dict[str, VenueQuote]:
        """Fetch quotes for several tickers (paced). Tickers that error/miss are skipped."""
        out: dict[str, VenueQuote] = {}
        for t in tickers:
            try:
                q = self.get_quote(t)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
                continue
            if q is not None:
                out[t] = q
        return out
