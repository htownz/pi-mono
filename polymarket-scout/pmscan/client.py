"""Read-only Polymarket API clients.

- GammaClient  -> https://gamma-api.polymarket.com  (market discovery/metadata, public, no auth)
- ClobClient   -> https://clob.polymarket.com        (order books, public read, no auth)

No credentials, no wallet, no order placement. Phase 1/1b is detection only.
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from typing import Iterator

from .models import BookLevel, Market, OrderBook

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
UA = "pmscan/0.2 (read-only scanner)"


def _get_json(url: str, *, body: bytes | None = None, method: str = "GET",
              retries: int = 4, timeout: int = 30):
    """HTTP JSON with simple exponential backoff. stdlib only (no hard deps)."""
    headers = {"User-Agent": UA, "Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            last_err = e
            code = getattr(e, "code", None)
            # don't retry hard client errors except rate limiting
            if code is not None and code != 429 and 400 <= code < 500:
                raise
            time.sleep(min(2 ** attempt, 8) + 0.25 * attempt)
    raise RuntimeError(f"request failed after {retries} attempts: {url} :: {last_err}")


class GammaClient:
    def iter_active_markets(self, max_markets: int = 1000, page_size: int = 100) -> Iterator[dict]:
        """Yield raw active, open, order-book-enabled market dicts.

        Gamma hard-caps at 100/page regardless of `limit`; we paginate with `offset`.
        """
        offset = 0
        fetched = 0
        while fetched < max_markets:
            limit = min(page_size, max_markets - fetched)
            url = (f"{GAMMA}/markets?active=true&closed=false&archived=false"
                   f"&limit={limit}&offset={offset}&order=volume24hr&ascending=false")
            page = _get_json(url)
            if not page:
                break
            for m in page:
                yield m
            fetched += len(page)
            offset += len(page)
            if len(page) < limit:
                break


class ClobClient:
    def get_books(self, token_ids: list[str], chunk: int = 50) -> dict[str, OrderBook]:
        """Batch-fetch order books via POST /books, chunked to stay friendly to the API."""
        out: dict[str, OrderBook] = {}
        for i in range(0, len(token_ids), chunk):
            batch = token_ids[i:i + chunk]
            body = json.dumps([{"token_id": t} for t in batch]).encode()
            res = _get_json(f"{CLOB}/books", body=body, method="POST")
            for b in (res or []):
                tid = b.get("asset_id") or b.get("token_id")
                if not tid:
                    continue
                out[str(tid)] = OrderBook(
                    token_id=str(tid),
                    bids=[BookLevel(float(x["price"]), float(x["size"])) for x in (b.get("bids") or [])],
                    asks=[BookLevel(float(x["price"]), float(x["size"])) for x in (b.get("asks") or [])],
                    tick_size=float(b.get("tick_size", 0.01) or 0.01),
                    neg_risk=bool(b.get("neg_risk", False)),
                )
            time.sleep(0.15)  # gentle pacing; ~10 req/s soft limit
        return out


def _event_title(raw: dict) -> str | None:
    """Best-effort parent-event title from the Gamma `events` array."""
    events = raw.get("events")
    if isinstance(events, list) and events:
        ev = events[0]
        if isinstance(ev, dict):
            return ev.get("title") or ev.get("slug")
    return None


def parse_market(raw: dict) -> Market | None:
    """Convert a raw Gamma market dict into a normalized binary Market, or None if unusable.

    NegRisk outcome markets are themselves binary (Yes/No) and pass this filter; the NegRisk
    linkage fields are captured so Phase 1b can regroup them into events.
    """
    if not raw.get("enableOrderBook") or not raw.get("active") or raw.get("closed"):
        return None
    if raw.get("acceptingOrders") is False:
        return None
    try:
        token_ids = json.loads(raw.get("clobTokenIds") or "[]")
        outcomes = json.loads(raw.get("outcomes") or "[]")
    except (json.JSONDecodeError, TypeError):
        return None
    if len(token_ids) != 2 or len(outcomes) != 2:
        return None  # binary shape only; true >2-way is modeled as a NegRisk event group
    return Market(
        venue="polymarket",
        market_id=raw.get("conditionId", ""),
        question=raw.get("question", ""),
        slug=raw.get("slug", ""),
        outcomes=outcomes,
        token_ids=[str(t) for t in token_ids],
        fees_enabled=bool(raw.get("feesEnabled", False)),
        neg_risk=bool(raw.get("negRisk", False)),
        enable_order_book=True,
        accepting_orders=raw.get("acceptingOrders", True),
        volume_24hr=float(raw.get("volume24hr") or 0.0),
        liquidity=float(raw.get("liquidityNum") or 0.0),
        tick_size=float(raw.get("orderPriceMinTickSize") or 0.01),
        end_date=raw.get("endDate"),
        neg_risk_request_id=(raw.get("negRiskRequestID") or raw.get("negRiskRequestId") or None),
        neg_risk_other=bool(raw.get("negRiskOther", False)),
        group_title=_event_title(raw),
    )
