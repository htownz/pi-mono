"""Kalshi orderbook depth parsing + fill-quality math.

The /markets/<ticker>/orderbook endpoint returns the raw bid book for both
Yes and No sides. Kalshi's invariant: a Yes bid at price X is equivalent to
a No ask at 100-X (and vice versa). The walker below uses that to derive
tradable ask prices for either side from either book.

Two outputs the engine cares about:

  - tradable_price(side):     top-of-book ask cents on that side
  - fillable_at_size(side, n): the weighted average fill price for n
                              contracts walked through the book

A "fillable at size" of `None` means the book doesn't have enough depth to
honor the requested size — in that case the headline edge is illusory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class OrderbookLevel:
    price_cents: int
    size_contracts: int


@dataclass(frozen=True)
class Orderbook:
    """Snapshot of one market's order book.

    `yes_bids` are bids on the Yes side (sorted descending by price).
    `no_bids` are bids on the No side (sorted descending by price).
    Yes asks are derived as 100 - no_bid for each no_bids level.
    """
    market_ticker: str
    yes_bids: tuple[OrderbookLevel, ...]
    no_bids: tuple[OrderbookLevel, ...]

    @property
    def yes_asks(self) -> tuple[OrderbookLevel, ...]:
        """Yes ask = 100 - no_bid. Ascending by ask price (most aggressive first)."""
        derived = [
            OrderbookLevel(price_cents=100 - lvl.price_cents, size_contracts=lvl.size_contracts)
            for lvl in self.no_bids
        ]
        # Sort by price ascending — cheapest Yes ask first.
        return tuple(sorted(derived, key=lambda x: x.price_cents))

    @property
    def no_asks(self) -> tuple[OrderbookLevel, ...]:
        derived = [
            OrderbookLevel(price_cents=100 - lvl.price_cents, size_contracts=lvl.size_contracts)
            for lvl in self.yes_bids
        ]
        return tuple(sorted(derived, key=lambda x: x.price_cents))

    def top_ask(self, side: str) -> Optional[int]:
        """Best (cheapest) ask price for `side`, or None if no liquidity."""
        levels = self.yes_asks if side == "yes" else self.no_asks
        return levels[0].price_cents if levels else None

    def fillable_at_size(self, side: str, size_contracts: int) -> Optional["FillQuote"]:
        """Walk the book and compute the average fill price for `size_contracts`.

        Returns None if the book has zero depth on `side`. If depth is less
        than requested, returns a partial fill with `partial=True` and
        `filled_size` < requested.
        """
        levels = self.yes_asks if side == "yes" else self.no_asks
        if not levels:
            return None
        remaining = size_contracts
        spent = 0
        filled = 0
        worst = levels[0].price_cents
        for lvl in levels:
            if remaining <= 0:
                break
            take = min(remaining, lvl.size_contracts)
            spent += take * lvl.price_cents
            filled += take
            remaining -= take
            worst = max(worst, lvl.price_cents)
        if filled == 0:
            return None
        return FillQuote(
            side=side,
            requested_size=size_contracts,
            filled_size=filled,
            avg_price_cents=spent / filled,
            worst_price_cents=worst,
            partial=filled < size_contracts,
        )


@dataclass(frozen=True)
class FillQuote:
    """The book's answer to 'can I buy N contracts and what's it cost'?"""
    side: str                    # 'yes' or 'no'
    requested_size: int
    filled_size: int
    avg_price_cents: float
    worst_price_cents: int
    partial: bool

    def edge_against(self, fair_prob: float) -> float:
        """Realized edge (in decimal probability units) given a fair prob.

        For Yes: edge = fair - avg_ask/100
        For No:  edge = (1 - fair) - avg_ask/100
        """
        ask = self.avg_price_cents / 100.0
        return (fair_prob - ask) if self.side == "yes" else ((1.0 - fair_prob) - ask)


# -- Parsing -----------------------------------------------------------------

def parse_orderbook(raw: dict, market_ticker: str = "") -> Orderbook:
    """Parse Kalshi /orderbook response into a typed Orderbook.

    Kalshi's response shape:
        {
          "orderbook": {
            "yes": [[price_cents, contracts], ...],   # bids
            "no":  [[price_cents, contracts], ...]    # bids
          }
        }

    Some responses wrap differently; we accept either the wrapped or
    unwrapped form.
    """
    book = raw.get("orderbook") if isinstance(raw, dict) and "orderbook" in raw else raw
    if not isinstance(book, dict):
        return Orderbook(market_ticker=market_ticker, yes_bids=(), no_bids=())

    def _levels(side_key: str) -> tuple[OrderbookLevel, ...]:
        raw_levels = book.get(side_key) or []
        out: list[OrderbookLevel] = []
        for item in raw_levels:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    price = int(item[0])
                    size = int(item[1])
                except (TypeError, ValueError):
                    continue
                if size <= 0 or price <= 0 or price >= 100:
                    continue
                out.append(OrderbookLevel(price_cents=price, size_contracts=size))
        # Sort bids descending (most aggressive / highest price first).
        out.sort(key=lambda x: x.price_cents, reverse=True)
        return tuple(out)

    return Orderbook(
        market_ticker=market_ticker,
        yes_bids=_levels("yes"),
        no_bids=_levels("no"),
    )
