"""Cross-bracket arbitrage detection across mutually-exclusive Kalshi events.

In a Kalshi event with N mutually-exclusive brackets, exactly one resolves
Yes and the rest resolve No. Two no-arbitrage bounds follow:

    Σ yes_asks ≥ 100c   - else buy Yes on every bracket: guaranteed 100c payout
                          (only one wins) for cost Σ yes_asks. Profit if Σ < 100.

    Σ yes_bids ≤ 100c   - else sell Yes on every bracket (= buy No on every):
                          collect Σ yes_bids upfront. Exactly one Yes wins,
                          you pay 100 to settle that single position. Profit
                          = Σ yes_bids - 100 - N × fee_per_leg.

The right side is fee-sensitive: Kalshi's per-trade fee depends on side and
size. We approximate with a flat per-leg cents value (default 2c, override
via CLI). True arb survives the fee; "edge" reported is after-fee.

This module is **category-agnostic** — it operates on KalshiEvent shape only.
Same code applies to weather, sports, politics, econ data, entertainment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from kalshi_scout.models import KalshiEvent


@dataclass(frozen=True)
class EventArbitrage:
    """One event's arbitrage analysis. None values indicate missing prices
    on at least one leg; those events are surfaced but not actionable."""
    event_ticker: str
    n_brackets: int
    n_priced_brackets: int               # legs with non-None yes_ask
    sum_yes_asks_cents: Optional[int]     # None if any leg missing yes_ask
    sum_yes_bids_cents: Optional[int]     # None if any leg missing yes_bid
    fee_per_leg_cents: int

    # Gross = before fees; Net = after N × fee_per_leg.
    yes_basket_gross_edge_cents: Optional[int]
    yes_basket_net_edge_cents: Optional[int]
    no_basket_gross_edge_cents: Optional[int]
    no_basket_net_edge_cents: Optional[int]

    # Best side & edge — "yes" or "no" or None, regardless of sign.
    # The ranker filters on net > min threshold; here we just record.
    best_side: Optional[str]
    best_net_edge_cents: Optional[int]

    market_tickers: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def is_actionable(self) -> bool:
        return (
            self.best_net_edge_cents is not None
            and self.best_net_edge_cents > 0
        )


def compute_event_arbitrage(
    event: KalshiEvent,
    fee_per_leg_cents: int = 2,
) -> Optional[EventArbitrage]:
    """Compute arbitrage analysis for one event. Returns None if the event
    has fewer than 2 markets (not mutually-exclusive — no arb possible)."""
    if not event.markets or len(event.markets) < 2:
        return None

    n = len(event.markets)
    yes_asks = [m.yes_ask for m in event.markets]
    yes_bids = [m.yes_bid for m in event.markets]

    # Count legs where yes_ask is a real positive price.
    n_priced = sum(1 for a in yes_asks if a is not None and a > 0)

    sum_asks: Optional[int] = None
    if all(a is not None and a > 0 for a in yes_asks):
        sum_asks = sum(yes_asks)

    sum_bids: Optional[int] = None
    if all(b is not None and b > 0 for b in yes_bids):
        sum_bids = sum(yes_bids)

    total_fee = n * fee_per_leg_cents

    yes_gross = (100 - sum_asks) if sum_asks is not None else None
    yes_net = (yes_gross - total_fee) if yes_gross is not None else None

    no_gross = (sum_bids - 100) if sum_bids is not None else None
    no_net = (no_gross - total_fee) if no_gross is not None else None

    best_side: Optional[str] = None
    best_edge: Optional[int] = None
    if yes_net is not None:
        best_edge = yes_net
        best_side = "yes"
    if no_net is not None and (best_edge is None or no_net > best_edge):
        best_edge = no_net
        best_side = "no"

    notes_list: list[str] = []
    if n_priced < n:
        notes_list.append(f"{n - n_priced}/{n} legs missing yes_ask")
    if sum_asks is None and sum_bids is None:
        notes_list.append("no prices on any leg")

    return EventArbitrage(
        event_ticker=event.event_ticker,
        n_brackets=n,
        n_priced_brackets=n_priced,
        sum_yes_asks_cents=sum_asks,
        sum_yes_bids_cents=sum_bids,
        fee_per_leg_cents=fee_per_leg_cents,
        yes_basket_gross_edge_cents=yes_gross,
        yes_basket_net_edge_cents=yes_net,
        no_basket_gross_edge_cents=no_gross,
        no_basket_net_edge_cents=no_net,
        best_side=best_side,
        best_net_edge_cents=best_edge,
        market_tickers=tuple(m.ticker for m in event.markets),
        notes=tuple(notes_list),
    )


def rank_arbitrage_opportunities(
    events: list[KalshiEvent],
    fee_per_leg_cents: int = 2,
    min_net_edge_cents: int = 1,
) -> list[EventArbitrage]:
    """Compute arb per event, drop those below threshold, sort by net edge."""
    out: list[EventArbitrage] = []
    for event in events:
        arb = compute_event_arbitrage(event, fee_per_leg_cents=fee_per_leg_cents)
        if arb is None or arb.best_net_edge_cents is None:
            continue
        if arb.best_net_edge_cents < min_net_edge_cents:
            continue
        out.append(arb)
    out.sort(key=lambda a: -(a.best_net_edge_cents or 0))
    return out
