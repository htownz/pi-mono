"""Cross-venue parity detector (DRAFT scaffold). DETECTION ONLY.

The structural, non-latency edge the whole architecture points at: the *same* real-world
binary outcome priced on two venues (Polymarket + Kalshi). If you can buy YES on the cheaper
venue and the complementary NO on the other for less than $1 combined, exactly one pays $1 —
a locked cross-venue arb, independent of which way the event resolves:

    construction 1:  buy YES@A + buy NO@B   cost = yes_ask_A + no_ask_B
    construction 2:  buy YES@B + buy NO@A   cost = yes_ask_B + no_ask_A
    edge = $1 - min(cost over available constructions)

This is the cross-venue analogue of the within-market merge edge, except the two legs live
on different venues. Unlike the single-venue case, NOTHING here is mechanical: the lock is
real ONLY if both venues settle the identical event identically (same resolution source, same
date, same bucket boundaries). That settlement-equivalence is the parity analogue of NegRisk
exhaustiveness — unprovable from price data alone — so we carry it as an explicit, manually
asserted flag and refuse to call anything verified without it.

What this scaffold is and isn't:
  - IS: the venue-agnostic quote shape, the lock math, the opportunity model, adapters from
    Polymarket books and Kalshi cents, all synthetic-tested.
  - IS NOT (yet): the cross-venue *matching* layer (which Polymarket token == which Kalshi
    ticker). That is its own entity-resolution problem; v1 uses an explicit hand-curated
    registry of ParityLinks and treats auto-matching as a later phase. No live Kalshi feed is
    wired in here — a Kalshi adapter populating VenueQuote is the integration seam.

stdlib only; no dependency on kalshi_scout (a thin adapter bridges the two).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

from .models import Market, OrderBook


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class VenueQuote:
    """A normalized two-sided quote for one binary outcome on one venue. Prices in dollars
    (0..1); sizes in shares/contracts. Any side may be None when the book is one-sided."""
    venue: str                 # "polymarket" | "kalshi" | ...
    market_key: str            # venue-native id: Polymarket token_id, Kalshi ticker
    label: str                 # human outcome label
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    no_bid: Optional[float] = None
    no_ask: Optional[float] = None
    yes_ask_size: Optional[float] = None
    no_ask_size: Optional[float] = None


@dataclass
class ParityLink:
    """A hand-asserted claim that two venue quotes resolve the SAME real-world YES outcome.

    `settlement_verified` is the human assertion that both venues settle identically (source,
    date, bucket). Until cross-venue matching is automated, this registry entry IS the matcher.
    """
    name: str
    a: VenueQuote
    b: VenueQuote
    settlement_verified: bool = False
    note: str = ""


@dataclass
class ParityOpportunity:
    """A detected cross-venue lock (buy YES one venue, NO the other). DETECTION ONLY.

    `settlement_verified` mirrors the NegRisk exhaustiveness flag: when False the price lock is
    real-if-the-events-match but that precondition is unproven, so it must not be trusted blind.
    """
    ts: str
    name: str
    side: str                  # "A_yes+B_no" | "B_yes+A_no"
    yes_venue: str
    yes_key: str
    yes_ask: float
    no_venue: str
    no_key: str
    no_ask: float
    cost_sum: float
    edge_cents: float
    capturable_sets: Optional[float]   # min leg size when both known, else None
    net_profit_usd: Optional[float]    # per-set edge net of fees, × capturable when known
    settlement_verified: bool
    note: str = ""
    kind: str = "parity"

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


# --------------------------------------------------------------------------- #
# Adapters — populate the venue-agnostic quote from each venue's native shape
# --------------------------------------------------------------------------- #
def pm_venue_quote(market: Market, books: dict[str, OrderBook], *, label: str = "") -> VenueQuote | None:
    """Build a VenueQuote from a Polymarket binary Market + its YES/NO order books.

    Resolves the YES token by label via Market.yes_token() (not a hard index-0 assumption), so
    markets whose outcomes aren't ordered ['Yes','No'] don't get their NO book read as YES.
    """
    if not market.is_binary or len(market.token_ids) != 2:
        return None
    yes_tok = market.yes_token()
    no_tok = next((t for t in market.token_ids if t != yes_tok), None)
    if yes_tok is None or no_tok is None:
        return None
    yes_book = books.get(yes_tok)
    no_book = books.get(no_tok)
    if yes_book is None or no_book is None:
        return None
    ya, yb = yes_book.best_ask(), yes_book.best_bid()
    na, nb = no_book.best_ask(), no_book.best_bid()
    return VenueQuote(
        venue="polymarket",
        market_key=yes_tok,
        label=label or market.question,
        yes_bid=yb.price if yb else None,
        yes_ask=ya.price if ya else None,
        no_bid=nb.price if nb else None,
        no_ask=na.price if na else None,
        yes_ask_size=ya.size if ya else None,
        no_ask_size=na.size if na else None,
    )


def kalshi_venue_quote(
    ticker: str, *, label: str,
    yes_bid_c: Optional[int], yes_ask_c: Optional[int],
    no_bid_c: Optional[int], no_ask_c: Optional[int],
    yes_ask_size: Optional[float] = None, no_ask_size: Optional[float] = None,
) -> VenueQuote:
    """Build a VenueQuote from Kalshi cents fields (KalshiMarket.yes_bid/ask, no_bid/ask).

    Kept dependency-free: a bridge in/around kalshi_scout calls this with the raw cents so
    pmscan never imports the Kalshi package. Cents → dollars (÷100)."""
    c = lambda v: (v / 100.0) if v is not None else None
    return VenueQuote(
        venue="kalshi", market_key=ticker, label=label,
        yes_bid=c(yes_bid_c), yes_ask=c(yes_ask_c), no_bid=c(no_bid_c), no_ask=c(no_ask_c),
        yes_ask_size=yes_ask_size, no_ask_size=no_ask_size,
    )


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def scan_parity(
    link: ParityLink,
    *,
    fee_per_leg: float = 0.0,
    gas_usd: float = 0.0,
) -> ParityOpportunity | None:
    """Detect the better cross-venue lock for one linked outcome, or None if neither crosses.

    Considers both constructions (YES@A+NO@B and YES@B+NO@A), takes the cheaper basket, and
    emits when its cost < $1. The opportunity carries `settlement_verified` from the link —
    caller decides whether to trust unverified ones (mirrors scan-and-flag policy).
    """
    a, b = link.a, link.b
    candidates = []  # (cost, side, yes_venue, yes_key, yes_ask, no_venue, no_key, no_ask, sets)
    if a.yes_ask is not None and b.no_ask is not None:
        sets = _min_opt(a.yes_ask_size, b.no_ask_size)
        candidates.append((a.yes_ask + b.no_ask, "A_yes+B_no",
                           a.venue, a.market_key, a.yes_ask, b.venue, b.market_key, b.no_ask, sets))
    if b.yes_ask is not None and a.no_ask is not None:
        sets = _min_opt(b.yes_ask_size, a.no_ask_size)
        candidates.append((b.yes_ask + a.no_ask, "B_yes+A_no",
                           b.venue, b.market_key, b.yes_ask, a.venue, a.market_key, a.no_ask, sets))
    if not candidates:
        return None
    cost, side, yv, yk, ya, nv, nk, na, sets = min(candidates, key=lambda c: c[0])
    edge = 1.0 - cost
    if edge <= 0:
        return None  # no crossing — the cheaper basket still costs ≥ $1

    net = None
    if sets is not None:
        gross = edge * sets
        fees = 2.0 * fee_per_leg * sets   # one leg per venue
        net = round(gross - fees - gas_usd, 6)

    return ParityOpportunity(
        ts=_now(),
        name=link.name,
        side=side,
        yes_venue=yv, yes_key=yk, yes_ask=round(ya, 6),
        no_venue=nv, no_key=nk, no_ask=round(na, 6),
        cost_sum=round(cost, 6),
        edge_cents=round(edge * 100.0, 4),
        capturable_sets=sets,
        net_profit_usd=net,
        settlement_verified=link.settlement_verified,
        note=link.note,
    )


def scan_parity_links(links: list[ParityLink], **kw) -> list[ParityOpportunity]:
    """Map scan_parity over a registry of links; returns crossing opportunities, best edge first."""
    opps = [o for lk in links if (o := scan_parity(lk, **kw)) is not None]
    opps.sort(key=lambda o: o.edge_cents, reverse=True)
    return opps


def _min_opt(x: Optional[float], y: Optional[float]) -> Optional[float]:
    if x is None or y is None:
        return None
    return min(x, y)
