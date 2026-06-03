"""Normalized, venue-agnostic data models.

This is the Phase 0 "common internal representation." Polymarket is implemented now;
the same Market/OrderBook shapes are what a future Kalshi adapter would populate, so the
scanner logic downstream never has to know which venue a market came from.

Phase 1b extends the model with NegRisk (multi-outcome, mutually-exclusive) event
grouping and a dedicated NegRiskOpportunity. The binary Opportunity is left untouched.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class BookLevel:
    price: float          # USDC per share, 0..1
    size: float           # shares available at this price


@dataclass
class OrderBook:
    token_id: str
    bids: list[BookLevel] = field(default_factory=list)   # may be in any order
    asks: list[BookLevel] = field(default_factory=list)
    tick_size: float = 0.01
    neg_risk: bool = False

    def best_ask(self) -> Optional[BookLevel]:
        """Lowest-priced ask (cheapest place to BUY). Computed, not index-assumed."""
        return min(self.asks, key=lambda l: l.price) if self.asks else None

    def best_bid(self) -> Optional[BookLevel]:
        """Highest-priced bid (best place to SELL). Computed, not index-assumed."""
        return max(self.bids, key=lambda l: l.price) if self.bids else None

    def mid(self) -> Optional[float]:
        """Mid price if both sides exist, else the single available side. None if empty."""
        a, b = self.best_ask(), self.best_bid()
        if a is not None and b is not None:
            return (a.price + b.price) / 2.0
        if a is not None:
            return a.price
        if b is not None:
            return b.price
        return None


@dataclass
class Market:
    """A normalized binary market. outcomes/token_ids are index-aligned: [0]=YES, [1]=NO.

    NegRisk outcome markets are themselves binary (Yes/No), so they populate this same
    shape. The NegRisk linkage fields below let Phase 1b regroup them into events.
    """
    venue: str
    market_id: str                  # Polymarket conditionId
    question: str
    slug: str
    outcomes: list[str]
    token_ids: list[str]
    fees_enabled: bool = False
    neg_risk: bool = False
    enable_order_book: bool = True
    accepting_orders: bool = True
    volume_24hr: float = 0.0
    liquidity: float = 0.0
    tick_size: float = 0.01
    end_date: Optional[str] = None
    # --- NegRisk linkage (Phase 1b) ---
    neg_risk_request_id: Optional[str] = None   # shared id grouping outcomes of one event
    neg_risk_other: bool = False                # event exposes an implicit "Other/None" bucket
    group_title: Optional[str] = None           # parent event title, if Gamma provided one

    @property
    def is_binary(self) -> bool:
        return len(self.token_ids) == 2 and len(self.outcomes) == 2

    def yes_token(self) -> Optional[str]:
        """Token id for the YES outcome. Matches the 'Yes' label if present, else index 0."""
        if not self.token_ids:
            return None
        for i, o in enumerate(self.outcomes):
            if str(o).strip().lower() in ("yes", "true"):
                return self.token_ids[i]
        return self.token_ids[0]


@dataclass
class Opportunity:
    """A detected within-market sum-to-one mispricing. DETECTION ONLY — never executed."""
    ts: str
    venue: str
    market_id: str
    question: str
    slug: str
    side: str                       # "merge" (buy YES+NO < $1) or "split" (sell YES+NO > $1)
    yes_price: float                # ask (merge) or bid (split) for YES
    no_price: float                 # ask (merge) or bid (split) for NO
    price_sum: float
    edge_cents: float               # gross per-set edge in cents
    capturable_sets: float          # top-of-book size-limited set count
    gross_profit_usd: float
    net_profit_usd: float           # after modeled fees + gas
    fees_enabled: bool
    neg_risk: bool
    volume_24hr: float
    kind: str = "binary"

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


@dataclass
class NegRiskEvent:
    """A mutually-exclusive outcome set linked by a shared negRiskRequestID.

    Exactly one outcome resolves YES (pays $1); the rest resolve NO ($0). Each outcome is
    its own binary Market. `has_other` flags an implicit "Other/None" bucket — when set,
    we cannot prove the listed outcomes are exhaustive, so the naive buy-all-YES identity
    is not guaranteed (handled by flagging the opportunity uncertain rather than dropping it).
    """
    request_id: str
    outcomes: list[Market] = field(default_factory=list)
    title: Optional[str] = None
    has_other: bool = False

    @property
    def n(self) -> int:
        return len(self.outcomes)


@dataclass
class NegRiskSnapshot:
    """One timestamped basket reading for a NegRisk event — crossing or not.

    Unlike NegRiskOpportunity (emitted only when ask_sum < $1), a snapshot is recorded every
    cycle regardless of crossing, so the temporal detector has the event's *baseline* ask_sum
    to measure dips against. A structural/phantom event sits flat here; a real dislocation is
    a transient drop below its own baseline.
    """
    ts: str
    request_id: str
    title: str
    legs: int
    ask_sum: float
    bid_sum: float
    implied_mass: float
    has_other: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


@dataclass
class NegRiskOpportunity:
    """A detected cross-outcome NegRisk mispricing (buy-all-YES). DETECTION ONLY.

    edge = $1 - Σ best_ask(YES_i). Buying one YES per outcome costs Σ asks; exactly one
    resolves $1, so a sub-$1 basket is a structural edge — *iff* the set is exhaustive.
    `exhaustive_verified` records whether that precondition held; when False the edge is
    real-if-complete but the completeness couldn't be proven (see uncertainty_reason).
    """
    ts: str
    venue: str
    request_id: str
    title: str
    side: str                       # "buy_all_yes"
    legs: int                       # N outcomes in the basket
    outcomes: list[str]             # outcome labels / questions, index-aligned to legs
    ask_sum: float                  # Σ best_ask(YES_i) — cost to buy one YES per outcome
    bid_sum: float                  # Σ best_bid(YES_i) over legs that have a bid
    spread: float                   # ask_sum - bid_sum (NaN unless every leg has a bid)
    implied_mass: float             # Σ mid(YES_i) — ≈ 1.0 for an exhaustive, fairly-priced set
    implied_other: float            # max(0, 1 - implied_mass): est. prob on unlisted outcomes
    edge_cents: float               # gross per-basket edge in cents (1 - ask_sum) * 100
    capturable_sets: float          # min_i ask_size_i (top-of-book, conservative)
    gross_profit_usd: float
    net_profit_usd: float           # after modeled per-leg fees + gas
    exhaustive_verified: bool       # True only if no Other bucket AND implied_mass in band
    uncertainty_reason: str         # "" when verified; else why completeness is unproven
    fees_enabled: bool
    volume_24hr: float              # summed across legs
    kind: str = "negrisk"

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))
