"""Pre-flight risk aggregation for open positions.

Reads `positions` (open only) from the SnapshotStore, joins to the most
recent snapshot for each position's market_ticker to pull city / metric /
market_date / regime, and rolls up exposure by several buckets.

The flagship check is **event collision**: Kalshi event contracts are
mutually exclusive (only one bracket within an event can settle Yes), so
holding Yes contracts across multiple brackets of the same event guarantees
a partial loss. The risk report calls these out explicitly.

What this is NOT:
  - Not auto-trading. We don't open/close anything. The operator records
    positions by hand via `kalshi-scout positions add` (V1.1 will pull
    from Kalshi's authenticated API).
  - Not portfolio P&L. We only track max-loss exposure (price paid per
    contract); realized P&L is computed at settlement.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from kalshi_scout.store import PositionRow, SnapshotStore


@dataclass(frozen=True)
class EnrichedPosition:
    """A position plus the city / metric / date / regime context pulled from
    its most recent snapshot. When no snapshot exists for the ticker, all
    enrichment fields are None and the position falls into the 'unknown'
    bucket of every aggregation."""
    position: PositionRow
    city_slug: Optional[str]
    metric: Optional[str]
    market_date: Optional[date]
    regime: Optional[str]


@dataclass
class RiskBucket:
    label: str
    n_positions: int
    total_contracts: int
    total_max_loss_cents: int
    market_tickers: list[str] = field(default_factory=list)

    @property
    def total_max_loss_dollars(self) -> float:
        return self.total_max_loss_cents / 100.0


@dataclass
class EventCollision:
    """One event with multiple Yes positions across different brackets.

    Holding Yes on multiple brackets of the same event = guaranteed loss
    on all but at most one. The collision report names this explicitly.
    """
    event_ticker: str
    yes_positions: list[PositionRow]
    total_max_loss_cents: int

    @property
    def guaranteed_loss_cents(self) -> int:
        """At most one bracket pays out. Worst case: the cheapest one wins,
        we forfeit cost basis on every other Yes position."""
        if not self.yes_positions:
            return 0
        sorted_by_cost = sorted(
            self.yes_positions, key=lambda p: p.cost_basis_cents
        )
        # We can win at most the most expensive one; the others are dead.
        return sum(p.cost_basis_cents for p in sorted_by_cost[:-1])


@dataclass
class RiskReport:
    enriched: list[EnrichedPosition]
    by_city: dict[str, RiskBucket]
    by_market_date: dict[str, RiskBucket]
    by_regime: dict[str, RiskBucket]
    by_event: dict[str, RiskBucket]
    event_collisions: list[EventCollision]
    total_open_positions: int
    total_open_contracts: int
    total_max_loss_cents: int

    @property
    def total_max_loss_dollars(self) -> float:
        return self.total_max_loss_cents / 100.0


def enrich_positions(
    store: SnapshotStore,
    positions: list[PositionRow],
) -> list[EnrichedPosition]:
    """Join each position to the most recent snapshot for its ticker.

    Used to derive city / metric / market_date / regime context — which the
    aggregator then buckets exposure across.
    """
    enriched: list[EnrichedPosition] = []
    for p in positions:
        snaps = store.query_snapshots(market_ticker=p.market_ticker, limit=1)
        if snaps:
            s = snaps[0]
            enriched.append(EnrichedPosition(
                position=p,
                city_slug=s.city_slug,
                metric=s.metric,
                market_date=s.market_date,
                regime=s.regime,
            ))
        else:
            enriched.append(EnrichedPosition(
                position=p,
                city_slug=None, metric=None, market_date=None, regime=None,
            ))
    return enriched


def aggregate_risk(store: SnapshotStore) -> RiskReport:
    """Read open positions from the store and roll up exposure.

    Buckets emitted: city, market_date, regime, event_ticker. Plus the
    explicit event-collision list."""
    positions = store.query_positions(open_only=True)
    enriched = enrich_positions(store, positions)

    by_city = _bucket(enriched, key=lambda e: e.city_slug or "unknown")
    by_date = _bucket(enriched, key=lambda e: e.market_date.isoformat() if e.market_date else "unknown")
    by_regime = _bucket(enriched, key=lambda e: e.regime or "unknown")
    by_event = _bucket(enriched, key=lambda e: e.position.event_ticker)

    # Event collisions: events with >1 distinct YES position.
    yes_by_event: dict[str, list[PositionRow]] = defaultdict(list)
    for e in enriched:
        if e.position.side == "yes":
            yes_by_event[e.position.event_ticker].append(e.position)
    collisions = [
        EventCollision(
            event_ticker=ev,
            yes_positions=ps,
            total_max_loss_cents=sum(p.cost_basis_cents for p in ps),
        )
        for ev, ps in yes_by_event.items()
        if len({p.market_ticker for p in ps}) > 1
    ]
    collisions.sort(key=lambda c: c.guaranteed_loss_cents, reverse=True)

    total_contracts = sum(p.position.size_contracts for p in enriched)
    total_loss = sum(p.position.max_loss_cents for p in enriched)

    return RiskReport(
        enriched=enriched,
        by_city=by_city,
        by_market_date=by_date,
        by_regime=by_regime,
        by_event=by_event,
        event_collisions=collisions,
        total_open_positions=len(enriched),
        total_open_contracts=total_contracts,
        total_max_loss_cents=total_loss,
    )


def _bucket(enriched: list[EnrichedPosition], key) -> dict[str, RiskBucket]:
    out: dict[str, RiskBucket] = {}
    for e in enriched:
        k = key(e)
        b = out.get(k)
        if b is None:
            b = RiskBucket(label=k, n_positions=0, total_contracts=0,
                           total_max_loss_cents=0, market_tickers=[])
            out[k] = b
        b.n_positions += 1
        b.total_contracts += e.position.size_contracts
        b.total_max_loss_cents += e.position.max_loss_cents
        if e.position.market_ticker not in b.market_tickers:
            b.market_tickers.append(e.position.market_ticker)
    return out
