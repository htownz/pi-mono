"""Cross-bracket arbitrage detection across mutually-exclusive Kalshi events.

In a Kalshi event with N mutually-exclusive brackets, exactly one resolves
Yes and the rest resolve No. Two no-arbitrage bounds follow:

    Σ yes_asks ≥ 100c   - else buy Yes on every bracket: guaranteed 100c payout
                          (only one wins) for cost Σ yes_asks. Profit if Σ < 100.

    Σ yes_bids ≤ 100c   - else sell Yes on every bracket (= buy No on every):
                          collect Σ yes_bids upfront. Exactly one Yes wins,
                          you pay 100 to settle that single position. Profit
                          = Σ yes_bids - 100 - N × fee_per_leg.

**CRITICAL**: this math only applies to events whose brackets actually partition
the outcome space (weather temp brackets, election vote-share bands, etc.).
Kalshi groups many *non*-mutually-exclusive markets under one event_ticker too
— artist-streams pairwise comparisons, overlapping price thresholds — for
which Σ yes_bids legitimately exceeds 100 without arbitrage.

We gate the math two ways:

  1. `MUTUALLY_EXCLUSIVE_SERIES`: a hand-curated whitelist of series_ticker
     prefixes whose bracket structure has been confirmed MEX (today: weather
     temperature, `KXHIGH*` / `KXLOW*` / `HIGH*`).

  2. `detect_numeric_partition`: algorithmic fallback that parses numeric
     ranges from each market's yes_sub_title and confirms they tile a
     contiguous numeric axis with no overlaps. Promotes any event whose
     brackets look like `60-61°`, `61-62°`, ... regardless of series_ticker.

The fallback is conservative by design: an event with any market whose
yes_sub_title doesn't parse as a numeric interval is rejected (catches
artist-streams pairwise comparisons cleanly). Pass `--strict-mex` on the
arbitrage CLI to disable the fallback and use whitelist-only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from kalshi_scout.kalshi import TEMPERATURE_SERIES
from kalshi_scout.models import KalshiEvent


# Series_ticker prefixes whose events have been verified mutually exclusive.
# Add new entries only after confirming the brackets form a disjoint partition
# of the outcome space (one and only one bracket can settle Yes).
MUTUALLY_EXCLUSIVE_SERIES: frozenset[str] = frozenset(TEMPERATURE_SERIES.keys())


# -- Algorithmic MEX detection -----------------------------------------------

# Matches "60-61", "60 to 61", "60° to 61°", "$3.00–$3.50", etc.
# The interval marker can be a hyphen, "to", en-dash, or em-dash.
_RANGE_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(?:°|°[FCK])?\s*"
    r"(?:to|-|–|—)\s*"
    r"(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
# "above N" / "over N" / ">N"
_ABOVE_PREFIX_RE = re.compile(
    r"\b(?:above|over|greater than|at least|>=?)\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
# "N or above" / "N or higher" / "N+"
_ABOVE_SUFFIX_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*°?\s*(?:or above|or higher|or more|or greater|\+)",
    re.IGNORECASE,
)
_BELOW_PREFIX_RE = re.compile(
    r"\b(?:below|under|less than|at most|<=?)\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_BELOW_SUFFIX_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*°?\s*(?:or below|or lower|or fewer|or less)",
    re.IGNORECASE,
)
_EXACT_RE = re.compile(
    r"\b(?:exactly|equal to|=)\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Gap tolerance between adjacent intervals (in interval units). 1.5 is
# generous enough for integer-bucket weather (`60-61` then `62-63` has a
# 1-unit gap that's actually the strict-inequality boundary) and tight
# enough that an event missing a whole bucket would be rejected.
_GAP_TOLERANCE = 1.5


def _parse_interval(text: Optional[str]) -> Optional[tuple[float, float]]:
    """Return (lo, hi) extracted from a yes_sub_title-like string.

    Endpoints may be ±inf for "above N" / "below N" tail brackets. Returns
    None when no recognizable interval pattern is found — caller treats
    that as "this market can't participate in the partition check".
    """
    if not text:
        return None
    cleaned = text.strip().replace("$", "").replace(",", "")
    # Strip everything before the first digit; phrases like "between 60 and 61"
    # confuse the BELOW pattern otherwise.
    m = _RANGE_RE.search(cleaned)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return (min(a, b), max(a, b))
    for above_re in (_ABOVE_SUFFIX_RE, _ABOVE_PREFIX_RE):
        m = above_re.search(cleaned)
        if m:
            return (float(m.group(1)), float("inf"))
    for below_re in (_BELOW_SUFFIX_RE, _BELOW_PREFIX_RE):
        m = below_re.search(cleaned)
        if m:
            return (float("-inf"), float(m.group(1)))
    m = _EXACT_RE.search(cleaned)
    if m:
        v = float(m.group(1))
        return (v, v)
    return None


def _is_partition(intervals: list[tuple[float, float]]) -> tuple[bool, str]:
    """True iff intervals (any order) tile a contiguous numeric axis with no
    overlaps and no significant gaps. Returns (verdict, reason) for diagnostics.
    """
    if len(intervals) < 2:
        return False, "fewer than 2 intervals"
    finite = [iv for iv in intervals if iv[0] != float("-inf") and iv[1] != float("inf")]
    if not finite:
        return False, "no finite intervals to anchor the partition"
    sorted_iv = sorted(intervals)
    for prev, curr in zip(sorted_iv, sorted_iv[1:]):
        if curr[0] < prev[1] - 0.01:
            return False, f"overlap: {prev} and {curr}"
        if (curr[0] - prev[1]) > _GAP_TOLERANCE:
            return False, f"gap of {curr[0] - prev[1]:g} between {prev} and {curr}"
    return True, f"partitions axis across {len(intervals)} intervals"


@dataclass(frozen=True)
class MexDetection:
    """Audit-friendly result from algorithmic MEX detection."""
    is_mex: bool
    reason: str
    n_markets: int
    n_parsed: int          # markets whose yes_sub_title yielded an interval


def detect_numeric_partition(event: KalshiEvent) -> MexDetection:
    """Heuristic MEX classifier: does this event's bracket structure tile a
    numeric axis cleanly?

    Returns True when every market in the event has a parseable numeric
    interval (from yes_sub_title, falling back to title) AND those intervals
    form a contiguous non-overlapping partition. Either condition failing
    drops the event back into "treat as non-MEX" territory.
    """
    n = len(event.markets)
    if n < 2:
        return MexDetection(False, "fewer than 2 markets", n, 0)
    intervals: list[tuple[float, float]] = []
    n_parsed = 0
    for m in event.markets:
        iv = _parse_interval(m.yes_sub_title) or _parse_interval(m.title)
        if iv is None:
            return MexDetection(
                False, f"market {m.ticker} has no parseable numeric range "
                f"(yes_sub_title={m.yes_sub_title!r})",
                n, n_parsed,
            )
        n_parsed += 1
        intervals.append(iv)
    ok, reason = _is_partition(intervals)
    return MexDetection(ok, reason, n, n_parsed)


def is_mutually_exclusive_event(event: KalshiEvent, *, strict: bool = False) -> bool:
    """Best-effort MEX check.

    Pass-conditions (any one suffices):
      - Event's series_ticker (or event_ticker prefix) is in the curated
        `MUTUALLY_EXCLUSIVE_SERIES` whitelist.
      - `detect_numeric_partition(event)` returns True — the brackets tile
        a numeric axis cleanly. Skipped when `strict=True`.

    The detector adds expressive power without introducing the artist-streams
    false positives: non-MEX events overwhelmingly use non-numeric labels
    ("Team X wins") that fail interval parsing.
    """
    series = event.series_ticker or (
        event.event_ticker.split("-", 1)[0] if event.event_ticker else ""
    )
    if series.upper() in MUTUALLY_EXCLUSIVE_SERIES:
        return True
    if strict:
        return False
    return detect_numeric_partition(event).is_mex


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
    require_mex: bool = True,
    strict_mex: bool = False,
) -> list[EventArbitrage]:
    """Compute arb per event, drop those below threshold, sort by net edge.

    `require_mex=True` (default): only events that pass MEX gating are
    scored. By default the gate accepts both the curated whitelist AND
    events whose brackets pass `detect_numeric_partition`. Pass
    `strict_mex=True` to use the whitelist only (and reject everything
    else regardless of bracket structure) — the V1.1 behavior.

    `require_mex=False`: bypass the gate entirely for diagnostic dumps
    where you want to see raw Σ-deviation across all series. Most hits
    will be false positives from non-MEX events; do not trade them.
    """
    out: list[EventArbitrage] = []
    for event in events:
        if require_mex and not is_mutually_exclusive_event(event, strict=strict_mex):
            continue
        arb = compute_event_arbitrage(event, fee_per_leg_cents=fee_per_leg_cents)
        if arb is None or arb.best_net_edge_cents is None:
            continue
        if arb.best_net_edge_cents < min_net_edge_cents:
            continue
        out.append(arb)
    out.sort(key=lambda a: -(a.best_net_edge_cents or 0))
    return out
