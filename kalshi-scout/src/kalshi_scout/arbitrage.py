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
# Only INCLUSIVE tail forms are accepted. Strict-open operators (`above N`,
# `below N`, `>N`, `<N`) are ambiguous wrt boundary coverage: `below 50` +
# `above 50` leaves an exact-50 settlement with no winning bracket, so
# promoting events with strict labels would surface false-positive arbs.
# Word-boundary `\b` sits inside the letter alternative so symbolic
# operators (`>=N`, `<=N`) also match at string start.
_ABOVE_PREFIX_RE = re.compile(
    r"(?:\b(?:at least)|>=)\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_ABOVE_SUFFIX_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*°?\s*(?:or above|or higher|or more|or greater|\+)",
    re.IGNORECASE,
)
_BELOW_PREFIX_RE = re.compile(
    r"(?:\b(?:at most)|<=)\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_BELOW_SUFFIX_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*°?\s*(?:or below|or lower|or fewer|or less)",
    re.IGNORECASE,
)
_EXACT_RE = re.compile(
    r"(?:\b(?:exactly|equal to)|=)\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Adjacent finite intervals must touch (gap <= EPSILON). A 1.5-unit blanket
# tolerance silently accepts real missing buckets in decimal-denominated
# series (e.g. $3.00-$3.25 then $4.70-$5.00 has a 1.45 gap whose prices have
# no winning bracket). Tail brackets are exempted by `_is_partition` since
# they extend coverage to ±inf.
_GAP_EPSILON = 0.01


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
    """True iff intervals tile the entire real line: a low tail bracket and
    a high tail bracket on the ends AND finite intervals between them tile
    adjacently — including across the tail/finite boundaries — with no
    overlaps or gaps. Returns (verdict, reason) for diagnostics.

    Requiring tail brackets (low-tail = `-inf`-bounded, high-tail = `+inf`-
    bounded) is the gate's exhaustive-coverage proof: a finite-only set like
    `$3.00 to $3.25`, `$3.25 to $3.50`, `$3.50 to $3.75` has no winning
    bracket for prices outside that span, so the no-arbitrage math is invalid.

    Tail adjacency matters too: `2 or below` + `3 to 4` + `4 or above` has
    a real (2, 3) gap that no bracket covers; without the tail-finite
    boundary check, the gap-against-finite-only sweep wouldn't catch it.
    """
    if len(intervals) < 2:
        return False, "fewer than 2 intervals"
    low_tails = [iv for iv in intervals if iv[0] == float("-inf")]
    high_tails = [iv for iv in intervals if iv[1] == float("inf")]
    if not (low_tails and high_tails):
        missing = []
        if not low_tails:
            missing.append("below-tail (e.g. 'N or below')")
        if not high_tails:
            missing.append("above-tail (e.g. 'N or above')")
        return False, (
            f"finite-only partition — missing {', '.join(missing)}; "
            "outcomes outside the listed span have no winning bracket"
        )
    # The low-tail's right edge must be the lowest right edge among the
    # low tails (a market labeled `5 or below` covers (-inf, 5]).
    low_tail_hi = max(iv[1] for iv in low_tails)
    high_tail_lo = min(iv[0] for iv in high_tails)
    finite = sorted(
        iv for iv in intervals
        if iv[0] != float("-inf") and iv[1] != float("inf")
    )
    if not finite:
        # Pure low-tail + high-tail with no finite bucket between them is
        # always ambiguous at the boundary: with INCLUSIVE labels (`50 or
        # below` + `50 or above`) the value 50 is in BOTH brackets; with
        # STRICT labels (`below 50` + `above 50`) it's in NEITHER. Either
        # way it's not a true partition.
        return False, (
            "pure-tails partition with no finite bucket between them; "
            "boundary value is either double-covered or uncovered"
        )
    # Finite buckets present: low tail must touch the lowest finite lo,
    # finite intervals must tile adjacently, and the highest finite hi
    # must touch the high tail lo. All within `_GAP_EPSILON`.
    first_lo, last_hi = finite[0][0], finite[-1][1]
    if first_lo < low_tail_hi - _GAP_EPSILON:
        return False, (
            f"low-tail (-inf, {low_tail_hi:g}) overlaps first finite "
            f"bucket {finite[0]}"
        )
    if (first_lo - low_tail_hi) > _GAP_EPSILON:
        return False, (
            f"gap of {first_lo - low_tail_hi:g} between low-tail "
            f"(-inf, {low_tail_hi:g}) and first finite bucket {finite[0]}"
        )
    for prev, curr in zip(finite, finite[1:]):
        if curr[0] < prev[1] - _GAP_EPSILON:
            return False, f"overlap: {prev} and {curr}"
        if (curr[0] - prev[1]) > _GAP_EPSILON:
            return False, f"gap of {curr[0] - prev[1]:g} between {prev} and {curr}"
    if high_tail_lo < last_hi - _GAP_EPSILON:
        return False, (
            f"high-tail ({high_tail_lo:g}, +inf) overlaps last finite "
            f"bucket {finite[-1]}"
        )
    if (high_tail_lo - last_hi) > _GAP_EPSILON:
        return False, (
            f"gap of {high_tail_lo - last_hi:g} between last finite "
            f"bucket {finite[-1]} and high-tail ({high_tail_lo:g}, +inf)"
        )
    return True, (
        f"tiles axis with {len(intervals)} intervals "
        f"({len(finite)} finite + low/high tails, all adjacent)"
    )


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
    interval (read **only** from `yes_sub_title` — the field that carries
    the bracket label) AND those intervals form a contiguous partition with
    low+high tail brackets bounding the axis. Either condition failing
    drops the event back into "treat as non-MEX" territory.

    The detector deliberately does NOT fall back to `market.title` for
    parsing: the title is the question (e.g. "Will Houston's high be
    above 80°?") and its numeric content doesn't describe the bracket
    structure. Reading it would weaken the gate enough for non-MEX events
    with numeric questions to slip through.
    """
    n = len(event.markets)
    if n < 2:
        return MexDetection(False, "fewer than 2 markets", n, 0)
    intervals: list[tuple[float, float]] = []
    n_parsed = 0
    for m in event.markets:
        iv = _parse_interval(m.yes_sub_title)
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
