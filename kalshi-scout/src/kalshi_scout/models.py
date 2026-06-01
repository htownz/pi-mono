"""Core data model shared across crawler, parser, state engine, and ranker.

The scout treats every Kalshi temperature contract as the answer to a question
about an *interval* over the daily extreme. Three shapes:

    above(t)        -> "settlement >= t"          ticker suffix: T<t>  (yes_sub_title: "<t>° or above")
    below(t)        -> "settlement <  t"          ticker suffix: T<t>  (yes_sub_title: "<t>° or below")  # t already excluded
    between(lo, hi) -> "lo <= settlement <= hi"   ticker suffix: B<lo>-<hi>

Brackets are integer-degree on the Kalshi side. We keep them as floats so the
state engine can compare against real station readings that can carry .1 / .5
precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Optional


class Metric(str, Enum):
    HIGH = "high"
    LOW = "low"

    @property
    def is_high(self) -> bool:
        return self is Metric.HIGH

    @property
    def is_low(self) -> bool:
        return self is Metric.LOW


class BracketKind(str, Enum):
    """Matches Kalshi's GLOBALTEMPERATURE rulebook operators exactly.

    See `AGENTS.md` invariant I6. The colloquial yes_sub_title shapes seen on
    real Kalshi markets map as follows:

        "X° or above"  -> GTE  (rulebook: "at least X")
        "X° or below"  -> LTE  (rulebook: "at most X")
        "above X°"     -> GT   (rulebook: "above X", strict)
        "below X°"     -> LT   (rulebook: "below X", strict)
        "lo°–hi°"      -> BETWEEN
        "exactly X°"   -> EQ   (rulebook: equal to X rounded to 1dp)
    """
    GT = "gt"            # > threshold (rulebook "above")
    LT = "lt"            # < threshold (rulebook "below")
    GTE = "gte"          # >= threshold (rulebook "at least", colloquial "or above")
    LTE = "lte"          # <= threshold (rulebook "at most", colloquial "or below")
    EQ = "eq"            # == threshold rounded to 1dp (rulebook "exactly")
    BETWEEN = "between"  # lo <= t <= hi inclusive both ends


@dataclass(frozen=True)
class Bracket:
    """A single Kalshi temperature contract bracket.

    For one-sided kinds (GT/LT/GTE/LTE/EQ) the threshold is stored in `lo` for
    GT/GTE/EQ and in `hi` for LT/LTE — matching the conventional "lower bound
    for above-side / upper bound for below-side" mental model.
    """
    kind: BracketKind
    lo: Optional[float]  # inclusive lower bound (or strict for GT)
    hi: Optional[float]  # inclusive upper bound (or strict for LT)

    def contains(self, t: float) -> bool:
        if self.kind is BracketKind.GT:
            assert self.lo is not None
            return t > self.lo
        if self.kind is BracketKind.GTE:
            assert self.lo is not None
            return t >= self.lo
        if self.kind is BracketKind.LT:
            assert self.hi is not None
            return t < self.hi
        if self.kind is BracketKind.LTE:
            assert self.hi is not None
            return t <= self.hi
        if self.kind is BracketKind.EQ:
            assert self.lo is not None
            return round(t, 1) == round(self.lo, 1)
        assert self.lo is not None and self.hi is not None
        return self.lo <= t <= self.hi

    def label(self) -> str:
        if self.kind is BracketKind.GTE:
            return f"{self.lo:g}° or above"
        if self.kind is BracketKind.LTE:
            return f"{self.hi:g}° or below"
        if self.kind is BracketKind.GT:
            return f"above {self.lo:g}°"
        if self.kind is BracketKind.LT:
            return f"below {self.hi:g}°"
        if self.kind is BracketKind.EQ:
            return f"exactly {self.lo:g}°"
        return f"{self.lo:g}–{self.hi:g}°"


@dataclass(frozen=True)
class Station:
    """An NWS observation station that serves as the official settlement source.

    `neighbors` is a tuple of nearby ICAO codes used as a cross-check + fallback
    for the primary station. The neighbors do NOT settle the contract — only the
    primary's CLI does — but their observations provide a redundancy signal
    when the primary's ASOS goes offline and a microclimate-divergence signal
    when the primary diverges from the surrounding network (sea breeze pinching
    KHOU vs KIAH, marine layer pinching KLAX vs KBUR, lake breeze pinching
    KORD vs KDPA, etc).
    """
    icao: str               # e.g. "KHOU"
    name: str               # e.g. "Houston Hobby Airport"
    city_slug: str          # matches Kalshi market grouping, e.g. "HOUSTON"
    tz: str                 # IANA timezone, e.g. "America/Chicago"
    cli_product: str        # NWS CLI product ID, e.g. "CLIHOU"
    latitude: float
    longitude: float
    neighbors: tuple[str, ...] = ()   # supporting ICAO codes; cross-check only


class SettlementProvenance(str, Enum):
    """Where the scout's settlement decision came from.

    `RESOLVER` = parsed from the market's rules_primary text (authoritative).
    `REGISTRY` = fell back to the hand-curated stations.py map (lower trust).
    `UNVERIFIED` = neither produced a station — market must grade F.
    """
    RESOLVER = "resolver"
    REGISTRY = "registry"
    UNVERIFIED = "unverified"


@dataclass(frozen=True)
class Settlement:
    """Per-market settlement metadata extracted from Kalshi's rules text.

    This is the V0.4 resolver's output. It tells the engine exactly which
    NWS station and CLI product define the contract's Expiration Value.
    """
    station: Optional[Station]
    source_agency: str          # e.g. "National Weather Service"
    area_description: str       # raw <area> substitution from rules text
    provenance: SettlementProvenance
    notes: tuple[str, ...] = ()


@dataclass
class KalshiMarket:
    """One Kalshi contract (a single ticker). Multiple markets per event."""
    ticker: str
    event_ticker: str
    title: str
    yes_sub_title: str
    status: str
    close_time: Optional[datetime]
    yes_bid: Optional[int]    # cents
    yes_ask: Optional[int]    # cents
    no_bid: Optional[int]
    no_ask: Optional[int]
    last_price: Optional[int]
    volume: int
    open_interest: int
    raw: dict = field(default_factory=dict)


@dataclass
class KalshiEvent:
    """One Kalshi event grouping markets. For weather: one city/date/metric."""
    event_ticker: str
    series_ticker: str
    title: str
    sub_title: str
    markets: list[KalshiMarket] = field(default_factory=list)
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedContract:
    """The result of parsing a Kalshi market into its settlement question."""
    market_ticker: str
    event_ticker: str
    city_slug: str
    metric: Metric
    market_date: date          # the local-time settlement date
    bracket: Bracket


@dataclass
class StationReading:
    """A single point-in-time observation from the official station."""
    observed_at: datetime
    temperature_f: float
    dewpoint_f: Optional[float] = None
    wind_speed_mph: Optional[float] = None
    sky: Optional[str] = None


@dataclass
class StationState:
    """The current settlement-relevant state for a station inside a market day.

    `running_max_f` / `running_min_f` are the highest/lowest values observed so
    far within the market day's local window. They are the only values that
    affect a high/low contract's settlement once they cross a strike.

    `neighbor_running_max_f` / `neighbor_running_min_f` aggregate the same
    window's extrema across the primary's `Station.neighbors`. They are NOT
    settlement-conclusive — the contract still settles on the primary's CLI —
    but they expose two operational signals:

      1. Fallback: if the primary's ASOS is offline (`running_max_f is None`
         but neighbors have data), the engine can fall back to the neighbor
         median as a best-effort estimate, with a note attached.
      2. Microclimate divergence: when the primary diverges from the neighbor
         median by a large margin (sea breeze, marine layer, lake breeze),
         the engine widens its uncertainty band to reflect lower forecast
         skill in that regime.

    `neighbor_sample_count` is the total number of neighbor readings inside
    the window (across all neighbor stations), useful for gauging signal
    confidence in the divergence check.
    """
    station: Station
    market_date: date
    window_start: datetime    # localized
    window_end: datetime      # exclusive
    running_max_f: Optional[float]
    running_min_f: Optional[float]
    latest: Optional[StationReading]
    cli_report_date: Optional[date]   # None if no CLI matches the market date
    cli_max_f: Optional[float]
    cli_min_f: Optional[float]
    observations: list[StationReading] = field(default_factory=list)
    neighbor_running_max_f: Optional[float] = None
    neighbor_running_min_f: Optional[float] = None
    neighbor_sample_count: int = 0
    neighbor_icaos: tuple[str, ...] = ()   # neighbors actually queried

    #: Minimum dewpoint observed inside the market-day window so far, in °F.
    #: Physical floor on the daily LOW: the air rarely cools more than ~1°F
    #: below this without dry advection. Used by fair_probability LOW path
    #: to clip absurd cold-cooling forecasts; never used to lock a contract.
    #: None when no observations carry dewpoint data.
    dewpoint_floor_f: Optional[float] = None

    @property
    def cli_matches_market_date(self) -> bool:
        return self.cli_report_date == self.market_date

    @property
    def effective_running_max_f(self) -> Optional[float]:
        """Primary's running max, or the neighbor's if the primary is missing.

        Used by fair_probability so a transient ASOS outage on the primary
        doesn't collapse fair_prob to the no-data fallback. The neighbor value
        is NEVER used to lock a contract (LOCKED_YES / DEAD_NO) — only the
        primary's reading does that.
        """
        if self.running_max_f is not None:
            return self.running_max_f
        return self.neighbor_running_max_f

    @property
    def effective_running_min_f(self) -> Optional[float]:
        if self.running_min_f is not None:
            return self.running_min_f
        return self.neighbor_running_min_f


class ContractState(str, Enum):
    """Where a contract sits relative to the official station reading.

    A+/A grades fall out of states that already have settlement-conclusive
    evidence; B/C/D grades require forecast judgement.
    """
    LOCKED_YES = "locked_yes"            # settlement already proves Yes
    DEAD_NO = "dead_no"                  # settlement already proves No
    BRACKET_HIT_VULNERABLE = "bracket_hit_vulnerable"  # in the bracket now, remaining bust risk
    NOT_REACHED = "not_reached"          # path still has to get here
    FORECAST_DEPENDENT = "forecast_dependent"  # no decisive station data yet


@dataclass
class ContractEvaluation:
    contract: ParsedContract
    market: KalshiMarket
    state: ContractState
    reason: str
    fair_prob_low: float        # 0..1
    fair_prob_high: float       # 0..1
    yes_ask_cents: Optional[int]
    no_ask_cents: Optional[int]
    edge_yes: Optional[float]   # fair_mid - yes_ask/100, or None
    edge_no: Optional[float]
    grade: str                  # "A+", "A", "B+", "B", "C", "D", "F"
    notes: list[str] = field(default_factory=list)
