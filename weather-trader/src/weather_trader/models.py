"""Core data model shared across the client, parser, forecaster, and grader.

Every Kalshi temperature contract is the answer to a question about an
*interval* over the day's extreme temperature. The bracket shapes match
Kalshi's GLOBALTEMPERATURE rulebook operators exactly:

    "X° or above"  -> GTE  (rulebook "at least X")
    "X° or below"  -> LTE  (rulebook "at most X")
    "above X°"     -> GT   (rulebook "above X", strict)
    "below X°"     -> LT   (rulebook "below X", strict)
    "lo°–hi°"      -> BETWEEN (inclusive both ends)
    "exactly X°"   -> EQ   (equal to X rounded to 1dp)

Brackets are integer-degree on the Kalshi side but stored as floats so the
forecaster can compare them against real readings carrying .1 precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo


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
    GT = "gt"            # > threshold   (rulebook "above")
    LT = "lt"            # < threshold   (rulebook "below")
    GTE = "gte"          # >= threshold  (colloquial "or above")
    LTE = "lte"          # <= threshold  (colloquial "or below")
    EQ = "eq"            # == threshold rounded to 1dp ("exactly")
    BETWEEN = "between"  # lo <= t <= hi inclusive both ends


@dataclass(frozen=True)
class Bracket:
    """A single Kalshi temperature contract bracket.

    For one-sided kinds the threshold is stored in `lo` for GT/GTE/EQ and in
    `hi` for LT/LTE — the "lower bound for above-side, upper bound for
    below-side" mental model.
    """
    kind: BracketKind
    lo: Optional[float]
    hi: Optional[float]

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
    """A weather station the bot can forecast + settle a city against."""
    icao: str               # e.g. "KHOU"
    name: str               # e.g. "Houston Hobby Airport"
    city_slug: str          # matches Kalshi market grouping, e.g. "HOUSTON"
    tz: str                 # IANA timezone, e.g. "America/Chicago"
    latitude: float
    longitude: float


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
class Contract:
    """A Kalshi market parsed into its settlement question."""
    market_ticker: str
    event_ticker: str
    city_slug: str
    metric: Metric
    market_date: date           # the local-time settlement date
    bracket: Bracket


@dataclass
class Evaluation:
    """The bot's verdict on one contract: fair prob, edge, and grade.

    `dist` fields are flattened scalars (not the full ForecastDistribution
    object) so this type stays import-cycle-free and trivially serializable.
    """
    contract: Contract
    market: KalshiMarket
    fair_prob_low: float        # 0..1
    fair_prob_high: float       # 0..1
    fair_prob_mid: float        # 0..1
    forecast_mean_f: Optional[float]
    band_width_f: Optional[float]   # forecast spread proxy (q90 - q10), °F
    locked: bool                # day fully observed -> outcome determined
    yes_ask_cents: Optional[int]
    no_ask_cents: Optional[int]
    edge_yes: Optional[float]   # fair_mid - yes_ask/100, or None
    edge_no: Optional[float]    # (1 - fair_mid) - no_ask/100, or None
    grade: str                  # "A+", "A", "B+", "B", "C", "D", "F"
    notes: list[str] = field(default_factory=list)

    @property
    def best_edge(self) -> Optional[float]:
        edges = [e for e in (self.edge_yes, self.edge_no) if e is not None]
        return max(edges) if edges else None

    @property
    def best_side(self) -> Optional[str]:
        if self.edge_yes is None and self.edge_no is None:
            return None
        ey = self.edge_yes if self.edge_yes is not None else -1.0
        en = self.edge_no if self.edge_no is not None else -1.0
        return "yes" if ey >= en else "no"


def market_day_window(market_date: date, tz: str) -> tuple[datetime, datetime]:
    """Return the [start, end] localized window for a market date.

    Settlement is based on observations inside the station's local calendar
    day. Returned datetimes are timezone-aware so callers convert to UTC for
    API calls. `end` is the final second of the day (inclusive).
    """
    z = ZoneInfo(tz)
    start = datetime(market_date.year, market_date.month, market_date.day, 0, 0, 0, tzinfo=z)
    end = datetime(market_date.year, market_date.month, market_date.day, 23, 59, 59, tzinfo=z)
    return start, end
