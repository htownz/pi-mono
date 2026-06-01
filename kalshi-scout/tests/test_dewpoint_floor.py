"""Tests for the dewpoint-floor physics signal on LOW markets.

Physics: the daily LOW almost never drops more than ~1°F below the minimum
dewpoint observed during the cooling window. Once T = T_d, condensation
releases latent heat and halts further cooling.

The engine uses this in `fair_probability`'s LOW branch only — clipping
the projected daily-min to physics and widening the residual band when
the forecast extrapolation disagrees. It does NOT enter `classify()`:
dewpoint isn't a settlement fact and can never lock a contract.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from kalshi_scout.models import (
    Bracket,
    BracketKind,
    ContractState,
    Metric,
    ParsedContract,
    StationReading,
    StationState,
)
from kalshi_scout.nws import HourlyPoint
from kalshi_scout.state import build_station_state, fair_probability
from kalshi_scout.stations import get_station


# -- Helpers (shared with test_forecast_residual) ----------------------------

def _ss(running_max=None, running_min=None, dewpoint_floor=None) -> StationState:
    station = get_station("HOUSTON")
    assert station is not None
    z = ZoneInfo(station.tz)
    md = date(2026, 5, 31)
    return StationState(
        station=station, market_date=md,
        window_start=datetime(md.year, md.month, md.day, 0, 0, tzinfo=z),
        window_end=datetime(md.year, md.month, md.day, 23, 59, 59, tzinfo=z),
        running_max_f=running_max, running_min_f=running_min,
        latest=None, cli_report_date=None, cli_max_f=None, cli_min_f=None,
        dewpoint_floor_f=dewpoint_floor,
    )


def _contract(metric: Metric, bracket: Bracket) -> ParsedContract:
    return ParsedContract(
        market_ticker="X", event_ticker="Y", city_slug="HOUSTON",
        metric=metric, market_date=date(2026, 5, 31), bracket=bracket,
    )


def _forecast(now_utc: datetime, *temps_f: float) -> list[HourlyPoint]:
    return [
        HourlyPoint(start=now_utc + timedelta(hours=i + 1), temperature_f=t)
        for i, t in enumerate(temps_f)
    ]


# -- build_station_state populates dewpoint_floor_f --------------------------

class _StubNws:
    """Minimal NwsClient stub for build_station_state. observations() returns
    canned readings carrying dewpoint; everything else returns None/[]."""
    def __init__(self, primary_obs):
        self.primary_obs = primary_obs

    def observations(self, icao, start=None, end=None, limit=500):
        if icao.upper() == "KHOU":
            return self.primary_obs
        return []

    def latest_observation(self, icao):
        return None

    def latest_cli(self, location_id):
        return None


def test_build_station_state_populates_dewpoint_floor_from_min_dewpoint():
    """The lowest dewpoint across all populated readings becomes the floor."""
    station = get_station("HOUSTON")
    assert station is not None
    window_start = datetime(2026, 5, 27, 5, 0, tzinfo=timezone.utc)
    obs = [
        StationReading(observed_at=window_start, temperature_f=78.0, dewpoint_f=72.0),
        StationReading(observed_at=window_start + timedelta(hours=2),
                       temperature_f=76.0, dewpoint_f=70.0),
        StationReading(observed_at=window_start + timedelta(hours=4),
                       temperature_f=75.0, dewpoint_f=68.0),   # the floor
        StationReading(observed_at=window_start + timedelta(hours=6),
                       temperature_f=80.0, dewpoint_f=71.0),
    ]
    ss = build_station_state(
        _StubNws(obs), station, market_date=date(2026, 5, 27),
        now_utc=datetime(2026, 5, 28, 4, 0, tzinfo=timezone.utc),
    )
    assert ss.dewpoint_floor_f == 68.0


def test_build_station_state_skips_readings_with_no_dewpoint():
    """A reading with dewpoint_f=None must not be counted as the floor."""
    station = get_station("HOUSTON")
    assert station is not None
    window_start = datetime(2026, 5, 27, 5, 0, tzinfo=timezone.utc)
    obs = [
        StationReading(observed_at=window_start, temperature_f=78.0, dewpoint_f=72.0),
        StationReading(observed_at=window_start + timedelta(hours=2),
                       temperature_f=76.0, dewpoint_f=None),
        StationReading(observed_at=window_start + timedelta(hours=4),
                       temperature_f=75.0, dewpoint_f=70.0),
    ]
    ss = build_station_state(
        _StubNws(obs), station, market_date=date(2026, 5, 27),
        now_utc=datetime(2026, 5, 28, 4, 0, tzinfo=timezone.utc),
    )
    assert ss.dewpoint_floor_f == 70.0


def test_build_station_state_dewpoint_floor_none_when_no_dewpoint_anywhere():
    """No dewpoint readings → floor stays None → fair_prob LOW path
    behaves identically to pre-PR behavior."""
    station = get_station("HOUSTON")
    assert station is not None
    window_start = datetime(2026, 5, 27, 5, 0, tzinfo=timezone.utc)
    obs = [
        StationReading(observed_at=window_start, temperature_f=78.0, dewpoint_f=None),
        StationReading(observed_at=window_start + timedelta(hours=2),
                       temperature_f=76.0, dewpoint_f=None),
    ]
    ss = build_station_state(
        _StubNws(obs), station, market_date=date(2026, 5, 27),
        now_utc=datetime(2026, 5, 28, 4, 0, tzinfo=timezone.utc),
    )
    assert ss.dewpoint_floor_f is None


# -- fair_probability LOW path uses dewpoint floor ---------------------------

def test_dewpoint_floor_collapses_low_yes_when_bracket_below_physics():
    """Setup: market 'LOW ≤ 60°F' (NOT_REACHED) — forecast says it'll cool
    to 55°F overnight, but the minimum dewpoint observed is 72°F. Physics
    says the low can't drop more than ~1°F below dewpoint floor (71°F), so
    the projected min should be clipped to 71°F. With a 71°F projection
    and a bracket cap at 60°F, the contract is almost impossible to win
    (no overlap with the bracket → fair_prob YES ≈ 0).
    """
    contract = _contract(Metric.LOW, Bracket(BracketKind.LTE, lo=None, hi=60.0))
    ss = _ss(running_min=None, dewpoint_floor=72.0)
    now = ss.window_start + timedelta(hours=10)
    # Forecast extrapolates a 55°F low — clearly below dewpoint physics.
    forecast = _forecast(now.astimezone(timezone.utc), 60.0, 55.0, 56.0, 58.0)

    lo, hi = fair_probability(
        contract, ss, ContractState.FORECAST_DEPENDENT,
        forecast, now_utc=now.astimezone(timezone.utc),
    )
    # Bracket cap 60°F, projected min clipped to 71°F → bracket out of reach.
    assert (lo + hi) / 2 < 0.15, (
        f"expected very low YES fair_prob; got mid={(lo + hi) / 2:.3f}"
    )


def test_dewpoint_floor_no_effect_when_forecast_above_physics():
    """When the forecast already respects physics (forecast min ≥ dewpoint
    floor - 1°F), the projection is unchanged. The bracket overlap math
    runs on the same proj_lo / proj_hi as before."""
    contract = _contract(Metric.LOW, Bracket(BracketKind.LTE, lo=None, hi=72.0))
    ss_with_floor = _ss(running_min=None, dewpoint_floor=70.0)
    ss_without_floor = _ss(running_min=None, dewpoint_floor=None)
    now = ss_with_floor.window_start + timedelta(hours=10)
    # Forecast min 71°F — already at the dewpoint floor (70 - 1 = 69°F), so
    # projection clip is a no-op.
    forecast = _forecast(now.astimezone(timezone.utc), 73.0, 71.0, 72.0)

    lo_with, hi_with = fair_probability(
        contract, ss_with_floor, ContractState.FORECAST_DEPENDENT,
        forecast, now_utc=now.astimezone(timezone.utc),
    )
    lo_without, hi_without = fair_probability(
        contract, ss_without_floor, ContractState.FORECAST_DEPENDENT,
        forecast, now_utc=now.astimezone(timezone.utc),
    )
    assert (lo_with, hi_with) == (lo_without, hi_without)


def test_dewpoint_floor_widens_residual_when_forecast_disagrees():
    """When the forecast extrapolation drops below physics, we clip the
    projection AND widen the residual by 1°F. The widening reflects honest
    forecast-vs-physics disagreement: we trust physics for the central
    estimate but acknowledge the band is less reliable. Manifests as a
    different fair_prob output vs the same setup without the dewpoint
    signal — usually a wider band or shifted bucket."""
    contract = _contract(Metric.LOW, Bracket(BracketKind.BETWEEN, lo=68.0, hi=72.0))
    ss_with_floor = _ss(running_min=None, dewpoint_floor=70.0)
    ss_without_floor = _ss(running_min=None, dewpoint_floor=None)
    now = ss_with_floor.window_start + timedelta(hours=10)
    # Forecast says 65°F — below dewpoint floor (69°F).
    forecast = _forecast(now.astimezone(timezone.utc), 70.0, 65.0, 67.0)

    result_with = fair_probability(
        contract, ss_with_floor, ContractState.FORECAST_DEPENDENT,
        forecast, now_utc=now.astimezone(timezone.utc),
    )
    result_without = fair_probability(
        contract, ss_without_floor, ContractState.FORECAST_DEPENDENT,
        forecast, now_utc=now.astimezone(timezone.utc),
    )
    # Different outputs — dewpoint floor materially shifted the answer.
    assert result_with != result_without


def test_dewpoint_floor_does_not_affect_high_markets():
    """HIGH-side fair_prob runs through a different branch and must be
    untouched by dewpoint floor logic — daytime cap physics is much softer
    and not in scope for this signal."""
    contract = _contract(Metric.HIGH, Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0))
    ss_with_floor = _ss(running_max=79.5, dewpoint_floor=72.0)
    ss_without_floor = _ss(running_max=79.5, dewpoint_floor=None)
    now = ss_with_floor.window_start + timedelta(hours=10)
    forecast = _forecast(now.astimezone(timezone.utc), 80.5, 79.0, 76.0)

    result_with = fair_probability(
        contract, ss_with_floor, ContractState.BRACKET_HIT_VULNERABLE,
        forecast, now_utc=now.astimezone(timezone.utc),
    )
    result_without = fair_probability(
        contract, ss_without_floor, ContractState.BRACKET_HIT_VULNERABLE,
        forecast, now_utc=now.astimezone(timezone.utc),
    )
    assert result_with == result_without


def test_dewpoint_floor_does_not_alter_deterministic_states():
    """LOCKED_YES / DEAD_NO short-circuit before any forecast logic runs,
    so dewpoint floor must not enter the deterministic eps-locked range."""
    contract = _contract(Metric.LOW, Bracket(BracketKind.LTE, lo=None, hi=60.0))
    ss = _ss(running_min=58.0, dewpoint_floor=72.0)   # impossible-looking floor
    now = ss.window_start + timedelta(hours=10)
    forecast = _forecast(now.astimezone(timezone.utc), 60.0)

    # LOCKED_YES — min already dropped to 58°F, settlement-conclusive.
    lo_locked, hi_locked = fair_probability(
        contract, ss, ContractState.LOCKED_YES,
        forecast, now_utc=now.astimezone(timezone.utc),
    )
    assert lo_locked >= 0.97 and hi_locked == 1.0    # eps-locked Yes

    # DEAD_NO — same deterministic short-circuit.
    lo_dead, hi_dead = fair_probability(
        contract, ss, ContractState.DEAD_NO,
        forecast, now_utc=now.astimezone(timezone.utc),
    )
    assert lo_dead == 0.0 and hi_dead <= 0.03    # eps-locked No
