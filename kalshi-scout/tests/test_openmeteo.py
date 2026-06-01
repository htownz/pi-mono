"""Tests for the Open-Meteo ensemble client + ensemble fair-prob path.

The settlement source is never Open-Meteo — only the primary station's
CLI settles a contract. These tests cover:

  - parse_ensemble_response: handles real, degraded, and broken responses
  - OpenMeteoClient: caches per (lat, lon, tz, model)
  - fair_probability_from_ensemble: returns None when unsupported, counts
    members correctly, applies Wilson CI for sampling uncertainty, falls
    back gracefully on thin ensembles
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx
import respx

from kalshi_scout.config import RankerConfig
from kalshi_scout.models import (
    Bracket,
    BracketKind,
    ContractState,
    Metric,
    ParsedContract,
    StationState,
)
from kalshi_scout.openmeteo import (
    DEFAULT_ENSEMBLE_MODEL,
    EnsembleHourlyPoint,
    OPENMETEO_ENSEMBLE_URL,
    OpenMeteoClient,
    parse_ensemble_response,
)
from kalshi_scout.state import fair_probability_from_ensemble
from kalshi_scout.stations import get_station


# -- parser ------------------------------------------------------------------

def _make_response(times: list[str], members: list[list[float]]) -> dict:
    """Helper: build a minimal Open-Meteo-shaped response from times +
    a list of per-member series."""
    h = {"time": times}
    for i, series in enumerate(members):
        h[f"temperature_2m_member{i+1:02d}"] = series
    return {"timezone": "America/Chicago", "hourly": h}


def test_parse_returns_empty_on_missing_hourly():
    assert parse_ensemble_response({}, tz="America/Chicago") == []
    assert parse_ensemble_response({"hourly": {}}, tz="America/Chicago") == []


def test_parse_uses_deterministic_when_no_members():
    """Some Open-Meteo responses (or different model params) carry only the
    deterministic `temperature_2m` series. Parser treats that as a
    single-member ensemble — produces zero std but still gives the engine
    a usable forecast curve."""
    data = {
        "timezone": "America/Chicago",
        "hourly": {
            "time": ["2026-05-31T12:00", "2026-05-31T13:00"],
            "temperature_2m": [78.0, 79.0],
        },
    }
    points = parse_ensemble_response(data, tz="America/Chicago")
    assert len(points) == 2
    assert points[0].members_f == (78.0,)
    assert points[0].std_f == 0.0
    assert points[0].mean_f == 78.0


def test_parse_extracts_members_and_returns_utc():
    times = ["2026-05-31T12:00", "2026-05-31T13:00"]
    members = [
        [78.0, 79.0],
        [78.5, 79.5],
        [77.5, 80.0],
    ]
    points = parse_ensemble_response(
        _make_response(times, members), tz="America/Chicago",
    )
    assert len(points) == 2
    assert points[0].members_f == (78.0, 78.5, 77.5)
    # Converted to UTC: noon Central (CDT, UTC-5) → 17:00 UTC
    assert points[0].start == datetime(2026, 5, 31, 17, 0, tzinfo=timezone.utc)
    # mean ≈ (78 + 78.5 + 77.5) / 3 = 78.0
    assert round(points[0].mean_f, 3) == 78.0


def test_parse_drops_hours_with_no_valid_members():
    """A bad/None entry at one hour doesn't poison the rest."""
    data = _make_response(
        ["2026-05-31T12:00", "2026-05-31T13:00", "2026-05-31T14:00"],
        [[78.0, None, 79.0]],
    )
    points = parse_ensemble_response(data, tz="America/Chicago")
    # Hour 1 had None as only member → dropped. Hours 0 and 2 retained.
    assert len(points) == 2
    assert all(p.members_f for p in points)


def test_parse_ragged_members_drops_partials_per_hour():
    """Member series of different lengths: hours past the shortest are
    parsed only with the members that have data at that index."""
    data = _make_response(
        ["2026-05-31T12:00", "2026-05-31T13:00"],
        [[78.0, 79.0], [78.5]],   # second member only has 1 point
    )
    points = parse_ensemble_response(data, tz="America/Chicago")
    # Hour 0: both members → 2 members
    assert points[0].members_f == (78.0, 78.5)
    # Hour 1: only first member → 1 member
    assert points[1].members_f == (79.0,)


# -- client --------------------------------------------------------------

@respx.mock
def test_client_caches_repeated_calls_per_lat_lon():
    """Two calls with the same (lat, lon, tz, model) hit the API once."""
    route = respx.get(OPENMETEO_ENSEMBLE_URL).mock(
        return_value=httpx.Response(200, json=_make_response(
            ["2026-05-31T12:00"], [[78.0], [79.0]],
        )),
    )
    client = OpenMeteoClient()
    pts1 = client.ensemble_hourly_temperature(29.65, -95.28, "America/Chicago")
    pts2 = client.ensemble_hourly_temperature(29.65, -95.28, "America/Chicago")
    assert pts1 == pts2
    assert route.call_count == 1


@respx.mock
def test_client_returns_empty_on_http_failure():
    """A 500 doesn't propagate — the caller treats empty-list as
    'ensemble unavailable, use NWS-only path'."""
    respx.get(OPENMETEO_ENSEMBLE_URL).mock(return_value=httpx.Response(500))
    client = OpenMeteoClient()
    pts = client.ensemble_hourly_temperature(29.65, -95.28, "America/Chicago")
    assert pts == []


@respx.mock
def test_client_returns_empty_on_garbage_json():
    """Non-JSON body must not crash the engine — fall back to NWS-only."""
    respx.get(OPENMETEO_ENSEMBLE_URL).mock(
        return_value=httpx.Response(200, content=b"<html>oops</html>"),
    )
    client = OpenMeteoClient()
    pts = client.ensemble_hourly_temperature(29.65, -95.28, "America/Chicago")
    assert pts == []


# -- fair_probability_from_ensemble -----------------------------------------

def _ss(running_max=None, running_min=None) -> StationState:
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
    )


def _contract(metric: Metric, bracket: Bracket) -> ParsedContract:
    return ParsedContract(
        market_ticker="X", event_ticker="Y", city_slug="HOUSTON",
        metric=metric, market_date=date(2026, 5, 31), bracket=bracket,
    )


def _ensemble(now_utc: datetime, members_by_hour: list[list[float]]) -> list[EnsembleHourlyPoint]:
    """Build an ensemble forecast 1h apart starting at now_utc."""
    return [
        EnsembleHourlyPoint(
            start=now_utc + timedelta(hours=i + 1),
            members_f=tuple(members),
        )
        for i, members in enumerate(members_by_hour)
    ]


def test_ensemble_returns_none_on_deterministic_states():
    """LOCKED_YES / DEAD_NO are conclusive — ensemble adds no info; caller
    uses the eps-locked deterministic range."""
    ss = _ss(running_max=85.0)
    now = ss.window_start + timedelta(hours=14)
    ens = _ensemble(now.astimezone(timezone.utc), [[78.0] * 30])
    bracket = Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0)
    contract = _contract(Metric.HIGH, bracket)
    assert fair_probability_from_ensemble(
        contract, ss, ContractState.LOCKED_YES, ens,
        now_utc=now.astimezone(timezone.utc),
    ) is None
    assert fair_probability_from_ensemble(
        contract, ss, ContractState.DEAD_NO, ens,
        now_utc=now.astimezone(timezone.utc),
    ) is None


def test_ensemble_returns_none_on_empty_or_out_of_window():
    contract = _contract(Metric.HIGH, Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0))
    ss = _ss(running_max=None)
    now = ss.window_start + timedelta(hours=14)
    assert fair_probability_from_ensemble(
        contract, ss, ContractState.FORECAST_DEPENDENT, [],
        now_utc=now.astimezone(timezone.utc),
    ) is None
    # All ensemble points are AFTER the window ends.
    far_future = ss.window_end.astimezone(timezone.utc) + timedelta(hours=5)
    ens = [EnsembleHourlyPoint(start=far_future + timedelta(hours=i), members_f=(78.0,) * 30)
           for i in range(3)]
    assert fair_probability_from_ensemble(
        contract, ss, ContractState.FORECAST_DEPENDENT, ens,
        now_utc=now.astimezone(timezone.utc),
    ) is None


def test_ensemble_returns_none_when_below_min_members():
    """Thin ensemble (single deterministic member) shouldn't drive grading
    without further calibration — caller falls back to NWS-only path."""
    contract = _contract(Metric.HIGH, Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0))
    ss = _ss(running_max=None)
    now = ss.window_start + timedelta(hours=14)
    # 3 hours of single-member ensemble.
    ens = _ensemble(now.astimezone(timezone.utc), [[78.0], [79.5], [80.5]])
    assert fair_probability_from_ensemble(
        contract, ss, ContractState.FORECAST_DEPENDENT, ens,
        now_utc=now.astimezone(timezone.utc),
        min_members=10,
    ) is None


def test_ensemble_high_fair_prob_when_all_members_inside_bracket():
    """All 30 members project a daily max inside [79, 80] → empirical
    fraction = 1.0 → Wilson upper bound is 1.0, lower is slightly under."""
    contract = _contract(Metric.HIGH, Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0))
    ss = _ss(running_max=None)
    now = ss.window_start + timedelta(hours=14)
    # 30 members all peaking inside the bracket (clustered near 79.5).
    members_hour_1 = [79.0 + 0.03 * i for i in range(30)]   # 79.0 .. 79.87
    members_hour_2 = [78.5] * 30                            # cooling
    ens = _ensemble(now.astimezone(timezone.utc), [members_hour_1, members_hour_2])
    out = fair_probability_from_ensemble(
        contract, ss, ContractState.FORECAST_DEPENDENT, ens,
        now_utc=now.astimezone(timezone.utc),
    )
    assert out is not None
    lo, hi = out
    # All members inside bracket → mid > 0.85 by Wilson interval shape.
    assert (lo + hi) / 2 > 0.85
    assert hi <= 1.0


def test_ensemble_low_fair_prob_when_all_members_escape_bracket():
    """All 30 members project a daily max ABOVE the bracket → fair_prob ≈ 0."""
    contract = _contract(Metric.HIGH, Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0))
    ss = _ss(running_max=None)
    now = ss.window_start + timedelta(hours=14)
    # All members peaking at 82-83°F, well above the bracket.
    members_hot = [82.0 + 0.05 * i for i in range(30)]
    ens = _ensemble(now.astimezone(timezone.utc), [members_hot, [80.0] * 30])
    out = fair_probability_from_ensemble(
        contract, ss, ContractState.FORECAST_DEPENDENT, ens,
        now_utc=now.astimezone(timezone.utc),
    )
    assert out is not None
    lo, hi = out
    assert (lo + hi) / 2 < 0.15


def test_ensemble_mid_fair_prob_when_members_split():
    """Half the members project inside, half above the bracket → ~0.5
    fair_prob with a wider Wilson band reflecting sampling uncertainty."""
    contract = _contract(Metric.HIGH, Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0))
    ss = _ss(running_max=None)
    now = ss.window_start + timedelta(hours=14)
    # 15 members inside [79, 80], 15 members above 80.
    inside = [79.5] * 15
    outside = [82.0] * 15
    ens = _ensemble(now.astimezone(timezone.utc), [inside + outside])
    out = fair_probability_from_ensemble(
        contract, ss, ContractState.FORECAST_DEPENDENT, ens,
        now_utc=now.astimezone(timezone.utc),
    )
    assert out is not None
    lo, hi = out
    # Empirical fraction is exactly 0.5; Wilson CI is roughly ±0.18 at N=30.
    assert 0.4 < (lo + hi) / 2 < 0.6
    assert hi - lo > 0.2     # band reflects sampling uncertainty
