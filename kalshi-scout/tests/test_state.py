"""Unit tests for the contract state machine.

We test the classifier directly against synthetic StationState objects — this
keeps tests independent of network access.
"""

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from kalshi_scout.models import (
    Bracket,
    BracketKind,
    ContractState,
    Metric,
    ParsedContract,
    StationState,
)
from kalshi_scout.state import classify
from kalshi_scout.stations import get_station


def _state(running_max=None, running_min=None, market_date=date(2026, 5, 27)):
    station = get_station("HOUSTON")
    assert station is not None
    z = ZoneInfo(station.tz)
    return StationState(
        station=station,
        market_date=market_date,
        window_start=datetime(market_date.year, market_date.month, market_date.day, 0, 0, tzinfo=z),
        window_end=datetime(market_date.year, market_date.month, market_date.day, 23, 59, 59, tzinfo=z),
        running_max_f=running_max,
        running_min_f=running_min,
        latest=None,
        cli_report_date=None,
        cli_max_f=None,
        cli_min_f=None,
        observations=[],
    )


def _contract(metric, bracket, market_date=date(2026, 5, 27)):
    return ParsedContract(
        market_ticker="X",
        event_ticker="Y",
        city_slug="HOUSTON",
        metric=metric,
        market_date=market_date,
        bracket=bracket,
    )


# -- HIGH temperature ----------------------------------------------------------

def test_high_above_locked_yes_when_max_reaches_strike():
    c = _contract(Metric.HIGH, Bracket(BracketKind.GTE, lo=85.0, hi=None))
    state, _ = classify(c, _state(running_max=86.0))
    assert state is ContractState.LOCKED_YES


def test_high_above_not_reached_when_max_below_strike():
    c = _contract(Metric.HIGH, Bracket(BracketKind.GTE, lo=85.0, hi=None))
    state, _ = classify(c, _state(running_max=80.0))
    assert state is ContractState.NOT_REACHED


def test_high_below_dead_no_when_max_exceeds_strike():
    """The classic 'already-killed' scenario from the spec."""
    c = _contract(Metric.HIGH, Bracket(BracketKind.LTE, lo=None, hi=78.0))
    state, _ = classify(c, _state(running_max=79.0))
    assert state is ContractState.DEAD_NO


def test_high_below_still_alive_when_max_at_strike():
    c = _contract(Metric.HIGH, Bracket(BracketKind.LTE, lo=None, hi=78.0))
    state, _ = classify(c, _state(running_max=78.0))
    assert state is ContractState.FORECAST_DEPENDENT


def test_high_between_bracket_hit_vulnerable():
    """The 79–80 scenario from the spec: bracket hit but day not over."""
    c = _contract(Metric.HIGH, Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0))
    state, reason = classify(c, _state(running_max=79.0))
    assert state is ContractState.BRACKET_HIT_VULNERABLE
    assert "reaching" in reason


def test_high_between_dead_when_blown_through():
    c = _contract(Metric.HIGH, Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0))
    state, _ = classify(c, _state(running_max=81.0))
    assert state is ContractState.DEAD_NO


def test_high_between_not_reached_when_max_below():
    c = _contract(Metric.HIGH, Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0))
    state, _ = classify(c, _state(running_max=77.0))
    assert state is ContractState.NOT_REACHED


# -- LOW temperature -----------------------------------------------------------

def test_low_below_locked_yes_when_min_drops_to_strike():
    """Spec quote: 'a minimum cannot be undone by later warming.'"""
    c = _contract(Metric.LOW, Bracket(BracketKind.LTE, lo=None, hi=70.0))
    state, _ = classify(c, _state(running_min=70.0))
    assert state is ContractState.LOCKED_YES


def test_low_below_still_alive_when_min_above_strike():
    c = _contract(Metric.LOW, Bracket(BracketKind.LTE, lo=None, hi=70.0))
    state, _ = classify(c, _state(running_min=72.0))
    assert state is ContractState.FORECAST_DEPENDENT


def test_low_above_dead_when_min_drops_below():
    c = _contract(Metric.LOW, Bracket(BracketKind.GTE, lo=74.0, hi=None))
    state, _ = classify(c, _state(running_min=72.0))
    assert state is ContractState.DEAD_NO


def test_low_between_bracket_hit_vulnerable_remaining_risk_downward():
    """Spec: bracket 70-71 hit at 71; remaining risk is dropping below 70."""
    c = _contract(Metric.LOW, Bracket(BracketKind.BETWEEN, lo=70.0, hi=71.0))
    state, reason = classify(c, _state(running_min=71.0))
    assert state is ContractState.BRACKET_HIT_VULNERABLE
    assert "69" in reason or "dropping" in reason


def test_low_between_dead_when_min_below_bracket():
    c = _contract(Metric.LOW, Bracket(BracketKind.BETWEEN, lo=70.0, hi=71.0))
    state, _ = classify(c, _state(running_min=68.0))
    assert state is ContractState.DEAD_NO


def test_low_between_not_reached_when_min_above_bracket():
    c = _contract(Metric.LOW, Bracket(BracketKind.BETWEEN, lo=70.0, hi=71.0))
    state, _ = classify(c, _state(running_min=74.0))
    assert state is ContractState.NOT_REACHED


def test_no_obs_means_forecast_dependent():
    c = _contract(Metric.HIGH, Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0))
    state, _ = classify(c, _state(running_max=None))
    assert state is ContractState.FORECAST_DEPENDENT
