"""Tests for V0.9 regime-shifted fair_probability integration.

Verifies that:
  - When config is None, fair_probability behavior matches V0.8 exactly.
  - When config has an applied regime shift, fair_prob bounds shift accordingly.
  - LOCKED_YES / DEAD_NO outputs are NEVER shifted (deterministic states).
  - Shifts are clamped so bounds stay in [0, 1].
  - Ranker uses config cutoffs when provided; defaults preserved otherwise.
"""

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from kalshi_scout.config import RankerConfig, RegimeShift, StateThresholds, regime_key
from kalshi_scout.models import (
    Bracket,
    BracketKind,
    ContractEvaluation,
    ContractState,
    KalshiMarket,
    Metric,
    ParsedContract,
    StationState,
)
from kalshi_scout.ranker import grade
from kalshi_scout.state import fair_probability
from kalshi_scout.stations import get_station


def _state(running_max=None, running_min=None,
           market_date=date(2026, 5, 27)) -> StationState:
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


def _contract(metric: Metric, bracket: Bracket) -> ParsedContract:
    return ParsedContract(
        market_ticker="X", event_ticker="Y",
        city_slug="HOUSTON", metric=metric,
        market_date=date(2026, 5, 27), bracket=bracket,
    )


# -- fair_probability shift application --------------------------------------

def test_no_config_unchanged_from_v08(monkeypatch):
    """No config + no regime -> identical to pre-V0.9 output."""
    c = _contract(Metric.HIGH, Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0))
    s = _state(running_max=None)
    # BRACKET_HIT_VULNERABLE not applicable (no running_max), so this should
    # be FORECAST_DEPENDENT with forecast None -> (0.25, 0.75)
    lo, hi = fair_probability(
        c, s, ContractState.FORECAST_DEPENDENT, forecast=None,
        regime=None, config=None,
    )
    assert (lo, hi) == (0.25, 0.75)


def test_locked_yes_never_shifted_even_with_regime(monkeypatch):
    """LOCKED_YES is settlement-conclusive — no regime shift applies."""
    c = _contract(Metric.HIGH, Bracket(BracketKind.GTE, lo=78.0, hi=None))
    s = _state(running_max=79.0)
    cfg = RankerConfig.default()
    cfg.regime_shifts[regime_key("rain_cooled", "high", "gte")] = \
        RegimeShift.of(0.10, n=100, applied=True)
    lo, hi = fair_probability(
        c, s, ContractState.LOCKED_YES, forecast=None,
        regime="rain_cooled", config=cfg,
    )
    # Default LOCKED_YES output is (0.98, 1.0); not shifted.
    assert hi == 1.0
    assert lo == 0.98


def test_dead_no_never_shifted_even_with_regime():
    c = _contract(Metric.HIGH, Bracket(BracketKind.LTE, lo=None, hi=78.0))
    s = _state(running_max=80.0)
    cfg = RankerConfig.default()
    cfg.regime_shifts[regime_key("rain_cooled", "high", "lte")] = \
        RegimeShift.of(0.10, n=100, applied=True)
    lo, hi = fair_probability(
        c, s, ContractState.DEAD_NO, forecast=None,
        regime="rain_cooled", config=cfg,
    )
    assert lo == 0.0
    assert hi == 0.02


def test_forecast_dependent_shifted_by_applied_regime():
    """FORECAST_DEPENDENT with no forecast -> baseline (0.25, 0.75).
    Apply +0.10 regime shift -> (0.35, 0.85)."""
    c = _contract(Metric.HIGH, Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0))
    s = _state()
    cfg = RankerConfig.default()
    cfg.regime_shifts[regime_key("rain_cooled", "high", "between")] = \
        RegimeShift.of(0.10, n=100, applied=True)
    lo, hi = fair_probability(
        c, s, ContractState.FORECAST_DEPENDENT, forecast=None,
        regime="rain_cooled", config=cfg,
    )
    assert abs(lo - 0.35) < 1e-9
    assert abs(hi - 0.85) < 1e-9


def test_bracket_hit_vulnerable_shifted_too():
    c = _contract(Metric.HIGH, Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0))
    s = _state()
    cfg = RankerConfig.default()
    cfg.regime_shifts[regime_key("clear_and_dry", "high", "between")] = \
        RegimeShift.of(-0.05, n=100, applied=True)
    lo, hi = fair_probability(
        c, s, ContractState.BRACKET_HIT_VULNERABLE, forecast=None,
        regime="clear_and_dry", config=cfg,
    )
    # Baseline (0.45, 0.85) shifted -0.05 -> (0.40, 0.80)
    assert abs(lo - 0.40) < 1e-9
    assert abs(hi - 0.80) < 1e-9


def test_shift_clamped_to_unit_interval():
    c = _contract(Metric.HIGH, Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0))
    s = _state()
    cfg = RankerConfig.default()
    cfg.regime_shifts[regime_key("rain_cooled", "high", "between")] = \
        RegimeShift.of(0.50, n=100, applied=True)  # clamped to 0.20
    lo, hi = fair_probability(
        c, s, ContractState.FORECAST_DEPENDENT, forecast=None,
        regime="rain_cooled", config=cfg,
    )
    # Baseline (0.25, 0.75) + 0.20 -> (0.45, 0.95)
    assert abs(lo - 0.45) < 1e-9
    assert abs(hi - 0.95) < 1e-9


def test_unapplied_shift_is_no_op():
    c = _contract(Metric.HIGH, Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0))
    s = _state()
    cfg = RankerConfig.default()
    cfg.regime_shifts[regime_key("rain_cooled", "high", "between")] = \
        RegimeShift.of(0.10, n=5, applied=False)
    lo, hi = fair_probability(
        c, s, ContractState.FORECAST_DEPENDENT, forecast=None,
        regime="rain_cooled", config=cfg,
    )
    # Unapplied -> baseline preserved
    assert (lo, hi) == (0.25, 0.75)


# -- Ranker with custom config ----------------------------------------------

def _market(yes_ask: int = 71) -> KalshiMarket:
    return KalshiMarket(
        ticker="K", event_ticker="E",
        title="", yes_sub_title="", status="open", close_time=None,
        yes_bid=yes_ask - 1, yes_ask=yes_ask,
        no_bid=100 - yes_ask - 1, no_ask=100 - yes_ask,
        last_price=yes_ask, volume=10, open_interest=100,
    )


def test_ranker_uses_tightened_locked_yes_cutoff_from_config():
    """If the config raises high_cutoff for LOCKED_YES from 0.08 to 0.15,
    a 0.10-edge LOCKED_YES that would have been A+ at defaults becomes A."""
    c = ParsedContract(
        market_ticker="K", event_ticker="E",
        city_slug="HOUSTON", metric=Metric.HIGH,
        market_date=date(2026, 5, 27),
        bracket=Bracket(BracketKind.GTE, lo=78.0, hi=None),
    )
    # Default would grade A+ at edge 0.10 (>= 0.08).
    default_eval = grade(c, _market(yes_ask=85),  # fair_mid 0.95, ask 0.85 -> edge 0.10
                         ContractState.LOCKED_YES, "", 0.95, 0.95)
    assert default_eval.grade == "A+"

    # Tightened config: high_cutoff raised to 0.15.
    cfg = RankerConfig.default()
    cfg.locked_yes = StateThresholds(high_cutoff=0.15, low_cutoff=0.05)
    tightened_eval = grade(c, _market(yes_ask=85),
                           ContractState.LOCKED_YES, "", 0.95, 0.95,
                           config=cfg)
    # 0.10 edge no longer meets the 0.15 cutoff -> A (next tier down)
    assert tightened_eval.grade == "A"


def test_ranker_uses_loosened_dead_no_cutoff_from_config():
    """Loosening dead_no.high_cutoff from 0.08 to 0.02 should promote
    a small-edge No-side trade from B to A+."""
    c = ParsedContract(
        market_ticker="K", event_ticker="E",
        city_slug="HOUSTON", metric=Metric.HIGH,
        market_date=date(2026, 5, 27),
        bracket=Bracket(BracketKind.LTE, lo=None, hi=78.0),
    )
    # Yes ask 95c -> no_ask 5c. fair_mid 0.01 -> no_edge = 0.99 - 0.05 = +0.94 (huge).
    # That's already A+ under defaults. Let's use a tiny-edge scenario instead:
    # fair_mid 0.01 (dead), no_ask 96 -> no_edge = (1 - 0.01) - 0.96 = +0.03
    # Under defaults: 0.03 < 0.08, falls to A cutoff 0.03 -> just barely A.
    default_eval = grade(c, _market(yes_ask=4),  # no_ask = 96
                         ContractState.DEAD_NO, "", 0.0, 0.02)
    assert default_eval.grade == "A"

    cfg = RankerConfig.default()
    cfg.dead_no = StateThresholds(high_cutoff=0.02, low_cutoff=0.01)
    loose_eval = grade(c, _market(yes_ask=4),
                       ContractState.DEAD_NO, "", 0.0, 0.02,
                       config=cfg)
    # 0.03 >= loosened 0.02 -> A+
    assert loose_eval.grade == "A+"
