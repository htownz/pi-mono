"""Tests for the forecast-residual capture + calibration loop (#3).

Three layers:
  - `project_extremum`: today's best point estimate of the daily extremum
    (max for HIGH, min for LOW). Stored on each snapshot so the tuner can
    compute (projected - realized) residuals after settlement.
  - `derive_forecast_residuals`: per-(station_icao, metric) median |residual|
    from settled snapshots, gated on `MIN_N_PER_RESIDUAL`.
  - `fair_probability`: reads the calibrated residual from config when one
    exists, falls back to the 2.0°F default otherwise.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from kalshi_scout.config import (
    DEFAULT_FORECAST_RESIDUAL_F,
    ForecastResidual,
    MIN_N_PER_RESIDUAL,
    RankerConfig,
    residual_key,
)
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
from kalshi_scout.nws import HourlyPoint
from kalshi_scout.state import fair_probability, project_extremum
from kalshi_scout.stations import get_station
from kalshi_scout.store import SnapshotStore, settlement_from_cli
from kalshi_scout.tuning import derive_forecast_residuals


def _settle(store: SnapshotStore, ticker: str, event_ticker: str,
            market_date: date, cli_value_f: float,
            bracket: Bracket = Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0)) -> None:
    """Build + persist a SettlementRow via the resolver helper. Encapsulates
    the (10-arg) constructor so test bodies stay readable."""
    s = settlement_from_cli(
        market_ticker=ticker, event_ticker=event_ticker, market_date=market_date,
        city_slug="HOUSTON", metric=Metric.HIGH, bracket=bracket,
        station_icao="KHOU", cli_product="CLIHOU",
        cli_report_date=market_date, cli_value_f=cli_value_f,
    )
    store.record_settlement(s)


# -- fixtures ----------------------------------------------------------------

def _station_state(running_max=None, running_min=None, market_date=date(2026, 5, 27)):
    station = get_station("HOUSTON")
    assert station is not None
    z = ZoneInfo(station.tz)
    return StationState(
        station=station, market_date=market_date,
        window_start=datetime(market_date.year, market_date.month, market_date.day, 0, 0, tzinfo=z),
        window_end=datetime(market_date.year, market_date.month, market_date.day, 23, 59, 59, tzinfo=z),
        running_max_f=running_max, running_min_f=running_min,
        latest=None,
        cli_report_date=None, cli_max_f=None, cli_min_f=None,
        observations=[],
    )


def _forecast(now_utc: datetime, *temps_f: float) -> list[HourlyPoint]:
    """Build a forecast with one HourlyPoint per temp, spaced 1h apart."""
    return [
        HourlyPoint(start=now_utc + timedelta(hours=i + 1), temperature_f=t)
        for i, t in enumerate(temps_f)
    ]


def _contract(metric, bracket, market_date=date(2026, 5, 27)) -> ParsedContract:
    return ParsedContract(
        market_ticker="X", event_ticker="Y", city_slug="HOUSTON",
        metric=metric, market_date=market_date, bracket=bracket,
    )


# -- project_extremum --------------------------------------------------------

def test_project_extremum_high_uses_max_of_observed_and_forecast():
    ss = _station_state(running_max=85.0)
    now = datetime(2026, 5, 27, 16, 0, tzinfo=ZoneInfo("America/Chicago"))
    forecast = _forecast(now.astimezone(timezone.utc), 88.0, 86.0, 80.0)
    p = project_extremum(Metric.HIGH, forecast, ss, now_utc=now.astimezone(timezone.utc))
    # max(observed 85, forecast max 88) = 88
    assert p == 88.0


def test_project_extremum_high_falls_back_to_observed_when_forecast_below():
    """When the day is past its peak, observed_max > all remaining forecast.
    The projection should equal the observed value, not the forecast max."""
    ss = _station_state(running_max=92.0)
    now = datetime(2026, 5, 27, 19, 0, tzinfo=ZoneInfo("America/Chicago"))
    forecast = _forecast(now.astimezone(timezone.utc), 88.0, 84.0, 78.0)
    p = project_extremum(Metric.HIGH, forecast, ss, now_utc=now.astimezone(timezone.utc))
    assert p == 92.0


def test_project_extremum_low_uses_min_of_observed_and_forecast():
    ss = _station_state(running_min=68.0)
    now = datetime(2026, 5, 27, 3, 0, tzinfo=ZoneInfo("America/Chicago"))
    forecast = _forecast(now.astimezone(timezone.utc), 65.0, 64.0, 70.0)
    p = project_extremum(Metric.LOW, forecast, ss, now_utc=now.astimezone(timezone.utc))
    # min(observed 68, forecast min 64) = 64
    assert p == 64.0


def test_project_extremum_observed_only_when_no_forecast():
    ss = _station_state(running_max=85.0, running_min=68.0)
    assert project_extremum(Metric.HIGH, None, ss) == 85.0
    assert project_extremum(Metric.LOW, None, ss) == 68.0


def test_project_extremum_forecast_only_when_no_observations_yet():
    """Pre-dawn scan: no observations yet, but the forecast covers the day."""
    ss = _station_state()  # observed max/min both None
    now = datetime(2026, 5, 27, 4, 0, tzinfo=ZoneInfo("America/Chicago"))
    forecast = _forecast(now.astimezone(timezone.utc), 72.0, 78.0, 84.0, 88.0)
    p = project_extremum(Metric.HIGH, forecast, ss, now_utc=now.astimezone(timezone.utc))
    assert p == 88.0


def test_project_extremum_returns_none_when_nothing_available():
    """Empty observation history + forecast that's entirely outside the window."""
    ss = _station_state()
    now = datetime(2026, 5, 28, 4, 0, tzinfo=ZoneInfo("America/Chicago"))
    # All forecast points are *after* window_end (window ends end of 5/27 local).
    forecast = _forecast(now.astimezone(timezone.utc), 72.0, 78.0)
    p = project_extremum(Metric.HIGH, forecast, ss, now_utc=now.astimezone(timezone.utc))
    assert p is None


# -- fair_probability uses calibrated residual -------------------------------

def test_fair_probability_uses_calibrated_residual_when_config_provided():
    """A tighter residual (0.5°F vs 2.0°F default) shifts the margin enough
    to change which discrete probability bucket the BRACKET_HIT case lands in.

    Setup: HIGH market at 79-80°, forecast max 80.5°F, observed max 79.5°F
    (inside bracket → BRACKET_HIT_VULNERABLE). Escape threshold is 81°F.

      With 2.0°F default: margin = 80.5 + 2.0 - 81 = +1.5 → p=0.35 bucket
      With 0.5°F tight:   margin = 80.5 + 0.5 - 81 = +0.0 → p=0.55 bucket
    """
    contract = _contract(Metric.HIGH, Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0))
    ss = _station_state(running_max=79.5)
    now = ss.window_start + timedelta(hours=14)
    forecast = _forecast(now.astimezone(timezone.utc), 80.5, 79.0, 76.0)

    # Default — no config → 2.0°F residual → p=0.35 → (0.27, 0.43)
    lo_default, hi_default = fair_probability(
        contract, ss, ContractState.BRACKET_HIT_VULNERABLE,
        forecast, now_utc=now.astimezone(timezone.utc),
    )
    assert round((lo_default + hi_default) / 2, 2) == 0.35

    # Calibrated — KHOU/high residual 0.5°F → p=0.55 → (0.47, 0.63)
    cfg = RankerConfig.default()
    cfg.forecast_residuals[residual_key("KHOU", "high")] = ForecastResidual.of(
        residual_f=0.5, n=50, applied=True,
    )
    lo_cal, hi_cal = fair_probability(
        contract, ss, ContractState.BRACKET_HIT_VULNERABLE,
        forecast, now_utc=now.astimezone(timezone.utc), config=cfg,
    )
    assert round((lo_cal + hi_cal) / 2, 2) == 0.55
    # Confirm the tightening actually moved the answer.
    assert lo_cal > lo_default


def test_fair_probability_unapplied_residual_no_ops():
    """A calibrated entry with applied=False (below sample threshold) is
    silently ignored — the engine output matches the no-config path exactly."""
    contract = _contract(Metric.HIGH, Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0))
    ss = _station_state(running_max=79.5)
    now = ss.window_start + timedelta(hours=14)
    forecast = _forecast(now.astimezone(timezone.utc), 80.5, 79.0, 76.0)

    cfg = RankerConfig.default()
    cfg.forecast_residuals[residual_key("KHOU", "high")] = ForecastResidual.of(
        residual_f=0.5, n=3, applied=False,
    )
    lo_default, hi_default = fair_probability(
        contract, ss, ContractState.BRACKET_HIT_VULNERABLE,
        forecast, now_utc=now.astimezone(timezone.utc),
    )
    lo_with_unapplied, hi_with_unapplied = fair_probability(
        contract, ss, ContractState.BRACKET_HIT_VULNERABLE,
        forecast, now_utc=now.astimezone(timezone.utc), config=cfg,
    )
    assert (lo_with_unapplied, hi_with_unapplied) == (lo_default, hi_default)


# -- derive_forecast_residuals -----------------------------------------------

def _eval_with_projection(
    ticker: str = "KXHIGHHOUSTON-26MAY27-B79-80",
    grade: str = "B",
    state: ContractState = ContractState.FORECAST_DEPENDENT,
) -> ContractEvaluation:
    contract = ParsedContract(
        market_ticker=ticker, event_ticker="KXHIGHHOUSTON-26MAY27",
        city_slug="HOUSTON", metric=Metric.HIGH,
        market_date=date(2026, 5, 27),
        bracket=Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0),
    )
    market = KalshiMarket(
        ticker=ticker, event_ticker="KXHIGHHOUSTON-26MAY27",
        title="", yes_sub_title="", status="open", close_time=None,
        yes_bid=40, yes_ask=42, no_bid=58, no_ask=60,
        last_price=None, volume=10, open_interest=100,
    )
    return ContractEvaluation(
        contract=contract, market=market, state=state, reason="forecast",
        fair_prob_low=0.30, fair_prob_high=0.50,
        yes_ask_cents=42, no_ask_cents=60,
        edge_yes=-0.02, edge_no=0.00, grade=grade, notes=[],
    )


@pytest.fixture
def store(tmp_path: Path):
    s = SnapshotStore(tmp_path / "test.db")
    yield s
    s.close()


def test_derive_forecast_residuals_no_op_on_empty_store(store: SnapshotStore):
    """Per-station residual calibration starts as a no-op — when the store
    has no settled snapshots with projections, the function returns empty
    dicts and the engine keeps using the 2.0°F default."""
    residuals, report = derive_forecast_residuals(store)
    assert residuals == {}
    assert report == []


def test_derive_forecast_residuals_below_threshold_marks_unapplied(store: SnapshotStore):
    """Three settled days at KHOU is below MIN_N_PER_RESIDUAL — the entry is
    captured for the audit trail but `applied=False` so the engine ignores it."""
    # Three siblings on the same day would dedupe to one — use three distinct
    # days to actually exercise the N=3 path.
    base = date(2026, 5, 27)
    for i, projected in enumerate([85.0, 86.0, 84.0]):
        day = base + timedelta(days=i)
        ticker = f"KXHIGHHOUSTON-{i}-B79-80"
        event = f"KXHIGHHOUSTON-{i}"
        e = _eval_with_projection(ticker=ticker)
        e.contract = ParsedContract(
            market_ticker=ticker, event_ticker=event,
            city_slug="HOUSTON", metric=Metric.HIGH, market_date=day,
            bracket=e.contract.bracket,
        )
        store.record_scan(
            evaluations=[e],
            station_state_map={
                ticker: {
                    "station_icao": "KHOU", "cli_product": "CLIHOU",
                    "source_provenance": "resolver", "regime": "clear_and_dry",
                    "running_max_f": projected - 1, "running_min_f": None,
                    "projected_extremum_f": projected,
                    "cli_report_date": None, "cli_max_f": None, "cli_min_f": None,
                },
            },
        )
        _settle(store, ticker, event, day, cli_value_f=85.5)

    residuals, report = derive_forecast_residuals(store)
    key = residual_key("KHOU", "high")
    assert key in residuals
    assert residuals[key].n_samples == 3
    assert residuals[key].applied is False
    assert len(report) == 1
    assert report[0].n_settled == 3
    assert "below threshold" in report[0].note


def test_derive_forecast_residuals_applies_when_enough_samples(store: SnapshotStore):
    """With N >= MIN_N_PER_RESIDUAL distinct days, the median absolute
    residual is computed and `applied=True`. The config's lookup then returns
    the calibrated value instead of the 2.0°F default."""
    base = date(2026, 5, 1)
    # Each day: projection = realized + i*0.1 (so residuals are 0.0, 0.1, 0.2, ...).
    n_days = MIN_N_PER_RESIDUAL + 5
    for i in range(n_days):
        day = base + timedelta(days=i)
        realized = 85.0
        projected = realized + (i * 0.1)
        ticker = f"KXHIGHHOUSTON-26MAY-D{i}"
        e = _eval_with_projection(ticker=ticker)
        # Override market_date on the contract for this day.
        e.contract = ParsedContract(
            market_ticker=ticker, event_ticker=f"KXHIGHHOUSTON-{i}",
            city_slug="HOUSTON", metric=Metric.HIGH, market_date=day,
            bracket=e.contract.bracket,
        )
        store.record_scan(
            evaluations=[e],
            station_state_map={
                ticker: {
                    "station_icao": "KHOU", "cli_product": "CLIHOU",
                    "source_provenance": "resolver", "regime": "clear_and_dry",
                    "running_max_f": realized, "running_min_f": None,
                    "projected_extremum_f": projected,
                    "cli_report_date": None, "cli_max_f": None, "cli_min_f": None,
                },
            },
        )
        _settle(store, ticker, f"KXHIGHHOUSTON-{i}", day, cli_value_f=realized)

    residuals, report = derive_forecast_residuals(store)
    key = residual_key("KHOU", "high")
    assert key in residuals
    assert residuals[key].n_samples == n_days
    assert residuals[key].applied is True
    # Median |residual| of [0.0, 0.1, 0.2, ..., (n-1)*0.1] ≈ (n-1)*0.05.
    expected_median = round((n_days - 1) * 0.05, 1)
    assert round(residuals[key].residual_f, 1) == expected_median


def test_derive_forecast_residuals_counts_unfillable_market_days(store: SnapshotStore):
    """Regression for the Codex/Copilot finding on PR #4: residual calibration
    must not depend on whether the market had a tradable quote at scan time.

    `backtest()` drops snapshots whose bracket has no fillable yes_ask/no_ask,
    so routing through it would silently exclude valid settled days in thin
    markets. The fix uses `get_settlement()` directly — this test creates
    a settled snapshot with explicit None prices and asserts the residual
    is still computed.
    """
    day = date(2026, 5, 27)
    ticker = "KXHIGHHOUSTON-THIN-B79-80"
    contract = ParsedContract(
        market_ticker=ticker, event_ticker="KXHIGHHOUSTON-THIN",
        city_slug="HOUSTON", metric=Metric.HIGH, market_date=day,
        bracket=Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0),
    )
    # Build an evaluation with NO usable price on either side — backtest()
    # would skip this snapshot entirely.
    unpriced = KalshiMarket(
        ticker=ticker, event_ticker="KXHIGHHOUSTON-THIN",
        title="", yes_sub_title="", status="open", close_time=None,
        yes_bid=None, yes_ask=None, no_bid=None, no_ask=None,
        last_price=None, volume=0, open_interest=0,
    )
    eval_ = ContractEvaluation(
        contract=contract, market=unpriced,
        state=ContractState.FORECAST_DEPENDENT, reason="thin",
        fair_prob_low=0.30, fair_prob_high=0.50,
        yes_ask_cents=None, no_ask_cents=None,
        edge_yes=None, edge_no=None, grade="D", notes=[],
    )
    store.record_scan(
        evaluations=[eval_],
        station_state_map={
            ticker: {
                "station_icao": "KHOU", "cli_product": "CLIHOU",
                "source_provenance": "resolver", "regime": "clear_and_dry",
                "running_max_f": 84.0, "running_min_f": None,
                "projected_extremum_f": 85.0,
                "cli_report_date": None, "cli_max_f": None, "cli_min_f": None,
            },
        },
    )
    _settle(store, ticker, "KXHIGHHOUSTON-THIN", day, cli_value_f=84.5)

    residuals, _ = derive_forecast_residuals(store)
    key = residual_key("KHOU", "high")
    # Old (backtest-gated) implementation: residuals would be {} here.
    # New implementation: the settled day counts.
    assert key in residuals
    assert residuals[key].n_samples == 1


def test_derive_forecast_residuals_dedupes_same_day_siblings(store: SnapshotStore):
    """Multiple bracket contracts for the same (station, metric, market_date)
    share the same projection and the same realized value — counting each one
    would inflate N and double-weight that day. The dedup key is per day."""
    day = date(2026, 5, 27)
    for sibling in ("B79-80", "B80-81", "B81-82"):
        ticker = f"KXHIGHHOUSTON-26MAY27-{sibling}"
        e = _eval_with_projection(ticker=ticker)
        store.record_scan(
            evaluations=[e],
            station_state_map={
                ticker: {
                    "station_icao": "KHOU", "cli_product": "CLIHOU",
                    "source_provenance": "resolver", "regime": "clear_and_dry",
                    "running_max_f": 84.0, "running_min_f": None,
                    "projected_extremum_f": 85.0,
                    "cli_report_date": None, "cli_max_f": None, "cli_min_f": None,
                },
            },
        )
        _settle(store, ticker, "KXHIGHHOUSTON-26MAY27", day, cli_value_f=84.5)

    residuals, _ = derive_forecast_residuals(store)
    key = residual_key("KHOU", "high")
    # 3 siblings, 1 day → N=1 (deduped).
    assert residuals[key].n_samples == 1
