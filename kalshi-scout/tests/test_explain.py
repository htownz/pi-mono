"""Tests for the `explain` CLI command's helpers.

`explain` runs the full pipeline for one market and dumps every intermediate.
The two helpers under test:

  - `_grade_derivation`: reproduces ranker._grade_value's reasoning in words.
  - `_explain_to_dict`: assembles the JSON-mode payload from all the
    intermediate objects.

The full command itself is integration-tested by running `kalshi-scout explain
<ticker>` against live APIs; here we just verify the building blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone

from kalshi_scout.cli import _explain_to_dict, _grade_derivation
from kalshi_scout.config import RankerConfig
from kalshi_scout.models import (
    Bracket,
    BracketKind,
    ContractEvaluation,
    ContractState,
    KalshiMarket,
    Metric,
    ParsedContract,
    Settlement,
    SettlementProvenance,
    Station,
    StationReading,
    StationState,
)


@dataclass
class _FakeRegime:
    """Minimal stand-in for RegimeReading — _explain_to_dict only reads .regime,
    .confidence, .reasoning, and .regime.value."""
    class _R:
        value = "clear_and_dry"
    regime = _R()
    confidence = 0.8
    reasoning = ("blue skies", "low humidity")


def _market(yes_ask=10, yes_bid=8) -> KalshiMarket:
    return KalshiMarket(
        ticker="KXHIGHTHOU-26MAY29-B94.5",
        event_ticker="KXHIGHTHOU-26MAY29",
        title="Will the maximum temperature be 94-95° on May 29, 2026?",
        yes_sub_title="94° to 95°",
        status="active",
        close_time=datetime(2026, 5, 30, 8, 0, tzinfo=timezone.utc),
        yes_bid=yes_bid, yes_ask=yes_ask,
        no_bid=100 - yes_ask, no_ask=100 - yes_bid,
        last_price=yes_ask, volume=1379, open_interest=909,
    )


def _contract() -> ParsedContract:
    return ParsedContract(
        market_ticker="KXHIGHTHOU-26MAY29-B94.5",
        event_ticker="KXHIGHTHOU-26MAY29",
        city_slug="HOUSTON",
        metric=Metric.HIGH,
        market_date=date(2026, 5, 29),
        bracket=Bracket(BracketKind.BETWEEN, lo=94.0, hi=95.0),
    )


def _station() -> Station:
    return Station(
        icao="KHOU", name="Houston Hobby Airport", city_slug="HOUSTON",
        tz="America/Chicago", cli_product="CLIHOU",
        latitude=29.65, longitude=-95.28,
    )


def _settlement(station: Station | None = None) -> Settlement:
    return Settlement(
        station=station,
        source_agency="National Weather Service",
        area_description="Houston",
        provenance=SettlementProvenance.RESOLVER,
        notes=("matched rules pattern KHOU",),
    )


def _station_state(running_max: float = 92.0) -> StationState:
    obs = [StationReading(
        observed_at=datetime(2026, 5, 29, 18, 0, tzinfo=timezone.utc),
        temperature_f=running_max,
    )]
    return StationState(
        station=_station(),
        market_date=date(2026, 5, 29),
        window_start=datetime(2026, 5, 29, 5, 0, tzinfo=timezone.utc),
        window_end=datetime(2026, 5, 30, 5, 0, tzinfo=timezone.utc),
        running_max_f=running_max,
        running_min_f=70.0,
        latest=obs[0],
        cli_report_date=None,
        cli_max_f=None,
        cli_min_f=None,
        observations=obs,
    )


# -- _grade_derivation -------------------------------------------------------

def test_grade_derivation_locked_yes_hits_high_cutoff():
    """edge_yes 0.12 ≥ 0.08 (default high_cutoff for locked_yes) → A+."""
    msg = _grade_derivation(
        ContractState.LOCKED_YES, edge_yes=0.12, edge_no=None,
        spread_cents=2, config=RankerConfig.default(),
    )
    assert "A+" in msg
    assert "0.120" in msg or "0.12" in msg


def test_grade_derivation_locked_yes_wide_spread_demotes_to_A():
    msg = _grade_derivation(
        ContractState.LOCKED_YES, edge_yes=0.15, edge_no=None,
        spread_cents=15, config=RankerConfig.default(),
    )
    assert "wide spread" in msg
    assert "A" in msg and "A+" not in msg


def test_grade_derivation_locked_yes_below_low_cutoff_is_B():
    msg = _grade_derivation(
        ContractState.LOCKED_YES, edge_yes=0.01, edge_no=None,
        spread_cents=2, config=RankerConfig.default(),
    )
    assert "B" in msg
    assert "below cutoff" in msg


def test_grade_derivation_locked_yes_missing_yes_ask_grades_F():
    msg = _grade_derivation(
        ContractState.LOCKED_YES, edge_yes=None, edge_no=None,
        spread_cents=None, config=RankerConfig.default(),
    )
    assert "F" in msg


def test_grade_derivation_forecast_dependent_uses_best_edge():
    """FORECAST_DEPENDENT picks the higher of edge_yes/edge_no."""
    msg = _grade_derivation(
        ContractState.FORECAST_DEPENDENT, edge_yes=0.04, edge_no=0.13,
        spread_cents=3, config=RankerConfig.default(),
    )
    # 0.13 ≥ 0.12 (high_cutoff for forecast_dependent) → B
    assert "B" in msg
    assert "0.130" in msg or "0.13" in msg


def test_grade_derivation_handles_no_state():
    msg = _grade_derivation(None, edge_yes=0.1, edge_no=0.1, spread_cents=2,
                            config=RankerConfig.default())
    assert "no state" in msg


# -- _explain_to_dict --------------------------------------------------------

def test_explain_to_dict_full_pipeline_payload():
    market = _market()
    contract = _contract()
    station = _station()
    settlement = _settlement(station)
    ss = _station_state()
    regime = _FakeRegime()
    eval_ = ContractEvaluation(
        contract=contract, market=market,
        state=ContractState.FORECAST_DEPENDENT,
        reason="hot day forecast",
        fair_prob_low=0.55, fair_prob_high=0.75,
        yes_ask_cents=10, no_ask_cents=90,
        edge_yes=0.55, edge_no=-0.45,
        grade="B", notes=["zero volume today"],
    )

    out = _explain_to_dict(
        market, contract, settlement, ss,
        forecast=[],
        regime_reading=regime,
        state_value=ContractState.FORECAST_DEPENDENT,
        state_reason="hot day forecast",
        fair_lo=0.55, fair_hi=0.75,
        eval_=eval_, config=RankerConfig.default(),
        now_utc=datetime(2026, 5, 29, 12, 0, tzinfo=timezone.utc),
    )

    assert out["market"]["ticker"] == "KXHIGHTHOU-26MAY29-B94.5"
    assert out["market"]["yes_ask"] == 10
    assert out["parsed_contract"]["city_slug"] == "HOUSTON"
    assert out["parsed_contract"]["bracket"]["kind"] == "between"
    assert out["parsed_contract"]["bracket"]["lo"] == 94.0
    assert out["settlement"]["station_icao"] == "KHOU"
    assert out["settlement"]["provenance"] == "resolver"
    assert out["station_state"]["running_max_f"] == 92.0
    assert out["station_state"]["n_observations"] == 1
    assert out["regime"]["regime"] == "clear_and_dry"
    assert out["state_machine"]["state"] == "forecast_dependent"
    assert out["fair_probability"]["mid"] == 0.65
    assert out["grade"]["grade"] == "B"
    assert "zero volume today" in out["grade"]["notes"]


def test_explain_to_dict_handles_failed_parse():
    """When parse_market returns None, the parsed_contract / station_state /
    grade fields are all None — payload must not crash."""
    market = _market()
    out = _explain_to_dict(
        market, contract=None, settlement=None, station_state=None,
        forecast=[], regime_reading=None,
        state_value=None, state_reason="",
        fair_lo=None, fair_hi=None, eval_=None,
        config=RankerConfig.default(),
        now_utc=datetime(2026, 5, 29, 12, 0, tzinfo=timezone.utc),
    )
    assert out["market"]["ticker"] == "KXHIGHTHOU-26MAY29-B94.5"
    assert out["parsed_contract"] is None
    assert out["settlement"] is None
    assert out["station_state"] is None
    assert out["regime"] is None
    assert out["state_machine"] is None
    assert out["fair_probability"] is None
    assert out["grade"] is None


def test_explain_to_dict_handles_unverified_settlement():
    """parser succeeded but the resolver couldn't pin a station — settlement
    is present but station=None."""
    market = _market()
    contract = _contract()
    settlement = Settlement(
        station=None,
        source_agency="?", area_description="",
        provenance=SettlementProvenance.UNVERIFIED,
        notes=("no rules text",),
    )
    out = _explain_to_dict(
        market, contract, settlement, station_state=None,
        forecast=[], regime_reading=None,
        state_value=None, state_reason="",
        fair_lo=None, fair_hi=None, eval_=None,
        config=RankerConfig.default(),
        now_utc=datetime(2026, 5, 29, 12, 0, tzinfo=timezone.utc),
    )
    assert out["settlement"]["station_icao"] is None
    assert out["settlement"]["provenance"] == "unverified"
    assert "no rules text" in out["settlement"]["notes"]
    assert out["station_state"] is None
    assert out["grade"] is None
