"""Tests for V0.8 calibration report.

Builds a small synthetic history in a SnapshotStore, runs calibrate(),
verifies the per-grade math.
"""

from datetime import date
from pathlib import Path

import pytest

from kalshi_scout.calibrate import calibrate, report_to_dict
from kalshi_scout.models import (
    Bracket,
    BracketKind,
    ContractEvaluation,
    ContractState,
    KalshiMarket,
    Metric,
    ParsedContract,
)
from kalshi_scout.store import SnapshotStore, settlement_from_cli


def _eval(ticker: str, grade: str, state: ContractState, yes_ask: int,
          edge_yes: float, bracket: Bracket | None = None) -> ContractEvaluation:
    bracket = bracket or Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0)
    contract = ParsedContract(
        market_ticker=ticker, event_ticker="E",
        city_slug="HOUSTON", metric=Metric.HIGH,
        market_date=date(2026, 5, 27), bracket=bracket,
    )
    market = KalshiMarket(
        ticker=ticker, event_ticker="E",
        title="", yes_sub_title="", status="open", close_time=None,
        yes_bid=yes_ask - 1, yes_ask=yes_ask,
        no_bid=100 - yes_ask - 1, no_ask=100 - yes_ask,
        last_price=yes_ask, volume=10, open_interest=100,
    )
    return ContractEvaluation(
        contract=contract, market=market, state=state, reason="",
        fair_prob_low=0.95, fair_prob_high=0.99,
        yes_ask_cents=yes_ask, no_ask_cents=100 - yes_ask,
        edge_yes=edge_yes, edge_no=None,
        grade=grade, notes=[],
    )


def _record_winning_pair(store: SnapshotStore, ticker: str, grade: str, yes_ask: int):
    """Store a LOCKED_YES eval + settlement that resolves Yes."""
    store.record_scan([_eval(ticker, grade, ContractState.LOCKED_YES,
                             yes_ask=yes_ask, edge_yes=0.95 - yes_ask / 100.0)])
    store.record_settlement(settlement_from_cli(
        market_ticker=ticker, event_ticker="E",
        market_date=date(2026, 5, 27),
        city_slug="HOUSTON", metric=Metric.HIGH,
        bracket=Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0),
        station_icao="KHOU", cli_product="CLIHOU",
        cli_report_date=date(2026, 5, 27),
        cli_value_f=80.0,
    ))


def _record_losing_pair(store: SnapshotStore, ticker: str, grade: str, yes_ask: int):
    """Store a LOCKED_YES eval + settlement that resolves No (the alert
    was wrong about the state — drift / misclassification)."""
    store.record_scan([_eval(ticker, grade, ContractState.LOCKED_YES,
                             yes_ask=yes_ask, edge_yes=0.95 - yes_ask / 100.0)])
    store.record_settlement(settlement_from_cli(
        market_ticker=ticker, event_ticker="E",
        market_date=date(2026, 5, 27),
        city_slug="HOUSTON", metric=Metric.HIGH,
        bracket=Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0),
        station_icao="KHOU", cli_product="CLIHOU",
        cli_report_date=date(2026, 5, 27),
        cli_value_f=82.0,  # above bracket -> resolves No
    ))


@pytest.fixture
def store(tmp_path: Path) -> SnapshotStore:
    s = SnapshotStore(tmp_path / "cal.db")
    yield s
    s.close()


def test_calibrate_empty_store_reports_no_data(store: SnapshotStore):
    report = calibrate(store)
    assert report.total_snapshots == 0
    assert report.settled_snapshots == 0
    assert not report.has_any_data()
    # Every tier present with zeros.
    for tier in ("A+", "A", "B+", "B", "C", "D"):
        assert tier in report.stats_by_grade
        assert report.stats_by_grade[tier].n == 0


def test_calibrate_counts_unsettled_snapshots_in_total_only(store: SnapshotStore):
    """A snapshot without a settlement increments total_snapshots but no
    grade-tier's stats."""
    store.record_scan([_eval("X", "A+", ContractState.LOCKED_YES,
                             yes_ask=71, edge_yes=0.28)])
    report = calibrate(store)
    assert report.total_snapshots == 1
    assert report.settled_snapshots == 0
    assert report.stats_by_grade["A+"].n == 0


def test_calibrate_a_plus_winners(store: SnapshotStore):
    """Three A+ wins at yes_ask=71 -> each pays +29c. Hit rate 100%."""
    _record_winning_pair(store, "K1", "A+", 71)
    _record_winning_pair(store, "K2", "A+", 71)
    _record_winning_pair(store, "K3", "A+", 71)
    report = calibrate(store)
    s = report.stats_by_grade["A+"]
    assert s.n == 3
    assert s.n_unique_markets == 3
    assert s.wins == 3
    assert s.hit_rate == 1.0
    assert s.avg_pnl_c == 29.0
    assert s.total_pnl_c == 87


def test_calibrate_mixed_winners_and_losers(store: SnapshotStore):
    """Two A+ wins (+29c each), one A+ loss (-71c). Hit rate 67%, avg -4.3c."""
    _record_winning_pair(store, "K1", "A+", 71)
    _record_winning_pair(store, "K2", "A+", 71)
    _record_losing_pair(store, "K3", "A+", 71)
    report = calibrate(store)
    s = report.stats_by_grade["A+"]
    assert s.n == 3
    assert s.wins == 2
    assert abs(s.hit_rate - 2 / 3) < 1e-9
    # 2 * +29 + 1 * -71 = -13. Avg = -13/3 ≈ -4.33
    assert s.total_pnl_c == -13
    assert abs(s.avg_pnl_c - (-13.0 / 3.0)) < 1e-6


def test_calibrate_separates_grades(store: SnapshotStore):
    _record_winning_pair(store, "K1", "A+", 71)
    _record_winning_pair(store, "K2", "A", 80)
    _record_losing_pair(store, "K3", "B+", 90)
    report = calibrate(store)
    assert report.stats_by_grade["A+"].n == 1
    assert report.stats_by_grade["A"].n == 1
    assert report.stats_by_grade["B+"].n == 1
    assert report.stats_by_grade["A+"].total_pnl_c == 29   # 100 - 71
    assert report.stats_by_grade["A"].total_pnl_c == 20    # 100 - 80
    assert report.stats_by_grade["B+"].total_pnl_c == -90  # bought at 90, lost


def test_calibrate_median_edge_within_grade(store: SnapshotStore):
    """Median of stored edges should be computable per grade tier."""
    _record_winning_pair(store, "K1", "A+", 71)  # edge 0.24
    _record_winning_pair(store, "K2", "A+", 65)  # edge 0.30
    _record_winning_pair(store, "K3", "A+", 80)  # edge 0.15
    report = calibrate(store)
    s = report.stats_by_grade["A+"]
    # Median of [0.24, 0.30, 0.15] = 0.24
    assert s.median_edge is not None
    assert abs(s.median_edge - 0.24) < 1e-6


def test_report_to_dict_is_json_serializable(store: SnapshotStore):
    _record_winning_pair(store, "K1", "A+", 71)
    report = calibrate(store)
    d = report_to_dict(report)
    import json as _json
    s = _json.dumps(d)
    assert "A+" in s
    parsed = _json.loads(s)
    assert parsed["by_grade"]["A+"]["n"] == 1
    assert parsed["by_grade"]["A+"]["hit_rate"] == 1.0
