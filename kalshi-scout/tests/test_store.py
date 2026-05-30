"""Tests for the V0.7 SQLite snapshot store, settlement derivation,
replay verification, and backtester.

These tests are offline — they create temporary SQLite databases per test.
They exercise the full snapshot -> settlement -> backtest -> replay loop
that activates AGENTS.md invariants D1 and D2.
"""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from kalshi_scout.models import (
    Bracket,
    BracketKind,
    ContractEvaluation,
    ContractState,
    KalshiMarket,
    Metric,
    ParsedContract,
)
import click

from kalshi_scout.cli import _parse_utc
from kalshi_scout.store import (
    SnapshotStore,
    backtest,
    replay,
    settlement_from_cli,
)


def _eval(
    ticker: str = "KXHIGHHOUSTON-26MAY27-B79-80",
    state: ContractState = ContractState.LOCKED_YES,
    bracket: Bracket | None = None,
    metric: Metric = Metric.HIGH,
    yes_ask: int | None = 71,
    fair_lo: float = 0.97,
    fair_hi: float = 0.99,
    grade: str = "A+",
) -> ContractEvaluation:
    bracket = bracket or Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0)
    contract = ParsedContract(
        market_ticker=ticker,
        event_ticker="KXHIGHHOUSTON-26MAY27",
        city_slug="HOUSTON",
        metric=metric,
        market_date=date(2026, 5, 27),
        bracket=bracket,
    )
    market = KalshiMarket(
        ticker=ticker,
        event_ticker="KXHIGHHOUSTON-26MAY27",
        title="",
        yes_sub_title="",
        status="open",
        close_time=None,
        yes_bid=yes_ask - 1 if yes_ask else None,
        yes_ask=yes_ask,
        no_bid=(100 - yes_ask - 1) if yes_ask else None,
        no_ask=(100 - yes_ask) if yes_ask else None,
        last_price=None, volume=10, open_interest=100,
    )
    return ContractEvaluation(
        contract=contract,
        market=market,
        state=state,
        reason="test",
        fair_prob_low=fair_lo, fair_prob_high=fair_hi,
        yes_ask_cents=yes_ask,
        no_ask_cents=(100 - yes_ask) if yes_ask else None,
        edge_yes=fair_lo - (yes_ask / 100.0) if yes_ask else None,
        edge_no=None,
        grade=grade,
        notes=["settlement: KHOU via resolver"],
    )


@pytest.fixture
def store(tmp_path: Path) -> SnapshotStore:
    s = SnapshotStore(tmp_path / "test.db")
    yield s
    s.close()


# -- Schema + basic round-trip ----------------------------------------------

def test_store_init_creates_schema(tmp_path: Path):
    s = SnapshotStore(tmp_path / "fresh.db")
    # Re-opening should not fail (IF NOT EXISTS used throughout).
    s.close()
    s2 = SnapshotStore(tmp_path / "fresh.db")
    s2.close()


def test_record_scan_persists_and_round_trips(store: SnapshotStore):
    scan_id = store.record_scan(
        evaluations=[_eval()],
        station_state_map={
            "KXHIGHHOUSTON-26MAY27-B79-80": {
                "station_icao": "KHOU",
                "cli_product": "CLIHOU",
                "source_provenance": "resolver",
                "running_max_f": 79.0,
                "running_min_f": 70.0,
                "cli_report_date": None,
                "cli_max_f": None,
                "cli_min_f": None,
            }
        },
    )
    assert scan_id

    rows = store.query_snapshots()
    assert len(rows) == 1
    row = rows[0]
    assert row.market_ticker == "KXHIGHHOUSTON-26MAY27-B79-80"
    assert row.grade == "A+"
    assert row.state == ContractState.LOCKED_YES.value
    assert row.running_max_f == 79.0
    assert row.station_icao == "KHOU"
    assert row.cli_product == "CLIHOU"
    assert row.source_provenance == "resolver"
    assert row.notes == ["settlement: KHOU via resolver"]


def test_query_snapshots_filters_by_grade_and_date(store: SnapshotStore):
    # Two snapshots at different grades
    store.record_scan(evaluations=[_eval(grade="A+")])
    store.record_scan(evaluations=[_eval(grade="C")])
    a_rows = store.query_snapshots(min_grade="A")
    c_rows = store.query_snapshots(min_grade="C")
    assert len(a_rows) == 1
    assert a_rows[0].grade == "A+"
    # min_grade=C should include both
    assert len(c_rows) == 2


# -- Ticker-history query + retention ----------------------------------------

def test_query_snapshots_filters_by_ticker(store: SnapshotStore):
    """Single-ticker history query — the use case that motivated #2."""
    store.record_scan(evaluations=[_eval(ticker="KXHIGHHOUSTON-26MAY27-B79-80")])
    store.record_scan(evaluations=[_eval(ticker="KXHIGHHOUSTON-26MAY27-B80-81")])
    store.record_scan(evaluations=[_eval(ticker="KXHIGHHOUSTON-26MAY27-B79-80")])
    rows = store.query_snapshots(market_ticker="KXHIGHHOUSTON-26MAY27-B79-80")
    assert len(rows) == 2
    assert {r.market_ticker for r in rows} == {"KXHIGHHOUSTON-26MAY27-B79-80"}


def test_query_snapshots_filters_by_since_until(store: SnapshotStore):
    early = datetime(2026, 5, 27, 10, 0, tzinfo=timezone.utc)
    mid = datetime(2026, 5, 27, 14, 0, tzinfo=timezone.utc)
    late = datetime(2026, 5, 27, 18, 0, tzinfo=timezone.utc)
    store.record_scan(evaluations=[_eval()], scanned_at=early)
    store.record_scan(evaluations=[_eval()], scanned_at=mid)
    store.record_scan(evaluations=[_eval()], scanned_at=late)

    # since alone
    after_noon = store.query_snapshots(
        since=datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc),
    )
    assert len(after_noon) == 2

    # since + until window
    window = store.query_snapshots(
        since=datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc),
        until=datetime(2026, 5, 27, 16, 0, tzinfo=timezone.utc),
    )
    assert len(window) == 1
    assert window[0].scanned_at_utc == mid


def test_count_snapshots_total_and_before(store: SnapshotStore):
    early = datetime(2026, 5, 27, 10, 0, tzinfo=timezone.utc)
    late = datetime(2026, 5, 27, 18, 0, tzinfo=timezone.utc)
    store.record_scan(evaluations=[_eval()], scanned_at=early)
    store.record_scan(evaluations=[_eval()], scanned_at=late)

    assert store.count_snapshots() == 2
    assert store.count_snapshots(
        before=datetime(2026, 5, 27, 12, 0, tzinfo=timezone.utc),
    ) == 1


def test_prune_snapshots_deletes_old_rows(store: SnapshotStore):
    # Microseconds zeroed so the stored ISO timestamp round-trips exactly.
    now = datetime.now(timezone.utc).replace(microsecond=0)
    old = now - timedelta(days=45)
    recent = now - timedelta(days=5)
    store.record_scan(evaluations=[_eval()], scanned_at=old)
    store.record_scan(evaluations=[_eval()], scanned_at=recent)

    deleted = store.prune_snapshots(before=now - timedelta(days=30))
    assert deleted == 1
    remaining = store.query_snapshots()
    assert len(remaining) == 1
    assert remaining[0].scanned_at_utc == recent


def test_prune_snapshots_preserves_kept_grades(store: SnapshotStore):
    """`keep_grades` lets us prune noise while keeping the calibration loop's
    settled history intact — A+/A picks drive the next config iteration."""
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=45)
    store.record_scan(evaluations=[_eval(grade="A+")], scanned_at=old)
    store.record_scan(evaluations=[_eval(grade="B")], scanned_at=old)
    store.record_scan(evaluations=[_eval(grade="D")], scanned_at=old)

    deleted = store.prune_snapshots(
        before=now - timedelta(days=30),
        keep_grades=("A+", "A"),
    )
    assert deleted == 2  # B and D pruned, A+ kept
    remaining = {r.grade for r in store.query_snapshots()}
    assert remaining == {"A+"}


def test_prune_snapshots_is_noop_when_nothing_qualifies(store: SnapshotStore):
    now = datetime.now(timezone.utc)
    store.record_scan(evaluations=[_eval()], scanned_at=now - timedelta(hours=1))
    deleted = store.prune_snapshots(before=now - timedelta(days=30))
    assert deleted == 0
    assert len(store.query_snapshots()) == 1


def test_parse_utc_accepts_date_and_datetime_forms():
    """The `--since` / `--until` CLI options accept multiple shorthand forms."""
    expected = datetime(2026, 5, 29, 0, 0, tzinfo=timezone.utc)
    assert _parse_utc("2026-05-29") == expected

    with_hour = datetime(2026, 5, 29, 12, 30, tzinfo=timezone.utc)
    assert _parse_utc("2026-05-29T12:30") == with_hour
    assert _parse_utc("2026-05-29 12:30") == with_hour
    assert _parse_utc("2026-05-29T12:30:00") == with_hour


def test_parse_utc_rejects_garbage():
    with pytest.raises(click.BadParameter):
        _parse_utc("not a date")


# -- Settlement derivation ---------------------------------------------------

def test_settlement_from_cli_resolves_yes_for_max_in_bracket():
    bracket = Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0)
    s = settlement_from_cli(
        market_ticker="KXHIGHHOUSTON-26MAY27-B79-80",
        event_ticker="KXHIGHHOUSTON-26MAY27",
        market_date=date(2026, 5, 27),
        city_slug="HOUSTON", metric=Metric.HIGH, bracket=bracket,
        station_icao="KHOU", cli_product="CLIHOU",
        cli_report_date=date(2026, 5, 27),
        cli_value_f=80.0,
    )
    assert s.resolved_yes is True


def test_settlement_from_cli_resolves_no_above_bracket():
    bracket = Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0)
    s = settlement_from_cli(
        market_ticker="X", event_ticker="X", market_date=date(2026, 5, 27),
        city_slug="HOUSTON", metric=Metric.HIGH, bracket=bracket,
        station_icao="KHOU", cli_product="CLIHOU",
        cli_report_date=date(2026, 5, 27), cli_value_f=81.0,
    )
    assert s.resolved_yes is False


def test_settlement_from_cli_lte_low_market():
    # "70° or below" LOW market: yes if min <= 70
    bracket = Bracket(BracketKind.LTE, lo=None, hi=70.0)
    yes = settlement_from_cli(
        market_ticker="X", event_ticker="X", market_date=date(2026, 5, 28),
        city_slug="HOUSTON", metric=Metric.LOW, bracket=bracket,
        station_icao="KHOU", cli_product="CLIHOU",
        cli_report_date=date(2026, 5, 28), cli_value_f=68.0,
    )
    no = settlement_from_cli(
        market_ticker="Y", event_ticker="Y", market_date=date(2026, 5, 28),
        city_slug="HOUSTON", metric=Metric.LOW, bracket=bracket,
        station_icao="KHOU", cli_product="CLIHOU",
        cli_report_date=date(2026, 5, 28), cli_value_f=72.0,
    )
    assert yes.resolved_yes is True
    assert no.resolved_yes is False


def test_record_settlement_round_trips(store: SnapshotStore):
    bracket = Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0)
    s = settlement_from_cli(
        market_ticker="KXHIGHHOUSTON-26MAY27-B79-80",
        event_ticker="KXHIGHHOUSTON-26MAY27",
        market_date=date(2026, 5, 27),
        city_slug="HOUSTON", metric=Metric.HIGH, bracket=bracket,
        station_icao="KHOU", cli_product="CLIHOU",
        cli_report_date=date(2026, 5, 27), cli_value_f=80.0,
    )
    store.record_settlement(s)
    fetched = store.get_settlement("KXHIGHHOUSTON-26MAY27-B79-80")
    assert fetched is not None
    assert fetched.resolved_yes is True
    assert fetched.cli_value_f == 80.0


# -- Backtest ----------------------------------------------------------------

def test_backtest_winning_locked_yes_pays_out(store: SnapshotStore):
    # A+ LOCKED_YES alert at 71c Yes ask, settlement resolves Yes -> +29c
    store.record_scan(evaluations=[_eval(yes_ask=71, grade="A+",
                                          state=ContractState.LOCKED_YES)])
    bracket = Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0)
    store.record_settlement(settlement_from_cli(
        market_ticker="KXHIGHHOUSTON-26MAY27-B79-80",
        event_ticker="KXHIGHHOUSTON-26MAY27",
        market_date=date(2026, 5, 27),
        city_slug="HOUSTON", metric=Metric.HIGH, bracket=bracket,
        station_icao="KHOU", cli_product="CLIHOU",
        cli_report_date=date(2026, 5, 27), cli_value_f=80.0,
    ))
    rows = backtest(store, min_grade="A")
    assert len(rows) == 1
    assert rows[0].side == "yes"
    assert rows[0].price_paid_cents == 71
    assert rows[0].won is True
    assert rows[0].pnl_cents == 29


def test_backtest_losing_dead_no_takes_no_side(store: SnapshotStore):
    """DEAD_NO alert: take No side; if settlement actually resolves No,
    we win the No side payout."""
    store.record_scan(evaluations=[_eval(
        ticker="KXHIGHHOUSTON-26MAY27-LTE78",
        bracket=Bracket(BracketKind.LTE, lo=None, hi=78.0),
        state=ContractState.DEAD_NO,
        yes_ask=15,  # cheap Yes
        fair_lo=0.0, fair_hi=0.02,  # fair mid 0.01 -> we take No side
        grade="A",
    )])
    store.record_settlement(settlement_from_cli(
        market_ticker="KXHIGHHOUSTON-26MAY27-LTE78",
        event_ticker="KXHIGHHOUSTON-26MAY27",
        market_date=date(2026, 5, 27),
        city_slug="HOUSTON", metric=Metric.HIGH,
        bracket=Bracket(BracketKind.LTE, lo=None, hi=78.0),
        station_icao="KHOU", cli_product="CLIHOU",
        cli_report_date=date(2026, 5, 27), cli_value_f=80.0,  # > 78 -> resolves No
    ))
    rows = backtest(store, min_grade="A")
    assert len(rows) == 1
    assert rows[0].side == "no"
    assert rows[0].resolved_yes is False
    assert rows[0].won is True
    # Bought No at 85c (100-15), pays 100c -> +15c
    assert rows[0].price_paid_cents == 85
    assert rows[0].pnl_cents == 15


def test_backtest_skips_snapshots_without_settlement(store: SnapshotStore):
    store.record_scan(evaluations=[_eval(grade="A+")])
    rows = backtest(store, min_grade="A")
    assert rows == []


def test_backtest_filters_by_grade(store: SnapshotStore):
    store.record_scan(evaluations=[_eval(grade="C")])
    store.record_settlement(settlement_from_cli(
        market_ticker="KXHIGHHOUSTON-26MAY27-B79-80",
        event_ticker="KXHIGHHOUSTON-26MAY27",
        market_date=date(2026, 5, 27),
        city_slug="HOUSTON", metric=Metric.HIGH,
        bracket=Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0),
        station_icao="KHOU", cli_product="CLIHOU",
        cli_report_date=date(2026, 5, 27), cli_value_f=80.0,
    ))
    a_rows = backtest(store, min_grade="A")
    c_rows = backtest(store, min_grade="C")
    assert a_rows == []
    assert len(c_rows) == 1


# -- Replay (invariant D1) ---------------------------------------------------

def test_replay_matches_when_engine_unchanged(store: SnapshotStore):
    """A LOCKED_YES snapshot on a GTE bracket with running_max past the
    strike must replay to the same state + grade. Bonus: this test also
    surfaces the kind of inconsistency replay is designed to catch
    (LOCKED_YES on a BETWEEN with max inside the bracket would be
    BRACKET_HIT_VULNERABLE, not LOCKED_YES)."""
    store.record_scan(
        evaluations=[_eval(
            bracket=Bracket(BracketKind.GTE, lo=78.0, hi=None),
            state=ContractState.LOCKED_YES,
            grade="A+",
        )],
        station_state_map={
            "KXHIGHHOUSTON-26MAY27-B79-80": {
                "station_icao": "KHOU",
                "cli_product": "CLIHOU",
                "source_provenance": "resolver",
                "running_max_f": 79.0,  # >= 78 -> LOCKED_YES against GTE 78
                "running_min_f": None,
                "cli_report_date": None,
                "cli_max_f": None,
                "cli_min_f": None,
            }
        },
    )
    snap = store.query_snapshots()[0]
    result = replay(store, snap.id)
    assert result.matches is True, f"drift: {result.drift_reason}"
    assert result.replayed_state == ContractState.LOCKED_YES.value
    assert result.replayed_grade == "A+"


def test_replay_detects_state_drift_when_running_max_changes(store: SnapshotStore):
    """If the snapshot's running_max would have produced a different state
    under the current engine, replay must report drift.

    Built-in regression: we manually corrupt a snapshot's running_max_f to
    simulate engine drift, then assert replay catches it.
    """
    store.record_scan(
        evaluations=[_eval(state=ContractState.LOCKED_YES, grade="A+",
                           bracket=Bracket(BracketKind.GTE, lo=78.0, hi=None))],
        station_state_map={
            "KXHIGHHOUSTON-26MAY27-B79-80": {
                "station_icao": "KHOU", "cli_product": "CLIHOU",
                "source_provenance": "resolver",
                "running_max_f": 79.0,  # actually drives LOCKED_YES against GTE 78
                "running_min_f": None,
                "cli_report_date": None, "cli_max_f": None, "cli_min_f": None,
            }
        },
    )
    # Corrupt the stored running_max to a value where the bracket would
    # produce NOT_REACHED instead of LOCKED_YES.
    store._conn.execute(
        "UPDATE snapshots SET running_max_f = 70.0 WHERE 1=1"
    )
    snap = store.query_snapshots()[0]
    result = replay(store, snap.id)
    assert result.matches is False
    assert "state" in (result.drift_reason or "")


def test_replay_returns_not_found_for_missing_id(store: SnapshotStore):
    result = replay(store, 99999)
    assert result.matches is False
    assert "not found" in (result.drift_reason or "")
