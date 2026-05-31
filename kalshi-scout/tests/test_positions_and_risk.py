"""Tests for V1.0 position tracking + risk aggregation."""

from datetime import date
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
from kalshi_scout.risk import aggregate_risk, enrich_positions
from kalshi_scout.store import SnapshotStore


@pytest.fixture
def store(tmp_path: Path) -> SnapshotStore:
    s = SnapshotStore(tmp_path / "risk.db")
    yield s
    s.close()


def _snapshot_for(store: SnapshotStore, market_ticker: str,
                  city: str = "HOUSTON", regime: str = "clear_and_dry",
                  bracket: Bracket | None = None) -> None:
    """Persist a single snapshot so the enricher has city/regime context."""
    bracket = bracket or Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0)
    contract = ParsedContract(
        market_ticker=market_ticker, event_ticker=market_ticker.rsplit("-", 1)[0],
        city_slug=city, metric=Metric.HIGH,
        market_date=date(2026, 5, 27), bracket=bracket,
    )
    market = KalshiMarket(
        ticker=market_ticker, event_ticker=contract.event_ticker,
        title="", yes_sub_title="", status="open", close_time=None,
        yes_bid=70, yes_ask=71, no_bid=29, no_ask=30,
        last_price=71, volume=10, open_interest=100,
    )
    eval_ = ContractEvaluation(
        contract=contract, market=market,
        state=ContractState.BRACKET_HIT_VULNERABLE, reason="",
        fair_prob_low=0.6, fair_prob_high=0.7,
        yes_ask_cents=71, no_ask_cents=29,
        edge_yes=None, edge_no=None, grade="B", notes=[],
    )
    store.record_scan([eval_], station_state_map={
        market_ticker: {"regime": regime, "station_icao": "KHOU",
                        "cli_product": "CLIHOU", "source_provenance": "resolver"}
    })


# -- Position store ----------------------------------------------------------

def test_add_position_round_trips(store: SnapshotStore):
    pid = store.add_position(
        market_ticker="K1", event_ticker="E1",
        side="yes", size_contracts=100, avg_price_cents=71,
    )
    open_positions = store.query_positions(open_only=True)
    assert len(open_positions) == 1
    p = open_positions[0]
    assert p.id == pid
    assert p.market_ticker == "K1"
    assert p.size_contracts == 100
    assert p.avg_price_cents == 71
    assert p.cost_basis_cents == 7100
    assert p.is_open is True


def test_close_position_excludes_from_open_list(store: SnapshotStore):
    pid = store.add_position("K1", "E1", "yes", 50, 71)
    assert store.close_position(pid) is True
    assert store.query_positions(open_only=True) == []
    assert len(store.query_positions(open_only=False)) == 1
    # Closing twice is a no-op (already closed).
    assert store.close_position(pid) is False


def test_close_position_captures_exit_price_for_pnl(store: SnapshotStore):
    """Exit price recorded at close → realized_pnl_cents computable on read."""
    pid = store.add_position("K1", "E1", "yes", size_contracts=10, avg_price_cents=30)
    assert store.close_position(pid, at_price_cents=100) is True   # settled YES win

    rows = store.query_positions(open_only=False)
    assert len(rows) == 1
    row = rows[0]
    assert row.closed_at_price_cents == 100
    # (exit - entry) × size = (100 - 30) × 10 = 700c.
    assert row.realized_pnl_cents == 700


def test_close_position_without_exit_price_leaves_pnl_unknown(store: SnapshotStore):
    """Backwards-compat: old close() calls and operators who don't pass
    --at-price should still work, just with realized_pnl_cents=None."""
    pid = store.add_position("K1", "E1", "yes", size_contracts=10, avg_price_cents=30)
    assert store.close_position(pid) is True
    row = store.query_positions(open_only=False)[0]
    assert row.closed_at_price_cents is None
    assert row.realized_pnl_cents is None


def test_realized_pnl_handles_losing_settlement(store: SnapshotStore):
    """Settled YES that lost → exit price is 0 → P&L = -cost_basis."""
    pid = store.add_position("K1", "E1", "yes", size_contracts=10, avg_price_cents=30)
    store.close_position(pid, at_price_cents=0)
    row = store.query_positions(open_only=False)[0]
    assert row.realized_pnl_cents == -300   # (0 - 30) × 10
    assert row.cost_basis_cents == 300


def test_add_position_validates_inputs(store: SnapshotStore):
    with pytest.raises(ValueError):
        store.add_position("K", "E", side="maybe", size_contracts=10, avg_price_cents=50)
    with pytest.raises(ValueError):
        store.add_position("K", "E", side="yes", size_contracts=0, avg_price_cents=50)
    with pytest.raises(ValueError):
        store.add_position("K", "E", side="yes", size_contracts=10, avg_price_cents=0)
    with pytest.raises(ValueError):
        store.add_position("K", "E", side="yes", size_contracts=10, avg_price_cents=100)


# -- Enrichment --------------------------------------------------------------

def test_enrich_picks_up_city_and_regime_from_snapshot(store: SnapshotStore):
    _snapshot_for(store, "K1", city="HOUSTON", regime="rain_cooled")
    store.add_position("K1", "E1", "yes", 50, 71)
    rows = enrich_positions(store, store.query_positions())
    assert rows[0].city_slug == "HOUSTON"
    assert rows[0].regime == "rain_cooled"
    assert rows[0].metric == "high"


def test_enrich_falls_back_to_unknown_when_no_snapshot(store: SnapshotStore):
    store.add_position("K_NEW", "E_NEW", "yes", 50, 71)
    rows = enrich_positions(store, store.query_positions())
    assert rows[0].city_slug is None
    assert rows[0].regime is None


# -- Risk aggregation --------------------------------------------------------

def test_risk_report_totals_match(store: SnapshotStore):
    _snapshot_for(store, "K1", city="HOUSTON")
    _snapshot_for(store, "K2", city="NYC")
    store.add_position("K1", "E1", "yes", 100, 71)  # cost 7100c
    store.add_position("K2", "E2", "no", 50, 30)    # cost 1500c
    report = aggregate_risk(store)
    assert report.total_open_positions == 2
    assert report.total_open_contracts == 150
    assert report.total_max_loss_cents == 7100 + 1500


def test_risk_buckets_by_city(store: SnapshotStore):
    _snapshot_for(store, "K1", city="HOUSTON")
    _snapshot_for(store, "K2", city="HOUSTON")
    _snapshot_for(store, "K3", city="NYC")
    store.add_position("K1", "E1", "yes", 50, 71)
    store.add_position("K2", "E2", "yes", 50, 71)
    store.add_position("K3", "E3", "yes", 50, 50)
    report = aggregate_risk(store)
    assert report.by_city["HOUSTON"].n_positions == 2
    assert report.by_city["HOUSTON"].total_max_loss_cents == 71 * 50 * 2
    assert report.by_city["NYC"].n_positions == 1


def test_risk_event_collision_detected(store: SnapshotStore):
    """Two YES positions across two brackets of the same event = collision."""
    _snapshot_for(store, "E-B79-80", city="HOUSTON",
                  bracket=Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0))
    _snapshot_for(store, "E-B81-82", city="HOUSTON",
                  bracket=Bracket(BracketKind.BETWEEN, lo=81.0, hi=82.0))
    store.add_position("E-B79-80", "E", "yes", 100, 71)  # cost 7100c
    store.add_position("E-B81-82", "E", "yes", 100, 30)  # cost 3000c
    report = aggregate_risk(store)
    assert len(report.event_collisions) == 1
    c = report.event_collisions[0]
    assert c.event_ticker == "E"
    assert len(c.yes_positions) == 2
    # Guaranteed loss: cheaper one (3000c) is the locked-in loss; we can
    # win at most the expensive one (7100c) so 3000c is guaranteed gone.
    assert c.guaranteed_loss_cents == 3000


def test_risk_no_collision_when_yes_and_no_in_same_event(store: SnapshotStore):
    """Yes + No on different brackets of the same event isn't a collision —
    they're hedged, not duplicated."""
    _snapshot_for(store, "E-B79-80", city="HOUSTON")
    _snapshot_for(store, "E-B81-82", city="HOUSTON",
                  bracket=Bracket(BracketKind.BETWEEN, lo=81.0, hi=82.0))
    store.add_position("E-B79-80", "E", "yes", 100, 71)
    store.add_position("E-B81-82", "E", "no", 100, 70)
    report = aggregate_risk(store)
    assert report.event_collisions == []


def test_risk_collision_with_three_yes_positions(store: SnapshotStore):
    """Three Yes across three brackets: we can win at most one (the most
    expensive); the other two are guaranteed dead."""
    for i, lo in enumerate((79, 81, 83)):
        _snapshot_for(store, f"E-B{lo}-{lo+1}",
                      bracket=Bracket(BracketKind.BETWEEN, lo=float(lo), hi=float(lo + 1)))
    store.add_position("E-B79-80", "E", "yes", 100, 20)  # cost 2000
    store.add_position("E-B81-82", "E", "yes", 100, 60)  # cost 6000
    store.add_position("E-B83-84", "E", "yes", 100, 30)  # cost 3000
    report = aggregate_risk(store)
    c = report.event_collisions[0]
    # Sorted by cost: [2000, 3000, 6000]. The 6000 can win; the other two die.
    # Guaranteed loss = 2000 + 3000 = 5000.
    assert c.guaranteed_loss_cents == 5000


def test_risk_closed_positions_excluded(store: SnapshotStore):
    pid = store.add_position("K1", "E1", "yes", 100, 71)
    store.close_position(pid)
    report = aggregate_risk(store)
    assert report.total_open_positions == 0
    assert report.total_max_loss_cents == 0
