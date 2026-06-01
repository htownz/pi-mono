"""Tests for `PositionMonitor` — take-profit + cut-loss on open positions.

The monitor walks `store.query_positions(open_only=True)` each scan and
closes positions when:

  1. The current bid for our side >= take_profit_bid_cents (default 95c)
  2. The snapshot's state flipped against us (NO + locked_yes, YES + dead_no)

Paper-mode closes via `store.close_position`. Live-mode is deferred —
the monitor logs `reason='live_skipped'` so the operator can see what
it would have done.

Tests cover:
  - take-profit at threshold
  - take-profit below threshold (no-op)
  - cut-loss on NO position when state flips to locked_yes
  - cut-loss on YES position when state flips to dead_no
  - cut-loss disabled via flag
  - no snapshot for position → skip
  - no bid for our side (NULL) → skip
  - live mode emits 'live_skipped' attempt without closing
  - audit jsonl is written when path provided
  - n_closed / n_examined return value accuracy
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
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
from kalshi_scout.store import SnapshotStore
from kalshi_scout.trading import PositionMonitor


# -- fixtures ----------------------------------------------------------------

@pytest.fixture
def store(tmp_path: Path) -> SnapshotStore:
    s = SnapshotStore(tmp_path / "monitor.db")
    yield s
    s.close()


def _record_snapshot(
    store: SnapshotStore,
    market_ticker: str,
    event_ticker: str = "EVT",
    state: str = "dead_no",
    yes_bid: int | None = 5,
    no_bid: int | None = 89,
    metric: Metric = Metric.LOW,
    bracket: Bracket = Bracket(BracketKind.GTE, lo=63.0, hi=None),
    fair_lo: float = 0.0,
    fair_hi: float = 0.02,
    grade: str = "A+",
) -> None:
    """Persist one minimal snapshot via the public record_scan API. Yes/no
    asks default to bid+1 (typical 1c spread)."""
    contract = ParsedContract(
        market_ticker=market_ticker, event_ticker=event_ticker,
        city_slug="DC", metric=metric, market_date=date(2026, 5, 30),
        bracket=bracket,
    )
    market = KalshiMarket(
        ticker=market_ticker, event_ticker=event_ticker,
        title="", yes_sub_title="", status="open", close_time=None,
        yes_bid=yes_bid,
        yes_ask=(yes_bid + 1) if yes_bid is not None else None,
        no_bid=no_bid,
        no_ask=(no_bid + 1) if no_bid is not None else None,
        last_price=None, volume=100, open_interest=200,
    )
    eval_ = ContractEvaluation(
        contract=contract, market=market,
        state=ContractState(state), reason="",
        fair_prob_low=fair_lo, fair_prob_high=fair_hi,
        yes_ask_cents=market.yes_ask, no_ask_cents=market.no_ask,
        edge_yes=None, edge_no=None, grade=grade, notes=[],
    )
    store.record_scan([eval_])


# -- take-profit -------------------------------------------------------------

def test_take_profit_closes_no_position_when_no_bid_at_threshold(store):
    """NO position bought at 50c, no_bid now 96c → close, realize 46c × size."""
    pid = store.add_position("MKT1", "EVT", "no", size_contracts=2, avg_price_cents=50)
    _record_snapshot(store, "MKT1", no_bid=96)

    monitor = PositionMonitor(store, take_profit_bid_cents=95, paper=True)
    n_closed, n_examined = monitor.run()

    assert (n_closed, n_examined) == (1, 1)
    closed = store.query_positions(open_only=False)
    assert len(closed) == 1
    assert closed[0].closed_at_price_cents == 96
    assert closed[0].realized_pnl_cents == (96 - 50) * 2


def test_take_profit_does_not_fire_below_threshold(store):
    """NO bid 94c, threshold 95c → hold."""
    pid = store.add_position("MKT1", "EVT", "no", 1, 50)
    _record_snapshot(store, "MKT1", no_bid=94)

    monitor = PositionMonitor(store, take_profit_bid_cents=95, paper=True)
    n_closed, n_examined = monitor.run()

    assert (n_closed, n_examined) == (0, 1)
    open_pos = store.query_positions(open_only=True)
    assert len(open_pos) == 1


def test_take_profit_works_for_yes_position(store):
    """YES position bought at 30c, yes_bid now 95c → close."""
    pid = store.add_position("MKT1", "EVT", "yes", 1, 30)
    _record_snapshot(store, "MKT1", yes_bid=95, no_bid=4, state="locked_yes")

    monitor = PositionMonitor(store, take_profit_bid_cents=95, paper=True)
    n_closed, _ = monitor.run()
    assert n_closed == 1
    closed = store.query_positions(open_only=False)[0]
    assert closed.closed_at_price_cents == 95


# -- cut-loss on state flip --------------------------------------------------

def test_cut_loss_closes_no_position_when_state_flips_to_locked_yes(store):
    """NO position + snapshot state=locked_yes means the bracket settled
    against us — we will pay $0 at expiration. Close at current no_bid to
    salvage whatever's left."""
    pid = store.add_position("MKT1", "EVT", "no", 1, 80)
    _record_snapshot(store, "MKT1", no_bid=5, state="locked_yes")
    # no_bid 5c is below take-profit, but state-flip should still fire.

    monitor = PositionMonitor(
        store, take_profit_bid_cents=95, cut_loss_on_state_flip=True, paper=True,
    )
    n_closed, _ = monitor.run()
    assert n_closed == 1
    closed = store.query_positions(open_only=False)[0]
    assert closed.closed_at_price_cents == 5
    # Realized -75c — a big loss but better than -80c at expiration.
    assert closed.realized_pnl_cents == (5 - 80) * 1


def test_cut_loss_closes_yes_position_when_state_flips_to_dead_no(store):
    """Symmetric for YES positions."""
    pid = store.add_position("MKT1", "EVT", "yes", 1, 60)
    _record_snapshot(store, "MKT1", yes_bid=3, no_bid=96, state="dead_no")

    monitor = PositionMonitor(
        store, take_profit_bid_cents=95, cut_loss_on_state_flip=True, paper=True,
    )
    n_closed, _ = monitor.run()
    assert n_closed == 1
    closed = store.query_positions(open_only=False)[0]
    assert closed.closed_at_price_cents == 3


def test_cut_loss_disabled_via_flag_holds_through_adverse_state(store):
    """When cut_loss_on_state_flip=False, an adverse state-flip is NOT
    enough to close. Take-profit is still active independently."""
    pid = store.add_position("MKT1", "EVT", "no", 1, 80)
    _record_snapshot(store, "MKT1", no_bid=5, state="locked_yes")

    monitor = PositionMonitor(
        store, take_profit_bid_cents=95, cut_loss_on_state_flip=False, paper=True,
    )
    n_closed, n_examined = monitor.run()
    assert (n_closed, n_examined) == (0, 1)
    assert len(store.query_positions(open_only=True)) == 1


def test_cut_loss_does_not_fire_on_intermediate_states(store):
    """bracket_hit_vulnerable / forecast_dependent / not_reached are NOT
    adverse — they're still in play. Only locked_yes / dead_no fire."""
    pid = store.add_position("MKT1", "EVT", "no", 1, 80)
    _record_snapshot(store, "MKT1", no_bid=50, state="bracket_hit_vulnerable")

    monitor = PositionMonitor(
        store, take_profit_bid_cents=95, cut_loss_on_state_flip=True, paper=True,
    )
    n_closed, _ = monitor.run()
    assert n_closed == 0


# -- edge cases --------------------------------------------------------------

def test_no_snapshot_for_position_is_skipped(store):
    """Position exists for a market with no snapshots — skip without error."""
    pid = store.add_position("MKT1", "EVT", "no", 1, 50)
    # No snapshot persisted.

    monitor = PositionMonitor(store, take_profit_bid_cents=95, paper=True)
    n_closed, n_examined = monitor.run()
    assert (n_closed, n_examined) == (0, 1)
    assert len(store.query_positions(open_only=True)) == 1


def test_null_bid_skips_position(store):
    """Snapshot exists but no_bid is None (one-sided book) → can't realize
    anything, so hold."""
    pid = store.add_position("MKT1", "EVT", "no", 1, 50)
    _record_snapshot(store, "MKT1", no_bid=None)

    monitor = PositionMonitor(store, take_profit_bid_cents=95, paper=True)
    n_closed, _ = monitor.run()
    assert n_closed == 0


def test_live_mode_logs_skipped_but_does_not_close(store, tmp_path):
    """In live mode (paper=False) the monitor records 'live_skipped' to the
    audit log without touching the position — sell-order placement is
    deferred to a follow-up PR."""
    audit = tmp_path / "exit.jsonl"
    pid = store.add_position("MKT1", "EVT", "no", 1, 50)
    _record_snapshot(store, "MKT1", no_bid=96)

    monitor = PositionMonitor(
        store, take_profit_bid_cents=95, paper=False, audit_log_path=audit,
    )
    n_closed, n_examined = monitor.run()
    assert (n_closed, n_examined) == (0, 1)
    # Position still open.
    assert len(store.query_positions(open_only=True)) == 1
    # Audit row written with reason='live_skipped'.
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["reason"] == "live_skipped"
    assert rows[0]["closed"] is False
    assert rows[0]["exit_price_cents"] == 96


# -- audit log ---------------------------------------------------------------

def test_audit_log_captures_take_profit_close(store, tmp_path):
    """Every close attempt — successful or skipped — writes one JSONL row."""
    audit = tmp_path / "exit.jsonl"
    pid = store.add_position("MKT1", "EVT", "no", 2, 50)
    _record_snapshot(store, "MKT1", no_bid=97)

    monitor = PositionMonitor(
        store, take_profit_bid_cents=95, paper=True, audit_log_path=audit,
    )
    monitor.run()

    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["reason"] == "take_profit"
    assert row["closed"] is True
    assert row["paper"] is True
    assert row["size_contracts"] == 2
    assert row["open_price_cents"] == 50
    assert row["exit_price_cents"] == 97
    assert row["realized_pnl_cents"] == (97 - 50) * 2
    assert row["position_id"] == pid


def test_audit_log_captures_cut_loss_close(store, tmp_path):
    audit = tmp_path / "exit.jsonl"
    pid = store.add_position("MKT1", "EVT", "no", 1, 80)
    _record_snapshot(store, "MKT1", no_bid=8, state="locked_yes")

    monitor = PositionMonitor(
        store, take_profit_bid_cents=95, cut_loss_on_state_flip=True,
        paper=True, audit_log_path=audit,
    )
    monitor.run()

    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["reason"] == "cut_loss_state_flip"
    assert rows[0]["closed"] is True


def test_no_audit_log_path_means_no_file_written(store, tmp_path):
    """Audit log is optional — None path means no file ever created."""
    pid = store.add_position("MKT1", "EVT", "no", 1, 50)
    _record_snapshot(store, "MKT1", no_bid=97)

    monitor = PositionMonitor(
        store, take_profit_bid_cents=95, paper=True, audit_log_path=None,
    )
    monitor.run()
    assert not (tmp_path / "exit.jsonl").exists()


# -- multiple positions ------------------------------------------------------

def test_multiple_positions_examined_correctly(store):
    """Three positions, one closes by take-profit, one by cut-loss, one
    holds → n_closed=2, n_examined=3."""
    p1 = store.add_position("MKT1", "EVT1", "no", 1, 50)   # → take-profit
    p2 = store.add_position("MKT2", "EVT2", "no", 1, 80)   # → cut-loss
    p3 = store.add_position("MKT3", "EVT3", "no", 1, 60)   # → hold

    _record_snapshot(store, "MKT1", no_bid=96, state="dead_no")
    _record_snapshot(store, "MKT2", no_bid=10, state="locked_yes")
    _record_snapshot(store, "MKT3", no_bid=70, state="dead_no")

    monitor = PositionMonitor(
        store, take_profit_bid_cents=95, cut_loss_on_state_flip=True, paper=True,
    )
    n_closed, n_examined = monitor.run()
    assert (n_closed, n_examined) == (2, 3)


# -- invariants --------------------------------------------------------------

def test_take_profit_bid_must_be_in_valid_range(store):
    """Invalid threshold (0, 100, or out of range) → ValueError at construction."""
    with pytest.raises(ValueError):
        PositionMonitor(store, take_profit_bid_cents=0)
    with pytest.raises(ValueError):
        PositionMonitor(store, take_profit_bid_cents=100)
    with pytest.raises(ValueError):
        PositionMonitor(store, take_profit_bid_cents=-1)
