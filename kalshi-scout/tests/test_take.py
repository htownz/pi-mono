"""Tests for the `take` CLI command's snapshot-driven side/price derivation.

The full CLI surface is integration-tested by running `kalshi-scout take
<ticker>` end-to-end; here we just verify the side-picker helper makes the
right call across state and edge combinations.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from kalshi_scout.cli import _derive_take_side
from kalshi_scout.store import SnapshotRow


def _snap(
    state: str = "forecast_dependent",
    yes_ask: int | None = 30,
    no_ask: int | None = 70,
    edge_yes: float | None = 0.1,
    edge_no: float | None = -0.05,
) -> SnapshotRow:
    """Synthesize a SnapshotRow with the fields `_derive_take_side` reads."""
    return SnapshotRow(
        id=1, scan_id="x",
        scanned_at_utc=datetime(2026, 5, 30, 16, 0, tzinfo=timezone.utc),
        market_ticker="K", event_ticker="E", city_slug="HOUSTON",
        metric="high", market_date=date(2026, 5, 30),
        bracket_kind="between", bracket_lo=79.0, bracket_hi=80.0,
        station_icao="KHOU", cli_product="CLIHOU",
        source_provenance="resolver", regime="clear_and_dry",
        running_max_f=None, running_min_f=None,
        cli_report_date=None, cli_max_f=None, cli_min_f=None,
        state=state, reason="",
        fair_prob_low=0.4, fair_prob_high=0.6,
        yes_bid=yes_ask - 1 if yes_ask else None, yes_ask=yes_ask,
        no_bid=no_ask - 1 if no_ask else None, no_ask=no_ask,
        last_price=None, volume=10, open_interest=100,
        edge_yes=edge_yes, edge_no=edge_no,
        grade="B", notes=[],
    )


def test_derive_take_side_locked_yes_picks_yes_regardless_of_edges():
    """A LOCKED_YES state always means buy yes — even if edge_no happens
    to be larger, the settlement-conclusive yes side is the actionable one."""
    snap = _snap(state="locked_yes", yes_ask=85, edge_yes=0.10, edge_no=0.50)
    side, price = _derive_take_side(snap)
    assert side == "yes"
    assert price == 85


def test_derive_take_side_dead_no_picks_no_regardless_of_edges():
    snap = _snap(state="dead_no", no_ask=5, edge_yes=0.50, edge_no=0.10)
    side, price = _derive_take_side(snap)
    assert side == "no"
    assert price == 5


def test_derive_take_side_picks_larger_edge_for_non_deterministic_states():
    """FORECAST_DEPENDENT / BRACKET_HIT_VULNERABLE pick whichever side
    has the larger edge."""
    yes_wins = _snap(state="forecast_dependent",
                     edge_yes=0.15, edge_no=0.03,
                     yes_ask=30, no_ask=70)
    assert _derive_take_side(yes_wins) == ("yes", 30)

    no_wins = _snap(state="bracket_hit_vulnerable",
                    edge_yes=0.02, edge_no=0.20,
                    yes_ask=30, no_ask=70)
    assert _derive_take_side(no_wins) == ("no", 70)


def test_derive_take_side_tolerates_none_edges():
    """Missing edges shouldn't crash the picker — default to yes side."""
    snap = _snap(state="not_reached",
                 edge_yes=None, edge_no=None,
                 yes_ask=30, no_ask=70)
    side, _ = _derive_take_side(snap)
    assert side == "yes"


def test_derive_take_side_returns_none_price_when_ask_missing():
    """Caller is responsible for raising when no price is available;
    the picker just returns what's there."""
    snap = _snap(state="locked_yes", yes_ask=None,
                 edge_yes=0.10, edge_no=None)
    side, price = _derive_take_side(snap)
    assert side == "yes"
    assert price is None
