"""Tests for V1.0 FastAPI dashboard.

Uses FastAPI's TestClient (no live HTTP); each test gets a fresh sqlite
file via tmp_path.
"""

from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kalshi_scout.models import (
    Bracket,
    BracketKind,
    ContractEvaluation,
    ContractState,
    KalshiMarket,
    Metric,
    ParsedContract,
)
from kalshi_scout.server import create_app
from kalshi_scout.store import SnapshotStore, settlement_from_cli


def _seed_store(path: Path) -> None:
    """Drop one A+ snapshot and one open YES position into a store."""
    with SnapshotStore(path) as store:
        contract = ParsedContract(
            market_ticker="KXHIGHHOUSTON-26MAY27-B79-80",
            event_ticker="KXHIGHHOUSTON-26MAY27",
            city_slug="HOUSTON", metric=Metric.HIGH,
            market_date=date(2026, 5, 27),
            bracket=Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0),
        )
        market = KalshiMarket(
            ticker=contract.market_ticker, event_ticker=contract.event_ticker,
            title="", yes_sub_title="", status="open", close_time=None,
            yes_bid=70, yes_ask=71, no_bid=29, no_ask=30,
            last_price=71, volume=10, open_interest=100,
        )
        eval_ = ContractEvaluation(
            contract=contract, market=market,
            state=ContractState.LOCKED_YES, reason="max already hit 79",
            fair_prob_low=0.97, fair_prob_high=0.99,
            yes_ask_cents=71, no_ask_cents=29,
            edge_yes=0.27, edge_no=None, grade="A+", notes=[],
        )
        store.record_scan([eval_], station_state_map={
            contract.market_ticker: {
                "regime": "clear_and_dry",
                "station_icao": "KHOU", "cli_product": "CLIHOU",
                "source_provenance": "resolver",
            }
        })
        store.add_position(
            market_ticker=contract.market_ticker,
            event_ticker=contract.event_ticker,
            side="yes", size_contracts=100, avg_price_cents=71,
        )


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    db = tmp_path / "dash.db"
    _seed_store(db)
    app = create_app(db)
    return TestClient(app)


# -- Health + smoke ----------------------------------------------------------

def test_health_reports_ok(tmp_path: Path):
    db = tmp_path / "empty.db"
    SnapshotStore(db).close()  # init empty schema
    client = TestClient(create_app(db))
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_health_reports_has_data(client: TestClient):
    resp = client.get("/api/health")
    assert resp.json()["has_data"] is True


# -- HTML pages --------------------------------------------------------------

def test_index_renders_with_recent_snapshot(client: TestClient):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.text
    assert "kalshi-scout" in body
    assert "KXHIGHHOUSTON-26MAY27-B79-80" in body
    assert "A+" in body
    # Risk summary should show the position.
    assert "open positions" in body


def test_calibration_renders(client: TestClient):
    resp = client.get("/calibration")
    assert resp.status_code == 200
    assert "Calibration" in resp.text


def test_risk_page_renders(client: TestClient):
    resp = client.get("/risk")
    assert resp.status_code == 200
    assert "Open position risk" in resp.text


# -- JSON API ----------------------------------------------------------------

def test_api_snapshots_returns_seeded_row(client: TestClient):
    resp = client.get("/api/snapshots?limit=10&min_grade=A")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["grade"] == "A+"
    assert rows[0]["market_ticker"] == "KXHIGHHOUSTON-26MAY27-B79-80"
    assert rows[0]["regime"] == "clear_and_dry"


def test_api_risk_returns_position(client: TestClient):
    resp = client.get("/api/risk")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_open_positions"] == 1
    assert body["total_open_contracts"] == 100
    assert body["total_max_loss_cents"] == 7100
    assert "HOUSTON" in body["by_city"]


def test_api_calibration_serializes(client: TestClient):
    resp = client.get("/api/calibration")
    assert resp.status_code == 200
    body = resp.json()
    assert "by_grade" in body
    assert "A+" in body["by_grade"]
