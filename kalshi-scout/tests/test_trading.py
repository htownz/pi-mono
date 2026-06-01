"""Tests for the authenticated Kalshi trading client + RiskGuard + AutoTrader.

The signing logic is unit-tested via a generated RSA keypair so we don't need
real Kalshi credentials. The HTTP layer is mocked via respx. Risk-guard
tests build SnapshotRow / Alert fixtures directly and assert the (allowed,
reason) decisions.

Live-API verification (does Kalshi actually accept our signature?) is the
operator's responsibility — outbound to api.elections.kalshi.com is
sandboxed off here.
"""

from __future__ import annotations

import base64
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
import respx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from kalshi_scout.notify import Alert
from kalshi_scout.store import SnapshotRow, SnapshotStore
from kalshi_scout.trading import (
    AutoTrader,
    KALSHI_API_BASE,
    KalshiTradingClient,
    KillSwitch,
    RiskDecision,
    RiskGuard,
    RiskLimits,
    TradeAttempt,
    _signature_payload,
    _sign,
    auto_close_settled_positions,
)


# -- fixtures ----------------------------------------------------------------

@pytest.fixture
def keypair(tmp_path: Path) -> tuple[rsa.RSAPrivateKey, Path]:
    """Generate a fresh RSA keypair, write the private key to a PEM file,
    and return (private_key_object, pem_path) so tests can both sign-with
    and load-from the same key."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem_path = tmp_path / "kalshi.pem"
    pem_path.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    return key, pem_path


@pytest.fixture
def store(tmp_path: Path) -> SnapshotStore:
    s = SnapshotStore(tmp_path / "trading.db")
    yield s
    s.close()


def _snap(
    ticker: str = "KXLOWTDC-26MAY30-T62",
    event: str = "KXLOWTDC-26MAY30",
    state: str = "dead_no",
    metric: str = "low",
    bracket_kind: str = "gte",
    bracket_lo: float | None = 63.0,
    bracket_hi: float | None = None,
    running_max: float | None = None,
    running_min: float | None = 62.0,   # well below 63 → safe rounding margin
    yes_ask: int | None = 15,
    no_ask: int | None = 89,
    fair_lo: float = 0.0,
    fair_hi: float = 0.02,
    edge_yes: float | None = -0.14,
    edge_no: float | None = 0.10,
    grade: str = "A+",
    snap_id: int = 1,
) -> SnapshotRow:
    return SnapshotRow(
        id=snap_id, scan_id="x",
        scanned_at_utc=datetime(2026, 5, 30, 23, 0, tzinfo=timezone.utc),
        market_ticker=ticker, event_ticker=event,
        city_slug="DC", metric=metric,
        market_date=date(2026, 5, 30),
        bracket_kind=bracket_kind, bracket_lo=bracket_lo, bracket_hi=bracket_hi,
        station_icao="KDCA", cli_product="CLIDCA",
        source_provenance="resolver", regime="clear_and_dry",
        running_max_f=running_max, running_min_f=running_min,
        projected_extremum_f=None,
        cli_report_date=None, cli_max_f=None, cli_min_f=None,
        state=state, reason="",
        fair_prob_low=fair_lo, fair_prob_high=fair_hi,
        yes_bid=(yes_ask - 1) if yes_ask else None, yes_ask=yes_ask,
        no_bid=(no_ask - 1) if no_ask else None, no_ask=no_ask,
        last_price=None, volume=100, open_interest=200,
        edge_yes=edge_yes, edge_no=edge_no,
        grade=grade, notes=[],
    )


def _alert(ticker: str = "KXLOWTDC-26MAY30-T62",
           event: str = "KXLOWTDC-26MAY30", grade: str = "A+") -> Alert:
    return Alert(
        fired_at_utc=datetime(2026, 5, 30, 23, 30, tzinfo=timezone.utc),
        market_ticker=ticker, event_ticker=event,
        city_slug="DC", market_date="2026-05-30",
        bracket="63° or above", metric="low",
        state="dead_no", reason="observed min below bracket",
        grade=grade, previous_grade="B+",
        yes_ask_cents=15, no_ask_cents=89,
        edge_yes=-0.14, edge_no=0.10,
        fair_prob_low=0.0, fair_prob_high=0.02,
        notes=[],
    )


# -- Signing -----------------------------------------------------------------

def test_signature_payload_includes_ts_method_path():
    """Kalshi signs `f"{timestamp_ms}{METHOD}{path}"` exactly."""
    payload = _signature_payload(1730000000000, "post", "/trade-api/v2/portfolio/orders")
    assert payload == "1730000000000POST/trade-api/v2/portfolio/orders"


def test_sign_produces_verifiable_pss_signature(keypair):
    """The signature we emit should verify with PSS-SHA256 against the
    matching public key. Confirms our padding choice matches Kalshi's spec."""
    private, _ = keypair
    payload = "1730000000000POST/trade-api/v2/portfolio/orders"
    sig_b64 = _sign(private, payload)
    sig = base64.b64decode(sig_b64)
    # If padding/hash were wrong, .verify would raise — the assertion is
    # implicit in not raising.
    private.public_key().verify(
        sig, payload.encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )


def test_client_sends_kalshi_auth_headers(keypair):
    """Every authenticated request carries the three KALSHI-ACCESS-* headers
    with the right shape."""
    _, pem = keypair
    with respx.mock(base_url=KALSHI_API_BASE) as mock:
        route = mock.post("/portfolio/orders").respond(200, json={"order": {"order_id": "ord_1"}})
        client = KalshiTradingClient(key_id="kid_abc", private_key_path=pem)
        resp = client.place_order(
            ticker="K", action="buy", side="yes",
            count=1, price_cents=15, order_type="limit",
        )
        client.close()
    assert resp == {"order": {"order_id": "ord_1"}}
    req = route.calls.last.request
    assert req.headers["KALSHI-ACCESS-KEY"] == "kid_abc"
    assert "KALSHI-ACCESS-TIMESTAMP" in req.headers
    assert int(req.headers["KALSHI-ACCESS-TIMESTAMP"]) > 0
    sig_b64 = req.headers["KALSHI-ACCESS-SIGNATURE"]
    # Base64-decodable → looks like a signature, not garbage.
    base64.b64decode(sig_b64)


def test_place_order_body_uses_correct_price_field_for_side(keypair):
    """Kalshi expects `yes_price` for yes-side limits and `no_price` for
    no-side limits. Test both."""
    _, pem = keypair
    with respx.mock(base_url=KALSHI_API_BASE) as mock:
        yes_route = mock.post("/portfolio/orders").respond(200, json={"order": {}})
        client = KalshiTradingClient(key_id="kid", private_key_path=pem)
        client.place_order(ticker="K", action="buy", side="yes",
                           count=1, price_cents=15, order_type="limit")
        yes_body = json.loads(yes_route.calls.last.request.content)
        assert yes_body["yes_price"] == 15
        assert "no_price" not in yes_body

        client.place_order(ticker="K", action="buy", side="no",
                           count=2, price_cents=89, order_type="limit")
        no_body = json.loads(yes_route.calls.last.request.content)
        assert no_body["no_price"] == 89
        assert "yes_price" not in no_body
        client.close()


def test_place_order_validates_inputs(keypair):
    _, pem = keypair
    client = KalshiTradingClient(key_id="kid", private_key_path=pem)
    with pytest.raises(ValueError, match="action must be"):
        client.place_order(ticker="K", action="hodl", side="yes", count=1, price_cents=15)
    with pytest.raises(ValueError, match="side must be"):
        client.place_order(ticker="K", action="buy", side="maybe", count=1, price_cents=15)
    with pytest.raises(ValueError, match="count must be"):
        client.place_order(ticker="K", action="buy", side="yes", count=0, price_cents=15)
    with pytest.raises(ValueError, match="price_cents must be"):
        client.place_order(ticker="K", action="buy", side="yes", count=1, price_cents=100)
    client.close()


# -- KillSwitch --------------------------------------------------------------

def test_kill_switch_active_when_file_exists(tmp_path):
    kill = tmp_path / "scout.kill"
    sw = KillSwitch(kill)
    assert sw.is_active() is False
    kill.touch()
    assert sw.is_active() is True


# -- RiskGuard ---------------------------------------------------------------

def _guard(store: SnapshotStore, tmp_path: Path, **limit_overrides) -> RiskGuard:
    limits = RiskLimits(**limit_overrides) if limit_overrides else RiskLimits()
    return RiskGuard(limits, store, KillSwitch(tmp_path / "kill"))


def test_risk_guard_allows_safe_dead_no(store, tmp_path):
    """The baseline: dead_no with running_min comfortably below the bracket
    floor passes all checks."""
    g = _guard(store, tmp_path)
    snap = _snap()  # running_min=62.0, bracket_lo=63.0 → 1.0°F gap
    decision = g.can_place(snap, side="no", price_cents=89, size_contracts=1)
    assert decision.allowed is True, decision.reason


def test_risk_guard_blocks_when_kill_switch_active(store, tmp_path):
    kill = tmp_path / "kill"
    kill.touch()
    g = _guard(store, tmp_path)
    decision = g.can_place(_snap(), side="no", price_cents=89, size_contracts=1)
    assert decision.allowed is False
    assert "kill switch" in decision.reason


def test_risk_guard_rejects_oversized_order(store, tmp_path):
    g = _guard(store, tmp_path, max_position_size_contracts=5)
    decision = g.can_place(_snap(), side="no", price_cents=89, size_contracts=6)
    assert decision.allowed is False
    assert "max_position_size_contracts" in decision.reason


def test_risk_guard_rejects_oversized_cost(store, tmp_path):
    g = _guard(store, tmp_path, max_position_cost_cents=500)
    # 6 contracts @ 89c = 534c > 500c cap.
    decision = g.can_place(_snap(), side="no", price_cents=89, size_contracts=6)
    assert decision.allowed is False
    # Note: size cap fires first (5) — set a higher size cap so cost is the
    # binding check.
    g2 = _guard(store, tmp_path,
                max_position_size_contracts=100, max_position_cost_cents=500)
    decision = g2.can_place(_snap(), side="no", price_cents=89, size_contracts=6)
    assert decision.allowed is False
    assert "max_position_cost_cents" in decision.reason


def test_risk_guard_rejects_event_concentration(store, tmp_path):
    """An existing open position on the same event consumes part of the
    concentration budget; the next order is refused once cumulative cost
    exceeds the cap."""
    g = _guard(store, tmp_path,
               max_concentration_per_event_cents=200,
               max_position_size_contracts=100,
               max_position_cost_cents=1000)
    # Pre-existing $1.50 position on the same event.
    store.add_position(
        market_ticker="KXLOWTDC-26MAY30-T63",   # sibling bracket
        event_ticker="KXLOWTDC-26MAY30",
        side="no", size_contracts=2, avg_price_cents=75,
    )
    decision = g.can_place(_snap(), side="no", price_cents=89, size_contracts=1)
    # 150c existing + 89c new = 239c > 200c cap → refuse.
    assert decision.allowed is False
    assert "concentration" in decision.reason


def test_risk_guard_kills_after_daily_loss_threshold(store, tmp_path):
    g = _guard(store, tmp_path, max_daily_loss_cents=100)
    # Close a position with a -150c realized P&L, today.
    pid = store.add_position(
        market_ticker="OLD-1", event_ticker="OLD",
        side="yes", size_contracts=10, avg_price_cents=30,
    )
    store.close_position(pid, at_price_cents=15)  # (15-30)*10 = -150c
    decision = g.can_place(_snap(), side="no", price_cents=89, size_contracts=1)
    assert decision.allowed is False
    assert "daily loss" in decision.reason


def test_risk_guard_doesnt_count_winners_toward_daily_loss(store, tmp_path):
    """Wins don't reset or extend the kill threshold — only losses count."""
    g = _guard(store, tmp_path, max_daily_loss_cents=100)
    pid = store.add_position(
        market_ticker="WINNER", event_ticker="WINNER",
        side="yes", size_contracts=10, avg_price_cents=30,
    )
    store.close_position(pid, at_price_cents=100)  # +700c
    decision = g.can_place(_snap(), side="no", price_cents=89, size_contracts=1)
    assert decision.allowed is True, decision.reason


def test_risk_guard_enforces_min_edge(store, tmp_path):
    """Engine's grading min is 1c; the auto-trader's default is 5c to absorb
    fees + slippage. Edge below threshold is refused."""
    g = _guard(store, tmp_path, min_edge_cents=20)
    # Default snap: no_ask=89, fair_no = 1 - 0.01 = 0.99 → edge = 99 - 89 = 10c.
    decision = g.can_place(_snap(), side="no", price_cents=89, size_contracts=1)
    assert decision.allowed is False
    assert "edge" in decision.reason


def test_risk_guard_rounding_risk_low_dead_no_too_close(store, tmp_path):
    """The KXLOWTDC-T62 trap: running_min within 0.5°F of bracket_lo means
    the CLI report could round the official daily-min across the boundary
    and flip the outcome. Refuse."""
    g = _guard(store, tmp_path)
    # bracket_lo=63, running_min=62.6 → gap = 0.4°F < 0.5°F buffer
    risky = _snap(running_min=62.6)
    decision = g.can_place(risky, side="no", price_cents=89, size_contracts=1)
    assert decision.allowed is False
    assert "rounding risk" in decision.reason
    assert "62.6" in decision.reason


def test_risk_guard_rounding_risk_high_dead_no_too_close(store, tmp_path):
    """HIGH dead_no: running_max within 0.5°F of bracket_hi is risky too."""
    g = _guard(store, tmp_path)
    snap = _snap(
        metric="high", bracket_lo=None, bracket_hi=72.0,
        running_max=72.3, running_min=None,
    )
    decision = g.can_place(snap, side="no", price_cents=89, size_contracts=1)
    assert decision.allowed is False
    assert "rounding risk" in decision.reason


def test_risk_guard_rounding_risk_locked_yes(store, tmp_path):
    """HIGH locked_yes: running_max barely above bracket_lo. CLI rounding
    down could flip it to NO."""
    g = _guard(store, tmp_path)
    snap = _snap(
        metric="high", state="locked_yes", bracket_lo=72.0, bracket_hi=None,
        running_max=72.3, running_min=None,
        yes_ask=92, no_ask=10, edge_yes=0.08, edge_no=None,
        fair_lo=0.98, fair_hi=1.0, grade="A+",
    )
    decision = g.can_place(snap, side="yes", price_cents=92, size_contracts=1)
    assert decision.allowed is False
    assert "rounding risk" in decision.reason


def test_risk_guard_rounding_risk_passes_with_comfortable_margin(store, tmp_path):
    """Same dead_no shape, but running_min 1.0°F below bracket. Safe."""
    g = _guard(store, tmp_path)
    safe = _snap(running_min=62.0)  # 1.0°F below bracket_lo=63
    decision = g.can_place(safe, side="no", price_cents=89, size_contracts=1)
    assert decision.allowed is True, decision.reason


# -- AutoTrader --------------------------------------------------------------

def _trader(store: SnapshotStore, tmp_path: Path, *, paper: bool = True,
            client: KalshiTradingClient | None = None,
            **limit_overrides) -> AutoTrader:
    limits = RiskLimits(**limit_overrides) if limit_overrides else RiskLimits()
    guard = RiskGuard(limits, store, KillSwitch(tmp_path / "kill"))
    audit = tmp_path / "audit.jsonl"
    return AutoTrader(
        client=client, guard=guard, store=store,
        default_size=1, paper=paper, audit_log_path=audit,
    )


def test_auto_trader_paper_records_position_without_api_call(store, tmp_path):
    """Paper mode runs the full pipeline (derive, risk-check, record
    position, audit log) but skips the trading API entirely."""
    trader = _trader(store, tmp_path, paper=True)
    attempt = trader.maybe_trade(_alert(), _snap())
    assert attempt.placed is True
    assert attempt.paper is True
    assert attempt.order_id is None   # paper: no real order
    assert attempt.position_id is not None
    # Position landed in the store with the right side/price.
    positions = store.query_positions(open_only=True)
    assert len(positions) == 1
    assert positions[0].side == "no"
    assert positions[0].avg_price_cents == 89


def test_auto_trader_skips_when_risk_refuses(store, tmp_path):
    """Risk refusal → no position created, audit log captures the reason."""
    trader = _trader(store, tmp_path, paper=True, max_position_size_contracts=0)
    attempt = trader.maybe_trade(_alert(), _snap())
    assert attempt.placed is False
    assert "max_position_size_contracts" in attempt.reason
    assert store.query_positions(open_only=True) == []


def test_auto_trader_skips_rounding_risk_alert(store, tmp_path):
    """The headline safety: an A+ alert that's actually a rounding-risk
    false positive (KXLOWTDC-T62-style) is silently refused. Audit log
    captures it for review."""
    trader = _trader(store, tmp_path, paper=True)
    risky_snap = _snap(running_min=62.6)  # 0.4°F gap
    attempt = trader.maybe_trade(_alert(), risky_snap)
    assert attempt.placed is False
    assert "rounding risk" in attempt.reason
    assert store.query_positions(open_only=True) == []


def test_auto_trader_writes_audit_log(store, tmp_path):
    trader = _trader(store, tmp_path, paper=True)
    trader.maybe_trade(_alert(), _snap())
    trader.maybe_trade(
        _alert(ticker="KXLOWTDC-26MAY30-T63"),
        _snap(running_min=62.6),  # gets refused for rounding risk
    )
    audit = (tmp_path / "audit.jsonl").read_text().strip().split("\n")
    assert len(audit) == 2
    placed, refused = json.loads(audit[0]), json.loads(audit[1])
    assert placed["placed"] is True and placed["paper"] is True
    assert refused["placed"] is False and "rounding risk" in refused["reason"]


def test_auto_trader_live_mode_places_real_order(store, tmp_path, keypair):
    """Live mode (paper=False) calls the trading API and records the
    returned order_id alongside the local position."""
    _, pem = keypair
    with respx.mock(base_url=KALSHI_API_BASE) as mock:
        # Fully-filled response: status=executed + explicit fill count.
        mock.post("/portfolio/orders").respond(
            200, json={"order": {
                "order_id": "ord_abc123",
                "status": "executed",
                "taker_fill_count": 1,
            }},
        )
        client = KalshiTradingClient(key_id="kid", private_key_path=pem)
        trader = _trader(store, tmp_path, paper=False, client=client)
        attempt = trader.maybe_trade(_alert(), _snap())
        client.close()
    assert attempt.placed is True
    assert attempt.paper is False
    assert attempt.order_id == "ord_abc123"
    # Audit + local position both reference the broker-side id via notes.
    pos = store.query_positions(open_only=True)[0]
    assert "ord_abc123" in pos.notes
    assert pos.size_contracts == 1


def test_auto_trader_live_mode_skips_unfilled_resting_order(store, tmp_path, keypair):
    """Codex P1: a limit order that's accepted but resting (no fill yet)
    must NOT be recorded as a local position. The local store would
    diverge from the broker; risk aggregation + auto-close would lie."""
    _, pem = keypair
    with respx.mock(base_url=KALSHI_API_BASE) as mock:
        mock.post("/portfolio/orders").respond(
            200, json={"order": {
                "order_id": "ord_resting",
                "status": "resting",
                "taker_fill_count": 0,
            }},
        )
        client = KalshiTradingClient(key_id="kid", private_key_path=pem)
        # Lift the caps so the 10-contract intent isn't refused by the
        # default $5 risk cap; we want the resting/fill check to be what
        # blocks the position write, not the risk guard.
        trader = _trader(store, tmp_path, paper=False, client=client,
                         max_position_size_contracts=10,
                         max_position_cost_cents=10000,
                         max_concentration_per_event_cents=10000)
        attempt = trader.maybe_trade(_alert(), _snap(), size=10)
        client.close()
    assert attempt.placed is False
    assert "unfilled" in attempt.reason
    assert "0/10" in attempt.reason
    # Order id still surfaced so the operator can chase it on Kalshi.
    assert attempt.order_id == "ord_resting"
    # No phantom position written.
    assert store.query_positions(open_only=True) == []


def test_auto_trader_live_mode_records_partial_fill_with_actual_size(store, tmp_path, keypair):
    """Codex P1: when Kalshi reports a partial fill, the local position
    records only the actually-filled size — not the requested size — so
    the local store mirrors the broker exactly."""
    _, pem = keypair
    with respx.mock(base_url=KALSHI_API_BASE) as mock:
        mock.post("/portfolio/orders").respond(
            200, json={"order": {
                "order_id": "ord_partial",
                "status": "resting",
                "taker_fill_count": 3,    # 3 of 10 immediately taken
            }},
        )
        client = KalshiTradingClient(key_id="kid", private_key_path=pem)
        trader = _trader(store, tmp_path, paper=False, client=client,
                         max_position_size_contracts=10,
                         max_position_cost_cents=10000,
                         max_concentration_per_event_cents=10000)
        attempt = trader.maybe_trade(_alert(), _snap(), size=10)
        client.close()
    assert attempt.placed is True
    assert attempt.size_contracts == 3       # not 10
    assert "partial fill 3/10" in attempt.reason
    pos = store.query_positions(open_only=True)[0]
    assert pos.size_contracts == 3
    assert "PARTIAL" in pos.notes


def test_fill_count_handles_alternate_field_names(store, tmp_path, keypair):
    """Kalshi has shipped both `taker_fill_count` and `fill_count` in
    different API versions; the parser tries both before falling back to
    the status field."""
    from kalshi_scout.trading import AutoTrader as AT
    # Explicit count wins regardless of status.
    assert AT._fill_count_from_response({"fill_count": 5}, requested=10) == 5
    assert AT._fill_count_from_response({"filled_quantity": 7}, requested=10) == 7
    # No count, but status=executed → full fill.
    assert AT._fill_count_from_response({"status": "executed"}, requested=10) == 10
    # No count, status=resting → 0.
    assert AT._fill_count_from_response({"status": "resting"}, requested=10) == 0
    # Unknown shape → conservative 0.
    assert AT._fill_count_from_response({}, requested=10) == 0
    # Bogus count clamped to [0, requested].
    assert AT._fill_count_from_response({"fill_count": 99}, requested=10) == 10
    assert AT._fill_count_from_response({"fill_count": -3}, requested=10) == 0


def test_auto_trader_honors_side_and_price_overrides(store, tmp_path):
    """Codex P1: when the caller passes side_override + price_override
    (e.g. from `fire` after the operator picked the side at the prompt),
    those values must reach the broker — not the snapshot-derived ones.

    Use a snap whose snapshot-derived side=no but where overriding to
    yes still passes risk (positive edge_yes). With the override path
    bypassing _derive_side_and_price, the order placed matches the
    operator's intent.
    """
    # fair_lo=0.55, fair_hi=0.75 → fair_mid=0.65. yes_ask=15 → edge_yes
    # = 0.65 - 0.15 = 0.50 (well above min_edge=5c). dead_no derivation
    # would default to side=no — the override should win.
    trader = _trader(store, tmp_path, paper=True)
    high_edge_snap = _snap(
        state="forecast_dependent", fair_lo=0.55, fair_hi=0.75,
        yes_ask=15, no_ask=85, edge_yes=0.50, edge_no=-0.50,
        running_min=62.0,
    )
    attempt = trader.maybe_trade(
        _alert(), high_edge_snap, size=1,
        side_override="yes", price_override=15,
    )
    assert attempt.placed is True, attempt.reason
    assert attempt.side == "yes"
    assert attempt.price_cents == 15
    pos = store.query_positions(open_only=True)[0]
    assert pos.side == "yes"
    assert pos.avg_price_cents == 15


def test_auto_trader_overrides_still_pass_through_risk_guard(store, tmp_path):
    """Overrides bypass the SNAPSHOT-derivation but NOT the risk guard.
    A nonsensical override (price=99c with no fair value to support it)
    is still refused by the min-edge check."""
    trader = _trader(store, tmp_path, paper=True)
    # Snap has fair_prob=[0, 0.02] → yes side fair ≈ 1c, no side fair ≈ 99c.
    # Override to side=yes, price=99 → edge_yes = 1 - 99 = -98c → REFUSED.
    attempt = trader.maybe_trade(
        _alert(), _snap(), size=1,
        side_override="yes", price_override=99,
    )
    assert attempt.placed is False
    assert "edge" in attempt.reason
    assert store.query_positions(open_only=True) == []


# -- refresh_quote (live-quote re-fetch before placement) --------------------

class _StubKalshiClient:
    """Drop-in for KalshiClient.get_market() in the refresh_quote path.
    Avoids respx-mocking the full Kalshi GET /markets/{ticker} response
    when we only care about which ask the trader uses."""
    def __init__(self, yes_ask: int | None = None, no_ask: int | None = None,
                 raises: Exception | None = None):
        self._yes_ask = yes_ask
        self._no_ask = no_ask
        self._raises = raises
        self.calls: list[str] = []

    def get_market(self, ticker: str):
        self.calls.append(ticker)
        if self._raises:
            raise self._raises
        from kalshi_scout.models import KalshiMarket
        return KalshiMarket(
            ticker=ticker, event_ticker="", title="", yes_sub_title="",
            status="active", close_time=None,
            yes_bid=None, yes_ask=self._yes_ask,
            no_bid=None, no_ask=self._no_ask,
            last_price=None, volume=0, open_interest=0,
        )

    def close(self) -> None:
        pass


def test_refresh_quote_uses_live_ask_in_place_of_snapshot(store, tmp_path):
    """When refresh_quote=True and a kalshi_client is wired, the order is
    placed at the live ask, not the snapshot's. The snapshot's ask is
    used only when refresh fails."""
    guard = RiskGuard(RiskLimits(min_edge_cents=0), store,
                      KillSwitch(tmp_path / "kill"))
    rt_client = _StubKalshiClient(no_ask=82)  # live ask is 82, snap is 89
    trader = AutoTrader(
        client=None, guard=guard, store=store,
        default_size=1, paper=True,
        audit_log_path=tmp_path / "audit.jsonl",
        kalshi_client=rt_client,
    )
    attempt = trader.maybe_trade(_alert(), _snap(), refresh_quote=True)
    assert attempt.placed is True
    assert attempt.price_cents == 82          # live, not snapshot's 89
    assert rt_client.calls == ["KXLOWTDC-26MAY30-T62"]


def test_refresh_quote_falls_back_to_snapshot_when_live_fetch_fails(store, tmp_path):
    """A network error on the live refresh shouldn't kill the order — the
    trader silently falls back to the snapshot ask. The audit log shows the
    snapshot price was used."""
    guard = RiskGuard(RiskLimits(min_edge_cents=0), store,
                      KillSwitch(tmp_path / "kill"))
    rt_client = _StubKalshiClient(raises=httpx.ConnectError("net down"))
    trader = AutoTrader(
        client=None, guard=guard, store=store,
        default_size=1, paper=True,
        audit_log_path=tmp_path / "audit.jsonl",
        kalshi_client=rt_client,
    )
    attempt = trader.maybe_trade(_alert(), _snap(), refresh_quote=True)
    assert attempt.placed is True
    assert attempt.price_cents == 89          # back to snapshot's


def test_refresh_quote_records_stale_drift_when_price_diverges(store, tmp_path):
    """If the live price differs from the snapshot's by more than 5c, the
    audit reason carries a 'stale-snapshot drift' note so the operator can
    spot a slow snapshot pipeline."""
    guard = RiskGuard(RiskLimits(min_edge_cents=0), store,
                      KillSwitch(tmp_path / "kill"))
    rt_client = _StubKalshiClient(no_ask=70)   # snap=89, drift -19c
    trader = AutoTrader(
        client=None, guard=guard, store=store,
        default_size=1, paper=True,
        audit_log_path=tmp_path / "audit.jsonl",
        kalshi_client=rt_client,
    )
    attempt = trader.maybe_trade(_alert(), _snap(), refresh_quote=True)
    assert attempt.placed is True
    assert "stale-snapshot drift" in attempt.reason
    assert "snap 89c → live 70c" in attempt.reason


def test_refresh_quote_no_drift_note_when_within_tolerance(store, tmp_path):
    """A 3c drift is within tolerance (default 5c); no audit warning."""
    guard = RiskGuard(RiskLimits(min_edge_cents=0), store,
                      KillSwitch(tmp_path / "kill"))
    rt_client = _StubKalshiClient(no_ask=86)   # snap=89, drift -3c
    trader = AutoTrader(
        client=None, guard=guard, store=store,
        default_size=1, paper=True,
        audit_log_path=tmp_path / "audit.jsonl",
        kalshi_client=rt_client,
    )
    attempt = trader.maybe_trade(_alert(), _snap(), refresh_quote=True)
    assert attempt.placed is True
    assert "stale-snapshot drift" not in attempt.reason


def test_refresh_quote_off_uses_snapshot_unchanged(store, tmp_path):
    """Default (refresh_quote=False) preserves the V1 snapshot-based
    behavior — no live call is made, even if kalshi_client is wired."""
    guard = RiskGuard(RiskLimits(min_edge_cents=0), store,
                      KillSwitch(tmp_path / "kill"))
    rt_client = _StubKalshiClient(no_ask=70)   # would have been used if refresh on
    trader = AutoTrader(
        client=None, guard=guard, store=store,
        default_size=1, paper=True,
        audit_log_path=tmp_path / "audit.jsonl",
        kalshi_client=rt_client,
    )
    # No refresh_quote=True passed.
    attempt = trader.maybe_trade(_alert(), _snap())
    assert attempt.placed is True
    assert attempt.price_cents == 89    # snapshot wins
    assert rt_client.calls == []        # live client was never called


def test_refresh_quote_ignored_when_overrides_passed(store, tmp_path):
    """If the operator passed --price explicitly via `fire`, the override
    wins over any live refresh. We don't second-guess them at this layer."""
    guard = RiskGuard(RiskLimits(min_edge_cents=0), store,
                      KillSwitch(tmp_path / "kill"))
    rt_client = _StubKalshiClient(no_ask=70)
    trader = AutoTrader(
        client=None, guard=guard, store=store,
        default_size=1, paper=True,
        audit_log_path=tmp_path / "audit.jsonl",
        kalshi_client=rt_client,
    )
    attempt = trader.maybe_trade(
        _alert(), _snap(),
        side_override="no", price_override=85,
        refresh_quote=True,
    )
    assert attempt.placed is True
    assert attempt.price_cents == 85    # operator override wins
    assert rt_client.calls == []        # refresh path is skipped under overrides


def test_auto_trader_handles_api_error(store, tmp_path, keypair):
    """A 4xx/5xx from Kalshi is captured in the attempt as `placed=False`
    with the exception text — no position is recorded."""
    _, pem = keypair
    with respx.mock(base_url=KALSHI_API_BASE) as mock:
        mock.post("/portfolio/orders").respond(503, text="overloaded")
        client = KalshiTradingClient(key_id="kid", private_key_path=pem)
        trader = _trader(store, tmp_path, paper=False, client=client)
        attempt = trader.maybe_trade(_alert(), _snap())
        client.close()
    assert attempt.placed is False
    assert "API error" in attempt.reason
    assert store.query_positions(open_only=True) == []


def test_auto_trader_constructor_requires_client_unless_paper(store, tmp_path):
    """Live mode without an injected client is a configuration error."""
    guard = RiskGuard(RiskLimits(), store, KillSwitch(tmp_path / "kill"))
    with pytest.raises(ValueError, match="client is required"):
        AutoTrader(client=None, guard=guard, store=store, paper=False)
    # paper=True is fine without a client.
    AutoTrader(client=None, guard=guard, store=store, paper=True)


# -- Settlement auto-close ---------------------------------------------------

def _record_settlement(store: SnapshotStore, market_ticker: str,
                       event_ticker: str, market_date: date,
                       cli_value_f: float, bracket_lo: float | None = 63.0,
                       bracket_hi: float | None = None) -> None:
    """Persist a settlement for the close test, using the same path the
    backfill-settlements CLI takes."""
    from kalshi_scout.models import Bracket, BracketKind, Metric
    from kalshi_scout.store import settlement_from_cli
    if bracket_lo is not None and bracket_hi is None:
        bracket = Bracket(BracketKind.GTE, lo=bracket_lo, hi=None)
    elif bracket_hi is not None and bracket_lo is None:
        bracket = Bracket(BracketKind.LTE, lo=None, hi=bracket_hi)
    else:
        bracket = Bracket(BracketKind.BETWEEN, lo=bracket_lo, hi=bracket_hi)
    s = settlement_from_cli(
        market_ticker=market_ticker, event_ticker=event_ticker,
        market_date=market_date, city_slug="DC", metric=Metric.LOW,
        bracket=bracket, station_icao="KDCA", cli_product="CLIDCA",
        cli_report_date=market_date, cli_value_f=cli_value_f,
    )
    store.record_settlement(s)


def test_auto_close_closes_winners_at_100(store, tmp_path):
    """No-side position whose bracket resolves No → exit price 100, full
    payout captured."""
    day = date(2026, 5, 30)
    pid = store.add_position(
        market_ticker="KXLOWTDC-26MAY30-T62",
        event_ticker="KXLOWTDC-26MAY30",
        side="no", size_contracts=10, avg_price_cents=89,
    )
    # bracket: gte 63 ("low >= 63"). cli_value 62 < 63 → resolved_yes=False.
    # No side wins.
    _record_settlement(store, "KXLOWTDC-26MAY30-T62", "KXLOWTDC-26MAY30", day,
                       cli_value_f=62.0, bracket_lo=63.0)
    closed = auto_close_settled_positions(store, on_settled_date=day)
    assert (pid, "KXLOWTDC-26MAY30-T62", 100) in closed
    pos = store.query_positions(open_only=False)[0]
    assert pos.closed_at_price_cents == 100
    assert pos.realized_pnl_cents == (100 - 89) * 10  # +110c


def test_auto_close_closes_losers_at_0(store, tmp_path):
    """No-side position whose bracket resolves Yes → exit price 0."""
    day = date(2026, 5, 30)
    pid = store.add_position(
        market_ticker="KXLOWTDC-26MAY30-T62",
        event_ticker="KXLOWTDC-26MAY30",
        side="no", size_contracts=10, avg_price_cents=89,
    )
    # CLI value 65 >= 63 → resolved_yes=True → no side LOSES.
    _record_settlement(store, "KXLOWTDC-26MAY30-T62", "KXLOWTDC-26MAY30", day,
                       cli_value_f=65.0, bracket_lo=63.0)
    closed = auto_close_settled_positions(store, on_settled_date=day)
    assert (pid, "KXLOWTDC-26MAY30-T62", 0) in closed
    pos = store.query_positions(open_only=False)[0]
    assert pos.realized_pnl_cents == (0 - 89) * 10   # -890c


def test_auto_close_skips_unsettled_and_other_dates(store, tmp_path):
    """Only positions whose settlement matches `on_settled_date` are touched.
    Unsettled positions, and positions from other days, stay open."""
    day = date(2026, 5, 30)
    other = date(2026, 5, 29)
    settled_pid = store.add_position(
        market_ticker="SETTLED", event_ticker="E", side="no",
        size_contracts=1, avg_price_cents=89,
    )
    unsettled_pid = store.add_position(
        market_ticker="UNSETTLED", event_ticker="E", side="no",
        size_contracts=1, avg_price_cents=89,
    )
    other_pid = store.add_position(
        market_ticker="OTHER-DAY", event_ticker="E", side="no",
        size_contracts=1, avg_price_cents=89,
    )
    _record_settlement(store, "SETTLED", "E", day, cli_value_f=62.0)
    _record_settlement(store, "OTHER-DAY", "E", other, cli_value_f=62.0)

    closed = auto_close_settled_positions(store, on_settled_date=day)
    closed_tickers = {t for _, t, _ in closed}
    assert closed_tickers == {"SETTLED"}
    open_tickers = {p.market_ticker for p in store.query_positions(open_only=True)}
    assert open_tickers == {"UNSETTLED", "OTHER-DAY"}
