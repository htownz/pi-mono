"""Authenticated Kalshi trading client + risk-gated auto-trader.

This module is what turns the scout from a read-only signal generator into
an actual bot. Three layers:

  KalshiTradingClient — RSA-PSS-SHA256-signed requests to the trading
                        endpoints (place_order, get_balance, get_positions).
                        Auth lives here and nowhere else.

  RiskGuard          — pre-flight checks every order has to pass:
                          - per-order size + cost cap
                          - per-event concentration cap
                          - rolling daily-loss kill threshold
                          - minimum edge requirement (above the engine's)
                          - rounding-risk filter for dead_no / locked_yes
                            alerts whose extremum is dangerously close to
                            the bracket boundary (the trap from the
                            KXLOWTDC-26MAY30-T62 debrief)
                          - kill-switch file (panic stop)

  AutoTrader         — orchestrator. Given an Alert + the snapshot that
                       produced it, derives side/price, asks RiskGuard,
                       places the order (or paper-logs it), and records
                       the resulting position to the snapshot store.

The trading API base URL is configurable so the user can point at Kalshi's
demo environment for testing before going live.

Safety patterns:
  - Kill file checked before EVERY order, not just on startup.
  - All risk limits returned as (allowed, reason) — no silent refusals.
  - `paper=True` short-circuits the API call so the audit trail builds
    without real money at stake.
  - Audit log (JSONL) captures every order attempt + decision for replay.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from kalshi_scout.notify import Alert
from kalshi_scout.store import SnapshotRow, SnapshotStore


KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"


# -- Auth + transport --------------------------------------------------------

def _signature_payload(timestamp_ms: int, method: str, path: str) -> str:
    """Kalshi's signed payload: `f"{ts_ms}{METHOD}{path}"`. `path` is the URL
    path including the `/trade-api/v2/...` prefix but excluding any query
    string."""
    return f"{timestamp_ms}{method.upper()}{path}"


def _load_private_key(path: Path) -> rsa.RSAPrivateKey:
    """Load an RSA private key from a PEM file. Raises if the key is
    encrypted (Kalshi-issued keys are unencrypted by default)."""
    pem = Path(path).read_bytes()
    key = serialization.load_pem_private_key(pem, password=None)
    if not isinstance(key, rsa.RSAPrivateKey):
        raise ValueError(f"key at {path} is not an RSA private key")
    return key


def _sign(key: rsa.RSAPrivateKey, payload: str) -> str:
    """RSA-PSS-SHA256 signature with SHA-256-length salt, base64-encoded."""
    sig = key.sign(
        payload.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode("ascii")


class KalshiTradingClient:
    """Authenticated HTTP client for Kalshi's trading endpoints.

    The signing payload is `f"{ts_ms}{METHOD}{path}"` where `path` includes
    the `/trade-api/v2/...` prefix. The signature is sent in the
    `KALSHI-ACCESS-SIGNATURE` header alongside the key id + timestamp.
    """
    def __init__(
        self,
        key_id: str,
        private_key_path: Path,
        base_url: str = KALSHI_API_BASE,
        timeout: float = 10.0,
        client: Optional[httpx.Client] = None,
    ):
        self.key_id = key_id
        self._key = _load_private_key(private_key_path)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = client or httpx.Client(timeout=timeout)

    def _headers(self, method: str, path: str) -> dict:
        ts = int(time.time() * 1000)
        sig = _sign(self._key, _signature_payload(ts, method, path))
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": str(ts),
            "KALSHI-ACCESS-SIGNATURE": sig,
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, json_body: Optional[dict] = None) -> dict:
        # `path` is the API path including /trade-api/v2/... so the signature
        # matches exactly what the server sees.
        url = self.base_url.rsplit("/trade-api/v2", 1)[0] + path
        headers = self._headers(method, path)
        resp = self._client.request(
            method, url, headers=headers,
            json=json_body if json_body is not None else None,
        )
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def place_order(
        self,
        ticker: str,
        action: str,          # "buy" — we don't expose sell here yet
        side: str,            # "yes" or "no"
        count: int,           # contracts
        price_cents: int,     # for limit orders; 1..99
        order_type: str = "limit",
    ) -> dict:
        """POST a single order. Returns Kalshi's raw response body.

        Limit orders set `yes_price` or `no_price` to `price_cents` per
        Kalshi's API convention (the field name carries the side).
        """
        if action not in ("buy",):
            raise ValueError(f"action must be 'buy', got {action!r}")
        if side not in ("yes", "no"):
            raise ValueError(f"side must be 'yes' or 'no', got {side!r}")
        if order_type not in ("limit", "market"):
            raise ValueError(f"order_type must be 'limit' or 'market', got {order_type!r}")
        if count <= 0:
            raise ValueError(f"count must be > 0, got {count}")
        body = {
            "ticker": ticker, "action": action, "side": side,
            "count": count, "type": order_type,
        }
        if order_type == "limit":
            if not (0 < price_cents < 100):
                raise ValueError(f"price_cents must be 1..99 for limit, got {price_cents}")
            body["yes_price" if side == "yes" else "no_price"] = price_cents
        return self._request("POST", "/trade-api/v2/portfolio/orders", json_body=body)

    def get_balance_cents(self) -> int:
        """Returns the cleared cash balance in cents."""
        resp = self._request("GET", "/trade-api/v2/portfolio/balance")
        # Kalshi returns `balance` in cents (integer).
        return int(resp.get("balance", 0))

    def get_positions(self) -> list[dict]:
        """List open portfolio positions."""
        resp = self._request("GET", "/trade-api/v2/portfolio/positions")
        return list(resp.get("market_positions", []))

    def close(self) -> None:
        self._client.close()


# -- Risk + kill switch ------------------------------------------------------

class KillSwitch:
    """File-based emergency halt. Existence of `kill_path` blocks every order.

    Use `touch /data/scout.kill` to halt the bot instantly without restarting
    the daemon. Remove the file to resume.
    """
    def __init__(self, kill_path: Path):
        self.kill_path = Path(kill_path)

    def is_active(self) -> bool:
        return self.kill_path.exists()


@dataclass(frozen=True)
class RiskLimits:
    """Per-order and per-day risk caps. Defaults match the "Small" preset:
    $5 per position, $50 daily loss kill, $25 per event, 5 contracts per
    order. Bump these once you've watched the bot run cleanly for a week."""
    max_position_size_contracts: int = 5
    max_position_cost_cents: int = 500
    max_daily_loss_cents: int = 5000
    max_concentration_per_event_cents: int = 2500
    #: Refuse orders whose snapshot edge (cents) is below this; the engine's
    #: default min is 1c which is too tight to absorb fees + slippage.
    min_edge_cents: int = 5
    #: For dead_no/locked_yes alerts, refuse if the running extremum is
    #: within this many °F of the bracket boundary. The CLI report rounds
    #: to whole degrees, so an observation right at the boundary can flip
    #: the outcome under standard rounding.
    rounding_risk_buffer_f: float = 0.5


@dataclass(frozen=True)
class RiskDecision:
    """Outcome of a pre-trade check. `allowed=False` carries the human
    reason so the audit log captures why the bot didn't act."""
    allowed: bool
    reason: str


class RiskGuard:
    """Pre-trade risk checks. Stateful — tracks today's realized P&L from
    the snapshot store so the daily-loss kill threshold can engage."""
    def __init__(
        self,
        limits: RiskLimits,
        store: SnapshotStore,
        kill_switch: KillSwitch,
    ):
        self.limits = limits
        self.store = store
        self.kill_switch = kill_switch

    def _today_realized_loss_cents(self) -> int:
        """Sum of |realized_pnl_cents| across positions closed today (UTC)
        that lost money. Only the losing side counts toward the kill
        threshold; winners don't extend it."""
        today_start = datetime.combine(
            datetime.now(timezone.utc).date(), datetime.min.time(),
            tzinfo=timezone.utc,
        )
        loss = 0
        for p in self.store.query_positions(open_only=False):
            if p.closed_at_utc is None or p.closed_at_utc < today_start:
                continue
            pnl = p.realized_pnl_cents
            if pnl is None or pnl >= 0:
                continue
            loss += -pnl
        return loss

    def _open_cost_on_event_cents(self, event_ticker: str) -> int:
        """Sum of cost_basis_cents across currently-open positions on the
        same event_ticker — used for the concentration cap."""
        total = 0
        for p in self.store.query_positions(open_only=True):
            if p.event_ticker == event_ticker:
                total += p.cost_basis_cents
        return total

    def _rounding_risk(self, snap: SnapshotRow) -> Optional[str]:
        """For dead_no / locked_yes, check the observed extremum's distance
        from the bracket boundary. Returns a refusal reason if the gap is
        within `rounding_risk_buffer_f`, else None.

        The state machine treats the raw 5-min METAR observation as the
        settlement value, but Kalshi settles off the CLI Climatological
        Daily Report which rounds to whole degrees. An observation within
        0.5°F of the bracket boundary can flip outcomes under standard
        nearest-integer rounding.
        """
        bracket_lo, bracket_hi = snap.bracket_lo, snap.bracket_hi
        buffer = self.limits.rounding_risk_buffer_f
        if snap.state == "dead_no":
            # The observed max (for HIGH) or min (for LOW) crossed a bracket
            # boundary against the Yes side. Check the closest one.
            if snap.metric == "high" and snap.running_max_f is not None and bracket_hi is not None:
                # HIGH dead_no: running_max > bracket_hi. Distance is
                # (running_max - bracket_hi). Tight means risky.
                gap = snap.running_max_f - bracket_hi
                if gap < buffer:
                    return (
                        f"rounding risk: HIGH running_max {snap.running_max_f:g}°F "
                        f"only {gap:g}°F above bracket cap {bracket_hi:g}°F "
                        f"(< {buffer}°F buffer); CLI report could round below"
                    )
            if snap.metric == "low" and snap.running_min_f is not None and bracket_lo is not None:
                # LOW dead_no: running_min < bracket_lo.
                gap = bracket_lo - snap.running_min_f
                if gap < buffer:
                    return (
                        f"rounding risk: LOW running_min {snap.running_min_f:g}°F "
                        f"only {gap:g}°F below bracket floor {bracket_lo:g}°F "
                        f"(< {buffer}°F buffer); CLI report could round above"
                    )
        elif snap.state == "locked_yes":
            # Symmetric: HIGH locked_yes means running_max already reached
            # bracket_lo, so the danger is the CLI rounding it DOWN.
            if snap.metric == "high" and snap.running_max_f is not None and bracket_lo is not None:
                gap = snap.running_max_f - bracket_lo
                if gap < buffer:
                    return (
                        f"rounding risk: HIGH running_max {snap.running_max_f:g}°F "
                        f"only {gap:g}°F above bracket floor {bracket_lo:g}°F "
                        f"(< {buffer}°F buffer); CLI report could round below"
                    )
            if snap.metric == "low" and snap.running_min_f is not None and bracket_hi is not None:
                gap = bracket_hi - snap.running_min_f
                if gap < buffer:
                    return (
                        f"rounding risk: LOW running_min {snap.running_min_f:g}°F "
                        f"only {gap:g}°F below bracket cap {bracket_hi:g}°F "
                        f"(< {buffer}°F buffer); CLI report could round above"
                    )
        return None

    def can_place(
        self,
        snap: SnapshotRow,
        side: str,
        price_cents: int,
        size_contracts: int,
    ) -> RiskDecision:
        """Run every check in priority order. First failure short-circuits."""
        if self.kill_switch.is_active():
            return RiskDecision(False, f"kill switch active ({self.kill_switch.kill_path})")

        if size_contracts <= 0:
            return RiskDecision(False, f"size {size_contracts} <= 0")
        if size_contracts > self.limits.max_position_size_contracts:
            return RiskDecision(
                False,
                f"size {size_contracts} > max_position_size_contracts "
                f"{self.limits.max_position_size_contracts}",
            )

        cost = size_contracts * price_cents
        if cost > self.limits.max_position_cost_cents:
            return RiskDecision(
                False,
                f"cost {cost}c > max_position_cost_cents "
                f"{self.limits.max_position_cost_cents}",
            )

        existing = self._open_cost_on_event_cents(snap.event_ticker)
        if existing + cost > self.limits.max_concentration_per_event_cents:
            return RiskDecision(
                False,
                f"event concentration {existing + cost}c "
                f"(existing {existing}c + this {cost}c) > "
                f"max_concentration_per_event_cents "
                f"{self.limits.max_concentration_per_event_cents}",
            )

        today_loss = self._today_realized_loss_cents()
        if today_loss >= self.limits.max_daily_loss_cents:
            return RiskDecision(
                False,
                f"daily loss kill: realized {today_loss}c today >= "
                f"max_daily_loss_cents {self.limits.max_daily_loss_cents}",
            )

        edge_cents = self._edge_for_side(snap, side, price_cents)
        if edge_cents is None or edge_cents < self.limits.min_edge_cents:
            return RiskDecision(
                False,
                f"edge {edge_cents}c < min_edge_cents {self.limits.min_edge_cents}",
            )

        round_risk = self._rounding_risk(snap)
        if round_risk is not None:
            return RiskDecision(False, round_risk)

        return RiskDecision(True, "ok")

    @staticmethod
    def _edge_for_side(snap: SnapshotRow, side: str, price_cents: int) -> Optional[int]:
        """Edge in cents = (fair * 100 - price) for yes side, mirrored for no.

        Uses the snapshot's fair_prob midpoint; rounded to int cents.
        """
        fair_mid = (snap.fair_prob_low + snap.fair_prob_high) / 2.0
        if side == "yes":
            return int(round(fair_mid * 100 - price_cents))
        return int(round((1.0 - fair_mid) * 100 - price_cents))


# -- Orchestrator ------------------------------------------------------------

@dataclass(frozen=True)
class TradeAttempt:
    """One end-to-end attempt: from alert to (placed | refused | failed).
    Logged to the audit jsonl regardless of outcome."""
    fired_at_utc: datetime
    market_ticker: str
    event_ticker: str
    side: str
    price_cents: int
    size_contracts: int
    placed: bool
    paper: bool
    reason: str        # refusal reason or "ok" or API error
    order_id: Optional[str]
    position_id: Optional[int]
    snap_id: int
    grade: str

    def to_json_dict(self) -> dict:
        return {
            "fired_at_utc": self.fired_at_utc.astimezone(timezone.utc).isoformat(),
            "market_ticker": self.market_ticker,
            "event_ticker": self.event_ticker,
            "side": self.side,
            "price_cents": self.price_cents,
            "size_contracts": self.size_contracts,
            "cost_cents": self.size_contracts * self.price_cents,
            "placed": self.placed,
            "paper": self.paper,
            "reason": self.reason,
            "order_id": self.order_id,
            "position_id": self.position_id,
            "snap_id": self.snap_id,
            "grade": self.grade,
        }


class AutoTrader:
    """Orchestrates alert → derive side/price → risk check → place order →
    record position. Stateless wrt market data — every call re-pulls the
    latest snapshot from the store.

    `paper=True` short-circuits the trading-API call so the rest of the
    pipeline (risk guard, audit log, position recording) runs unchanged.
    Use this for a multi-day soak before going live.
    """
    def __init__(
        self,
        client: Optional[KalshiTradingClient],
        guard: RiskGuard,
        store: SnapshotStore,
        default_size: int = 1,
        order_type: str = "limit",
        paper: bool = False,
        audit_log_path: Optional[Path] = None,
    ):
        if not paper and client is None:
            raise ValueError("client is required when paper=False")
        self.client = client
        self.guard = guard
        self.store = store
        self.default_size = default_size
        self.order_type = order_type
        self.paper = paper
        self.audit_log_path = Path(audit_log_path) if audit_log_path else None
        if self.audit_log_path:
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    def maybe_trade(
        self,
        alert: Alert,
        snap: SnapshotRow,
        size: Optional[int] = None,
        *,
        side_override: Optional[str] = None,
        price_override: Optional[int] = None,
    ) -> TradeAttempt:
        """Single-alert entry. Returns the TradeAttempt regardless of
        outcome; the audit log captures it for replay.

        `side_override` / `price_override` bypass the snapshot-derivation
        for callers that have already resolved them (e.g. the `fire` CLI
        command, where the operator explicitly picked the side at the
        confirmation prompt). Without these the caller is at the mercy of
        the snapshot, which may be stale or disagree with the operator's
        intent.
        """
        size = size if size is not None else self.default_size
        if side_override is not None and price_override is not None:
            side, price = side_override, price_override
        else:
            side, price = self._derive_side_and_price(snap)

        attempt_base = dict(
            fired_at_utc=alert.fired_at_utc,
            market_ticker=alert.market_ticker,
            event_ticker=alert.event_ticker,
            snap_id=snap.id,
            grade=snap.grade,
            paper=self.paper,
        )

        if side is None or price is None or price <= 0 or price >= 100:
            attempt = TradeAttempt(
                **attempt_base, side=side or "—",
                price_cents=price or 0, size_contracts=size,
                placed=False,
                reason=f"no fillable {side or 'side'} on snapshot (price {price})",
                order_id=None, position_id=None,
            )
            self._audit(attempt)
            return attempt

        decision = self.guard.can_place(snap, side, price, size)
        if not decision.allowed:
            attempt = TradeAttempt(
                **attempt_base, side=side, price_cents=price,
                size_contracts=size, placed=False,
                reason=decision.reason, order_id=None, position_id=None,
            )
            self._audit(attempt)
            return attempt

        order_id: Optional[str] = None
        filled_count = size   # paper mode assumes full fill
        order_status: Optional[str] = None
        if not self.paper:
            try:
                resp = self.client.place_order(
                    ticker=alert.market_ticker, action="buy", side=side,
                    count=size, price_cents=price, order_type=self.order_type,
                )
            except Exception as exc:
                attempt = TradeAttempt(
                    **attempt_base, side=side, price_cents=price,
                    size_contracts=size, placed=False,
                    reason=f"API error: {exc}",
                    order_id=None, position_id=None,
                )
                self._audit(attempt)
                return attempt
            order = resp.get("order") or {}
            order_id = order.get("order_id")
            order_status = (order.get("status") or "").lower() or None
            filled_count = self._fill_count_from_response(order, requested=size)
            if filled_count == 0:
                # Limit accepted but resting on the book (or canceled).
                # Don't record a phantom position — the local store would
                # diverge from the broker. Surface the order id in the
                # audit so the operator can follow up via the Kalshi UI.
                attempt = TradeAttempt(
                    **attempt_base, side=side, price_cents=price,
                    size_contracts=0, placed=False,
                    reason=(
                        f"order {order_status or 'accepted'} but unfilled "
                        f"(0/{size} contracts); not recorded as position"
                    ),
                    order_id=order_id, position_id=None,
                )
                self._audit(attempt)
                return attempt

        # Record the position with the size that ACTUALLY filled. For paper
        # mode that's the requested size (no real broker to disagree). For
        # live, partial fills here mean the local store stays in sync with
        # the broker even when the book moved between snapshot and placement.
        note_parts = [
            f"auto-trade: grade={snap.grade} state={snap.state} "
            f"fair={snap.fair_prob_low * 100:.0f}-{snap.fair_prob_high * 100:.0f}%",
        ]
        if self.paper:
            note_parts.append("PAPER (not actually placed)")
        if order_id:
            note_parts.append(f"order_id={order_id}")
        if filled_count < size:
            note_parts.append(
                f"PARTIAL fill: {filled_count}/{size} (rest may be resting)"
            )
        position_id = self.store.add_position(
            market_ticker=alert.market_ticker,
            event_ticker=alert.event_ticker,
            side=side, size_contracts=filled_count, avg_price_cents=price,
            notes="; ".join(note_parts),
        )

        reason = "placed" + (" (paper)" if self.paper else "")
        if filled_count < size:
            reason += f" — partial fill {filled_count}/{size}"
        attempt = TradeAttempt(
            **attempt_base, side=side, price_cents=price,
            size_contracts=filled_count, placed=True, reason=reason,
            order_id=order_id, position_id=position_id,
        )
        self._audit(attempt)
        return attempt

    @staticmethod
    def _fill_count_from_response(order: dict, *, requested: int) -> int:
        """Best-effort fill count from Kalshi's order response.

        The field name has varied across Kalshi API versions; check the
        common shapes before falling back to status inference. Conservative:
        when nothing definitive is present, treat as 0 (resting) rather
        than assuming a full fill — better to under-record locally than to
        diverge from the broker.
        """
        for k in ("taker_fill_count", "fill_count", "filled_quantity", "filled_count"):
            v = order.get(k)
            if v is not None:
                try:
                    return max(0, min(requested, int(v)))
                except (TypeError, ValueError):
                    continue
        status = (order.get("status") or "").lower()
        if status == "executed":
            # Some API versions only return status; "executed" means fully
            # filled per Kalshi's documented order lifecycle.
            return requested
        if status in ("resting", "queued", "canceled", "cancelled"):
            return 0
        # Unknown shape — be defensive and assume nothing filled.
        return 0

    @staticmethod
    def _derive_side_and_price(snap: SnapshotRow) -> tuple[Optional[str], Optional[int]]:
        """Same logic as cli._derive_take_side but operates on a SnapshotRow.

        LOCKED_YES → buy yes at yes_ask.
        DEAD_NO    → buy no at no_ask.
        Other     → side with the larger fair edge; price is that side's ask.
        """
        if snap.state == "locked_yes":
            return "yes", snap.yes_ask
        if snap.state == "dead_no":
            return "no", snap.no_ask
        yes_e = snap.edge_yes if snap.edge_yes is not None else float("-inf")
        no_e = snap.edge_no if snap.edge_no is not None else float("-inf")
        if no_e > yes_e:
            return "no", snap.no_ask
        return "yes", snap.yes_ask

    def _audit(self, attempt: TradeAttempt) -> None:
        if self.audit_log_path is None:
            return
        with self.audit_log_path.open("a") as f:
            f.write(json.dumps(attempt.to_json_dict()) + "\n")


# -- Settlement-driven auto-close --------------------------------------------

def auto_close_settled_positions(
    store: SnapshotStore,
    on_settled_date: date,
) -> list[tuple[int, str, int]]:
    """For every open position whose market settled on `on_settled_date`,
    close it with the realized exit price (100 if won, 0 if lost).

    Returns a list of `(position_id, market_ticker, exit_price_cents)`
    for the operator's audit. Caller is responsible for printing or
    persisting that summary.
    """
    closed: list[tuple[int, str, int]] = []
    for pos in store.query_positions(open_only=True):
        settlement = store.get_settlement(pos.market_ticker)
        if settlement is None:
            continue
        if settlement.market_date != on_settled_date:
            continue
        side_won = (
            (pos.side == "yes" and settlement.resolved_yes)
            or (pos.side == "no" and not settlement.resolved_yes)
        )
        exit_price = 100 if side_won else 0
        if store.close_position(pos.id, at_price_cents=exit_price):
            closed.append((pos.id, pos.market_ticker, exit_price))
    return closed
