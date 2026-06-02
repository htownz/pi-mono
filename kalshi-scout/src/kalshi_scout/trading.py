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

from kalshi_scout.kalshi import KalshiClient
from kalshi_scout.notify import Alert
from kalshi_scout.store import PositionRow, SnapshotRow, SnapshotStore


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


def _fill_count_from_response(order: dict, *, requested: int) -> int:
    """Best-effort fill count from Kalshi's order response.

    The field name has varied across Kalshi API versions; check the
    common shapes before falling back to status inference. Conservative:
    when nothing definitive is present, treat as 0 (resting) rather
    than assuming a full fill — better to under-record locally than to
    diverge from the broker. Shared by AutoTrader (buy fills) and
    PositionMonitor (sell fills)."""
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
        if resp.is_error:
            # Kalshi puts the actual rejection reason in the response BODY
            # (e.g. {"error":{"code":"market_not_active",...}}). The default
            # raise_for_status() discards it, leaving a useless bare
            # "400 Bad Request" in the audit log. Surface the body so a
            # failed order says WHY. Keep the exception type as
            # httpx.HTTPStatusError so existing `except` handlers are
            # unaffected.
            detail = self._error_detail(resp)
            raise httpx.HTTPStatusError(
                f"{resp.status_code} {resp.reason_phrase} for {path}: {detail}",
                request=resp.request,
                response=resp,
            )
        return resp.json() if resp.content else {}

    @staticmethod
    def _error_detail(resp: httpx.Response) -> str:
        """Extract the most useful human string from an error response body.

        Handles the common Kalshi shapes — {"error": {"code","message"}},
        top-level {"code"/"message"}, plain text — and falls back to a
        truncated raw body. Never raises; returns a placeholder if the body
        is empty or unparseable."""
        try:
            data = resp.json()
        except Exception:
            text = (resp.text or "").strip()
            return text[:300] if text else "(empty body)"
        if isinstance(data, dict):
            for container in (data.get("error"), data):
                if not isinstance(container, dict):
                    continue
                code = container.get("code") or container.get("error_code")
                msg = container.get("message") or container.get("detail")
                parts = [str(p) for p in (code, msg) if p]
                if parts:
                    return " — ".join(parts)
        try:
            return json.dumps(data)[:300]
        except Exception:
            return "(unparseable body)"

    def place_order(
        self,
        ticker: str,
        action: str,          # "buy" or "sell"
        side: str,            # "yes" or "no"
        count: int,           # contracts
        price_cents: int,     # for limit orders; 1..99
        order_type: str = "limit",
    ) -> dict:
        """POST a single order. Returns Kalshi's raw response body.

        Limit orders set `yes_price` or `no_price` to `price_cents` per
        Kalshi's API convention (the field name carries the side, NOT the
        direction — for both buy and sell of the NO side you set `no_price`).

        `action="sell"` closes an existing position by selling contracts of
        the side we hold back into the book. A sell limit priced at the
        current bid is marketable and should fill immediately as a taker;
        if the book moved and it rests, the caller is responsible for
        canceling it (see PositionMonitor) so it can't double-fill later.
        """
        if action not in ("buy", "sell"):
            raise ValueError(f"action must be 'buy' or 'sell', got {action!r}")
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

    def cancel_order(self, order_id: str) -> dict:
        """Cancel a resting order by id. Used by the position monitor to
        retract a sell limit that didn't fill immediately, so it can't
        fill on a later book move and create an unintended short."""
        return self._request(
            "DELETE", f"/trade-api/v2/portfolio/orders/{order_id}"
        )

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
    #: Hard cap on TOTAL cost basis across ALL open positions, in cents.
    #: The per-event cap limits any single event, but with a broad eligible
    #: universe (many cities × metrics × brackets × multiple days) the bot
    #: can deploy most of a bankroll while every individual order stays
    #: under the per-order and per-event caps. This is the portfolio-level
    #: backstop. 0 = unlimited (backward-compatible default); set it to a
    #: fraction of bankroll (e.g. 40%) in production.
    max_total_deployment_cents: int = 0
    #: Refuse a YES buy when we already hold YES on a DIFFERENT bracket of
    #: the same event. Kalshi temperature brackets are mutually exclusive
    #: (the day's extremum lands in exactly one), so holding YES on two
    #: brackets of one event guarantees at least one loses. NO-side stacking
    #: is fine (many brackets can all settle NO) and is unaffected.
    mex_guard_yes_siblings: bool = True


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

    def _total_open_cost_cents(self) -> int:
        """Sum of cost_basis_cents across ALL currently-open positions —
        used for the portfolio-level deployment cap."""
        return sum(
            p.cost_basis_cents for p in self.store.query_positions(open_only=True)
        )

    def _holds_side_on_sibling_bracket(
        self, event_ticker: str, market_ticker: str, side: str,
    ) -> bool:
        """True if an open position holds `side` on a DIFFERENT bracket
        (market_ticker) of the same event. Used by the MEX guard to block
        guaranteed-loss YES stacking across mutually-exclusive brackets."""
        for p in self.store.query_positions(open_only=True):
            if (
                p.event_ticker == event_ticker
                and p.market_ticker != market_ticker
                and p.side == side
            ):
                return True
        return False

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

        # Portfolio-level deployment cap (0 = unlimited). This is the
        # backstop that the per-event cap can't provide: it bounds total
        # capital at risk across the whole open book, not just one event.
        if self.limits.max_total_deployment_cents > 0:
            total_open = self._total_open_cost_cents()
            if total_open + cost > self.limits.max_total_deployment_cents:
                return RiskDecision(
                    False,
                    f"total deployment {total_open + cost}c "
                    f"(open {total_open}c + this {cost}c) > "
                    f"max_total_deployment_cents "
                    f"{self.limits.max_total_deployment_cents}",
                )

        # MEX guard: refuse YES on a bracket when we already hold YES on a
        # sibling bracket of the same event. Only one bracket can settle YES,
        # so the second YES is a guaranteed loss. NO-side stacking is allowed
        # (many brackets settle NO) — the guard is YES-only by design.
        if (
            self.limits.mex_guard_yes_siblings
            and side == "yes"
            and self._holds_side_on_sibling_bracket(
                snap.event_ticker, snap.market_ticker, "yes"
            )
        ):
            return RiskDecision(
                False,
                f"MEX guard: already hold YES on a sibling bracket of "
                f"{snap.event_ticker}; only one bracket can settle YES "
                f"(guaranteed-loss avoidance)",
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
        kalshi_client: Optional[KalshiClient] = None,
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
        # Read-only Kalshi client (unauthenticated, separate from trading
        # client) used to refresh the quote right before placement when
        # `refresh_quote=True` is passed to maybe_trade. Snapshots are
        # typically 5+ minutes old and the book can move significantly in
        # that window.
        self.kalshi_client = kalshi_client
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
        refresh_quote: bool = False,
    ) -> TradeAttempt:
        """Single-alert entry. Returns the TradeAttempt regardless of
        outcome; the audit log captures it for replay.

        `side_override` / `price_override` bypass the snapshot-derivation
        for callers that have already resolved them (e.g. the `fire` CLI
        command, where the operator explicitly picked the side at the
        confirmation prompt). Without these the caller is at the mercy of
        the snapshot, which may be stale or disagree with the operator's
        intent.

        `refresh_quote=True` makes a live `KalshiClient.get_market()` call
        immediately before placement and uses the live ask instead of the
        snapshot's. This guards against acting on a snapshot that's 5+
        minutes old when the book has moved. When the live price differs
        from the snapshot's by more than 5c, a "stale-snapshot drift" note
        is appended to the audit reason so the operator can see the
        snapshot pipeline is falling behind. `price_override` (when set)
        takes precedence over the live refresh.
        """
        size = size if size is not None else self.default_size
        stale_note: Optional[str] = None
        if side_override is not None and price_override is not None:
            side, price = side_override, price_override
        else:
            side, snap_price = self._derive_side_and_price(snap)
            price = snap_price
            if (
                refresh_quote
                and side is not None
                and self.kalshi_client is not None
            ):
                live_price = self._fetch_live_ask(alert.market_ticker, side)
                if live_price is not None and live_price > 0:
                    if snap_price is not None:
                        drift = live_price - snap_price
                        if abs(drift) > 5:
                            stale_note = (
                                f"stale-snapshot drift: snap {snap_price}c "
                                f"→ live {live_price}c ({drift:+d}c)"
                            )
                    price = live_price

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
        if stale_note:
            note_parts.append(stale_note)
        position_id = self.store.add_position(
            market_ticker=alert.market_ticker,
            event_ticker=alert.event_ticker,
            side=side, size_contracts=filled_count, avg_price_cents=price,
            notes="; ".join(note_parts),
        )

        reason = "placed" + (" (paper)" if self.paper else "")
        if filled_count < size:
            reason += f" — partial fill {filled_count}/{size}"
        if stale_note:
            reason += f"; {stale_note}"
        attempt = TradeAttempt(
            **attempt_base, side=side, price_cents=price,
            size_contracts=filled_count, placed=True, reason=reason,
            order_id=order_id, position_id=position_id,
        )
        self._audit(attempt)
        return attempt

    def _fetch_live_ask(self, ticker: str, side: str) -> Optional[int]:
        """Pull the live `yes_ask` / `no_ask` for `ticker` from the read-only
        Kalshi client. Returns None on any failure — the caller falls back
        to the snapshot's ask, with a "no live quote" note in the audit log.
        """
        try:
            market = self.kalshi_client.get_market(ticker)
        except Exception:
            return None
        return market.yes_ask if side == "yes" else market.no_ask

    @staticmethod
    def _fill_count_from_response(order: dict, *, requested: int) -> int:
        """Delegates to the module-level helper. Kept as a staticmethod for
        backward compatibility with callers/tests that reference it here."""
        return _fill_count_from_response(order, requested=requested)

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


# -- In-flight position monitor ---------------------------------------------

@dataclass(frozen=True)
class ExitAttempt:
    """One end-to-end attempt to close an open position: take-profit or
    cut-loss. Logged to the exit audit jsonl regardless of outcome.

    Sibling of `TradeAttempt` for the entry side. Lives in its own jsonl
    so the entry audit's daily totals stay aligned with the existing
    `audit` command and dashboard panel; exits are a separate ledger.
    """
    fired_at_utc: datetime
    position_id: int
    market_ticker: str
    event_ticker: str
    side: str                        # 'yes' or 'no' — the side we hold
    size_contracts: int
    open_price_cents: int
    exit_price_cents: int            # snapshot's bid on our side at decision
    realized_pnl_cents: int          # (exit - open) × size
    reason: str                      # take_profit | cut_loss_state_flip | live_skipped
    closed: bool                     # did the close actually go through?
    paper: bool
    snap_id: Optional[int]

    def to_json_dict(self) -> dict:
        return {
            "fired_at_utc": self.fired_at_utc.astimezone(timezone.utc).isoformat(),
            "position_id": self.position_id,
            "market_ticker": self.market_ticker,
            "event_ticker": self.event_ticker,
            "side": self.side,
            "size_contracts": self.size_contracts,
            "open_price_cents": self.open_price_cents,
            "exit_price_cents": self.exit_price_cents,
            "realized_pnl_cents": self.realized_pnl_cents,
            "reason": self.reason,
            "closed": self.closed,
            "paper": self.paper,
            "snap_id": self.snap_id,
        }


class PositionMonitor:
    """Walks open positions and applies two exit triggers each scan:

      1. Take-profit: when the current bid for our side reaches a high
         threshold (default 95c), close to lock in the gain and free
         capital for the next opportunity. Hold-to-expiration captures
         ~5c more per contract; closing early ~7 hours earlier on a
         typical day frees that capital sooner — higher hourly ROI on
         the freed bankroll usually wins.

      2. Cut-loss on state flip: when the snapshot's state machine now
         says the bracket settled against the side we hold (NO position
         + state==LOCKED_YES; YES + state==DEAD_NO), close at the
         current bid. The forecast was wrong; capturing whatever the
         opposite side will still pay beats holding to a $0 settlement.

    Both triggers are snapshot-driven — the scan that just ran wrote
    fresh quotes and state. No extra Kalshi calls needed to DECIDE.

    Execution:
      - Paper mode: close locally via `store.close_position`.
      - Live mode WITHOUT a `trading_client`: log `reason='live_skipped'`
        (the monitor decides but can't act — useful for observing what it
        would do before arming real exits).
      - Live mode WITH a `trading_client`: place a sell limit at the
        current bid (marketable → fills immediately as a taker). On a
        full fill, close the position locally at the fill price. If the
        book moved and the order rests, CANCEL it immediately so it can't
        fill on a later scan and create an unintended short — then leave
        the position open for the next scan to retry fresh. This
        cancel-on-no-fill is the core safety against double-selling a
        position that has no order-tracking state in the local store.
    """

    def __init__(
        self,
        store: SnapshotStore,
        take_profit_bid_cents: int = 95,
        cut_loss_on_state_flip: bool = True,
        paper: bool = True,
        audit_log_path: Optional[Path] = None,
        trading_client: Optional["KalshiTradingClient"] = None,
    ) -> None:
        if not (0 < take_profit_bid_cents < 100):
            raise ValueError(
                f"take_profit_bid_cents must be in (0, 100), got "
                f"{take_profit_bid_cents}"
            )
        self.store = store
        self.take_profit_bid_cents = take_profit_bid_cents
        self.cut_loss_on_state_flip = cut_loss_on_state_flip
        self.paper = paper
        self.audit_log_path = audit_log_path
        self.trading_client = trading_client

    def run(self, now_utc: Optional[datetime] = None) -> tuple[int, int]:
        """Returns (n_closed, n_examined). Examined = open positions checked."""
        now_utc = now_utc or datetime.now(timezone.utc)
        n_closed = 0
        positions = self.store.query_positions(open_only=True)
        for position in positions:
            decision = self._should_close(position)
            if decision is None:
                continue
            exit_price_cents, reason, snap_id = decision
            attempt = self._close(position, exit_price_cents, reason, snap_id, now_utc)
            if attempt.closed:
                n_closed += 1
            if self.audit_log_path is not None:
                self._write_audit(attempt)
        return n_closed, len(positions)

    def _should_close(
        self, position: PositionRow,
    ) -> Optional[tuple[int, str, Optional[int]]]:
        """Returns (exit_price_cents, reason, snap_id) or None to hold."""
        latest = self.store.query_snapshots(
            market_ticker=position.market_ticker, limit=1,
        )
        if not latest:
            return None
        snap = latest[0]
        # Pick the bid for our side. NULLs mean no resting order on that
        # side — we can't realize anything, so hold.
        if position.side == "no":
            bid = snap.no_bid
        else:
            bid = snap.yes_bid
        if bid is None:
            return None

        # 1. Take-profit
        if bid >= self.take_profit_bid_cents:
            return bid, "take_profit", snap.id

        # 2. State-flip cut-loss
        if self.cut_loss_on_state_flip:
            adverse = (
                (position.side == "no" and snap.state == "locked_yes")
                or (position.side == "yes" and snap.state == "dead_no")
            )
            if adverse:
                return bid, "cut_loss_state_flip", snap.id
        return None

    def _close(
        self,
        position: PositionRow,
        exit_price_cents: int,
        reason: str,
        snap_id: Optional[int],
        now_utc: datetime,
    ) -> ExitAttempt:
        realized = (exit_price_cents - position.avg_price_cents) * position.size_contracts

        def _attempt(reason_: str, closed: bool, paper: bool) -> ExitAttempt:
            return ExitAttempt(
                fired_at_utc=now_utc, position_id=position.id,
                market_ticker=position.market_ticker,
                event_ticker=position.event_ticker, side=position.side,
                size_contracts=position.size_contracts,
                open_price_cents=position.avg_price_cents,
                exit_price_cents=exit_price_cents,
                realized_pnl_cents=realized,
                reason=reason_, closed=closed, paper=paper, snap_id=snap_id,
            )

        if self.paper:
            ok = self.store.close_position(
                position.id, closed_at=now_utc, at_price_cents=exit_price_cents,
            )
            return _attempt(reason, closed=ok, paper=True)

        # -- Live --------------------------------------------------------
        if self.trading_client is None:
            # No client wired (live exits not armed) — decide but don't act.
            return _attempt("live_skipped", closed=False, paper=False)
        return self._close_live(position, exit_price_cents, reason, _attempt)

    def _close_live(
        self,
        position: PositionRow,
        exit_price_cents: int,
        reason: str,
        _attempt,
    ) -> ExitAttempt:
        """Place a real sell limit at the bid; reconcile by actual fill.

        Fills fully → close locally. Doesn't fill → cancel so the resting
        order can't double-fill on a later scan, then leave open to retry.
        Any API exception leaves the position untouched (safe: we'd rather
        hold to settlement than corrupt local state on a flaky call).
        """
        assert self.trading_client is not None
        size = position.size_contracts
        try:
            resp = self.trading_client.place_order(
                ticker=position.market_ticker,
                action="sell",
                side=position.side,
                count=size,
                price_cents=exit_price_cents,
                order_type="limit",
            )
        except Exception as exc:
            return _attempt(f"sell_error: {exc}", closed=False, paper=False)

        order = resp.get("order") or {}
        order_id = order.get("order_id")
        filled = _fill_count_from_response(order, requested=size)

        if filled >= size:
            ok = self.store.close_position(
                position.id, at_price_cents=exit_price_cents,
            )
            tag = "" if order_id is None else f" order_id={order_id}"
            return _attempt(f"{reason} (live sold {filled}/{size}){tag}",
                            closed=ok, paper=False)

        # Not fully filled — retract so it can't fill later (double-sell guard).
        cancel_note = ""
        if order_id is not None:
            try:
                self.trading_client.cancel_order(order_id)
                cancel_note = "; canceled unfilled remainder"
            except Exception as exc:
                # Loud: a lingering resting sell is the one thing that could
                # double-fill. Surface it so the operator can cancel by hand.
                cancel_note = f"; WARN cancel FAILED ({exc}) — check Kalshi UI"
        return _attempt(
            f"{reason}_unfilled (sold {filled}/{size}){cancel_note}",
            closed=False, paper=False,
        )

    def _write_audit(self, attempt: ExitAttempt) -> None:
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
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
