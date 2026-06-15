"""Order execution. Paper by default; live trading is a guarded TODO.

`PaperExecutor` records intended orders (optionally to a JSONL log) and never
touches the network — this is the default and what the CLI uses. It lets you
run the full forecast -> grade -> execute pipeline and review what the bot
*would* have done.

`LiveKalshiExecutor` is scaffolded but intentionally **not wired to place real
orders**. Authenticated Kalshi trading requires RSA-PSS request signing and
moves real money; enabling it is the next milestone and must be done
deliberately with credentials. Calling `submit` raises until then.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol

from weather_trader.models import Evaluation


@dataclass
class Order:
    ticker: str
    side: str                # "yes" | "no"
    count: int
    limit_price_cents: int   # max price to pay per contract
    reason: str


@dataclass
class Fill:
    order: Order
    status: str              # "paper" | "submitted" | "rejected"
    detail: str = ""


def order_from_eval(
    e: Evaluation, count: int = 1, min_edge: float = 0.05
) -> Optional[Order]:
    """Build a limit order on the contract's best side, or None if no edge.

    The limit is the current ask on that side — we only take positive expected
    value at a price already showing in the book.
    """
    side = e.best_side
    edge = e.best_edge
    if side is None or edge is None or edge < min_edge:
        return None
    price = e.yes_ask_cents if side == "yes" else e.no_ask_cents
    if price is None:
        return None
    return Order(
        ticker=e.market.ticker,
        side=side,
        count=count,
        limit_price_cents=price,
        reason=f"grade {e.grade}, {side} edge {edge * 100:+.1f}c, fair {e.fair_prob_mid * 100:.0f}%",
    )


class Executor(Protocol):
    def submit(self, order: Order) -> Fill: ...


class PaperExecutor:
    """Records intended orders without sending anything. The safe default."""

    def __init__(self, log_path: Optional[str] = None) -> None:
        self.log_path = Path(log_path) if log_path else None
        self.orders: list[Order] = []

    def submit(self, order: Order) -> Fill:
        self.orders.append(order)
        fill = Fill(order=order, status="paper", detail="paper trade — not sent")
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                row = {"ts_utc": datetime.now(timezone.utc).isoformat(), **asdict(order)}
                fh.write(json.dumps(row) + "\n")
        return fill


class LiveKalshiExecutor:
    """Placeholder for authenticated live trading. Deliberately inert.

    Wiring this up (RSA-PSS request signing, balance/position checks, real
    order POST) is the next milestone. Until then `submit` refuses to run so
    the bot can never spend money by accident.
    """

    def __init__(self, key_id: Optional[str] = None, private_key_pem: Optional[str] = None) -> None:
        self.key_id = key_id
        self._private_key_pem = private_key_pem

    def submit(self, order: Order) -> Fill:
        raise NotImplementedError(
            "Live Kalshi trading is not wired yet. Use PaperExecutor. "
            "Enabling live execution (RSA-PSS signing, real orders) is a "
            "deliberate next milestone — see README 'Execution safety'."
        )
