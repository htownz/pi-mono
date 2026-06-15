"""Alert delivery: fire when a contract grades at or above a threshold.

Sinks are intentionally tiny. `StdoutSink` is for humans watching a terminal;
`JsonlSink` appends one JSON object per alert for dashboards / downstream
tooling. The dispatcher fires every evaluation at or above `min_grade`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol

from weather_trader.grade import GRADE_ORDER
from weather_trader.models import Evaluation


@dataclass
class Alert:
    ts_utc: str
    ticker: str
    event: str
    city: str
    metric: str
    market_date: str
    bracket: str
    grade: str
    side: Optional[str]
    edge_cents: Optional[float]
    fair_mid: float
    yes_ask_cents: Optional[int]
    no_ask_cents: Optional[int]
    forecast_mean_f: Optional[float]
    band_width_f: Optional[float]
    locked: bool


def alert_from_eval(e: Evaluation, now_utc: Optional[datetime] = None) -> Alert:
    now_utc = now_utc or datetime.now(timezone.utc)
    return Alert(
        ts_utc=now_utc.isoformat(),
        ticker=e.market.ticker,
        event=e.market.event_ticker,
        city=e.contract.city_slug,
        metric=e.contract.metric.value,
        market_date=e.contract.market_date.isoformat(),
        bracket=e.contract.bracket.label(),
        grade=e.grade,
        side=e.best_side,
        edge_cents=round(e.best_edge * 100, 1) if e.best_edge is not None else None,
        fair_mid=round(e.fair_prob_mid, 3),
        yes_ask_cents=e.yes_ask_cents,
        no_ask_cents=e.no_ask_cents,
        forecast_mean_f=round(e.forecast_mean_f, 1) if e.forecast_mean_f is not None else None,
        band_width_f=round(e.band_width_f, 1) if e.band_width_f is not None else None,
        locked=e.locked,
    )


class AlertSink(Protocol):
    def emit(self, alert: Alert) -> None: ...


class StdoutSink:
    def __init__(self, stream=None) -> None:
        self._stream = stream

    def emit(self, alert: Alert) -> None:
        side = alert.side or "—"
        edge = f"{alert.edge_cents:+.1f}c" if alert.edge_cents is not None else "—"
        line = (
            f"[{alert.grade}] {alert.ticker}  {alert.bracket}  "
            f"{side} edge {edge}  fair {alert.fair_mid * 100:.0f}%  "
            f"mean {alert.forecast_mean_f}°F"
        )
        print(line, file=self._stream)


class JsonlSink:
    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def emit(self, alert: Alert) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(alert)) + "\n")


class AlertDispatcher:
    def __init__(self, sinks: list[AlertSink], min_grade: str = "B") -> None:
        self.sinks = sinks
        if min_grade not in GRADE_ORDER:
            raise ValueError(f"min_grade must be one of {GRADE_ORDER}")
        self.cutoff = GRADE_ORDER.index(min_grade)

    def dispatch(self, evals: list[Evaluation], now_utc: Optional[datetime] = None) -> list[Alert]:
        fired: list[Alert] = []
        for e in evals:
            if e.grade not in GRADE_ORDER or GRADE_ORDER.index(e.grade) > self.cutoff:
                continue
            alert = alert_from_eval(e, now_utc=now_utc)
            for sink in self.sinks:
                sink.emit(alert)
            fired.append(alert)
        return fired
