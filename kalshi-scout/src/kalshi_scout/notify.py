"""Alert delivery framework.

Alerts fire on **grade-improvement transitions**: a contract whose most
recent stored snapshot was at a worse grade than its current evaluation
(or had no prior snapshot at all). This makes alerts naturally replayable
(invariant I8): the snapshot store is the source of truth for "have we
already alerted on this".

Three sink types in this slice:

  StdoutSink   - print to console (always-on default during testing)
  JsonlSink    - append one JSON object per alert to a file
  WebhookSink  - POST JSON to a configurable URL (Slack/Discord/ntfy etc)

Adding new sinks: implement AlertSink.emit().
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol

import httpx

from kalshi_scout.models import ContractEvaluation
from kalshi_scout.store import SnapshotRow, SnapshotStore


GRADE_ORDER = ["A+", "A", "B+", "B", "C", "D", "F"]


def _grade_rank(grade: str) -> int:
    return GRADE_ORDER.index(grade) if grade in GRADE_ORDER else 99


def _is_better(new_grade: str, old_grade: Optional[str]) -> bool:
    """True if new_grade is strictly better than old_grade (or no old)."""
    if old_grade is None:
        return True
    return _grade_rank(new_grade) < _grade_rank(old_grade)


@dataclass(frozen=True)
class Alert:
    """One alert event. JSON-serializable; the natural transport unit."""
    fired_at_utc: datetime
    market_ticker: str
    event_ticker: str
    city_slug: str
    market_date: str
    bracket: str
    metric: str
    state: str
    reason: str
    grade: str
    previous_grade: Optional[str]
    yes_ask_cents: Optional[int]
    no_ask_cents: Optional[int]
    edge_yes: Optional[float]
    edge_no: Optional[float]
    fair_prob_low: float
    fair_prob_high: float
    notes: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict:
        d = asdict(self)
        d["fired_at_utc"] = self.fired_at_utc.astimezone(timezone.utc).isoformat()
        return d


# -- Sinks -------------------------------------------------------------------

class AlertSink(Protocol):
    def emit(self, alert: Alert) -> None: ...


class StdoutSink:
    """Prints a one-line summary to stdout — useful for shell pipelines."""
    def emit(self, alert: Alert) -> None:
        prev = alert.previous_grade or "—"
        edge = (
            f"yes_edge={alert.edge_yes:+.2f}" if alert.edge_yes is not None
            else f"no_edge={alert.edge_no:+.2f}" if alert.edge_no is not None
            else "edge=—"
        )
        print(
            f"[ALERT] {alert.fired_at_utc.strftime('%H:%M:%S')} "
            f"{alert.market_ticker} {alert.grade} (was {prev}) "
            f"state={alert.state} {edge}"
        )


class JsonlSink:
    """Appends one JSON-encoded alert per line to a file.

    Size-based rotation: when the current file exceeds `max_bytes`, it's
    renamed to `<path>.1`, `<path>.1` shifts to `<path>.2`, etc., up to
    `backup_count` historical files. The newest writes always land in the
    base path so external tail-followers don't need to chase renames.
    """
    def __init__(
        self,
        path: Path | str,
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
    ):
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _maybe_rotate(self) -> None:
        if self.max_bytes <= 0 or not self.path.exists():
            return
        if self.path.stat().st_size < self.max_bytes:
            return
        # Shift older backups up: .4 -> .5, .3 -> .4, ..., .1 -> .2
        for i in range(self.backup_count, 0, -1):
            src = self.path.with_suffix(self.path.suffix + f".{i}")
            if i == self.backup_count and src.exists():
                src.unlink()
                continue
            dst = self.path.with_suffix(self.path.suffix + f".{i + 1}")
            if src.exists():
                src.replace(dst)
        # Current -> .1
        self.path.replace(self.path.with_suffix(self.path.suffix + ".1"))

    def emit(self, alert: Alert) -> None:
        self._maybe_rotate()
        with self.path.open("a") as f:
            f.write(json.dumps(alert.to_json_dict()) + "\n")


class WebhookSink:
    """POSTs the alert JSON to a URL. Wire to Slack/Discord/ntfy/etc.

    By default uses a 5-second timeout and a single retry. Failures are
    logged via the failure_log callable (default: print) so a downed
    webhook never blocks scan completion.
    """
    def __init__(
        self,
        url: str,
        timeout: float = 5.0,
        headers: Optional[dict] = None,
        client: Optional[httpx.Client] = None,
        failure_log=print,
    ):
        self.url = url
        self.timeout = timeout
        self.headers = headers or {"Content-Type": "application/json"}
        self._client = client or httpx.Client(timeout=timeout)
        self._failure_log = failure_log

    def emit(self, alert: Alert) -> None:
        try:
            resp = self._client.post(
                self.url,
                json=alert.to_json_dict(),
                headers=self.headers,
            )
            resp.raise_for_status()
        except Exception as exc:
            self._failure_log(f"webhook {self.url} failed: {exc}")

    def close(self) -> None:
        self._client.close()


# -- Dispatcher --------------------------------------------------------------

class AlertDispatcher:
    """Decides which evaluations are 'transitions worth alerting on'.

    Transitions are detected by looking up the most recent prior snapshot
    for each evaluation's market_ticker in the SnapshotStore — alerts fire
    only when the new grade is strictly better than the prior, and meets
    the dispatcher's `min_grade` threshold.

    No prior snapshot? First time we've seen this market — fire if it
    meets the threshold.

    The dispatcher does NOT write the snapshots itself; the caller is
    responsible for `store.record_scan(...)` either before or after
    dispatching. (Most callers will write *after* dispatching so the
    prior-grade lookup excludes the current scan.)
    """
    def __init__(
        self,
        sinks: list[AlertSink],
        store: SnapshotStore,
        min_grade: str = "A",
    ):
        if min_grade not in GRADE_ORDER:
            raise ValueError(f"min_grade must be one of {GRADE_ORDER}")
        self.sinks = sinks
        self.store = store
        self.min_grade = min_grade
        self._cutoff = _grade_rank(min_grade)

    def dispatch(
        self,
        evaluations: list[ContractEvaluation],
        now_utc: Optional[datetime] = None,
    ) -> list[Alert]:
        now_utc = now_utc or datetime.now(timezone.utc)
        fired: list[Alert] = []
        for e in evaluations:
            if _grade_rank(e.grade) > self._cutoff:
                continue
            prior = self._most_recent_prior_grade(e.market.ticker)
            if not _is_better(e.grade, prior):
                continue
            alert = Alert(
                fired_at_utc=now_utc,
                market_ticker=e.market.ticker,
                event_ticker=e.market.event_ticker,
                city_slug=e.contract.city_slug,
                market_date=e.contract.market_date.isoformat(),
                bracket=e.contract.bracket.label(),
                metric=e.contract.metric.value,
                state=e.state.value,
                reason=e.reason,
                grade=e.grade,
                previous_grade=prior,
                yes_ask_cents=e.yes_ask_cents,
                no_ask_cents=e.no_ask_cents,
                edge_yes=e.edge_yes,
                edge_no=e.edge_no,
                fair_prob_low=e.fair_prob_low,
                fair_prob_high=e.fair_prob_high,
                notes=list(e.notes),
            )
            for sink in self.sinks:
                sink.emit(alert)
            fired.append(alert)
        return fired

    def _most_recent_prior_grade(self, market_ticker: str) -> Optional[str]:
        rows: list[SnapshotRow] = self.store.query_snapshots(
            market_ticker=market_ticker, limit=1
        )
        if not rows:
            return None
        return rows[0].grade
