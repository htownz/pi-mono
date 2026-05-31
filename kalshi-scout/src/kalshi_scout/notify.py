"""Alert delivery framework.

Alerts fire on **grade-improvement transitions**: a contract whose most
recent stored snapshot was at a worse grade than its current evaluation
(or had no prior snapshot at all). This makes alerts naturally replayable
(invariant I8): the snapshot store is the source of truth for "have we
already alerted on this".

Five sink types:

  StdoutSink    - print to console (always-on default during testing)
  JsonlSink     - append one JSON object per alert to a file
  WebhookSink   - POST raw alert JSON to a configurable URL
  NtfySink      - push notification via ntfy.sh / self-hosted ntfy
  DiscordSink   - rich-embed message via Discord webhook

`WebhookSink` is the generic escape hatch — anything that can consume the
raw alert JSON. `NtfySink` / `DiscordSink` pre-format the alert for those
specific destinations so the operator gets a phone-readable notification
without a translator service in between.

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


def _format_alert_text(alert: Alert) -> str:
    """Compact phone-friendly summary used by ntfy and Discord sinks.

    The price line names the actionable side so a DEAD_NO alert shows
    `no_ask: 7c` instead of `yes_ask: —` — matching the side that the
    edge line reports.
    """
    prev = f" (was {alert.previous_grade})" if alert.previous_grade else ""
    side, price, edge_str = _embed_side(alert)
    edge_line = f"edge_{side}: {edge_str.split()[1]}" if edge_str != "—" else ""
    price_line = f"{side}_ask: {price}"
    fair_line = f"fair: {alert.fair_prob_low * 100:.0f}-{alert.fair_prob_high * 100:.0f}%"
    return (
        f"{alert.market_ticker} -> {alert.grade}{prev}\n"
        f"state: {alert.state}\n"
        f"{price_line}  {fair_line}\n"
        f"{edge_line}"
    ).rstrip()


def _ntfy_priority(grade: str) -> str:
    """Map grade to ntfy 1-5 priority. A+/A get max so the phone buzzes loud."""
    return {"A+": "5", "A": "5", "B+": "4", "B": "3"}.get(grade, "2")


class NtfySink:
    """Push notification via ntfy (https://docs.ntfy.sh).

    Spec form: `ntfy:<topic>` for the public ntfy.sh server, or
    `ntfy:<full-https-url>` for a self-hosted instance. The body is the
    formatted alert text; title / priority / tags ride in HTTP headers.
    """
    def __init__(
        self,
        topic_or_url: str,
        timeout: float = 5.0,
        client: Optional[httpx.Client] = None,
        failure_log=print,
    ):
        if topic_or_url.startswith(("http://", "https://")):
            self.url = topic_or_url
        else:
            self.url = f"https://ntfy.sh/{topic_or_url}"
        self.timeout = timeout
        self._client = client or httpx.Client(timeout=timeout)
        self._failure_log = failure_log

    def emit(self, alert: Alert) -> None:
        body = _format_alert_text(alert)
        headers = {
            "Title": f"kalshi-scout [{alert.grade}] {alert.market_ticker}",
            "Priority": _ntfy_priority(alert.grade),
            "Tags": "chart_with_upwards_trend",
        }
        try:
            resp = self._client.post(
                self.url, content=body.encode("utf-8"), headers=headers,
            )
            resp.raise_for_status()
        except Exception as exc:
            self._failure_log(f"ntfy {self.url} failed: {exc}")

    def close(self) -> None:
        self._client.close()


def _discord_color(grade: str) -> int:
    """Embed sidebar color per grade — green for A-tier, yellow for B, gray else."""
    return {
        "A+": 0x1F8B4C,   # bright green
        "A": 0x2ECC71,    # green
        "B+": 0xF1C40F,   # yellow
        "B": 0xE67E22,    # orange
    }.get(grade, 0x95A5A6)  # gray for C/D


def _embed_side(alert: Alert) -> tuple[str, str, str]:
    """Pick the (side_label, price_string, edge_string) the alert's
    actionable side. The ranker grades LOCKED_YES off edge_yes and DEAD_NO
    off edge_no, so the side with the larger edge is the tradable one and
    that side's ask is the fillable price the operator needs.
    """
    # Default: yes side. Switch to no when edge_no clearly wins.
    side = "yes"
    edge: Optional[float] = alert.edge_yes
    if alert.edge_no is not None and (
        alert.edge_yes is None or alert.edge_no > alert.edge_yes
    ):
        side = "no"
        edge = alert.edge_no
    if side == "yes":
        price = f"{alert.yes_ask_cents}c" if alert.yes_ask_cents is not None else "—"
    else:
        price = f"{alert.no_ask_cents}c" if alert.no_ask_cents is not None else "—"
    edge_str = f"{side} {edge:+.2f}" if edge is not None else "—"
    return side, price, edge_str


class DiscordSink:
    """Rich-embed alert via Discord webhook.

    Spec form: `discord:<full-webhook-url>`. The embed shows the grade
    transition, state, fillable side and edge, and the fair-prob band.
    """
    def __init__(
        self,
        webhook_url: str,
        timeout: float = 5.0,
        client: Optional[httpx.Client] = None,
        failure_log=print,
    ):
        self.webhook_url = webhook_url
        self.timeout = timeout
        self._client = client or httpx.Client(timeout=timeout)
        self._failure_log = failure_log

    def emit(self, alert: Alert) -> None:
        prev = f" (was {alert.previous_grade})" if alert.previous_grade else ""
        # Pick the actionable side from the edges so the embed reports the
        # tradable price for THAT side. DEAD_NO alerts have edge_no > edge_yes
        # and a populated no_ask — showing yes_ask there is misleading
        # (often "—") and hides the fillable price from the operator.
        side_label, price, edge_value = _embed_side(alert)
        price_field = f"{side_label}_ask"
        fair = f"{alert.fair_prob_low * 100:.0f}–{alert.fair_prob_high * 100:.0f}%"
        embed = {
            "title": f"[{alert.grade}] {alert.market_ticker}{prev}",
            "description": f"**{alert.state}** — {alert.reason}",
            "color": _discord_color(alert.grade),
            "fields": [
                {"name": price_field, "value": price, "inline": True},
                {"name": "fair", "value": fair, "inline": True},
                {"name": "edge", "value": edge_value, "inline": True},
                {"name": "city", "value": alert.city_slug, "inline": True},
                {"name": "metric", "value": alert.metric, "inline": True},
                {"name": "date", "value": alert.market_date, "inline": True},
            ],
            "timestamp": alert.fired_at_utc.astimezone(timezone.utc).isoformat(),
        }
        payload = {"username": "kalshi-scout", "embeds": [embed]}
        try:
            resp = self._client.post(self.webhook_url, json=payload)
            resp.raise_for_status()
        except Exception as exc:
            self._failure_log(f"discord webhook failed: {exc}")

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
