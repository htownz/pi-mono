"""Audit-log reader + summary for the auto-trader's JSONL output.

Used by:
  - `kalshi-scout audit` CLI command for human-readable end-of-day summaries.
  - The dashboard's /auto-trade endpoint for the same content over HTTP.

The JSONL schema is `TradeAttempt.to_json_dict()` in trading.py — one line
per attempt. This module parses, filters, and rolls up.

Refusal reasons aren't enumerated (the RiskGuard emits human strings), so
we bucket them with `_classify_refusal_reason` — a small set of substrings
that maps the most common reasons into stable category labels for charting.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator, Optional


@dataclass(frozen=True)
class AuditEntry:
    """One row from the audit JSONL. Mirrors TradeAttempt.to_json_dict but
    keeps strict typing so summary code doesn't have to dict-dance."""
    fired_at_utc: datetime
    market_ticker: str
    event_ticker: str
    side: str
    price_cents: int
    size_contracts: int
    cost_cents: int
    placed: bool
    paper: bool
    reason: str
    order_id: Optional[str]
    position_id: Optional[int]
    snap_id: int
    grade: str

    @classmethod
    def from_json(cls, d: dict) -> "AuditEntry":
        ts = d.get("fired_at_utc") or ""
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return cls(
            fired_at_utc=datetime.fromisoformat(ts).astimezone(timezone.utc),
            market_ticker=d.get("market_ticker", ""),
            event_ticker=d.get("event_ticker", ""),
            side=d.get("side", "—"),
            price_cents=int(d.get("price_cents") or 0),
            size_contracts=int(d.get("size_contracts") or 0),
            cost_cents=int(d.get("cost_cents") or 0),
            placed=bool(d.get("placed", False)),
            paper=bool(d.get("paper", False)),
            reason=d.get("reason", ""),
            order_id=d.get("order_id"),
            position_id=d.get("position_id"),
            snap_id=int(d.get("snap_id") or 0),
            grade=d.get("grade", ""),
        )


def read_audit_log(path: Path) -> Iterator[AuditEntry]:
    """Stream-parse the JSONL audit log. Skips malformed lines silently so
    one bad row doesn't break a multi-day summary, but a count of dropped
    lines could be added later if it becomes a problem."""
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield AuditEntry.from_json(json.loads(line))
            except (json.JSONDecodeError, ValueError, KeyError):
                continue


# -- Refusal classification --------------------------------------------------

# Order matters: most specific matches first. Falls through to "other" if no
# substring matches. Keep this in sync with the messages RiskGuard emits.
_REFUSAL_BUCKETS: list[tuple[str, str]] = [
    ("kill switch", "kill switch"),
    ("rounding risk", "rounding risk"),
    ("daily loss kill", "daily loss kill"),
    ("max_concentration_per_event", "event concentration cap"),
    ("concentration", "event concentration cap"),
    ("max_position_cost_cents", "cost cap"),
    ("max_position_size_contracts", "size cap"),
    ("min_edge_cents", "edge below min"),
    ("edge ", "edge below min"),                # bare "edge Xc < min..."
    ("unfilled", "order resting unfilled"),
    ("API error", "API error"),
    ("no fillable", "no fillable side"),
    ("size", "size invalid"),
]


def _classify_refusal_reason(reason: str) -> str:
    for needle, label in _REFUSAL_BUCKETS:
        if needle in reason:
            return label
    return "other"


# -- Summary -----------------------------------------------------------------

@dataclass
class DaySummary:
    """One day's roll-up of audit activity."""
    day: date
    total_attempts: int = 0
    placed: int = 0
    placed_paper: int = 0
    placed_live_filled_full: int = 0
    placed_live_partial: int = 0
    refused: int = 0
    refusal_breakdown: Counter = field(default_factory=Counter)
    total_cost_cents: int = 0       # cost across placed attempts
    by_grade: Counter = field(default_factory=Counter)
    recent_placed: list[AuditEntry] = field(default_factory=list)
    recent_refused: list[AuditEntry] = field(default_factory=list)


@dataclass
class AuditSummary:
    """Multi-day audit roll-up returned by `summarize`."""
    days: list[DaySummary]
    total_attempts: int
    total_placed: int
    total_refused: int

    def to_dict(self) -> dict:
        return {
            "total_attempts": self.total_attempts,
            "total_placed": self.total_placed,
            "total_refused": self.total_refused,
            "days": [
                {
                    "day": d.day.isoformat(),
                    "total_attempts": d.total_attempts,
                    "placed": d.placed,
                    "placed_paper": d.placed_paper,
                    "placed_live_filled_full": d.placed_live_filled_full,
                    "placed_live_partial": d.placed_live_partial,
                    "refused": d.refused,
                    "refusal_breakdown": dict(d.refusal_breakdown),
                    "total_cost_cents": d.total_cost_cents,
                    "by_grade": dict(d.by_grade),
                    "recent_placed": [_entry_to_dict(e) for e in d.recent_placed],
                    "recent_refused": [_entry_to_dict(e) for e in d.recent_refused],
                }
                for d in self.days
            ],
        }


def _entry_to_dict(e: AuditEntry) -> dict:
    return {
        "fired_at_utc": e.fired_at_utc.isoformat(),
        "market_ticker": e.market_ticker,
        "side": e.side, "price_cents": e.price_cents,
        "size_contracts": e.size_contracts, "cost_cents": e.cost_cents,
        "placed": e.placed, "paper": e.paper,
        "reason": e.reason, "order_id": e.order_id, "grade": e.grade,
    }


def summarize(
    entries: list[AuditEntry],
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    ticker: Optional[str] = None,
    recent_n: int = 5,
) -> AuditSummary:
    """Bucket the filtered entries by UTC day and produce per-day stats.

    `since` / `until` clip the time range; `ticker` filters to one market.
    `recent_n` is the count of most-recent placed / refused entries kept per
    day for display (separate lists so a noisy day's refusals don't crowd
    out the actual orders).
    """
    by_day: dict[date, DaySummary] = {}
    for e in entries:
        if since and e.fired_at_utc < since:
            continue
        if until and e.fired_at_utc > until:
            continue
        if ticker and e.market_ticker != ticker:
            continue
        day = e.fired_at_utc.astimezone(timezone.utc).date()
        s = by_day.setdefault(day, DaySummary(day=day))
        s.total_attempts += 1
        s.by_grade[e.grade] += 1
        if e.placed:
            s.placed += 1
            s.total_cost_cents += e.cost_cents
            if e.paper:
                s.placed_paper += 1
            elif "partial" in e.reason.lower():
                s.placed_live_partial += 1
            else:
                s.placed_live_filled_full += 1
            s.recent_placed.append(e)
        else:
            s.refused += 1
            s.refusal_breakdown[_classify_refusal_reason(e.reason)] += 1
            s.recent_refused.append(e)

    # Keep only the N most-recent in each list (entries arrive in file order;
    # sort desc by ts).
    for s in by_day.values():
        s.recent_placed.sort(key=lambda e: e.fired_at_utc, reverse=True)
        s.recent_refused.sort(key=lambda e: e.fired_at_utc, reverse=True)
        s.recent_placed = s.recent_placed[:recent_n]
        s.recent_refused = s.recent_refused[:recent_n]

    days = sorted(by_day.values(), key=lambda d: d.day, reverse=True)
    return AuditSummary(
        days=days,
        total_attempts=sum(d.total_attempts for d in days),
        total_placed=sum(d.placed for d in days),
        total_refused=sum(d.refused for d in days),
    )
