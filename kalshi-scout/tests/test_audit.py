"""Tests for the audit-log reader + summarizer.

Builds a synthetic JSONL file in tmp_path and verifies `summarize()` rolls
it up correctly: per-day bucketing, refusal categorization, recent-N
trimming, and `--ticker` / `--since` filters.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from kalshi_scout.audit import (
    AuditEntry,
    AuditSummary,
    _classify_refusal_reason,
    read_audit_log,
    summarize,
)


# -- fixtures + helpers ------------------------------------------------------

def _attempt_dict(
    fired_at: datetime,
    market_ticker: str = "KXHIGHTHOU-26MAY30-B95.5",
    event_ticker: str = "KXHIGHTHOU-26MAY30",
    side: str = "yes",
    price_cents: int = 15,
    size_contracts: int = 1,
    placed: bool = True,
    paper: bool = True,
    reason: str = "placed (paper)",
    order_id: str | None = None,
    grade: str = "A+",
) -> dict:
    """Mirror of TradeAttempt.to_json_dict for test fixture lines."""
    return {
        "fired_at_utc": fired_at.astimezone(timezone.utc).isoformat(),
        "market_ticker": market_ticker, "event_ticker": event_ticker,
        "side": side, "price_cents": price_cents,
        "size_contracts": size_contracts,
        "cost_cents": price_cents * size_contracts,
        "placed": placed, "paper": paper, "reason": reason,
        "order_id": order_id, "position_id": 1 if placed else None,
        "snap_id": 1, "grade": grade,
    }


@pytest.fixture
def audit_path(tmp_path: Path) -> Path:
    p = tmp_path / "auto-trade.jsonl"
    return p


def _write(audit_path: Path, *attempts: dict) -> None:
    with audit_path.open("a") as f:
        for a in attempts:
            f.write(json.dumps(a) + "\n")


# -- AuditEntry.from_json ----------------------------------------------------

def test_audit_entry_round_trips_through_json():
    fired = datetime(2026, 6, 1, 14, 30, 5, tzinfo=timezone.utc)
    d = _attempt_dict(fired, order_id="ord_x")
    entry = AuditEntry.from_json(d)
    assert entry.market_ticker == "KXHIGHTHOU-26MAY30-B95.5"
    assert entry.fired_at_utc == fired
    assert entry.cost_cents == 15
    assert entry.placed is True
    assert entry.order_id == "ord_x"


def test_audit_entry_tolerates_trailing_z_in_timestamp():
    """The to_json_dict output uses isoformat which may end in '+00:00';
    older logs (and curl-pasted entries) sometimes carry a trailing 'Z'."""
    d = _attempt_dict(datetime(2026, 6, 1, tzinfo=timezone.utc))
    d["fired_at_utc"] = "2026-06-01T00:00:00Z"
    entry = AuditEntry.from_json(d)
    assert entry.fired_at_utc.year == 2026


def test_read_audit_log_skips_blank_and_malformed_lines(tmp_path):
    p = tmp_path / "audit.jsonl"
    fired = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    p.write_text(
        json.dumps(_attempt_dict(fired)) + "\n"
        "\n"                          # blank line
        "{this is not json}\n"        # malformed
        + json.dumps(_attempt_dict(fired.replace(minute=1))) + "\n"
    )
    entries = list(read_audit_log(p))
    assert len(entries) == 2


def test_read_audit_log_returns_empty_when_file_missing(tmp_path):
    assert list(read_audit_log(tmp_path / "does-not-exist.jsonl")) == []


# -- Refusal classification --------------------------------------------------

def test_classify_refusal_reason_buckets_known_strings():
    """Common refusal reasons from RiskGuard should map to stable labels."""
    assert _classify_refusal_reason(
        "rounding risk: LOW running_min 62.6°F only 0.4°F below..."
    ) == "rounding risk"
    assert _classify_refusal_reason(
        "edge 3c < min_edge_cents 5"
    ) == "edge below min"
    assert _classify_refusal_reason(
        "event concentration 350c > max_concentration_per_event_cents 250"
    ) == "event concentration cap"
    assert _classify_refusal_reason(
        "daily loss kill: realized 150c today >= max_daily_loss_cents 100"
    ) == "daily loss kill"
    assert _classify_refusal_reason(
        "kill switch active (/data/scout.kill)"
    ) == "kill switch"
    assert _classify_refusal_reason(
        "order resting but unfilled (0/5 contracts); not recorded"
    ) == "order resting unfilled"


def test_classify_refusal_reason_falls_back_to_other():
    assert _classify_refusal_reason("garbage we never emit") == "other"


# -- summarize ---------------------------------------------------------------

def test_summarize_empty_log_returns_empty_summary(audit_path):
    summary = summarize(list(read_audit_log(audit_path)))
    assert summary.total_attempts == 0
    assert summary.days == []


def test_summarize_groups_by_utc_day(audit_path):
    """Entries on different UTC days land in separate DaySummary objects."""
    day_a = datetime(2026, 6, 1, 23, 50, tzinfo=timezone.utc)
    day_b = datetime(2026, 6, 2, 0, 10, tzinfo=timezone.utc)
    _write(audit_path,
           _attempt_dict(day_a),
           _attempt_dict(day_a.replace(minute=55)),
           _attempt_dict(day_b))
    summary = summarize(list(read_audit_log(audit_path)))
    days = {d.day: d for d in summary.days}
    assert days[date(2026, 6, 1)].total_attempts == 2
    assert days[date(2026, 6, 2)].total_attempts == 1
    # Days returned newest-first.
    assert summary.days[0].day == date(2026, 6, 2)


def test_summarize_counts_placed_vs_refused_with_breakdown(audit_path):
    fired = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    _write(audit_path,
           _attempt_dict(fired),
           _attempt_dict(fired.replace(minute=5)),
           _attempt_dict(fired.replace(minute=10),
                         placed=False,
                         reason="rounding risk: LOW running_min 62.6"),
           _attempt_dict(fired.replace(minute=15),
                         placed=False,
                         reason="edge 3c < min_edge_cents 5"),
           _attempt_dict(fired.replace(minute=20),
                         placed=False,
                         reason="edge 2c < min_edge_cents 5"))
    summary = summarize(list(read_audit_log(audit_path)))
    day = summary.days[0]
    assert day.total_attempts == 5
    assert day.placed == 2
    assert day.refused == 3
    assert dict(day.refusal_breakdown) == {
        "rounding risk": 1,
        "edge below min": 2,
    }


def test_summarize_tracks_paper_vs_live_filled_vs_partial(audit_path):
    """Live partial fills are reported separately so the operator can spot
    a book-liquidity problem."""
    fired = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    _write(audit_path,
           _attempt_dict(fired, paper=True),   # paper
           _attempt_dict(fired.replace(minute=5), paper=False,
                         reason="placed"),     # full live
           _attempt_dict(fired.replace(minute=10), paper=False,
                         reason="placed — partial fill 2/5"))  # partial
    summary = summarize(list(read_audit_log(audit_path)))
    day = summary.days[0]
    assert day.placed_paper == 1
    assert day.placed_live_filled_full == 1
    assert day.placed_live_partial == 1


def test_summarize_ticker_filter_narrows_to_single_market(audit_path):
    fired = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    _write(audit_path,
           _attempt_dict(fired, market_ticker="A"),
           _attempt_dict(fired.replace(minute=5), market_ticker="B"),
           _attempt_dict(fired.replace(minute=10), market_ticker="A"))
    summary = summarize(list(read_audit_log(audit_path)), ticker="A")
    assert summary.days[0].total_attempts == 2


def test_summarize_since_filter_drops_older_entries(audit_path):
    early = datetime(2026, 6, 1, 8, 0, tzinfo=timezone.utc)
    late = datetime(2026, 6, 1, 18, 0, tzinfo=timezone.utc)
    cutoff = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    _write(audit_path,
           _attempt_dict(early),
           _attempt_dict(late))
    summary = summarize(list(read_audit_log(audit_path)), since=cutoff)
    assert summary.total_attempts == 1
    assert summary.days[0].total_attempts == 1


def test_summarize_recent_n_caps_per_day_lists(audit_path):
    """A noisy day with 50 placed shouldn't dump 50 lines into the
    recent_placed list — recent_n trims to a configurable window."""
    base = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    for i in range(20):
        _write(audit_path, _attempt_dict(base + timedelta(minutes=i)))
    summary = summarize(list(read_audit_log(audit_path)), recent_n=3)
    day = summary.days[0]
    assert day.placed == 20
    assert len(day.recent_placed) == 3
    # Newest first.
    assert day.recent_placed[0].fired_at_utc == base + timedelta(minutes=19)


def test_summarize_to_dict_is_json_safe(audit_path):
    fired = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    _write(audit_path,
           _attempt_dict(fired),
           _attempt_dict(fired.replace(minute=5), placed=False,
                         reason="kill switch active"))
    summary = summarize(list(read_audit_log(audit_path)))
    d = summary.to_dict()
    # Must serialize cleanly — used by the JSON CLI mode and the
    # dashboard's JSON endpoint.
    s = json.dumps(d)
    parsed = json.loads(s)
    assert parsed["total_attempts"] == 2
    assert parsed["days"][0]["refusal_breakdown"] == {"kill switch": 1}
