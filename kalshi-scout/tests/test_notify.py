"""Tests for V0.8 alert delivery: sinks, transition detection, dispatcher.

Webhook tests use respx (already in dev deps) to mock httpx calls.
"""

from datetime import date, datetime, timezone
from pathlib import Path

import httpx
import pytest
import respx

from kalshi_scout.models import (
    Bracket,
    BracketKind,
    ContractEvaluation,
    ContractState,
    KalshiMarket,
    Metric,
    ParsedContract,
)
from kalshi_scout.notify import (
    Alert,
    AlertDispatcher,
    JsonlSink,
    StdoutSink,
    WebhookSink,
    _grade_rank,
    _is_better,
)
from kalshi_scout.store import SnapshotStore


# -- Helpers -----------------------------------------------------------------

def _eval(
    ticker: str = "KXHIGHHOUSTON-26MAY27-B79-80",
    grade: str = "A+",
    state: ContractState = ContractState.LOCKED_YES,
    yes_ask: int | None = 71,
) -> ContractEvaluation:
    contract = ParsedContract(
        market_ticker=ticker,
        event_ticker="KXHIGHHOUSTON-26MAY27",
        city_slug="HOUSTON",
        metric=Metric.HIGH,
        market_date=date(2026, 5, 27),
        bracket=Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0),
    )
    market = KalshiMarket(
        ticker=ticker, event_ticker="KXHIGHHOUSTON-26MAY27",
        title="", yes_sub_title="79° to 80°", status="open", close_time=None,
        yes_bid=yes_ask - 1 if yes_ask else None,
        yes_ask=yes_ask,
        no_bid=None, no_ask=None,
        last_price=None, volume=10, open_interest=100,
    )
    return ContractEvaluation(
        contract=contract, market=market, state=state,
        reason="test", fair_prob_low=0.97, fair_prob_high=0.99,
        yes_ask_cents=yes_ask, no_ask_cents=None,
        edge_yes=0.28 if yes_ask else None, edge_no=None,
        grade=grade, notes=["test note"],
    )


@pytest.fixture
def store(tmp_path: Path) -> SnapshotStore:
    s = SnapshotStore(tmp_path / "notify.db")
    yield s
    s.close()


# -- Grade ordering ----------------------------------------------------------

def test_grade_rank_orders_plus_above_plain():
    assert _grade_rank("A+") < _grade_rank("A") < _grade_rank("B+") < _grade_rank("B")


def test_is_better_no_prior_is_always_better():
    assert _is_better("A+", None) is True
    assert _is_better("F", None) is True


def test_is_better_strictly():
    assert _is_better("A+", "A") is True
    assert _is_better("A", "A+") is False
    assert _is_better("A", "A") is False  # not strictly better


# -- StdoutSink --------------------------------------------------------------

def test_stdout_sink_prints_alert(capsys):
    sink = StdoutSink()
    alert = Alert(
        fired_at_utc=datetime(2026, 5, 27, 16, 30, tzinfo=timezone.utc),
        market_ticker="KXHIGHHOUSTON-26MAY27-B79-80",
        event_ticker="KXHIGHHOUSTON-26MAY27",
        city_slug="HOUSTON", market_date="2026-05-27",
        bracket="79–80°", metric="high",
        state="locked_yes", reason="max already hit 79",
        grade="A+", previous_grade="C",
        yes_ask_cents=71, no_ask_cents=29,
        edge_yes=0.28, edge_no=None,
        fair_prob_low=0.97, fair_prob_high=0.99,
        notes=[],
    )
    sink.emit(alert)
    captured = capsys.readouterr()
    assert "KXHIGHHOUSTON-26MAY27-B79-80" in captured.out
    assert "A+" in captured.out
    assert "was C" in captured.out


# -- JsonlSink ---------------------------------------------------------------

def test_jsonl_sink_appends_one_line_per_alert(tmp_path: Path):
    path = tmp_path / "alerts.jsonl"
    sink = JsonlSink(path)
    a = Alert(
        fired_at_utc=datetime(2026, 5, 27, 16, 30, tzinfo=timezone.utc),
        market_ticker="X", event_ticker="Y",
        city_slug="HOUSTON", market_date="2026-05-27",
        bracket="79–80°", metric="high",
        state="locked_yes", reason="",
        grade="A+", previous_grade=None,
        yes_ask_cents=71, no_ask_cents=None,
        edge_yes=0.28, edge_no=None,
        fair_prob_low=0.97, fair_prob_high=0.99,
        notes=["one", "two"],
    )
    sink.emit(a)
    sink.emit(a)
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    import json as _json
    parsed = _json.loads(lines[0])
    assert parsed["market_ticker"] == "X"
    assert parsed["grade"] == "A+"
    assert parsed["notes"] == ["one", "two"]


def test_jsonl_sink_creates_parent_dir(tmp_path: Path):
    nested = tmp_path / "deep" / "deeper" / "alerts.jsonl"
    sink = JsonlSink(nested)
    assert nested.parent.exists()


# -- WebhookSink -------------------------------------------------------------

def test_webhook_sink_posts_json_body():
    """Mock httpx via respx; verify the request body matches the alert dict."""
    with respx.mock(base_url="https://hook.example.com") as mock:
        route = mock.post("/alerts").respond(200)
        sink = WebhookSink(url="https://hook.example.com/alerts")
        alert = Alert(
            fired_at_utc=datetime(2026, 5, 27, 16, 30, tzinfo=timezone.utc),
            market_ticker="X", event_ticker="Y",
            city_slug="HOUSTON", market_date="2026-05-27",
            bracket="79–80°", metric="high", state="locked_yes",
            reason="r", grade="A+", previous_grade="B",
            yes_ask_cents=71, no_ask_cents=None,
            edge_yes=0.28, edge_no=None,
            fair_prob_low=0.97, fair_prob_high=0.99,
            notes=[],
        )
        sink.emit(alert)
        sink.close()
        assert route.call_count == 1
        sent = route.calls.last.request
        import json as _json
        body = _json.loads(sent.content)
        assert body["market_ticker"] == "X"
        assert body["grade"] == "A+"
        assert body["previous_grade"] == "B"


def test_webhook_sink_swallows_failures():
    """A downed webhook must NOT raise — alerts are best-effort."""
    failures: list[str] = []
    with respx.mock(base_url="https://hook.example.com") as mock:
        mock.post("/alerts").respond(503)
        sink = WebhookSink(
            url="https://hook.example.com/alerts",
            failure_log=lambda msg: failures.append(msg),
        )
        alert = Alert(
            fired_at_utc=datetime(2026, 5, 27, 16, 30, tzinfo=timezone.utc),
            market_ticker="X", event_ticker="Y", city_slug="H",
            market_date="2026-05-27", bracket="b", metric="high",
            state="locked_yes", reason="",
            grade="A", previous_grade=None,
            yes_ask_cents=None, no_ask_cents=None,
            edge_yes=None, edge_no=None,
            fair_prob_low=0.0, fair_prob_high=0.0, notes=[],
        )
        sink.emit(alert)  # must not raise
        sink.close()
    assert len(failures) == 1
    assert "failed" in failures[0]


# -- Dispatcher transition logic ---------------------------------------------

class _RecordingSink:
    def __init__(self): self.alerts: list[Alert] = []
    def emit(self, alert: Alert) -> None: self.alerts.append(alert)


def test_dispatcher_fires_on_first_appearance_when_grade_meets_min(store: SnapshotStore):
    """No prior snapshot, current grade A+, min A -> fire."""
    sink = _RecordingSink()
    d = AlertDispatcher(sinks=[sink], store=store, min_grade="A")
    fired = d.dispatch([_eval(grade="A+")])
    assert len(fired) == 1
    assert sink.alerts[0].grade == "A+"
    assert sink.alerts[0].previous_grade is None


def test_dispatcher_does_not_fire_when_grade_below_min(store: SnapshotStore):
    sink = _RecordingSink()
    d = AlertDispatcher(sinks=[sink], store=store, min_grade="A")
    fired = d.dispatch([_eval(grade="C")])
    assert fired == []
    assert sink.alerts == []


def test_dispatcher_fires_only_on_grade_improvement(store: SnapshotStore):
    """Store a prior C snapshot; dispatch a new A+ -> fires. Dispatch
    again with the same A+ -> does NOT fire (no improvement)."""
    store.record_scan([_eval(grade="C")])
    sink = _RecordingSink()
    d = AlertDispatcher(sinks=[sink], store=store, min_grade="A")

    # First A+ scan: prior was C, A+ is strictly better -> fire
    fired1 = d.dispatch([_eval(grade="A+")])
    assert len(fired1) == 1
    assert fired1[0].previous_grade == "C"

    # Persist the A+ as the new prior, then dispatch another A+:
    store.record_scan([_eval(grade="A+")])
    fired2 = d.dispatch([_eval(grade="A+")])
    assert fired2 == [], "should not re-fire on same grade"


def test_dispatcher_fires_on_degradation_path_back_up(store: SnapshotStore):
    """C -> A+ fires. A+ persisted. Then A -> shouldn't fire (worse).
    Then back to A+ -> shouldn't fire (no improvement from A+ stored)."""
    store.record_scan([_eval(grade="C")])
    sink = _RecordingSink()
    d = AlertDispatcher(sinks=[sink], store=store, min_grade="A")
    d.dispatch([_eval(grade="A+")])
    store.record_scan([_eval(grade="A+")])
    sink.alerts.clear()
    # A is worse than A+
    d.dispatch([_eval(grade="A")])
    assert sink.alerts == []
    store.record_scan([_eval(grade="A")])
    # Back to A+ — improvement over A
    d.dispatch([_eval(grade="A+")])
    assert len(sink.alerts) == 1
    assert sink.alerts[0].grade == "A+"
    assert sink.alerts[0].previous_grade == "A"


def test_dispatcher_routes_to_multiple_sinks(store: SnapshotStore):
    s1 = _RecordingSink()
    s2 = _RecordingSink()
    d = AlertDispatcher(sinks=[s1, s2], store=store, min_grade="A")
    d.dispatch([_eval(grade="A+")])
    assert len(s1.alerts) == 1
    assert len(s2.alerts) == 1


def test_dispatcher_rejects_invalid_min_grade(store: SnapshotStore):
    with pytest.raises(ValueError):
        AlertDispatcher(sinks=[], store=store, min_grade="Z")
