"""Tests for V0.8 alert delivery: sinks, transition detection, dispatcher.

Webhook tests use respx (already in dev deps) to mock httpx calls.
"""

import json
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
import click

from kalshi_scout.cli import _build_sinks
from kalshi_scout.notify import (
    Alert,
    AlertDispatcher,
    DiscordSink,
    JsonlSink,
    NtfySink,
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


# -- NtfySink ----------------------------------------------------------------

def _alert(grade: str = "A+", market_ticker: str = "KXHIGHTHOU-26MAY30-B95",
           previous_grade: str | None = "B", yes_ask: int | None = 71,
           edge_yes: float | None = 0.28) -> Alert:
    return Alert(
        fired_at_utc=datetime(2026, 5, 30, 16, 30, tzinfo=timezone.utc),
        market_ticker=market_ticker, event_ticker="KXHIGHTHOU-26MAY30",
        city_slug="HOUSTON", market_date="2026-05-30",
        bracket="94–95°", metric="high", state="locked_yes",
        reason="observed max already in bracket",
        grade=grade, previous_grade=previous_grade,
        yes_ask_cents=yes_ask, no_ask_cents=None,
        edge_yes=edge_yes, edge_no=None,
        fair_prob_low=0.97, fair_prob_high=0.99,
        notes=[],
    )


def test_ntfy_sink_defaults_to_ntfy_sh():
    """Bare topic string targets the public ntfy.sh server."""
    with respx.mock(base_url="https://ntfy.sh") as mock:
        route = mock.post("/kalshi-scout").respond(200)
        sink = NtfySink(topic_or_url="kalshi-scout")
        sink.emit(_alert())
        sink.close()
        assert route.call_count == 1
        req = route.calls.last.request
        # Body is plaintext (not JSON), with the formatted alert summary.
        body = req.content.decode("utf-8")
        assert "KXHIGHTHOU-26MAY30-B95" in body
        assert "-> A+" in body
        assert "(was B)" in body
        assert "yes_ask: 71c" in body


def test_ntfy_sink_sets_priority_and_title_headers():
    """A+ -> priority 5; title carries the grade so the lockscreen shows it."""
    with respx.mock(base_url="https://ntfy.sh") as mock:
        route = mock.post("/topic").respond(200)
        NtfySink(topic_or_url="topic").emit(_alert(grade="A+"))
        req = route.calls.last.request
        assert req.headers["Priority"] == "5"
        assert "[A+]" in req.headers["Title"]


def test_ntfy_sink_lower_priority_for_b_grade():
    with respx.mock(base_url="https://ntfy.sh") as mock:
        route = mock.post("/topic").respond(200)
        NtfySink(topic_or_url="topic").emit(_alert(grade="B"))
        assert route.calls.last.request.headers["Priority"] == "3"


def test_ntfy_sink_accepts_self_hosted_url():
    """Full https URL skips the ntfy.sh prefix — supports self-hosted instances."""
    with respx.mock(base_url="https://ntfy.example.com") as mock:
        route = mock.post("/private-topic").respond(200)
        sink = NtfySink(topic_or_url="https://ntfy.example.com/private-topic")
        sink.emit(_alert())
        sink.close()
        assert route.call_count == 1


def test_ntfy_sink_swallows_failures():
    failures: list[str] = []
    with respx.mock(base_url="https://ntfy.sh") as mock:
        mock.post("/topic").respond(503)
        sink = NtfySink(topic_or_url="topic",
                        failure_log=lambda msg: failures.append(msg))
        sink.emit(_alert())  # must not raise
        sink.close()
    assert len(failures) == 1
    assert "ntfy" in failures[0]


# -- DiscordSink -------------------------------------------------------------

def test_discord_sink_posts_rich_embed():
    """Discord webhook gets a structured embed, not raw alert JSON."""
    url = "https://discord.com/api/webhooks/123/abc"
    with respx.mock() as mock:
        route = mock.post(url).respond(204)
        DiscordSink(webhook_url=url).emit(_alert())
        body = json.loads(route.calls.last.request.content)

    assert body["username"] == "kalshi-scout"
    assert len(body["embeds"]) == 1
    embed = body["embeds"][0]
    assert "KXHIGHTHOU-26MAY30-B95" in embed["title"]
    assert "[A+]" in embed["title"]
    assert "(was B)" in embed["title"]
    assert embed["color"] == 0x1F8B4C   # bright green for A+

    field_map = {f["name"]: f["value"] for f in embed["fields"]}
    assert field_map["yes_ask"] == "71c"
    assert field_map["fair"] == "97–99%"  # 97–99% en-dash
    assert "yes" in field_map["edge"]
    assert field_map["city"] == "HOUSTON"


def test_discord_sink_uses_no_side_when_no_edge_larger():
    """When edge_no exceeds edge_yes, the embed reports the no side."""
    url = "https://discord.com/api/webhooks/x/y"
    with respx.mock() as mock:
        route = mock.post(url).respond(204)
        alert = Alert(
            fired_at_utc=datetime.now(timezone.utc),
            market_ticker="X", event_ticker="Y", city_slug="DC",
            market_date="2026-05-30", bracket="b", metric="low",
            state="dead_no", reason="r", grade="A", previous_grade=None,
            yes_ask_cents=None, no_ask_cents=7,
            edge_yes=None, edge_no=0.45,
            fair_prob_low=0.0, fair_prob_high=0.02, notes=[],
        )
        DiscordSink(webhook_url=url).emit(alert)
        body = json.loads(route.calls.last.request.content)
    field_map = {f["name"]: f["value"] for f in body["embeds"][0]["fields"]}
    assert "no" in field_map["edge"]
    assert "+0.45" in field_map["edge"]


def test_discord_sink_swallows_failures():
    failures: list[str] = []
    url = "https://discord.com/api/webhooks/dead/path"
    with respx.mock() as mock:
        mock.post(url).respond(500)
        sink = DiscordSink(webhook_url=url,
                           failure_log=lambda msg: failures.append(msg))
        sink.emit(_alert())  # must not raise
        sink.close()
    assert len(failures) == 1


# -- CLI sink-spec parsing ---------------------------------------------------

def test_build_sinks_parses_each_spec_form():
    sinks = _build_sinks((
        "stdout",
        "jsonl:/tmp/alerts.jsonl",
        "webhook:https://hook.example.com/x",
        "ntfy:kalshi-scout",
        "discord:https://discord.com/api/webhooks/1/abc",
    ))
    assert [type(s).__name__ for s in sinks] == [
        "StdoutSink", "JsonlSink", "WebhookSink", "NtfySink", "DiscordSink",
    ]
    # Verify the URLs/topics actually landed on the right sinks.
    assert sinks[3].url == "https://ntfy.sh/kalshi-scout"
    assert sinks[4].webhook_url == "https://discord.com/api/webhooks/1/abc"
    # Close httpx clients on the network-bound sinks to release the connection
    # pools; the test harness flags leaked connections on teardown otherwise.
    for s in sinks[2:]:
        s.close()


def test_build_sinks_rejects_unknown_spec():
    with pytest.raises(click.BadParameter, match="not recognized"):
        _build_sinks(("pushover:abc",))
    # Error message lists every valid form so the operator can self-correct.
    with pytest.raises(click.BadParameter, match="ntfy:TOPIC"):
        _build_sinks(("garbage",))


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
