"""Tests for V1.0 JsonlSink size-based rotation."""

from datetime import datetime, timezone
from pathlib import Path

from kalshi_scout.notify import Alert, JsonlSink


def _alert() -> Alert:
    return Alert(
        fired_at_utc=datetime(2026, 5, 27, 16, 30, tzinfo=timezone.utc),
        market_ticker="X", event_ticker="Y",
        city_slug="HOUSTON", market_date="2026-05-27",
        bracket="79–80°", metric="high",
        state="locked_yes", reason="r",
        grade="A+", previous_grade="C",
        yes_ask_cents=71, no_ask_cents=29,
        edge_yes=0.28, edge_no=None,
        fair_prob_low=0.97, fair_prob_high=0.99,
        notes=["a" * 200],  # padding to make each line large enough to rotate quickly
    )


def test_no_rotation_below_max_bytes(tmp_path: Path):
    path = tmp_path / "alerts.jsonl"
    sink = JsonlSink(path, max_bytes=10_000_000)  # 10 MB threshold
    for _ in range(5):
        sink.emit(_alert())
    assert path.exists()
    assert not (tmp_path / "alerts.jsonl.1").exists()


def test_rotation_when_exceeds_max_bytes(tmp_path: Path):
    """With a tiny threshold, rotation kicks in after the second alert."""
    path = tmp_path / "alerts.jsonl"
    sink = JsonlSink(path, max_bytes=200, backup_count=3)
    sink.emit(_alert())
    # Each line is ~500 bytes due to the 200-char note padding, so the
    # second emit() will trigger rotation BEFORE writing.
    sink.emit(_alert())
    # After rotation, the original lives at .1 and the new write is the
    # only line in alerts.jsonl.
    assert (tmp_path / "alerts.jsonl.1").exists()
    assert path.exists()
    assert path.read_text().count("\n") == 1


def test_rotation_shifts_old_backups(tmp_path: Path):
    """Multiple rotations should produce .1 / .2 / .3 archives in order."""
    path = tmp_path / "alerts.jsonl"
    sink = JsonlSink(path, max_bytes=200, backup_count=3)
    for _ in range(5):
        sink.emit(_alert())
    # After 5 emits, we should have alerts.jsonl + .1, .2, .3 (capped at backup_count)
    assert path.exists()
    assert (tmp_path / "alerts.jsonl.1").exists()
    assert (tmp_path / "alerts.jsonl.2").exists()
    assert (tmp_path / "alerts.jsonl.3").exists()
    # .4 should NOT exist — we only keep backup_count archives.
    assert not (tmp_path / "alerts.jsonl.4").exists()


def test_emit_creates_parent_dir(tmp_path: Path):
    nested = tmp_path / "deep" / "nested" / "alerts.jsonl"
    sink = JsonlSink(nested, max_bytes=1024)
    sink.emit(_alert())
    assert nested.exists()


def test_zero_max_bytes_disables_rotation(tmp_path: Path):
    """max_bytes=0 means never rotate."""
    path = tmp_path / "alerts.jsonl"
    sink = JsonlSink(path, max_bytes=0)
    for _ in range(10):
        sink.emit(_alert())
    assert not (tmp_path / "alerts.jsonl.1").exists()
