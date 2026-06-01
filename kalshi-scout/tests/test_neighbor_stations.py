"""Tests for multi-station ensemble support (Tier 1A enhancement).

The primary's CLI is still the only settlement source — neighbors are a
cross-check + ASOS-outage fallback. These tests verify:

  - build_station_state populates neighbor_running_max/min when the primary
    station has neighbors defined.
  - Stations with no neighbors continue to behave exactly as before.
  - effective_running_max/min falls back to neighbors when primary is silent
    but never overrides primary readings when present.
  - Per-neighbor query failures do not break the primary path.
  - The four target cities (Houston, LA, NYC, Chicago) have neighbors
    registered.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

from kalshi_scout.models import (
    Station,
    StationReading,
    StationState,
)
from kalshi_scout.state import build_station_state
from kalshi_scout.stations import get_station


# -- Stub NwsClient ----------------------------------------------------------

class _StubNws:
    """Minimal NwsClient stub: per-ICAO canned observations, optional failures.

    `obs_by_icao` maps each ICAO to either:
      - a list of (epoch_seconds_offset_from_window_start, temp_f) tuples, or
      - a sentinel `_StubNws.RAISE` to make .observations() raise.
    """
    RAISE = object()

    def __init__(self, window_start: datetime, obs_by_icao: dict):
        self.window_start = window_start
        self.obs_by_icao = obs_by_icao
        self.calls: list[str] = []   # records every icao observations() saw

    def observations(self, icao, start=None, end=None, limit=500):
        self.calls.append(icao)
        spec = self.obs_by_icao.get(icao.upper())
        if spec is None:
            return []
        if spec is self.RAISE:
            raise RuntimeError(f"stub: {icao} explodes")
        return [
            StationReading(
                observed_at=self.window_start.replace(tzinfo=timezone.utc),
                temperature_f=temp_f,
            )
            for _, temp_f in spec
        ]

    def latest_observation(self, icao):
        return None

    def latest_cli(self, location_id):
        return None


# -- Tests -------------------------------------------------------------------

def test_houston_la_nyc_chicago_have_neighbors_registered():
    """The four target cities ship with non-empty neighbors so the ensemble
    path actually fires for them in production. Other cities default to ()."""
    for slug in ("HOUSTON", "LA", "NYC", "CHICAGO", "CHICAGOORD"):
        station = get_station(slug)
        assert station is not None, f"missing station for {slug}"
        assert station.neighbors, f"{slug} has no neighbors configured"


def test_build_station_state_populates_neighbor_extrema_for_houston():
    """KHOU + 3 neighbors → neighbor_running_max/min reflect the network's
    extreme, not the primary's. Sample count is the total readings pulled."""
    station = get_station("HOUSTON")
    assert station is not None
    window_start = datetime(2026, 5, 27, 5, 0, tzinfo=timezone.utc)  # Houston midnight
    stub = _StubNws(window_start, {
        "KHOU": [(0, 88.0), (3600, 90.0)],   # primary: max 90
        "KIAH": [(0, 92.0), (3600, 94.0)],   # neighbor: max 94 (hotter)
        "KGLS": [(0, 85.0), (3600, 86.0)],   # neighbor: max 86
        "KEFD": [(0, 89.0)],                 # neighbor: max 89
    })

    ss = build_station_state(
        stub, station, market_date=date(2026, 5, 27),
        now_utc=datetime(2026, 5, 28, 4, 0, tzinfo=timezone.utc),
    )

    # Primary's own running max is unchanged.
    assert ss.running_max_f == 90.0
    # Neighbor aggregate uses max-of-maxes (most aggressive cross-check).
    assert ss.neighbor_running_max_f == 94.0
    # 2 + 2 + 1 readings from queried neighbors.
    assert ss.neighbor_sample_count == 5
    assert set(ss.neighbor_icaos) == {"KIAH", "KGLS", "KEFD"}


def test_build_station_state_tolerates_per_neighbor_failures():
    """One neighbor blowing up doesn't break the primary path or the other
    neighbors. The dead one is silently dropped."""
    station = get_station("HOUSTON")
    assert station is not None
    window_start = datetime(2026, 5, 27, 5, 0, tzinfo=timezone.utc)
    stub = _StubNws(window_start, {
        "KHOU": [(0, 88.0)],
        "KIAH": _StubNws.RAISE,   # exploding neighbor
        "KGLS": [(0, 85.0)],
        "KEFD": [(0, 89.0)],
    })

    ss = build_station_state(
        stub, station, market_date=date(2026, 5, 27),
        now_utc=datetime(2026, 5, 28, 4, 0, tzinfo=timezone.utc),
    )

    assert ss.running_max_f == 88.0
    # Only KGLS and KEFD contributed.
    assert ss.neighbor_running_max_f == 89.0
    assert set(ss.neighbor_icaos) == {"KGLS", "KEFD"}


def test_build_station_state_no_neighbors_path_unchanged():
    """A station with neighbors=() must produce a StationState with all
    neighbor fields at their zero values — no spurious queries, no behavior
    change for the 20 cities not in the target set."""
    station = get_station("DENVER")   # KDEN has no neighbors configured
    assert station is not None
    assert station.neighbors == ()

    window_start = datetime(2026, 5, 27, 6, 0, tzinfo=timezone.utc)
    stub = _StubNws(window_start, {"KDEN": [(0, 75.0), (3600, 77.0)]})

    ss = build_station_state(
        stub, station, market_date=date(2026, 5, 27),
        now_utc=datetime(2026, 5, 28, 5, 0, tzinfo=timezone.utc),
    )

    assert ss.running_max_f == 77.0
    assert ss.neighbor_running_max_f is None
    assert ss.neighbor_running_min_f is None
    assert ss.neighbor_sample_count == 0
    assert ss.neighbor_icaos == ()
    # Stub only ever called for the primary.
    assert stub.calls == ["KDEN"]


def test_effective_running_max_prefers_primary_when_present():
    """The neighbor fallback only kicks in when primary is None — primary
    readings are never overridden by a hotter neighbor."""
    station = get_station("HOUSTON")
    assert station is not None
    ss = StationState(
        station=station, market_date=date(2026, 5, 27),
        window_start=datetime(2026, 5, 27, 5, 0, tzinfo=timezone.utc),
        window_end=datetime(2026, 5, 28, 5, 0, tzinfo=timezone.utc),
        running_max_f=90.0, running_min_f=70.0,
        latest=None, cli_report_date=None, cli_max_f=None, cli_min_f=None,
        neighbor_running_max_f=94.0,   # hotter neighbor
        neighbor_running_min_f=65.0,   # colder neighbor
    )
    # Primary wins for both.
    assert ss.effective_running_max_f == 90.0
    assert ss.effective_running_min_f == 70.0


def test_effective_running_max_falls_back_to_neighbor_when_primary_missing():
    """Primary ASOS offline → effective_* exposes the neighbor's extremum."""
    station = get_station("HOUSTON")
    assert station is not None
    ss = StationState(
        station=station, market_date=date(2026, 5, 27),
        window_start=datetime(2026, 5, 27, 5, 0, tzinfo=timezone.utc),
        window_end=datetime(2026, 5, 28, 5, 0, tzinfo=timezone.utc),
        running_max_f=None, running_min_f=None,
        latest=None, cli_report_date=None, cli_max_f=None, cli_min_f=None,
        neighbor_running_max_f=92.0,
        neighbor_running_min_f=68.0,
    )
    assert ss.effective_running_max_f == 92.0
    assert ss.effective_running_min_f == 68.0
