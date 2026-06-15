"""NWS (api.weather.gov) client: station observations + hourly forecast.

We pull two things per station:

  - observations (ASOS samples) within a time window — the observed-so-far
    truth that anchors the scenario ensemble (and, after the day closes, the
    realized daily high/low for the backfill loop).
  - the deterministic hourly forecast — added as weighted scenarios so the
    official NWS forecast anchors the remaining-day distribution.

The NWS API requires a User-Agent identifying the caller; we always send one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx

from weather_trader.models import Station

NWS_BASE_URL = "https://api.weather.gov"
USER_AGENT = "weather-trader/0.1 (ben.melson@gmail.com)"
DEFAULT_TIMEOUT = httpx.Timeout(15.0, connect=5.0)


def c_to_f(c: Optional[float]) -> Optional[float]:
    return None if c is None else c * 9.0 / 5.0 + 32.0


def mps_to_mph(m: Optional[float]) -> Optional[float]:
    return None if m is None else m * 2.236936


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value).astimezone(timezone.utc)
    except ValueError:
        return None


@dataclass
class Observation:
    observed_at: datetime       # UTC
    temperature_f: float


@dataclass
class HourlyPoint:
    start: datetime             # UTC
    temperature_f: float


class NwsClient:
    def __init__(
        self,
        base_url: str = NWS_BASE_URL,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(
            timeout=timeout,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/geo+json, application/ld+json, application/json",
            },
        )
        self._gridpoint_cache: dict[str, dict] = {}

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "NwsClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def _get(self, url: str, params: Optional[dict] = None) -> dict:
        if not url.startswith("http"):
            url = f"{self.base_url}{url}"
        resp = self._client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    def observations(
        self,
        icao: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 500,
    ) -> list[Observation]:
        params: dict = {"limit": limit}
        if start is not None:
            params["start"] = start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if end is not None:
            params["end"] = end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        data = self._get(f"/stations/{icao.upper()}/observations", params=params)
        out: list[Observation] = []
        for feat in data.get("features") or []:
            p = feat.get("properties") or {}
            t = _parse_iso(p.get("timestamp"))
            if t is None:
                continue
            temp_c = (p.get("temperature") or {}).get("value")
            if temp_c is None:
                continue
            out.append(Observation(observed_at=t, temperature_f=c_to_f(temp_c)))  # type: ignore[arg-type]
        out.sort(key=lambda r: r.observed_at)
        return out

    def _gridpoint(self, station: Station) -> dict:
        key = f"{station.latitude:.4f},{station.longitude:.4f}"
        cached = self._gridpoint_cache.get(key)
        if cached is not None:
            return cached
        data = self._get(f"/points/{key}")
        props = data.get("properties") or {}
        self._gridpoint_cache[key] = props
        return props

    def hourly_forecast(self, station: Station) -> list[HourlyPoint]:
        grid = self._gridpoint(station)
        url = grid.get("forecastHourly")
        if not url:
            return []
        data = self._get(url)
        periods = (data.get("properties") or {}).get("periods") or []
        out: list[HourlyPoint] = []
        for p in periods:
            start = _parse_iso(p.get("startTime"))
            temp = p.get("temperature")
            unit = (p.get("temperatureUnit") or "F").upper()
            if start is None or temp is None:
                continue
            temp_f = float(temp) if unit == "F" else c_to_f(float(temp))
            assert temp_f is not None
            out.append(HourlyPoint(start=start, temperature_f=temp_f))
        return out


# -- Helpers ----------------------------------------------------------------------


def observed_extremum(observations: list[Observation], is_high: bool) -> Optional[float]:
    """Running max (is_high) or min over a list of observations. None if empty."""
    temps = [o.temperature_f for o in observations]
    if not temps:
        return None
    return max(temps) if is_high else min(temps)
