"""Open-Meteo ensemble forecast client.

Open-Meteo's ensemble endpoint exposes per-member hourly temperatures from the
operational GFS/ICON/ECMWF ensembles as plain JSON. Each member becomes one
scenario in the forecaster's distribution — this is where the calibrated
spread comes from. Free tier: 10K req/day, no key.

Docs: https://open-meteo.com/en/docs/ensemble-api
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import httpx

OPENMETEO_ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
USER_AGENT = "weather-trader/0.1 (ben.melson@gmail.com)"
DEFAULT_TIMEOUT = httpx.Timeout(15.0, connect=5.0)

#: `gfs025` = 0.25° GFS ensemble (31 members), well-tuned for US temps,
#: refreshed 4x/day.
DEFAULT_ENSEMBLE_MODEL = "gfs025"

#: Most Kalshi temp markets settle same-day; 2 days covers today + tomorrow.
DEFAULT_FORECAST_DAYS = 2


@dataclass(frozen=True)
class EnsembleHourlyPoint:
    """One forecast hour, with one temperature per ensemble member."""
    start: datetime             # UTC
    members_f: tuple[float, ...]

    @property
    def mean_f(self) -> float:
        if not self.members_f:
            return float("nan")
        return statistics.fmean(self.members_f)

    @property
    def std_f(self) -> float:
        if len(self.members_f) < 2:
            return 0.0
        return statistics.pstdev(self.members_f)


class OpenMeteoClient:
    """Minimal ensemble client. One call per (lat, lon) per scan, cached."""

    def __init__(
        self,
        base_url: str = OPENMETEO_ENSEMBLE_URL,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = base_url
        self._client = client or httpx.Client(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )
        self._cache: dict[tuple[float, float, str, str], list[EnsembleHourlyPoint]] = {}

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpenMeteoClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def ensemble_hourly_temperature(
        self,
        latitude: float,
        longitude: float,
        tz: str,
        forecast_days: int = DEFAULT_FORECAST_DAYS,
        model: str = DEFAULT_ENSEMBLE_MODEL,
    ) -> list[EnsembleHourlyPoint]:
        """Fetch hourly ensemble temps for a point.

        Returns [] (not None) on any HTTP/parse failure so callers treat
        empty-list as "ensemble unavailable". Cached for this client's life by
        (lat, lon, tz, model).
        """
        key = (round(latitude, 4), round(longitude, 4), tz, model)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m",
            "models": model,
            "temperature_unit": "fahrenheit",
            "timezone": tz,
            "forecast_days": forecast_days,
        }
        try:
            resp = self._client.get(self.base_url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError):
            self._cache[key] = []
            return []
        points = parse_ensemble_response(data, tz=tz)
        self._cache[key] = points
        return points


def parse_ensemble_response(data: dict, tz: str) -> list[EnsembleHourlyPoint]:
    """Parse Open-Meteo's ensemble JSON into hourly points (members in °F, UTC).

    Tolerates a missing `hourly` block ([]), no member series (falls back to
    the deterministic `temperature_2m` as a single member), and NaN/None entries
    (dropped per hour).
    """
    hourly = data.get("hourly") or {}
    times: list = hourly.get("time") or []
    if not times:
        return []

    member_keys = sorted(k for k in hourly.keys() if k.startswith("temperature_2m_member"))
    if not member_keys:
        det = hourly.get("temperature_2m")
        if det is None:
            return []
        member_series = [det]
    else:
        member_series = [hourly[k] for k in member_keys]

    zone = ZoneInfo(tz)
    out: list[EnsembleHourlyPoint] = []
    for i, t_str in enumerate(times):
        try:
            local_dt = datetime.fromisoformat(t_str)
        except ValueError:
            continue
        if local_dt.tzinfo is None:
            local_dt = local_dt.replace(tzinfo=zone)
        utc_dt = local_dt.astimezone(timezone.utc)
        vals: list[float] = []
        for series in member_series:
            if i >= len(series):
                continue
            v = series[i]
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isnan(fv):
                continue
            vals.append(fv)
        if not vals:
            continue
        out.append(EnsembleHourlyPoint(start=utc_dt, members_f=tuple(vals)))
    return out
