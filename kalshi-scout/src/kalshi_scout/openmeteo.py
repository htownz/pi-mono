"""Open-Meteo ensemble forecast client.

The NWS public JSON API only exposes deterministic temperature forecasts.
Open-Meteo's ensemble endpoint exposes per-member hourly temperatures from
the operational GFS/ICON/ECMWF ensembles via plain JSON — same complexity
as our NWS client but with calibrated probabilistic content.

We use this as an OPTIONAL second opinion. The settlement source is still
the primary station's NWS CLI; Open-Meteo never touches that path. When
opted in via `--use-ensemble` (config.use_ensemble=True), the engine
computes fair_prob by counting ensemble members that would settle YES on
the contract, then falls back to the NWS-only path when the ensemble call
fails or returns insufficient data.

Free tier: 10K requests/day, no API key. We make one call per (lat, lon)
per scan iteration, cached for the duration of the scan to avoid
duplicate fetches across markets sharing a station.

Docs: https://open-meteo.com/en/docs/ensemble-api
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import httpx


OPENMETEO_ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
USER_AGENT = "kalshi-scout/0.3 (ben.melson@gmail.com)"
DEFAULT_TIMEOUT = httpx.Timeout(15.0, connect=5.0)

#: Default ensemble model. `gfs025` is the 0.25° GFS ensemble (31 members),
#: well-tuned for US temperature forecasts and updated 4×/day. ECMWF
#: (`ecmwf_ifs04`) is more accurate at longer leads but the IFS ensemble has
#: 51 members — more JSON traffic — and refreshes only 2×/day.
DEFAULT_ENSEMBLE_MODEL = "gfs025"

#: Forecast horizon. Most Kalshi temp markets settle same-day, so 2 days of
#: ensemble covers today + tomorrow's settlement window with margin.
DEFAULT_FORECAST_DAYS = 2


@dataclass(frozen=True)
class EnsembleHourlyPoint:
    """One hour of the forecast, with one temperature per ensemble member."""
    start: datetime
    members_f: tuple[float, ...]

    @property
    def mean_f(self) -> float:
        if not self.members_f:
            return float("nan")
        return statistics.fmean(self.members_f)

    @property
    def std_f(self) -> float:
        """Population std-dev across members. 0 when only one member exists."""
        if len(self.members_f) < 2:
            return 0.0
        return statistics.pstdev(self.members_f)


class OpenMeteoClient:
    """Minimal ensemble client. One call per (lat, lon) per scan iteration."""

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
        # Per-(lat, lon, tz, model) cache. Cleared by caller between scans
        # if it ever wants a fresh fetch.
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
        """Fetch hourly ensemble temps for the given point.

        Returns an empty list (NOT None) on any HTTP / parse failure so
        callers can treat empty-list as "ensemble unavailable, fall back".

        Cached for the lifetime of this client by (lat, lon, tz, model).
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
    """Parse Open-Meteo's ensemble JSON into hourly points.

    Response shape (relevant slice):
        {
          "timezone": "America/Chicago",
          "utc_offset_seconds": -18000,
          "hourly": {
            "time": ["2026-05-31T00:00", "2026-05-31T01:00", ...],
            "temperature_2m_member01": [78.2, 77.8, ...],
            "temperature_2m_member02": [78.5, 77.9, ...],
            ...
            "temperature_2m": [78.4, 77.9, ...]      // deterministic mean (optional)
          }
        }

    Tolerates:
      - Missing `hourly` block → returns [].
      - No member series (just deterministic) → treats the deterministic
        series as a single-member ensemble. Useful as a degraded-mode signal.
      - Member series with NaN/None entries → drops the entry for that hour.
    """
    hourly = data.get("hourly") or {}
    times: list = hourly.get("time") or []
    if not times:
        return []

    # Pick member series by key prefix. Open-Meteo names them as
    # `temperature_2m_member01`, `_member02`, etc. Fall back to the
    # deterministic `temperature_2m` when no members are present.
    member_keys = sorted(
        k for k in hourly.keys() if k.startswith("temperature_2m_member")
    )
    if not member_keys:
        det = hourly.get("temperature_2m")
        if det is None:
            return []
        member_series = [det]
    else:
        member_series = [hourly[k] for k in member_keys]

    # The timezone the response was rendered in. Open-Meteo returns naive
    # local-time strings; pair with tz to get aware datetimes, then convert
    # to UTC so they compose cleanly with the rest of the engine.
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
        # Members at this hour. Drop members whose value is None/NaN.
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
