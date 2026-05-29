"""NWS (api.weather.gov) client + CLI product parsing.

We pull three things from NWS for each station:

  - station observations (5-min ASOS samples) in a time window
  - the hourly forecast (so we can project remaining-day risk)
  - the daily CLI product (the official non-preliminary climate report)

The NWS API requires a User-Agent identifying the caller. We always send one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import httpx

from kalshi_scout.models import Station, StationReading

NWS_BASE_URL = "https://api.weather.gov"
USER_AGENT = "kalshi-scout/0.3 (ben.melson@gmail.com)"
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
class HourlyPoint:
    start: datetime
    temperature_f: float
    sky_cover_pct: Optional[float] = None
    probability_of_precip: Optional[float] = None
    wind_speed_mph: Optional[float] = None


@dataclass
class CliReport:
    product_id: str
    issued_at: datetime
    report_date: Optional[date]
    max_f: Optional[float]
    min_f: Optional[float]
    raw_text: str


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

    # -- Observations --------------------------------------------------------------

    def observations(
        self,
        icao: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 500,
    ) -> list[StationReading]:
        params: dict = {"limit": limit}
        if start is not None:
            params["start"] = start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if end is not None:
            params["end"] = end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        data = self._get(f"/stations/{icao.upper()}/observations", params=params)
        readings: list[StationReading] = []
        for feat in data.get("features") or []:
            p = feat.get("properties") or {}
            t = _parse_iso(p.get("timestamp"))
            if t is None:
                continue
            temp_c = (p.get("temperature") or {}).get("value")
            if temp_c is None:
                continue
            dew_c = (p.get("dewpoint") or {}).get("value")
            wind_mps = (p.get("windSpeed") or {}).get("value")
            sky = p.get("textDescription")
            readings.append(
                StationReading(
                    observed_at=t,
                    temperature_f=c_to_f(temp_c),  # type: ignore[arg-type]
                    dewpoint_f=c_to_f(dew_c),
                    wind_speed_mph=mps_to_mph(wind_mps),
                    sky=sky,
                )
            )
        readings.sort(key=lambda r: r.observed_at)
        return readings

    def latest_observation(self, icao: str) -> Optional[StationReading]:
        try:
            data = self._get(f"/stations/{icao.upper()}/observations/latest")
        except httpx.HTTPStatusError:
            return None
        p = (data.get("properties") or {})
        t = _parse_iso(p.get("timestamp"))
        temp_c = (p.get("temperature") or {}).get("value")
        if t is None or temp_c is None:
            return None
        return StationReading(
            observed_at=t,
            temperature_f=c_to_f(temp_c),  # type: ignore[arg-type]
            dewpoint_f=c_to_f((p.get("dewpoint") or {}).get("value")),
            wind_speed_mph=mps_to_mph((p.get("windSpeed") or {}).get("value")),
            sky=p.get("textDescription"),
        )

    # -- Forecast ------------------------------------------------------------------

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
            wind = p.get("windSpeed")  # e.g. "10 mph"
            wind_mph: Optional[float] = None
            if isinstance(wind, str):
                m = re.search(r"(\d+(?:\.\d+)?)", wind)
                if m:
                    wind_mph = float(m.group(1))
            out.append(
                HourlyPoint(
                    start=start,
                    temperature_f=temp_f,
                    probability_of_precip=(p.get("probabilityOfPrecipitation") or {}).get("value"),
                    wind_speed_mph=wind_mph,
                )
            )
        return out

    # -- CLI product ---------------------------------------------------------------

    def latest_cli(self, location_id: str) -> Optional[CliReport]:
        """Fetch the most recent CLI product for a given CLI location id, e.g. CLIHOU.

        Returns None if no product is available.
        """
        try:
            listing = self._get(f"/products/types/CLI/locations/{location_id}")
        except httpx.HTTPStatusError:
            return None
        items = (listing.get("@graph") or listing.get("graph") or [])
        if not items:
            return None
        # Most recent first by issuance timestamp
        items.sort(key=lambda p: p.get("issuanceTime") or "", reverse=True)
        latest = items[0]
        try:
            product = self._get(latest["@id"]) if latest.get("@id") else self._get(f"/products/{latest['id']}")
        except httpx.HTTPStatusError:
            return None
        text = product.get("productText") or ""
        issued = _parse_iso(product.get("issuanceTime")) or datetime.now(timezone.utc)
        return parse_cli_report(product_id=product.get("id") or location_id, text=text, issued_at=issued)


# -- CLI text parser --------------------------------------------------------------

_CLI_DATE_RE = re.compile(
    r"CLIMATE\s+REPORT.*?\n.*?\n"
    r"(?P<month>JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\w*\s+(?P<day>\d{1,2})\s+(?P<year>\d{4})",
    re.IGNORECASE | re.DOTALL,
)

_CLI_MAX_RE = re.compile(r"^MAXIMUM\s+(-?\d+)", re.IGNORECASE | re.MULTILINE)
_CLI_MIN_RE = re.compile(r"^MINIMUM\s+(-?\d+)", re.IGNORECASE | re.MULTILINE)

_MONTHS_3 = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def parse_cli_report(product_id: str, text: str, issued_at: datetime) -> CliReport:
    """Parse a CLI plain-text product into (report_date, max_f, min_f).

    CLI products are fixed-column text. The block we care about looks like:

        TEMPERATURE (F)
         MAXIMUM        79  504 PM   ...
         MINIMUM        72  600 AM   ...

    And the header carries the report date, e.g. "MAY 27 2026".
    """
    report_date: Optional[date] = None
    m = _CLI_DATE_RE.search(text)
    if m:
        mon = _MONTHS_3.get(m.group("month").upper())
        if mon:
            try:
                report_date = date(int(m.group("year")), mon, int(m.group("day")))
            except ValueError:
                report_date = None

    max_f: Optional[float] = None
    min_f: Optional[float] = None
    mm = _CLI_MAX_RE.search(text)
    if mm:
        try:
            max_f = float(mm.group(1))
        except ValueError:
            max_f = None
    mn = _CLI_MIN_RE.search(text)
    if mn:
        try:
            min_f = float(mn.group(1))
        except ValueError:
            min_f = None

    return CliReport(
        product_id=product_id,
        issued_at=issued_at,
        report_date=report_date,
        max_f=max_f,
        min_f=min_f,
        raw_text=text,
    )


# -- Helpers ----------------------------------------------------------------------

def market_day_window(market_date: date, tz: str) -> tuple[datetime, datetime]:
    """Return the [start, end) local-time window for a given market date.

    Settlement is based on observations *inside* the local calendar day. We
    return localized datetimes so callers can convert to UTC for API calls.
    """
    z = ZoneInfo(tz)
    start = datetime(market_date.year, market_date.month, market_date.day, 0, 0, 0, tzinfo=z)
    end = datetime(market_date.year, market_date.month, market_date.day, 23, 59, 59, tzinfo=z)
    # We treat the window as inclusive of the final second of the day. Callers
    # using it as half-open [start, end) won't include observations exactly at
    # 23:59:59, which is acceptable — ASOS samples won't perfectly align with
    # midnight anyway.
    return start, end
