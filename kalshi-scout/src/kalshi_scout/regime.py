"""Weather regime classifier.

Maps the current setup at a station (hourly forecast + recent observations)
to one of a small set of qualitative regimes. The regime is meant to be
human-readable context surfaced in evaluation notes — it does NOT auto-shift
fair_probability or grade in this slice.

Why notes-only: the spec's regime taxonomy is long and any auto-adjustment
must be backtest-supported per invariant I9. We need stored history before
we can let regime tags move the grade ladder. For V0.5 the classifier
produces the tag + reasoning; V0.8 backtest-tunes the weights.

Regime tags (deliberately small set; widen with backtest evidence):

  CLEAR_AND_DRY            sky mostly clear, dew-point spread > 10°F,
                           low precip probability. Symmetric — no bias.
  RAIN_COOLED              precip in window or recently; bias toward
                           lower-than-forecast highs.
  MARINE_LAYER             coastal station, low clouds, small dew-point
                           spread, light wind from sea direction; caps highs.
  COLD_FRONT_NEAR          rapidly falling forecast temps, increasing wind,
                           precip-then-clearing pattern; bias toward
                           early-day high then collapse.
  CALM_HUMID_RADIATIONAL   night/early-morning, calm wind, small dew-point
                           spread, clear sky. Optimal radiational cooling
                           setup; bias toward lower-than-forecast lows.
  UNKNOWN                  not enough data or no pattern matches.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from kalshi_scout.models import Station, StationReading
from kalshi_scout.nws import HourlyPoint


class WeatherRegime(str, Enum):
    CLEAR_AND_DRY = "clear_and_dry"
    RAIN_COOLED = "rain_cooled"
    MARINE_LAYER = "marine_layer"
    COLD_FRONT_NEAR = "cold_front_near"
    CALM_HUMID_RADIATIONAL = "calm_humid_radiational"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RegimeReading:
    regime: WeatherRegime
    confidence: float       # 0..1 — heuristic, not calibrated
    reasoning: tuple[str, ...]


# -- Station geography (rough; coastal vs inland) ---------------------------
# These are the only stations where MARINE_LAYER is a plausible call.
_COASTAL_ICAOS = frozenset({
    "KLAX", "KSFO", "KMIA", "KBOS", "KJFK", "KLGA", "KHOU", "KIAH",
})


def classify_regime(
    station: Station,
    forecast: Optional[list[HourlyPoint]],
    recent_obs: Optional[list[StationReading]],
    now_utc: Optional[datetime] = None,
) -> RegimeReading:
    """Classify the current weather regime at a station.

    Decision flow (priority order — first matching wins):
      1. Rain-cooled: high probability of precip in next 6h OR recently
         observed sky descriptor includes rain.
      2. Cold front near: forecast temp dropping > 8°F over next 6h AND
         forecast wind rising.
      3. Marine layer: coastal station, recent obs show small dew-point
         spread (< 4°F), light wind, sky overcast / mostly cloudy.
      4. Calm humid radiational: night/early-morning, calm wind (< 5 mph),
         dew-point spread < 6°F, no precip in forecast next 6h.
      5. Clear and dry: low precip prob, large dew-point spread.
      6. Unknown.
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    reasoning: list[str] = []

    fc_next6 = _forecast_window(forecast, now_utc, hours=6)
    obs_latest = _latest_observation(recent_obs)

    # -- 1. RAIN_COOLED --
    if fc_next6:
        max_pop = max((p.probability_of_precip or 0) for p in fc_next6)
        if max_pop >= 50:
            reasoning.append(f"forecast precip probability {max_pop:.0f}% within 6h")
            return RegimeReading(WeatherRegime.RAIN_COOLED, 0.7, tuple(reasoning))
    if obs_latest and obs_latest.sky:
        sky_l = obs_latest.sky.lower()
        if any(w in sky_l for w in ("rain", "shower", "thunderstorm", "drizzle")):
            reasoning.append(f"recent observation sky='{obs_latest.sky}'")
            return RegimeReading(WeatherRegime.RAIN_COOLED, 0.7, tuple(reasoning))

    # -- 2. COLD_FRONT_NEAR --
    if fc_next6 and len(fc_next6) >= 4:
        temps = [p.temperature_f for p in fc_next6]
        drop = max(temps) - min(temps)
        # Wind rising: later in window vs earlier
        early_wind = sum((p.wind_speed_mph or 0) for p in fc_next6[:2]) / 2
        late_wind = sum((p.wind_speed_mph or 0) for p in fc_next6[-2:]) / 2
        if drop >= 8 and late_wind > early_wind + 3:
            reasoning.append(f"forecast Δtemp={drop:.0f}°F, wind {early_wind:.0f}->{late_wind:.0f}mph")
            return RegimeReading(WeatherRegime.COLD_FRONT_NEAR, 0.6, tuple(reasoning))

    # -- 3. MARINE_LAYER --
    if station.icao in _COASTAL_ICAOS and obs_latest:
        spread = _dew_spread(obs_latest)
        wind = obs_latest.wind_speed_mph or 0
        sky_l = (obs_latest.sky or "").lower()
        if spread is not None and spread < 4 and wind < 12 and \
                any(w in sky_l for w in ("cloud", "overcast", "fog")):
            reasoning.append(
                f"coastal station, dew-spread {spread:.1f}°F, wind {wind:.0f}mph, sky='{obs_latest.sky}'"
            )
            return RegimeReading(WeatherRegime.MARINE_LAYER, 0.6, tuple(reasoning))

    # -- 4. CALM_HUMID_RADIATIONAL --
    if obs_latest:
        spread = _dew_spread(obs_latest)
        wind = obs_latest.wind_speed_mph or 0
        local_hour = _local_hour(station, now_utc)
        is_overnight = local_hour < 7 or local_hour >= 22
        no_precip_soon = True
        if fc_next6:
            no_precip_soon = max((p.probability_of_precip or 0) for p in fc_next6) < 20
        if (
            is_overnight
            and spread is not None and spread < 6
            and wind < 5
            and no_precip_soon
        ):
            reasoning.append(
                f"overnight (local hr {local_hour}), calm ({wind:.0f}mph), "
                f"dew-spread {spread:.1f}°F"
            )
            return RegimeReading(WeatherRegime.CALM_HUMID_RADIATIONAL, 0.6, tuple(reasoning))

    # -- 5. CLEAR_AND_DRY --
    if obs_latest and fc_next6:
        spread = _dew_spread(obs_latest)
        max_pop = max((p.probability_of_precip or 0) for p in fc_next6)
        if spread is not None and spread > 10 and max_pop < 20:
            reasoning.append(f"dew-spread {spread:.1f}°F, max forecast PoP {max_pop:.0f}%")
            return RegimeReading(WeatherRegime.CLEAR_AND_DRY, 0.5, tuple(reasoning))

    # -- 6. UNKNOWN --
    if not obs_latest and not fc_next6:
        reasoning.append("no observations or forecast data available")
    elif not obs_latest:
        reasoning.append("no recent observation")
    elif not fc_next6:
        reasoning.append("no forecast within next 6 hours")
    else:
        reasoning.append("no regime pattern matched")
    return RegimeReading(WeatherRegime.UNKNOWN, 0.3, tuple(reasoning))


# -- Helpers ----------------------------------------------------------------

def _forecast_window(
    forecast: Optional[list[HourlyPoint]], now_utc: datetime, hours: int
) -> list[HourlyPoint]:
    if not forecast:
        return []
    horizon = now_utc.timestamp() + hours * 3600
    return [p for p in forecast if now_utc.timestamp() <= p.start.timestamp() <= horizon]


def _latest_observation(obs: Optional[list[StationReading]]) -> Optional[StationReading]:
    if not obs:
        return None
    return max(obs, key=lambda r: r.observed_at)


def _dew_spread(reading: StationReading) -> Optional[float]:
    if reading.dewpoint_f is None:
        return None
    return reading.temperature_f - reading.dewpoint_f


def _local_hour(station: Station, now_utc: datetime) -> int:
    try:
        from zoneinfo import ZoneInfo
        return now_utc.astimezone(ZoneInfo(station.tz)).hour
    except Exception:
        return now_utc.hour
