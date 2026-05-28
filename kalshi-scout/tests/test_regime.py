"""Tests for V0.5 weather regime classifier.

The classifier is heuristic — these tests verify that each regime tag fires
on a clean instance of its triggering pattern, and that nothing fires on
ambiguous data (UNKNOWN fallback).
"""

from datetime import datetime, timedelta, timezone

from kalshi_scout.models import StationReading
from kalshi_scout.nws import HourlyPoint
from kalshi_scout.regime import WeatherRegime, classify_regime
from kalshi_scout.stations import get_station


HOUSTON = get_station("HOUSTON")
DENVER = get_station("DENVER")
NOW = datetime(2026, 5, 27, 18, 0, tzinfo=timezone.utc)  # 1pm CDT
NIGHT = datetime(2026, 5, 28, 8, 0, tzinfo=timezone.utc)  # 3am CDT


def _hourly(start: datetime, temp_f: float, pop: float = 0.0, wind: float = 5.0) -> HourlyPoint:
    return HourlyPoint(
        start=start, temperature_f=temp_f,
        probability_of_precip=pop, wind_speed_mph=wind,
    )


def _obs(temp_f: float, dew_f: float = 50.0, wind: float = 5.0,
         sky: str = "Clear", at: datetime = NOW) -> StationReading:
    return StationReading(
        observed_at=at, temperature_f=temp_f, dewpoint_f=dew_f,
        wind_speed_mph=wind, sky=sky,
    )


# -- RAIN_COOLED -------------------------------------------------------------

def test_rain_cooled_when_high_precip_probability_in_forecast():
    forecast = [_hourly(NOW + timedelta(hours=i), 80, pop=70) for i in range(6)]
    obs = [_obs(82)]
    r = classify_regime(HOUSTON, forecast, obs, now_utc=NOW)
    assert r.regime is WeatherRegime.RAIN_COOLED
    assert "precip probability" in r.reasoning[0]


def test_rain_cooled_when_recent_sky_describes_rain():
    forecast = [_hourly(NOW + timedelta(hours=i), 80, pop=0) for i in range(6)]
    obs = [_obs(75, sky="Thunderstorm")]
    r = classify_regime(HOUSTON, forecast, obs, now_utc=NOW)
    assert r.regime is WeatherRegime.RAIN_COOLED


# -- COLD_FRONT_NEAR ---------------------------------------------------------

def test_cold_front_near_when_temp_drops_and_wind_rises():
    """Temp drops 80 -> 68 over 6h, wind 5 -> 15 mph."""
    forecast = [
        _hourly(NOW + timedelta(hours=0), 80, wind=5),
        _hourly(NOW + timedelta(hours=1), 78, wind=5),
        _hourly(NOW + timedelta(hours=2), 75, wind=8),
        _hourly(NOW + timedelta(hours=3), 72, wind=10),
        _hourly(NOW + timedelta(hours=4), 70, wind=14),
        _hourly(NOW + timedelta(hours=5), 68, wind=16),
    ]
    obs = [_obs(80)]
    r = classify_regime(HOUSTON, forecast, obs, now_utc=NOW)
    assert r.regime is WeatherRegime.COLD_FRONT_NEAR


# -- MARINE_LAYER ------------------------------------------------------------

def test_marine_layer_when_coastal_and_low_clouds():
    obs = [_obs(72, dew_f=70, wind=8, sky="Overcast")]
    forecast = [_hourly(NOW + timedelta(hours=i), 75, pop=10, wind=8) for i in range(6)]
    r = classify_regime(HOUSTON, forecast, obs, now_utc=NOW)
    assert r.regime is WeatherRegime.MARINE_LAYER


def test_marine_layer_does_not_fire_for_inland_station():
    """Same conditions at Denver (inland, KDEN not in coastal set) should not match."""
    obs = [_obs(72, dew_f=70, wind=8, sky="Overcast")]
    forecast = [_hourly(NOW + timedelta(hours=i), 75, pop=10, wind=8) for i in range(6)]
    r = classify_regime(DENVER, forecast, obs, now_utc=NOW)
    assert r.regime is not WeatherRegime.MARINE_LAYER


# -- CALM_HUMID_RADIATIONAL --------------------------------------------------

def test_calm_humid_radiational_at_night_with_clear_and_calm():
    """3am local, calm wind, dew-spread = 65 - 62 = 3°F."""
    obs = [_obs(65, dew_f=62, wind=2, sky="Clear", at=NIGHT)]
    forecast = [_hourly(NIGHT + timedelta(hours=i), 64, pop=0, wind=2) for i in range(6)]
    r = classify_regime(HOUSTON, forecast, obs, now_utc=NIGHT)
    assert r.regime is WeatherRegime.CALM_HUMID_RADIATIONAL


def test_calm_humid_does_not_fire_in_afternoon():
    """1pm CDT — afternoon, not overnight."""
    obs = [_obs(78, dew_f=72, wind=2, sky="Clear")]
    forecast = [_hourly(NOW + timedelta(hours=i), 80, pop=0, wind=2) for i in range(6)]
    r = classify_regime(HOUSTON, forecast, obs, now_utc=NOW)
    assert r.regime is not WeatherRegime.CALM_HUMID_RADIATIONAL


# -- CLEAR_AND_DRY -----------------------------------------------------------

def test_clear_and_dry_when_low_humidity_and_no_precip_forecast():
    """Dew spread 30°F, max PoP 10%."""
    obs = [_obs(85, dew_f=55, wind=5, sky="Clear")]
    forecast = [_hourly(NOW + timedelta(hours=i), 87, pop=5, wind=5) for i in range(6)]
    r = classify_regime(HOUSTON, forecast, obs, now_utc=NOW)
    assert r.regime is WeatherRegime.CLEAR_AND_DRY


# -- UNKNOWN -----------------------------------------------------------------

def test_unknown_when_no_data():
    r = classify_regime(HOUSTON, [], [], now_utc=NOW)
    assert r.regime is WeatherRegime.UNKNOWN
    assert "no observations" in r.reasoning[0].lower() or "no forecast" in r.reasoning[0].lower()


def test_unknown_when_no_pattern_matches():
    """Mid-day, moderate wind, moderate humidity, no precip forecast — none
    of the clear patterns trigger."""
    obs = [_obs(75, dew_f=65, wind=10, sky="Partly Cloudy")]
    forecast = [_hourly(NOW + timedelta(hours=i), 76, pop=15, wind=10) for i in range(6)]
    r = classify_regime(HOUSTON, forecast, obs, now_utc=NOW)
    assert r.regime is WeatherRegime.UNKNOWN
