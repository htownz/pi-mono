from zoneinfo import ZoneInfo

from weather_trader.kalshi import TEMPERATURE_SERIES
from weather_trader.stations import all_cities, get_station


def test_every_series_city_resolves_to_a_station():
    missing = sorted({slug for _, slug in TEMPERATURE_SERIES.values() if get_station(slug) is None})
    assert not missing, f"series cities with no station registry entry: {missing}"


def test_stations_have_valid_tz_and_coords():
    for slug in all_cities():
        st = get_station(slug)
        assert st is not None
        ZoneInfo(st.tz)  # raises if the tz is invalid
        assert -90 <= st.latitude <= 90
        assert -180 <= st.longitude <= 180
        assert st.icao.startswith("K") and len(st.icao) == 4


def test_get_station_is_case_insensitive():
    assert get_station("nyc") is get_station("NYC")
