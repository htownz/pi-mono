"""Registry mapping Kalshi city slugs to weather stations.

Each city the bot trades maps to exactly one station: its ICAO (for NWS
observations + forecast) and its lat/lon (for the Open-Meteo ensemble). The
timezone defines the local market-day window the daily high/low is taken over.

Add a city by appending a Station here and a series-ticker entry in
`kalshi.TEMPERATURE_SERIES`.
"""

from __future__ import annotations

from typing import Optional

from weather_trader.models import Station

_STATIONS: dict[str, Station] = {
    s.city_slug: s
    for s in [
        Station("KHOU", "Houston Hobby Airport", "HOUSTON", "America/Chicago", 29.6453, -95.2768),
        Station("KIAH", "Houston Bush Intercontinental", "HOUSTONIAH", "America/Chicago", 29.9844, -95.3414),
        Station("KNYC", "New York Central Park", "NYC", "America/New_York", 40.7794, -73.9692),
        Station("KLGA", "LaGuardia Airport", "NYCLGA", "America/New_York", 40.7794, -73.8803),
        Station("KJFK", "John F. Kennedy Airport", "NYCJFK", "America/New_York", 40.6398, -73.7789),
        Station("KMDW", "Chicago Midway", "CHICAGO", "America/Chicago", 41.7868, -87.7522),
        Station("KORD", "Chicago O'Hare", "CHICAGOORD", "America/Chicago", 41.9786, -87.9047),
        Station("KDCA", "Reagan National", "DC", "America/New_York", 38.8521, -77.0377),
        Station("KMIA", "Miami International", "MIAMI", "America/New_York", 25.7933, -80.2906),
        Station("KLAX", "Los Angeles International", "LA", "America/Los_Angeles", 33.9425, -118.4081),
        Station("KSFO", "San Francisco International", "SF", "America/Los_Angeles", 37.6213, -122.3790),
        Station("KDEN", "Denver International", "DENVER", "America/Denver", 39.8561, -104.6737),
        Station("KATL", "Hartsfield-Jackson Atlanta", "ATLANTA", "America/New_York", 33.6407, -84.4277),
        Station("KAUS", "Austin-Bergstrom", "AUSTIN", "America/Chicago", 30.1975, -97.6664),
        Station("KPHL", "Philadelphia International", "PHILLY", "America/New_York", 39.8729, -75.2437),
        Station("KBOS", "Boston Logan", "BOSTON", "America/New_York", 42.3656, -71.0096),
        Station("KLAS", "Las Vegas Harry Reid", "LASVEGAS", "America/Los_Angeles", 36.0840, -115.1537),
        Station("KPHX", "Phoenix Sky Harbor", "PHOENIX", "America/Phoenix", 33.4373, -112.0078),
        Station("KSEA", "Seattle-Tacoma International", "SEATTLE", "America/Los_Angeles", 47.4502, -122.3088),
        Station("KDFW", "Dallas Fort Worth International", "DALLAS", "America/Chicago", 32.8998, -97.0403),
        Station("KSAT", "San Antonio International", "SANANTONIO", "America/Chicago", 29.5337, -98.4698),
        Station("KMSP", "Minneapolis-St Paul International", "MINNEAPOLIS", "America/Chicago", 44.8848, -93.2223),
        Station("KOKC", "Will Rogers Oklahoma City", "OKCITY", "America/Chicago", 35.3931, -97.6007),
        Station("KMSY", "Louis Armstrong New Orleans", "NEWORLEANS", "America/Chicago", 29.9934, -90.2580),
    ]
}


def get_station(city_slug: str) -> Optional[Station]:
    return _STATIONS.get(city_slug.upper())


def all_cities() -> list[str]:
    return sorted(_STATIONS.keys())
