"""Hand-curated registry mapping Kalshi city slugs to NWS settlement stations.

Each entry pins exactly which ICAO station and which CLI product the scout
treats as the official settlement source. Expand here when adding cities;
the rest of the scout reads from this registry only.

When adding a new city, verify the Kalshi market's actual settlement source
against the contract rules before adding — guessing the station is exactly
the "settlement-source confusion" pitfall the scout is built to avoid.
"""

from __future__ import annotations

from typing import Optional

from kalshi_scout.models import Station

_STATIONS: dict[str, Station] = {
    s.city_slug: s
    for s in [
        Station(
            icao="KHOU",
            name="Houston Hobby Airport",
            city_slug="HOUSTON",
            tz="America/Chicago",
            cli_product="CLIHOU",
            latitude=29.6453,
            longitude=-95.2768,
        ),
        Station(
            icao="KIAH",
            name="Houston Bush Intercontinental",
            city_slug="HOUSTONIAH",
            tz="America/Chicago",
            cli_product="CLIIAH",
            latitude=29.9844,
            longitude=-95.3414,
        ),
        Station(
            icao="KNYC",
            name="New York Central Park",
            city_slug="NYC",
            tz="America/New_York",
            cli_product="CLINYC",
            latitude=40.7794,
            longitude=-73.9692,
        ),
        Station(
            icao="KLGA",
            name="LaGuardia Airport",
            city_slug="NYCLGA",
            tz="America/New_York",
            cli_product="CLILGA",
            latitude=40.7794,
            longitude=-73.8803,
        ),
        Station(
            icao="KJFK",
            name="John F. Kennedy Airport",
            city_slug="NYCJFK",
            tz="America/New_York",
            cli_product="CLIJFK",
            latitude=40.6398,
            longitude=-73.7789,
        ),
        Station(
            icao="KMDW",
            name="Chicago Midway",
            city_slug="CHICAGO",
            tz="America/Chicago",
            cli_product="CLIMDW",
            latitude=41.7868,
            longitude=-87.7522,
        ),
        Station(
            icao="KORD",
            name="Chicago O'Hare",
            city_slug="CHICAGOORD",
            tz="America/Chicago",
            cli_product="CLIORD",
            latitude=41.9786,
            longitude=-87.9047,
        ),
        Station(
            icao="KDCA",
            name="Reagan National",
            city_slug="DC",
            tz="America/New_York",
            cli_product="CLIDCA",
            latitude=38.8521,
            longitude=-77.0377,
        ),
        Station(
            icao="KMIA",
            name="Miami International",
            city_slug="MIAMI",
            tz="America/New_York",
            cli_product="CLIMIA",
            latitude=25.7933,
            longitude=-80.2906,
        ),
        Station(
            icao="KLAX",
            name="Los Angeles International",
            city_slug="LA",
            tz="America/Los_Angeles",
            cli_product="CLILAX",
            latitude=33.9425,
            longitude=-118.4081,
        ),
        Station(
            icao="KSFO",
            name="San Francisco International",
            city_slug="SF",
            tz="America/Los_Angeles",
            cli_product="CLISFO",
            latitude=37.6213,
            longitude=-122.3790,
        ),
        Station(
            icao="KDEN",
            name="Denver International",
            city_slug="DENVER",
            tz="America/Denver",
            cli_product="CLIDEN",
            latitude=39.8561,
            longitude=-104.6737,
        ),
        Station(
            icao="KATL",
            name="Hartsfield-Jackson Atlanta",
            city_slug="ATLANTA",
            tz="America/New_York",
            cli_product="CLIATL",
            latitude=33.6407,
            longitude=-84.4277,
        ),
        Station(
            icao="KAUS",
            name="Austin-Bergstrom",
            city_slug="AUSTIN",
            tz="America/Chicago",
            cli_product="CLIAUS",
            latitude=30.1975,
            longitude=-97.6664,
        ),
        Station(
            icao="KPHL",
            name="Philadelphia International",
            city_slug="PHILLY",
            tz="America/New_York",
            cli_product="CLIPHL",
            latitude=39.8729,
            longitude=-75.2437,
        ),
        Station(
            icao="KBOS",
            name="Boston Logan",
            city_slug="BOSTON",
            tz="America/New_York",
            cli_product="CLIBOS",
            latitude=42.3656,
            longitude=-71.0096,
        ),
        Station(
            icao="KLAS",
            name="Las Vegas Harry Reid",
            city_slug="LASVEGAS",
            tz="America/Los_Angeles",
            cli_product="CLILAS",
            latitude=36.0840,
            longitude=-115.1537,
        ),
    ]
}


def get_station(city_slug: str) -> Optional[Station]:
    return _STATIONS.get(city_slug.upper())


def all_cities() -> list[str]:
    return sorted(_STATIONS.keys())
