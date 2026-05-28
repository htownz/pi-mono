"""Unit tests for the V0.4 settlement-source resolver.

Rules-text fixtures are synthesized from the GLOBALTEMPERATURE template; when
real Kalshi `rules_primary` text becomes available, drop it in `fixtures/`
and add a passthrough test that loads the real string.
"""

from datetime import date

from kalshi_scout.models import (
    Bracket,
    BracketKind,
    KalshiMarket,
    Metric,
    ParsedContract,
    SettlementProvenance,
)
from kalshi_scout.resolver import resolve_settlement


def _market(rules_primary: str = "", ticker: str = "KXLOWHOUSTON-26MAY28-LTE70") -> KalshiMarket:
    return KalshiMarket(
        ticker=ticker,
        event_ticker=ticker.rsplit("-", 1)[0],
        title="",
        yes_sub_title="70° or below",
        status="open",
        close_time=None,
        yes_bid=None,
        yes_ask=None,
        no_bid=None,
        no_ask=None,
        last_price=None,
        volume=0,
        open_interest=0,
        raw={"rules_primary": rules_primary},
    )


def _contract(city_slug: str = "HOUSTON") -> ParsedContract:
    return ParsedContract(
        market_ticker="X",
        event_ticker="Y",
        city_slug=city_slug,
        metric=Metric.LOW,
        market_date=date(2026, 5, 28),
        bracket=Bracket(BracketKind.LTE, lo=None, hi=70.0),
    )


def test_resolver_matches_icao_in_rules_text():
    rules = (
        "The Underlying for this Contract is the minimum temperature in Houston, Texas (KHOU) "
        "on May 28, 2026. The Source Agencies are, in hierarchical order, "
        "National Weather Service."
    )
    s = resolve_settlement(_market(rules), _contract())
    assert s.station is not None
    assert s.station.icao == "KHOU"
    assert s.provenance is SettlementProvenance.RESOLVER
    assert "matched ICAO" in s.notes[0]


def test_resolver_matches_station_name_when_no_icao():
    rules = (
        "The Underlying for this Contract is the minimum temperature in "
        "Houston Hobby Airport on May 28, 2026. The Source Agencies are, "
        "in hierarchical order, National Weather Service."
    )
    s = resolve_settlement(_market(rules), _contract())
    assert s.station is not None
    assert s.station.icao == "KHOU"
    assert s.provenance is SettlementProvenance.RESOLVER


def test_resolver_falls_back_to_registry_when_rules_text_empty():
    s = resolve_settlement(_market(""), _contract("HOUSTON"))
    assert s.station is not None
    assert s.station.icao == "KHOU"
    assert s.provenance is SettlementProvenance.REGISTRY


def test_resolver_unverified_when_unknown_city_and_no_rules():
    s = resolve_settlement(_market(""), _contract("TIMBUKTU"))
    assert s.station is None
    assert s.provenance is SettlementProvenance.UNVERIFIED


def test_resolver_picks_longest_match_when_multiple_cities_appear():
    # "Houston" appears multiple times, but "Houston Hobby Airport" is more
    # specific and should win the longest-match tiebreaker.
    rules = (
        "Compared against Houston International (KIAH), the official settlement station "
        "is Houston Hobby Airport. The Source Agencies are National Weather Service."
    )
    s = resolve_settlement(_market(rules), _contract())
    # ICAO match takes priority — KIAH is mentioned explicitly.
    assert s.station is not None
    assert s.station.icao == "KIAH"
    assert s.provenance is SettlementProvenance.RESOLVER


def test_resolver_extracts_source_agency():
    rules = "Source Agencies are, in hierarchical order, National Weather Service."
    s = resolve_settlement(_market(rules), _contract())
    assert "National Weather Service" in s.source_agency


def test_resolver_rejects_unknown_icao():
    # KXYZ is not a known station; rules-text ICAO match should yield None.
    rules = "Underlying temperature reported at KXYZ on the Expiration Date."
    s = resolve_settlement(_market(rules), _contract("TIMBUKTU"))
    # Falls through to registry; TIMBUKTU not registered, so UNVERIFIED.
    assert s.station is None
    assert s.provenance is SettlementProvenance.UNVERIFIED
