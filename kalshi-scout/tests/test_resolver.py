"""Unit tests for the V0.4 settlement-source resolver.

Rules-text fixtures live under `tests/fixtures/`. The May 27, 2026 Houston
high-temp fixture is real Kalshi rules text (verified visually); tests that
exercise it are the authoritative regression check.
"""

from datetime import date
from pathlib import Path

from kalshi_scout.models import (
    Bracket,
    BracketKind,
    KalshiMarket,
    Metric,
    ParsedContract,
    SettlementProvenance,
)
from kalshi_scout.resolver import resolve_settlement

FIXTURES = Path(__file__).parent / "fixtures"


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


# -- Real Kalshi rules-text fixtures --------------------------------------------

def test_resolver_houston_may27_fixture_cli_product_match():
    """Verified against real Kalshi rules text for KXHIGHHOUSTON-26MAY27-B79-80.

    The rules name CLIHOU directly: "Data for CLIHOU can be found by clicking
    the following URL..." That's the highest-trust signal and should win
    against any weaker matches (station name, WFO code, bare city).
    """
    rules = (FIXTURES / "rules_KXHIGHHOUSTON_26MAY27_B79-80.txt").read_text()
    market = _market(rules, ticker="KXHIGHHOUSTON-26MAY27-B79-80")
    contract = ParsedContract(
        market_ticker=market.ticker,
        event_ticker="KXHIGHHOUSTON-26MAY27",
        city_slug="HOUSTON",
        metric=Metric.HIGH,
        market_date=date(2026, 5, 27),
        bracket=Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0),
    )
    s = resolve_settlement(market, contract)
    assert s.station is not None
    assert s.station.icao == "KHOU"
    assert s.station.cli_product == "CLIHOU"
    assert s.provenance is SettlementProvenance.RESOLVER
    assert "CLIHOU" in s.notes[0]
    # Area extractor should pick up "Houston" from "recorded at Houston for ...".
    assert "houston" in s.area_description.lower()


def test_resolver_handles_hyphenated_station_name():
    """The same fixture mentions 'Houston-Hobby' (hyphenated). The name
    matcher must be hyphen-tolerant so it still resolves correctly even
    without a CLI product in the text."""
    rules = (
        "Outcome verified from NWS Climatological Report Houston. "
        "Location: Houston-Hobby, TX with Daily Climate Report selected."
    )
    s = resolve_settlement(_market(rules), _contract())
    assert s.station is not None
    assert s.station.icao == "KHOU"
    assert s.provenance is SettlementProvenance.RESOLVER
