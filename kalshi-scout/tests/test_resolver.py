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


def _fixture_contract(city_slug: str, metric: Metric, bracket: Bracket) -> ParsedContract:
    return ParsedContract(
        market_ticker="X",
        event_ticker="Y",
        city_slug=city_slug,
        metric=metric,
        market_date=date(2026, 5, 27),
        bracket=bracket,
    )


def test_fixture_lowhouston_lte68_resolves_via_cli_product():
    """KXLOWHOUSTON LOW market: same CLIHOU mention as the HIGH market.
    Validates that LOW/HIGH markets resolve identically — matchers are
    metric-agnostic."""
    rules = (FIXTURES / "rules_KXLOWHOUSTON_26MAY27_LTE68.txt").read_text()
    contract = _fixture_contract("HOUSTON", Metric.LOW, Bracket(BracketKind.LTE, lo=None, hi=68.0))
    s = resolve_settlement(_market(rules), contract)
    assert s.station is not None and s.station.icao == "KHOU"
    assert s.provenance is SettlementProvenance.RESOLVER
    assert "CLIHOU" in s.notes[0]


def test_fixture_austin_resolves_via_token_match():
    """Registry: 'Austin-Bergstrom'. Rules text: 'Austin Bergstrom'.
    Token matcher must treat the hyphen as a word separator."""
    rules = (FIXTURES / "rules_KXHIGHAUSTIN_26MAY27_LTE80.txt").read_text()
    contract = _fixture_contract("AUSTIN", Metric.HIGH, Bracket(BracketKind.LTE, lo=None, hi=80.0))
    s = resolve_settlement(_market(rules), contract)
    assert s.station is not None and s.station.icao == "KAUS"
    assert s.provenance is SettlementProvenance.RESOLVER


def test_fixture_miami_resolves_via_token_match():
    """Registry: 'Miami International'. Rules text: 'Miami International Airport'.
    Generic-suffix stripping makes the registry token set {'miami'}, which
    sits inside the rules-text token set."""
    rules = (FIXTURES / "rules_KXHIGHMIAMI_26MAY27_B89-90.txt").read_text()
    contract = _fixture_contract("MIAMI", Metric.HIGH, Bracket(BracketKind.BETWEEN, lo=89.0, hi=90.0))
    s = resolve_settlement(_market(rules), contract)
    assert s.station is not None and s.station.icao == "KMIA"
    assert s.provenance is SettlementProvenance.RESOLVER


def test_fixture_chicago_resolves_to_midway_not_ohare():
    """Kalshi's Chicago market settles at KMDW (Midway), not KORD (O'Hare).
    Registry now reflects that and the token matcher picks KMDW.

    Regression check: if someone reverts CHICAGO back to KORD, this fails
    and the test message tells them exactly why."""
    rules = (FIXTURES / "rules_KXHIGHCHICAGO_26MAY27_B81-82.txt").read_text()
    contract = _fixture_contract("CHICAGO", Metric.HIGH, Bracket(BracketKind.BETWEEN, lo=81.0, hi=82.0))
    s = resolve_settlement(_market(rules), contract)
    assert s.station is not None
    assert s.station.icao == "KMDW", (
        f"Chicago market resolved to {s.station.icao}; Kalshi rules text says 'Chicago Midway'"
    )
    assert s.provenance is SettlementProvenance.RESOLVER


def test_fixture_la_resolves_despite_airport_vs_international():
    """Registry: 'Los Angeles International'. Rules text: 'Los Angeles Airport'.
    Both contain 'los angeles' after suffix-stripping."""
    rules = (FIXTURES / "rules_KXHIGHLA_26MAY27_B66-67.txt").read_text()
    contract = _fixture_contract("LA", Metric.HIGH, Bracket(BracketKind.BETWEEN, lo=66.0, hi=67.0))
    s = resolve_settlement(_market(rules), contract)
    assert s.station is not None and s.station.icao == "KLAX"
    assert s.provenance is SettlementProvenance.RESOLVER


def test_fixture_nyc_resolves_despite_token_order_swap():
    """Registry: 'New York Central Park'. Rules text: 'Central Park, New York'.
    Substring matching would fail here — token-set matching is required."""
    rules = (FIXTURES / "rules_KXHIGHNYC_26MAY27_B81-82.txt").read_text()
    contract = _fixture_contract("NYC", Metric.HIGH, Bracket(BracketKind.BETWEEN, lo=81.0, hi=82.0))
    s = resolve_settlement(_market(rules), contract)
    assert s.station is not None and s.station.icao == "KNYC"
    assert s.provenance is SettlementProvenance.RESOLVER


def test_fixture_lasvegas_resolves_via_cli_product():
    """Las Vegas LOW market. The rules text names CLILAS explicitly, the
    same pattern as Houston's CLIHOU mention. Confirms the CLI-product
    matcher is city-agnostic.

    Also exercises a freshly-added registry entry (KLAS / CLILAS)."""
    rules = (FIXTURES / "rules_KXLOWLASVEGAS_26MAY27_B59-60.txt").read_text()
    contract = _fixture_contract("LASVEGAS", Metric.LOW, Bracket(BracketKind.BETWEEN, lo=59.0, hi=60.0))
    s = resolve_settlement(_market(rules), contract)
    assert s.station is not None and s.station.icao == "KLAS"
    assert s.station.cli_product == "CLILAS"
    assert s.provenance is SettlementProvenance.RESOLVER
    assert "CLILAS" in s.notes[0]
