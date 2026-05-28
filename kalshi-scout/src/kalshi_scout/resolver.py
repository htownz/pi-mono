"""Settlement-source resolver.

The Kalshi GLOBALTEMPERATURE rulebook (see AGENTS.md) defines the resolution
hierarchy: National Weather Service first, then per-area national agencies.
For US temperature markets, the relevant settlement source is an NWS station
and the day's official non-preliminary report (CLI).

This module parses `KalshiMarket.raw["rules_primary"]` (and `rules_secondary`
when present) to extract the settlement station explicitly. When the rules
text doesn't pin a station, we fall back to the hand registry in `stations.py`
— but the fallback is **tagged** in the result so the ranker can reflect lower
trust (per AGENTS.md invariant I4).

The resolver is intentionally tolerant: many Kalshi rules will mention a city
("Houston") without an ICAO; others will state an ICAO directly ("KHOU"); a
few may name a station by full name ("Houston Hobby Airport"). We try each
signal in priority order and the first that produces a known station wins.
"""

from __future__ import annotations

import re
from typing import Optional

from kalshi_scout.models import (
    KalshiMarket,
    ParsedContract,
    Settlement,
    SettlementProvenance,
    Station,
)
from kalshi_scout.stations import get_station

# Station lookups by various signals -------------------------------------------

# Kalshi's rules text often names the CLI product directly (e.g. "Data for
# CLIHOU can be found by..."). This is the strongest signal — the CLI is
# literally the settlement document. Look for the 6-character form CLI<XYZ>
# anywhere in the text.
_CLI_PRODUCT_RE = re.compile(r"\bCLI[A-Z]{3}\b")

# ICAO codes are stable 4-letter US identifiers starting with K. Look for them
# in the rules text directly — rules sometimes say "(KHOU)" parenthetically.
_ICAO_RE = re.compile(r"\bK[A-Z]{3}\b")

# NWS Weather Forecast Office codes (e.g. "wfo=hgx" → Houston/Galveston).
# Used as a weak auxiliary signal — narrows the region but doesn't pin a
# specific station, since one WFO can serve multiple climate sites.
_WFO_RE = re.compile(r"\bwfo[=\s:]+([a-z]{3})\b", re.IGNORECASE)

_AGENCY_RE = re.compile(
    r"(National Weather Service|Met Office|Australian Bureau of Meteorology|Environment Canada)",
    re.IGNORECASE,
)

# Areas in real Kalshi rules text appear in several shapes:
#   "temperature in <area> be ..."           (template form)
#   "temperature recorded at <area> for ..." (live form, May 2026)
#   "temperature at <area> on ..."           (variant)
_AREA_PATTERNS = (
    re.compile(
        r"temperature\s+(?:recorded\s+(?:at|in)|measured\s+(?:at|in)|at|in)\s+"
        r"(?P<area>[A-Z][A-Za-z .,'\-/]+?)\s+(?:for|on|during|between|be\b)",
        re.IGNORECASE,
    ),
)


def _all_stations() -> list[Station]:
    """Build a station list once per resolver call by walking the registry."""
    from kalshi_scout.stations import all_cities  # local import to avoid cycles
    out: list[Station] = []
    for slug in all_cities():
        s = get_station(slug)
        if s is not None:
            out.append(s)
    return out


def _find_station_by_cli_product(rules_text: str) -> Optional[Station]:
    """Highest-trust match: the CLI product ID is the settlement document.

    Real Kalshi rules text frequently contains a literal "Data for CLIHOU can
    be found..." — when that's present, settlement is unambiguous.
    """
    candidates = _CLI_PRODUCT_RE.findall(rules_text or "")
    if not candidates:
        return None
    cli_set = {c.upper() for c in candidates}
    for station in _all_stations():
        if station.cli_product.upper() in cli_set:
            return station
    return None


def _find_station_by_icao(rules_text: str) -> Optional[Station]:
    candidates = _ICAO_RE.findall(rules_text or "")
    if not candidates:
        return None
    icao_set = {c.upper() for c in candidates}
    for station in _all_stations():
        if station.icao.upper() in icao_set:
            return station
    return None


def _normalize(text: str) -> str:
    """Lowercase + collapse hyphens/underscores/slashes to single spaces.

    Real rules text writes "Houston-Hobby, TX" while our registry stores
    "Houston Hobby Airport". Normalize both before substring matching.
    """
    text = text.lower()
    text = re.sub(r"[\-_/]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _find_station_by_name(rules_text: str) -> Optional[Station]:
    if not rules_text:
        return None
    norm_text = _normalize(rules_text)
    # Prefer the longest match — "houston hobby airport" should beat
    # "houston" when both exist in the text. Match against both the full
    # registry name and just the city portion (registry name minus the
    # trailing " Airport" / " International" / etc).
    best: Optional[tuple[int, Station]] = None
    for station in _all_stations():
        norm_name = _normalize(station.name)
        # Try full name and a "short" form (drop trailing generic suffixes).
        short_name = re.sub(
            r"\s+(airport|international|intercontinental|national|regional|municipal)$",
            "",
            norm_name,
        )
        candidates = {norm_name, short_name}
        for cand in candidates:
            if cand and cand in norm_text:
                score = len(cand)
                if best is None or score > best[0]:
                    best = (score, station)
        # Also match city slug as a separated word.
        slug_pattern = rf"\b{re.escape(_normalize(station.city_slug))}\b"
        if re.search(slug_pattern, norm_text):
            score = len(station.city_slug)
            if best is None or score > best[0]:
                best = (score, station)
    return best[1] if best else None


def _extract_area(rules_text: str) -> str:
    if not rules_text:
        return ""
    for pattern in _AREA_PATTERNS:
        m = pattern.search(rules_text)
        if m:
            return m.group("area").strip().rstrip(".,;")
    return ""


def _extract_agency(rules_text: str) -> str:
    if not rules_text:
        return "National Weather Service"
    m = _AGENCY_RE.search(rules_text)
    if m:
        return m.group(1)
    return "National Weather Service"


def resolve_settlement(
    market: KalshiMarket,
    contract: Optional[ParsedContract] = None,
) -> Settlement:
    """Return a Settlement for the market.

    Resolution order (each fallback is tagged in the result):
      1. Rules text names a CLI product (e.g. "CLIHOU"). Strongest signal:
         the CLI is literally the settlement document.
      2. Rules text mentions a known ICAO (e.g. "KHOU").
      3. Rules text mentions a registered station name or city slug
         (hyphen/whitespace-tolerant).
      4. Contract's parsed city_slug maps to a registered station
         (REGISTRY fallback, tagged as lower trust).
      5. UNVERIFIED — engine must grade F per invariant I4.
    """
    rules_primary = market.raw.get("rules_primary", "") or ""
    rules_secondary = market.raw.get("rules_secondary", "") or ""
    rules_text = f"{rules_primary}\n{rules_secondary}"

    area = _extract_area(rules_text) or (contract.city_slug if contract else "")
    agency = _extract_agency(rules_text)
    notes: list[str] = []

    station = _find_station_by_cli_product(rules_text)
    if station is not None:
        return Settlement(
            station=station,
            source_agency=agency,
            area_description=area or station.name,
            provenance=SettlementProvenance.RESOLVER,
            notes=(f"matched CLI product {station.cli_product} in rules text",),
        )

    station = _find_station_by_icao(rules_text)
    if station is not None:
        return Settlement(
            station=station,
            source_agency=agency,
            area_description=area or station.name,
            provenance=SettlementProvenance.RESOLVER,
            notes=("matched ICAO in rules text",),
        )

    station = _find_station_by_name(rules_text)
    if station is not None:
        return Settlement(
            station=station,
            source_agency=agency,
            area_description=area or station.name,
            provenance=SettlementProvenance.RESOLVER,
            notes=("matched station/city name in rules text",),
        )

    # Fall back to ticker-derived city slug.
    if contract is not None:
        station = get_station(contract.city_slug)
        if station is not None:
            notes.append("rules text did not pin a station; using registry by ticker city_slug")
            return Settlement(
                station=station,
                source_agency=agency,
                area_description=area or station.name,
                provenance=SettlementProvenance.REGISTRY,
                notes=tuple(notes),
            )

    return Settlement(
        station=None,
        source_agency=agency,
        area_description=area,
        provenance=SettlementProvenance.UNVERIFIED,
        notes=("no settlement source could be derived from rules or registry",),
    )
