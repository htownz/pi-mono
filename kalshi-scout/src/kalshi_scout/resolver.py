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

# ICAO codes are stable 4-letter US identifiers starting with K. Look for them
# in the rules text directly — rules sometimes say "(KHOU)" parenthetically.
_ICAO_RE = re.compile(r"\bK[A-Z]{3}\b")

# Match an explicit station name reference. Each registry station can be
# referenced by its `name` substring (e.g. "Houston Hobby Airport") or just
# the city-specific portion ("Hobby Airport").
_AGENCY_RE = re.compile(
    r"(National Weather Service|Met Office|Australian Bureau of Meteorology|Environment Canada)",
    re.IGNORECASE,
)

# Areas as they appear in templated rules. The rulebook template substitutes
# `<area>` with a human description; we extract the chunk between "in" and
# the next operator phrase.
_AREA_RE = re.compile(
    r"temperature\s+in\s+(?P<area>.+?)(?=\s+(?:be\s+|in\s+|on\s+|during\s+|between\s+|above\s+|below\s+|at\s+(?:least|most)\s+|exactly\s+))",
    re.IGNORECASE | re.DOTALL,
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


def _find_station_by_icao(rules_text: str) -> Optional[Station]:
    candidates = _ICAO_RE.findall(rules_text or "")
    if not candidates:
        return None
    icao_set = {c.upper() for c in candidates}
    for station in _all_stations():
        if station.icao.upper() in icao_set:
            return station
    return None


def _find_station_by_name(rules_text: str) -> Optional[Station]:
    text = (rules_text or "").lower()
    if not text:
        return None
    # Prefer the longest match — "Houston Hobby Airport" should beat
    # "Houston" when both exist in the text.
    best: Optional[tuple[int, Station]] = None
    for station in _all_stations():
        name = station.name.lower()
        if name in text:
            score = len(name)
            if best is None or score > best[0]:
                best = (score, station)
        # Also try city slug — case-insensitive, separated word.
        slug_pattern = rf"\b{re.escape(station.city_slug.lower())}\b"
        if re.search(slug_pattern, text):
            score = len(station.city_slug)
            if best is None or score > best[0]:
                best = (score, station)
    return best[1] if best else None


def _extract_area(rules_text: str) -> str:
    if not rules_text:
        return ""
    m = _AREA_RE.search(rules_text)
    if not m:
        return ""
    return m.group("area").strip().rstrip(".,;")


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
      1. Rules text mentions an ICAO we know about.
      2. Rules text mentions a registered station name or city slug.
      3. Contract's parsed city_slug maps to a registered station.
      4. UNVERIFIED — engine must grade F per invariant I4.
    """
    rules_primary = market.raw.get("rules_primary", "") or ""
    rules_secondary = market.raw.get("rules_secondary", "") or ""
    rules_text = f"{rules_primary}\n{rules_secondary}"

    area = _extract_area(rules_text) or (contract.city_slug if contract else "")
    agency = _extract_agency(rules_text)
    notes: list[str] = []

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
