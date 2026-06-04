"""Curated cross-venue parity registry + link resolution (DRAFT).

Cross-venue matching — deciding which Polymarket market is the SAME real-world outcome as
which Kalshi ticker — can't be done safely by fuzzy string match alone (dates, bucket edges,
and resolution sources have to line up). Until that's automated, the matcher IS this
hand-curated registry: a human declares each pair and asserts whether the two venues settle
identically.

Each ParityCandidate is ONE outcome:
  - pm_match:        case-insensitive substring used to locate the Polymarket market by its
                     question/slug (e.g. "fed decision in june");
  - kalshi_ticker:   the Kalshi market ticker for the same YES outcome;
  - settlement_verified: a human's assertion that both venues resolve this outcome identically
                     (same source, date, bucket). Until you've actually checked, leave it False
                     — scan_parity will still surface the price lock, just flagged unverified.

The seed entries below are TEMPLATES (settlement_verified=False). Replace pm_match / the
Kalshi ticker with verified live identifiers before trusting any result. See parity_run.py.
"""
from __future__ import annotations

from dataclasses import dataclass

from .parity import ParityLink, VenueQuote


@dataclass(frozen=True)
class ParityCandidate:
    name: str                  # human description of the shared outcome
    pm_match: str              # case-insensitive substring to find the Polymarket market
    kalshi_ticker: str         # Kalshi market ticker for the same YES outcome
    settlement_verified: bool = False
    note: str = ""


# Seed templates — recurring markets that exist on both venues. Identifiers are PLACEHOLDERS
# to be replaced with live, verified values; settlement_verified stays False until checked.
REGISTRY: list[ParityCandidate] = [
    ParityCandidate(
        name="Fed rate decision (June): no change",
        pm_match="fed decision in june",
        kalshi_ticker="KXFED-26JUN-REPLACE",
        note="TEMPLATE: confirm PM question text + Kalshi ticker; verify both settle on the "
             "same FOMC announcement and the same outcome bucket.",
    ),
    ParityCandidate(
        name="2028 Democratic presidential nominee: <candidate>",
        pm_match="2028 democratic presidential nominee",
        kalshi_ticker="KXDEMNOM28-REPLACE",
        note="TEMPLATE: this is one CANDIDATE outcome; pm_match must be specific enough to hit "
             "the single Polymarket market, not the event.",
    ),
    ParityCandidate(
        name="Largest company end of 2026: <company>",
        pm_match="largest company end of 2026",
        kalshi_ticker="KXBIGCO-26-REPLACE",
        note="TEMPLATE: verify bucket/date alignment before flagging settlement_verified.",
    ),
]


def build_links(
    candidates: list[ParityCandidate],
    pm_quotes: list[VenueQuote],
    kalshi_quotes: dict[str, VenueQuote],
) -> tuple[list[ParityLink], list[str]]:
    """Pair each candidate with a live Polymarket quote (substring match on label) and a Kalshi
    quote (by ticker). Returns (links, unmatched_names). A candidate is skipped (and reported)
    when either side is missing."""
    links: list[ParityLink] = []
    unmatched: list[str] = []
    for c in candidates:
        needle = c.pm_match.lower()
        pm = next((q for q in pm_quotes if needle in q.label.lower()), None)
        kal = kalshi_quotes.get(c.kalshi_ticker)
        if pm is None or kal is None:
            unmatched.append(c.name)
            continue
        links.append(ParityLink(
            name=c.name, a=pm, b=kal,
            settlement_verified=c.settlement_verified, note=c.note,
        ))
    return links, unmatched
