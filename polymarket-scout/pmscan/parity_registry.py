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

The seed entries below are REAL candidates (NBA Finals + tonight's MLB game), but with
settlement_verified=False: Kalshi tickers are verified-real where noted, a couple are inferred,
and the Polymarket pm_match strings must be confirmed against live questions. The runner reports
any that don't resolve. Confirm identifiers and YES-side alignment before flipping verified. See
parity_run.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .models import Market, OrderBook
from .parity import ParityLink, pm_venue_quote


@dataclass(frozen=True)
class ParityCandidate:
    name: str                  # human description of the shared outcome
    pm_match: str              # case-insensitive substring to find the PM market (question OR slug)
    kalshi_ticker: str         # Kalshi market ticker for the same YES outcome
    settlement_verified: bool = False
    note: str = ""
    pm_outcome: Optional[str] = None   # for categorical PM markets (['Royals','Twins']): the
                                       # outcome label that is the shared YES. None for Yes/No.


# Seed templates — recurring markets that exist on both venues. Identifiers are PLACEHOLDERS
# to be replaced with live, verified values; settlement_verified stays False until checked.
REGISTRY: list[ParityCandidate] = [
    # --- Real candidates seeded 2026-06-04. Kalshi tickers are verified-real where noted;
    # Polymarket pm_match strings and a couple of Kalshi tickers are best-effort and must be
    # confirmed against the live markets (the runner reports any that don't resolve). All stay
    # settlement_verified=False until you've aligned the YES side on both venues. ---
    ParityCandidate(
        name="MLB 2026-06-04: Royals @ Twins (Twins win)",
        pm_match="twins",   # PM MLB game market for KC@MIN tonight; refine to the exact question
        kalshi_ticker="KXMLBGAME-26JUN041940KCMIN",   # VERIFIED real Kalshi market (Jun 4, 19:40, KC@MIN)
        note="VERIFY which team is the Kalshi YES side. If the PM market is categorical "
             "(outcomes like ['Royals','Twins']), set pm_outcome='Twins' to the SAME team as the "
             "Kalshi YES. Same game's winner => settlement equivalent; flip verified once aligned.",
    ),
    ParityCandidate(
        name="2026 NBA Champion: San Antonio Spurs",
        pm_match="spurs win the 2026 nba",   # in PM event polymarket.com/event/2026-nba-champion
        kalshi_ticker="KXNBA-26-SAS",         # best guess: NBA champion series KXNBA / event -26 / team SAS
        note="VERIFY: confirm the Kalshi NBA-champion ticker (series may be KXNBA / KXNBACHAMP) "
             "and the exact PM sub-market wording. PM Spurs ~0.64, Knicks ~0.36 as of Jun 4.",
    ),
    ParityCandidate(
        name="NBA Finals Game 2 (2026-06-05): NYK @ SAS (Spurs win)",
        pm_match="knicks vs. spurs",          # PM single-game market for Game 2; refine to exact text
        kalshi_ticker="KXNBAGAME-26JUN05NYKSAS",  # inferred from verified Game 1 (KXNBAGAME-26JUN03NYKSAS)
        note="Game 2 ticker inferred from the confirmed Game 1 pattern + the Jun 5 @ San Antonio "
             "schedule. VERIFY the exact ticker and which team is Kalshi YES; align pm_match.",
    ),
]


def build_links(
    candidates: list[ParityCandidate],
    pm_markets: list[Market],
    books: dict[str, OrderBook],
    kalshi_quotes: dict[str, VenueQuote],
) -> tuple[list[ParityLink], list[str]]:
    """Pair each candidate with a live Polymarket market and a Kalshi quote.

    Matching is by `pm_match` substring against the PM **question OR slug**. A candidate is
    skipped — and reported with a reason, never silently dropped — when:
      - the Kalshi quote is missing,
      - the substring matches no PM market, or MORE THAN ONE (ambiguous → tighten pm_match), or
      - the PM quote can't be built (e.g. a categorical market with no `pm_outcome` selector).
    Returns (links, unmatched_with_reasons).
    """
    links: list[ParityLink] = []
    unmatched: list[str] = []
    for c in candidates:
        needle = c.pm_match.lower()
        matches = [m for m in pm_markets
                   if needle in m.question.lower() or needle in (m.slug or "").lower()]
        kal = kalshi_quotes.get(c.kalshi_ticker)
        if len(matches) != 1 or kal is None:
            reason = (f"ambiguous: matches {len(matches)} PM markets" if len(matches) > 1
                      else ("no PM match" if not matches else "no Kalshi quote"))
            unmatched.append(f"{c.name} [{reason}]")
            continue
        pm = pm_venue_quote(matches[0], books, label=matches[0].question, yes_outcome=c.pm_outcome)
        if pm is None:
            unmatched.append(f"{c.name} [PM quote unbuildable — categorical market? set pm_outcome]")
            continue
        links.append(ParityLink(
            name=c.name, a=pm, b=kal,
            settlement_verified=c.settlement_verified, note=c.note,
        ))
    return links, unmatched
