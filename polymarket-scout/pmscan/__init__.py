"""pmscan — read-only Polymarket sum-to-one scanner (Phase 0/1/1b)."""
from .models import (
    BookLevel,
    Market,
    NegRiskEvent,
    NegRiskOpportunity,
    NegRiskSnapshot,
    OrderBook,
    Opportunity,
)
from .client import ClobClient, GammaClient, parse_event, parse_market
from .scanner import group_negrisk, negrisk_snapshot, scan_market, scan_negrisk
from .temporal import Dip, detect_dips, load_snapshots
from .parity import (
    ParityLink,
    ParityOpportunity,
    VenueQuote,
    kalshi_venue_quote,
    pm_venue_quote,
    scan_parity,
    scan_parity_links,
)

__version__ = "0.8.0"

__all__ = [
    "BookLevel",
    "Market",
    "OrderBook",
    "Opportunity",
    "NegRiskEvent",
    "NegRiskOpportunity",
    "NegRiskSnapshot",
    "GammaClient",
    "ClobClient",
    "parse_market",
    "parse_event",
    "scan_market",
    "group_negrisk",
    "scan_negrisk",
    "negrisk_snapshot",
    "detect_dips",
    "load_snapshots",
    "Dip",
    "VenueQuote",
    "ParityLink",
    "ParityOpportunity",
    "pm_venue_quote",
    "kalshi_venue_quote",
    "scan_parity",
    "scan_parity_links",
]
