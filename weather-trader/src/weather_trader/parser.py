"""Parse a Kalshi temperature market into a structured contract.

Two sources, used together:

  1. The event_ticker's series prefix (e.g. "KXHIGHHOU" from
     "KXHIGHHOU-26MAY28"), looked up in `kalshi.TEMPERATURE_SERIES` to get the
     (metric, city_slug) pair. Kalshi's naming is too inconsistent to regex.

  2. `yes_sub_title` (e.g. "79° to 80°" / "78° or below" / "85° or above") for
     the bracket. Six operator forms map to BracketKind.

The market_date comes from the date token in the event_ticker ("26MAY28").

If the series isn't known, the bracket can't be parsed, or the date can't be
parsed, we return None — a silent skip beats a wrong settlement guess.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Optional

from weather_trader.kalshi import TEMPERATURE_SERIES, derive_series_from_event_ticker
from weather_trader.models import Bracket, BracketKind, Contract, KalshiMarket, Metric

_DATE_TOKEN_RE = re.compile(r"^(\d{2})([A-Z]{3})(\d{1,2})$")
_BETWEEN_SUFFIX = re.compile(r"^B(?P<lo>\d+(?:\.\d+)?)-(?P<hi>\d+(?:\.\d+)?)$")
_THRESHOLD_SUFFIX = re.compile(r"^T(?P<t>\d+(?:\.\d+)?)$")

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

_DEGREE = "[°˚]?"
_BETWEEN_TITLE = re.compile(
    rf"(?P<lo>-?\d+(?:\.\d+)?){_DEGREE}\s*(?:to|[–-])\s*(?P<hi>-?\d+(?:\.\d+)?){_DEGREE}",
    re.IGNORECASE,
)
# "X° or below" / "at most X°" -> LTE (inclusive)
_LTE_TITLE = re.compile(
    rf"(?:(?P<t1>-?\d+(?:\.\d+)?){_DEGREE}\s*or\s*(?:below|under|less)|"
    rf"at\s*most\s*(?P<t2>-?\d+(?:\.\d+)?){_DEGREE})",
    re.IGNORECASE,
)
# "X° or above" / "at least X°" -> GTE (inclusive)
_GTE_TITLE = re.compile(
    rf"(?:(?P<t1>-?\d+(?:\.\d+)?){_DEGREE}\s*or\s*(?:above|over|higher|more)|"
    rf"at\s*least\s*(?P<t2>-?\d+(?:\.\d+)?){_DEGREE})",
    re.IGNORECASE,
)
_GT_TITLE = re.compile(rf"^\s*above\s+(?P<t>-?\d+(?:\.\d+)?){_DEGREE}\s*$", re.IGNORECASE)
_LT_TITLE = re.compile(rf"^\s*below\s+(?P<t>-?\d+(?:\.\d+)?){_DEGREE}\s*$", re.IGNORECASE)
_EQ_TITLE = re.compile(rf"exactly\s+(?P<t>-?\d+(?:\.\d+)?){_DEGREE}", re.IGNORECASE)


def bracket_from_title(yes_sub_title: str) -> Optional[Bracket]:
    """Best-effort parse of Kalshi's `yes_sub_title` shapes.

    Order matters: BETWEEN before LTE/GTE so "70 to 75°" isn't caught by a
    stray "above". Strict GT/LT only match titles that start with
    "above"/"below" *without* "or" (the colloquial "or"-form is GTE/LTE).
    """
    if not yes_sub_title:
        return None
    text = yes_sub_title.strip()
    m = _BETWEEN_TITLE.search(text)
    if m:
        lo, hi = float(m.group("lo")), float(m.group("hi"))
        if lo > hi:
            lo, hi = hi, lo
        return Bracket(BracketKind.BETWEEN, lo, hi)
    m = _LTE_TITLE.search(text)
    if m:
        t = float(m.group("t1") or m.group("t2"))
        return Bracket(BracketKind.LTE, lo=None, hi=t)
    m = _GTE_TITLE.search(text)
    if m:
        t = float(m.group("t1") or m.group("t2"))
        return Bracket(BracketKind.GTE, lo=t, hi=None)
    m = _EQ_TITLE.search(text)
    if m:
        return Bracket(BracketKind.EQ, lo=float(m.group("t")), hi=None)
    m = _GT_TITLE.search(text)
    if m:
        return Bracket(BracketKind.GT, lo=float(m.group("t")), hi=None)
    m = _LT_TITLE.search(text)
    if m:
        return Bracket(BracketKind.LT, lo=None, hi=float(m.group("t")))
    return None


def _bracket_from_suffix(suffix: str, yes_sub_title: str) -> Optional[Bracket]:
    """Parse the trailing ticker token. `B<lo>-<hi>` is unambiguous; `T<n>`
    can't disambiguate direction, so we defer to the title."""
    m = _BETWEEN_SUFFIX.match(suffix)
    if m:
        lo, hi = float(m.group("lo")), float(m.group("hi"))
        if lo > hi:
            lo, hi = hi, lo
        return Bracket(BracketKind.BETWEEN, lo, hi)
    if _THRESHOLD_SUFFIX.match(suffix):
        return bracket_from_title(yes_sub_title)
    return None


def parse_market(market: KalshiMarket) -> Optional[Contract]:
    """Return a Contract, or None if the market isn't a recognized temp market."""
    series_ticker = derive_series_from_event_ticker(market.event_ticker)
    if not series_ticker:
        return None
    entry = TEMPERATURE_SERIES.get(series_ticker)
    if entry is None:
        return None
    metric_str, city_slug = entry
    metric = Metric.HIGH if metric_str == "high" else Metric.LOW

    parts = market.event_ticker.upper().split("-", 1)
    if len(parts) < 2:
        return None
    market_date = _parse_date_token(parts[1])
    if market_date is None:
        return None

    suffix: Optional[str] = None
    if market.ticker.upper().startswith(market.event_ticker.upper() + "-"):
        suffix = market.ticker[len(market.event_ticker) + 1:]
    bracket = (
        _bracket_from_suffix(suffix, market.yes_sub_title) if suffix else None
    ) or bracket_from_title(market.yes_sub_title)
    if bracket is None:
        return None

    return Contract(
        market_ticker=market.ticker,
        event_ticker=market.event_ticker,
        city_slug=city_slug,
        metric=metric,
        market_date=market_date,
        bracket=bracket,
    )


def parse_event_date(event_ticker: str) -> Optional[date]:
    """Extract the market date from an event ticker, e.g. KXHIGHNYC-26JUN16 -> 2026-06-16.

    Lets callers filter events by date cheaply, before any network fetch.
    """
    if not event_ticker or "-" not in event_ticker:
        return None
    return _parse_date_token(event_ticker.split("-", 1)[1].upper())


def _parse_date_token(tok: str) -> Optional[date]:
    """`26MAY28` -> date(2026, 5, 28)."""
    m = _DATE_TOKEN_RE.match(tok)
    if not m:
        return None
    yy, mon, dd = m.groups()
    if mon not in _MONTHS:
        return None
    try:
        return date(2000 + int(yy), _MONTHS[mon], int(dd))
    except ValueError:
        return None
