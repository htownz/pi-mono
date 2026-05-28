"""Parse a Kalshi temperature market into a structured settlement question.

Two information sources, used together:

  1. Ticker shape, e.g. "KXHIGHHOUSTON-26MAY27-B79-80" or "KXLOWNYC-26MAY28-T70"
     - Series prefix tells us metric (HIGH vs LOW) and city slug.
     - Date token gives us the local-time market date (the city's local timezone).
     - Final suffix gives us the bracket: T<n> = threshold, B<lo>-<hi> = between.

  2. `yes_sub_title`, e.g. "79° to 80°" / "78° or below" / "85° or above".
     - Used to disambiguate T<n> tickers (above vs below) since the suffix
       alone doesn't reveal direction.

We deliberately do not try to parse arbitrary event titles. If neither source
fits, the parser returns None and the universe scanner skips the market —
better silent skip than wrong settlement guess.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

from kalshi_scout.models import (
    Bracket,
    BracketKind,
    KalshiMarket,
    Metric,
    ParsedContract,
)

_TICKER_RE = re.compile(
    r"""^
    KX(?P<metric>HIGH|LOW|TEMP)         # metric prefix
    (?P<city>[A-Z]+)                    # city slug
    -
    (?P<datetok>\d{2}[A-Z]{3}\d{1,2})   # e.g. 26MAY27
    -
    (?P<suffix>.+)$
    """,
    re.VERBOSE,
)

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
# "X° or below" / "at most X°" -> LTE (inclusive of X)
_LTE_TITLE = re.compile(
    rf"(?:(?P<t1>-?\d+(?:\.\d+)?){_DEGREE}\s*or\s*(?:below|under|less)|"
    rf"at\s*most\s*(?P<t2>-?\d+(?:\.\d+)?){_DEGREE})",
    re.IGNORECASE,
)
# "X° or above" / "at least X°" -> GTE (inclusive of X)
_GTE_TITLE = re.compile(
    rf"(?:(?P<t1>-?\d+(?:\.\d+)?){_DEGREE}\s*or\s*(?:above|over|higher|more)|"
    rf"at\s*least\s*(?P<t2>-?\d+(?:\.\d+)?){_DEGREE})",
    re.IGNORECASE,
)
# "above X°" without "or" -> GT (strict)
_GT_TITLE = re.compile(
    rf"^\s*above\s+(?P<t>-?\d+(?:\.\d+)?){_DEGREE}\s*$",
    re.IGNORECASE,
)
# "below X°" without "or" -> LT (strict)
_LT_TITLE = re.compile(
    rf"^\s*below\s+(?P<t>-?\d+(?:\.\d+)?){_DEGREE}\s*$",
    re.IGNORECASE,
)
_EQ_TITLE = re.compile(
    rf"exactly\s+(?P<t>-?\d+(?:\.\d+)?){_DEGREE}",
    re.IGNORECASE,
)


def _parse_date_token(tok: str) -> Optional[date]:
    # "26MAY27" -> 2026-05-27
    if len(tok) < 6:
        return None
    try:
        yy = int(tok[:2])
        mon = tok[2:5]
        day = int(tok[5:])
    except ValueError:
        return None
    if mon not in _MONTHS:
        return None
    year = 2000 + yy
    try:
        return date(year, _MONTHS[mon], day)
    except ValueError:
        return None


def _bracket_from_title(yes_sub_title: str) -> Optional[Bracket]:
    """Best-effort parse of Kalshi's `yes_sub_title` shapes.

    Order matters: BETWEEN before LTE/GTE so "70 to 75°" doesn't get caught by
    a stray "above" elsewhere. Strict GT/LT only match titles starting with
    "above"/"below" *without* "or" (the colloquial-with-or form is GTE/LTE).
    """
    if not yes_sub_title:
        return None
    text = yes_sub_title.strip()
    m = _BETWEEN_TITLE.search(text)
    if m:
        lo = float(m.group("lo"))
        hi = float(m.group("hi"))
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
        t = float(m.group("t"))
        return Bracket(BracketKind.EQ, lo=t, hi=None)
    m = _GT_TITLE.search(text)
    if m:
        t = float(m.group("t"))
        return Bracket(BracketKind.GT, lo=t, hi=None)
    m = _LT_TITLE.search(text)
    if m:
        t = float(m.group("t"))
        return Bracket(BracketKind.LT, lo=None, hi=t)
    return None


def _bracket_from_suffix(suffix: str, yes_sub_title: str) -> Optional[Bracket]:
    """Parse the trailing token of a Kalshi temperature market ticker.

    `B<lo>-<hi>` is unambiguous. `T<n>` could be either above or below the
    strike — we look at the yes_sub_title to disambiguate; if that fails we
    return None rather than guess.
    """
    m = _BETWEEN_SUFFIX.match(suffix)
    if m:
        lo = float(m.group("lo"))
        hi = float(m.group("hi"))
        if lo > hi:
            lo, hi = hi, lo
        return Bracket(BracketKind.BETWEEN, lo, hi)
    m = _THRESHOLD_SUFFIX.match(suffix)
    if m:
        # The T<n> ticker suffix alone cannot disambiguate operator direction
        # (above vs below, strict vs inclusive). Always defer to the title.
        return _bracket_from_title(yes_sub_title)
    return None


def parse_market(market: KalshiMarket) -> Optional[ParsedContract]:
    """Return a ParsedContract or None if the ticker isn't a recognized temp market."""
    ticker = market.ticker.upper()
    m = _TICKER_RE.match(ticker)
    if not m:
        return None

    metric_tok = m.group("metric")
    if metric_tok == "HIGH":
        metric = Metric.HIGH
    elif metric_tok == "LOW":
        metric = Metric.LOW
    else:
        # KXTEMP* — infer from title; this is rare. Default to HIGH only if
        # the title clearly says "high"; bail out otherwise.
        title_low = (market.title or "").lower()
        if "low" in title_low and "high" not in title_low:
            metric = Metric.LOW
        elif "high" in title_low and "low" not in title_low:
            metric = Metric.HIGH
        else:
            return None

    market_date = _parse_date_token(m.group("datetok"))
    if market_date is None:
        return None

    bracket = _bracket_from_suffix(m.group("suffix"), market.yes_sub_title) \
        or _bracket_from_title(market.yes_sub_title)
    if bracket is None:
        return None

    return ParsedContract(
        market_ticker=market.ticker,
        event_ticker=market.event_ticker,
        city_slug=m.group("city"),
        metric=metric,
        market_date=market_date,
        bracket=bracket,
    )
