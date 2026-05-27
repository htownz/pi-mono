from datetime import date

from kalshi_scout.models import BracketKind, KalshiMarket, Metric
from kalshi_scout.parser import parse_market


def _market(ticker: str, yes_sub_title: str = "", title: str = "") -> KalshiMarket:
    return KalshiMarket(
        ticker=ticker,
        event_ticker=ticker.rsplit("-", 1)[0] if "-" in ticker else "",
        title=title,
        yes_sub_title=yes_sub_title,
        status="open",
        close_time=None,
        yes_bid=None,
        yes_ask=None,
        no_bid=None,
        no_ask=None,
        last_price=None,
        volume=0,
        open_interest=0,
    )


def test_between_bracket_houston_high():
    m = _market("KXHIGHHOUSTON-26MAY27-B79-80", yes_sub_title="79° to 80°")
    p = parse_market(m)
    assert p is not None
    assert p.metric is Metric.HIGH
    assert p.city_slug == "HOUSTON"
    assert p.market_date == date(2026, 5, 27)
    assert p.bracket.kind is BracketKind.BETWEEN
    assert p.bracket.lo == 79.0
    assert p.bracket.hi == 80.0


def test_below_threshold_via_title():
    m = _market("KXHIGHHOUSTON-26MAY27-T78", yes_sub_title="78° or below")
    p = parse_market(m)
    assert p is not None
    assert p.bracket.kind is BracketKind.BELOW
    assert p.bracket.hi == 78.0
    assert p.bracket.lo is None


def test_above_threshold_via_title():
    m = _market("KXHIGHHOUSTON-26MAY27-T85", yes_sub_title="85° or above")
    p = parse_market(m)
    assert p is not None
    assert p.bracket.kind is BracketKind.ABOVE
    assert p.bracket.lo == 85.0
    assert p.bracket.hi is None


def test_low_market_tomorrow():
    m = _market("KXLOWHOUSTON-26MAY28-B70-71", yes_sub_title="70° to 71°")
    p = parse_market(m)
    assert p is not None
    assert p.metric is Metric.LOW
    assert p.market_date == date(2026, 5, 28)
    assert p.bracket.kind is BracketKind.BETWEEN
    assert p.bracket.lo == 70.0
    assert p.bracket.hi == 71.0


def test_ambiguous_threshold_without_title_returns_none():
    m = _market("KXHIGHHOUSTON-26MAY27-T78", yes_sub_title="")
    p = parse_market(m)
    # Without a title we cannot determine above-vs-below; safer to skip.
    assert p is None


def test_garbage_ticker_returns_none():
    assert parse_market(_market("GARBAGE")) is None
    assert parse_market(_market("KXSPORTSHOUSTON-26MAY27-B1-2")) is None


def test_em_dash_in_title():
    m = _market("KXHIGHHOUSTON-26MAY27-B79-80", yes_sub_title="79–80°")
    p = parse_market(m)
    assert p is not None
    assert p.bracket.kind is BracketKind.BETWEEN


def test_date_parsing_year():
    m = _market("KXLOWNYC-26DEC31-T20", yes_sub_title="20° or below")
    p = parse_market(m)
    assert p is not None
    assert p.market_date == date(2026, 12, 31)
    assert p.city_slug == "NYC"
