from datetime import date

from weather_trader.models import BracketKind, KalshiMarket, Metric
from weather_trader.parser import bracket_from_title, parse_event_date, parse_market


def _mkt(ticker: str, event: str, sub: str) -> KalshiMarket:
    return KalshiMarket(
        ticker=ticker, event_ticker=event, title="", yes_sub_title=sub, status="open",
        close_time=None, yes_bid=None, yes_ask=None, no_bid=None, no_ask=None,
        last_price=None, volume=0, open_interest=0,
    )


def test_bracket_between():
    b = bracket_from_title("79° to 80°")
    assert b is not None and b.kind is BracketKind.BETWEEN and b.lo == 79 and b.hi == 80


def test_bracket_lte_gte_strict_and_eq():
    assert bracket_from_title("78° or below").kind is BracketKind.LTE
    assert bracket_from_title("85° or above").kind is BracketKind.GTE
    assert bracket_from_title("at most 70°").kind is BracketKind.LTE
    assert bracket_from_title("at least 90°").kind is BracketKind.GTE
    assert bracket_from_title("above 80°").kind is BracketKind.GT
    assert bracket_from_title("below 75°").kind is BracketKind.LT
    assert bracket_from_title("exactly 80°").kind is BracketKind.EQ


def test_bracket_unparseable_returns_none():
    assert bracket_from_title("") is None
    assert bracket_from_title("warm and sunny") is None


def test_parse_market_full():
    m = _mkt("KXHIGHNYC-26JUN16-B79-80", "KXHIGHNYC-26JUN16", "79° to 80°")
    c = parse_market(m)
    assert c is not None
    assert c.city_slug == "NYC"
    assert c.metric is Metric.HIGH
    assert c.market_date == date(2026, 6, 16)
    assert c.bracket.kind is BracketKind.BETWEEN


def test_parse_market_low_and_threshold_suffix_uses_title():
    m = _mkt("KXLOWNYC-26JUN16-T68", "KXLOWNYC-26JUN16", "68° or below")
    c = parse_market(m)
    assert c is not None and c.metric is Metric.LOW
    assert c.bracket.kind is BracketKind.LTE and c.bracket.hi == 68


def test_parse_event_date():
    assert parse_event_date("KXHIGHNYC-26JUN16") == date(2026, 6, 16)
    assert parse_event_date("KXLOWTHOU-26JUN15") == date(2026, 6, 15)
    assert parse_event_date("nodash") is None
    assert parse_event_date("KXHIGHNYC-FOO") is None


def test_parse_market_unknown_series_is_none():
    assert parse_market(_mkt("KXNOPE-26JUN16-B1-2", "KXNOPE-26JUN16", "1° to 2°")) is None


def test_parse_market_bad_date_is_none():
    assert parse_market(_mkt("KXHIGHNYC-FOO-B79-80", "KXHIGHNYC-FOO", "79° to 80°")) is None
