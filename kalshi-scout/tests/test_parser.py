from datetime import date

from kalshi_scout.models import BracketKind, KalshiMarket, Metric
from kalshi_scout.parser import parse_market


def _market(ticker: str, yes_sub_title: str = "", title: str = "") -> KalshiMarket:
    # Real Kalshi event_tickers are <series>-<date> — first two segments.
    parts = ticker.split("-", 2)
    event_ticker = "-".join(parts[:2]) if len(parts) >= 2 else ""
    return KalshiMarket(
        ticker=ticker,
        event_ticker=event_ticker,
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
    assert p.bracket.kind is BracketKind.LTE
    assert p.bracket.hi == 78.0
    assert p.bracket.lo is None


def test_above_threshold_via_title():
    m = _market("KXHIGHHOUSTON-26MAY27-T85", yes_sub_title="85° or above")
    p = parse_market(m)
    assert p is not None
    assert p.bracket.kind is BracketKind.GTE
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


def test_at_least_phrasing_maps_to_gte():
    m = _market("KXHIGHHOUSTON-26MAY27-T80", yes_sub_title="at least 80°")
    p = parse_market(m)
    assert p is not None
    assert p.bracket.kind is BracketKind.GTE
    assert p.bracket.lo == 80.0


def test_at_most_phrasing_maps_to_lte():
    m = _market("KXLOWHOUSTON-26MAY28-T75", yes_sub_title="at most 75°")
    p = parse_market(m)
    assert p is not None
    assert p.bracket.kind is BracketKind.LTE
    assert p.bracket.hi == 75.0


def test_strict_above_phrasing_maps_to_gt():
    m = _market("KXHIGHHOUSTON-26MAY27-T80", yes_sub_title="above 80°")
    p = parse_market(m)
    assert p is not None
    assert p.bracket.kind is BracketKind.GT
    assert p.bracket.lo == 80.0


def test_strict_below_phrasing_maps_to_lt():
    m = _market("KXLOWHOUSTON-26MAY28-T75", yes_sub_title="below 75°")
    p = parse_market(m)
    assert p is not None
    assert p.bracket.kind is BracketKind.LT
    assert p.bracket.hi == 75.0


def test_exactly_phrasing_maps_to_eq():
    m = _market("KXHIGHHOUSTON-26MAY27-T80", yes_sub_title="exactly 80°")
    p = parse_market(m)
    assert p is not None
    assert p.bracket.kind is BracketKind.EQ
    assert p.bracket.lo == 80.0


# -- Real Kalshi series naming styles (verified against /series 2026-05) ---

def test_short_city_code_houston_high():
    """KXHIGHHOU — Kalshi's actual Houston high series."""
    m = _market("KXHIGHHOU-26MAY28-B79-80", yes_sub_title="79° to 80°")
    p = parse_market(m)
    assert p is not None
    assert p.metric is Metric.HIGH
    assert p.city_slug == "HOUSTON"
    assert p.market_date == date(2026, 5, 28)


def test_no_kx_prefix_nyc_high():
    """HIGHNY — older Kalshi naming with no KX prefix."""
    m = _market("HIGHNY-26MAY27-T85", yes_sub_title="85° or above")
    p = parse_market(m)
    assert p is not None
    assert p.metric is Metric.HIGH
    assert p.city_slug == "NYC"


def test_low_t_prefix_houston():
    """KXLOWTHOU — 'Daily Low Temperature Houston' style with extra T."""
    m = _market("KXLOWTHOU-26MAY28-T70", yes_sub_title="70° or below")
    p = parse_market(m)
    assert p is not None
    assert p.metric is Metric.LOW
    assert p.city_slug == "HOUSTON"
    assert p.bracket.kind is BracketKind.LTE


def test_city_first_naming_houston():
    """KXHOUHIGH — city-first variant (rare but Kalshi has them)."""
    m = _market("KXHOUHIGH-26MAY27-B79-80", yes_sub_title="79° to 80°")
    p = parse_market(m)
    assert p is not None
    assert p.city_slug == "HOUSTON"
    assert p.metric is Metric.HIGH


def test_t_variant_boston_high():
    """KXHIGHTBOS — Boston with the extra T."""
    m = _market("KXHIGHTBOS-26MAY27-T75", yes_sub_title="75° or above")
    p = parse_market(m)
    assert p is not None
    assert p.city_slug == "BOSTON"


def test_unknown_series_returns_none():
    """A series not in TEMPERATURE_SERIES is silently skipped (invariant I5)."""
    m = _market("KXUNKNOWN-26MAY27-T80", yes_sub_title="80° or above")
    assert parse_market(m) is None
