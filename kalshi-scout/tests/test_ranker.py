from datetime import date

from kalshi_scout.models import (
    Bracket,
    BracketKind,
    ContractState,
    KalshiMarket,
    Metric,
    ParsedContract,
)
from kalshi_scout.ranker import grade


def _market(yes_ask=None, no_ask=None, yes_bid=None, no_bid=None, volume=100, oi=500) -> KalshiMarket:
    return KalshiMarket(
        ticker="KXHIGHHOUSTON-26MAY27-B79-80",
        event_ticker="KXHIGHHOUSTON-26MAY27",
        title="",
        yes_sub_title="79° to 80°",
        status="open",
        close_time=None,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        last_price=None,
        volume=volume,
        open_interest=oi,
    )


def _contract(metric=Metric.HIGH, bracket=None) -> ParsedContract:
    return ParsedContract(
        market_ticker="X",
        event_ticker="Y",
        city_slug="HOUSTON",
        metric=metric,
        market_date=date(2026, 5, 27),
        bracket=bracket or Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0),
    )


def test_locked_yes_with_stale_price_grades_A_plus():
    """Spec A+ scenario: settlement state proves Yes, market priced at 71c."""
    c = _contract(bracket=Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0))
    m = _market(yes_ask=71, yes_bid=70, no_ask=29, no_bid=28)
    e = grade(c, m, ContractState.LOCKED_YES, "max already hit 79", fair_lo=0.97, fair_hi=0.99)
    assert e.grade == "A+"
    assert e.edge_yes is not None and e.edge_yes > 0.25


def test_dead_no_grades_A_when_no_side_cheap():
    """Spec: high already at 79 → '78 or below' is dead → buy No."""
    c = _contract(bracket=Bracket(BracketKind.LTE, lo=None, hi=78.0))
    m = _market(yes_ask=15, yes_bid=14, no_ask=85, no_bid=84)
    e = grade(c, m, ContractState.DEAD_NO, "max 79 > 78", fair_lo=0.0, fair_hi=0.02)
    # No-side edge: 1 - 0.01 - 0.85 = 0.14 → A+/A
    assert e.edge_no is not None and e.edge_no > 0.10
    assert e.grade in ("A+", "A")


def test_wide_spread_demotes_grade():
    c = _contract()
    m = _market(yes_ask=80, yes_bid=60, no_ask=40, no_bid=20)  # 20c spread
    e = grade(c, m, ContractState.LOCKED_YES, "", fair_lo=0.97, fair_hi=0.99)
    assert "wide spread" in " ".join(e.notes)
    # A+ would require tight spread; should be downgraded
    assert e.grade != "A+"


def test_derives_yes_ask_from_no_bid_when_yes_ask_missing():
    c = _contract()
    m = _market(yes_ask=None, yes_bid=70, no_ask=None, no_bid=28)
    e = grade(c, m, ContractState.LOCKED_YES, "", fair_lo=0.97, fair_hi=0.99)
    # yes_ask should be derived as 100 - 28 = 72
    assert e.yes_ask_cents == 72


def test_dead_no_with_no_market_price_is_F():
    c = _contract()
    m = _market(yes_ask=None, yes_bid=None, no_ask=None, no_bid=None)
    e = grade(c, m, ContractState.DEAD_NO, "", fair_lo=0.0, fair_hi=0.02)
    assert e.grade == "F"
