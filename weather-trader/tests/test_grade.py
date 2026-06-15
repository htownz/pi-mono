from datetime import date

from weather_trader.forecast import ForecastDistribution, Scenario
from weather_trader.grade import _grade, evaluate, sort_key
from weather_trader.models import Bracket, BracketKind, Contract, KalshiMarket, Metric, Station

STATION = Station("KTST", "Test", "TEST", "UTC", 0.0, 0.0)


def test_grade_locked_ladder():
    assert _grade(locked=True, best_edge=0.10, band_width_f=0.0, spread_cents=2, volume=10) == "A+"
    assert _grade(locked=True, best_edge=0.10, band_width_f=0.0, spread_cents=12, volume=10) == "A"   # wide
    assert _grade(locked=True, best_edge=0.10, band_width_f=0.0, spread_cents=2, volume=0) == "A"      # illiquid
    assert _grade(locked=True, best_edge=0.05, band_width_f=0.0, spread_cents=2, volume=10) == "A"
    assert _grade(locked=True, best_edge=0.02, band_width_f=0.0, spread_cents=2, volume=10) == "B"
    assert _grade(locked=True, best_edge=0.0, band_width_f=0.0, spread_cents=2, volume=10) == "D"


def test_grade_forecast_ladder():
    assert _grade(locked=False, best_edge=0.20, band_width_f=3.0, spread_cents=2, volume=10) == "B+"
    assert _grade(locked=False, best_edge=0.13, band_width_f=6.0, spread_cents=2, volume=10) == "B"
    assert _grade(locked=False, best_edge=0.13, band_width_f=10.0, spread_cents=2, volume=10) == "C"  # broad
    assert _grade(locked=False, best_edge=0.10, band_width_f=6.0, spread_cents=2, volume=10) == "C"
    assert _grade(locked=False, best_edge=0.10, band_width_f=3.0, spread_cents=2, volume=10) == "B"   # narrow upgrade
    assert _grade(locked=False, best_edge=0.03, band_width_f=6.0, spread_cents=2, volume=10) == "D"
    assert _grade(locked=False, best_edge=0.20, band_width_f=3.0, spread_cents=12, volume=10) == "B"  # wide downgrade
    assert _grade(locked=False, best_edge=-0.1, band_width_f=3.0, spread_cents=2, volume=10) == "D"
    assert _grade(locked=False, best_edge=None, band_width_f=3.0, spread_cents=2, volume=10) == "F"


def _contract(bracket):
    return Contract("KXHIGHTEST-26JUN16-X", "KXHIGHTEST-26JUN16", "TEST",
                    Metric.HIGH, date(2026, 6, 16), bracket)


def _market(yes_ask=None, yes_bid=None, no_ask=None, volume=10):
    return KalshiMarket(
        ticker="KXHIGHTEST-26JUN16-X", event_ticker="KXHIGHTEST-26JUN16", title="",
        yes_sub_title="80° or above", status="open", close_time=None,
        yes_bid=yes_bid, yes_ask=yes_ask, no_bid=None, no_ask=no_ask,
        last_price=None, volume=volume, open_interest=0,
    )


def _locked_dist(value):
    return ForecastDistribution(
        Metric.HIGH, date(2026, 6, 16), STATION,
        [Scenario(value, "observed", 1.0)], value, True, 0.0, 0, [],
    )


def test_evaluate_locked_yes_computes_edge_and_grades_a_plus():
    bracket = Bracket(BracketKind.GTE, lo=80, hi=None)
    dist = _locked_dist(85.0)                      # observed 85 -> GTE80 settles YES
    e = evaluate(_contract(bracket), _market(yes_ask=90, yes_bid=88, no_ask=12), dist)
    assert e.fair_prob_mid == 1.0
    assert e.edge_yes is not None and abs(e.edge_yes - 0.10) < 1e-9
    assert e.best_side == "yes"
    assert e.grade == "A+"


def test_evaluate_unusable_dist_is_F():
    bracket = Bracket(BracketKind.GTE, lo=80, hi=None)
    empty = ForecastDistribution(Metric.HIGH, date(2026, 6, 16), STATION, [], None, False, 0.0, 0, [])
    e = evaluate(_contract(bracket), _market(yes_ask=50), empty)
    assert e.grade == "F"


def test_evaluate_no_price_is_F():
    bracket = Bracket(BracketKind.GTE, lo=80, hi=None)
    e = evaluate(_contract(bracket), _market(yes_ask=None, no_ask=None), _locked_dist(85.0))
    assert e.best_edge is None
    assert e.grade == "F"


def test_sort_key_orders_best_first():
    bracket = Bracket(BracketKind.GTE, lo=80, hi=None)
    good = evaluate(_contract(bracket), _market(yes_ask=90, yes_bid=88), _locked_dist(85.0))
    bad = evaluate(_contract(bracket), _market(yes_ask=None, no_ask=None), _locked_dist(85.0))
    ordered = sorted([bad, good], key=sort_key)
    assert ordered[0] is good and ordered[1] is bad
