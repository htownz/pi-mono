"""Tests for V1.1 cross-bracket arbitrage detection.

The math (per AGENTS.md invariant I7 — events are mutually exclusive):

  Yes-basket: pay Σ yes_asks, receive 100c (one bracket wins).
              Profit = 100 - Σ yes_asks - N × fee_per_leg.

  No-basket:  collect Σ yes_bids upfront (= sell yes on every bracket).
              Pay 100c to one winner. Profit = Σ yes_bids - 100 - N × fee.

These tests verify the math, threshold filtering, missing-price handling,
and the >2-bracket eligibility check.
"""

from kalshi_scout.arbitrage import (
    compute_event_arbitrage,
    rank_arbitrage_opportunities,
)
from kalshi_scout.models import KalshiEvent, KalshiMarket


def _market(ticker: str, yes_ask: int | None = None, yes_bid: int | None = None,
            event_ticker: str = "E") -> KalshiMarket:
    return KalshiMarket(
        ticker=ticker, event_ticker=event_ticker,
        title="", yes_sub_title="", status="open", close_time=None,
        yes_bid=yes_bid, yes_ask=yes_ask,
        no_bid=(100 - yes_ask) if yes_ask is not None else None,
        no_ask=(100 - yes_bid) if yes_bid is not None else None,
        last_price=None, volume=0, open_interest=0,
    )


def _event(yes_asks: list[int], yes_bids: list[int] | None = None,
           event_ticker: str = "KXHIGHHOU-26MAY29",
           series_ticker: str = "KXHIGHHOU") -> KalshiEvent:
    """Default event uses a known-MEX series (weather) so most tests bypass
    the mutual-exclusivity gate. Tests that exercise the gate use a
    non-whitelisted series_ticker explicitly."""
    yes_bids = yes_bids if yes_bids is not None else [a - 1 for a in yes_asks]
    markets = [
        _market(f"{event_ticker}-B{i}", yes_ask=a, yes_bid=b, event_ticker=event_ticker)
        for i, (a, b) in enumerate(zip(yes_asks, yes_bids))
    ]
    return KalshiEvent(
        event_ticker=event_ticker, series_ticker=series_ticker,
        title="", sub_title="", markets=markets,
    )


# -- Yes-basket arb -----------------------------------------------------------

def test_yes_basket_arb_when_sum_below_100():
    """5 brackets at 15c each = Σ 75c. Yes-basket gross edge = 25c, after
    5 × 2c fees = 15c net. The actionable side."""
    arb = compute_event_arbitrage(_event([15] * 5))
    assert arb is not None
    assert arb.sum_yes_asks_cents == 75
    assert arb.yes_basket_gross_edge_cents == 25
    assert arb.yes_basket_net_edge_cents == 15
    assert arb.best_side == "yes"
    assert arb.best_net_edge_cents == 15
    assert arb.is_actionable


def test_no_arb_when_sum_equals_100():
    """5 brackets at 20c each = Σ 100c, no gross edge, net = -fees."""
    arb = compute_event_arbitrage(_event([20] * 5))
    assert arb is not None
    assert arb.yes_basket_gross_edge_cents == 0
    assert arb.yes_basket_net_edge_cents == -10  # 5 × 2c
    # No-basket side: Σ yes_bids = 95, gross = -5, net = -15.
    assert arb.no_basket_net_edge_cents == -15
    # Best side still labelled, but not actionable.
    assert arb.best_side == "yes"  # -10 > -15
    assert not arb.is_actionable


# -- No-basket arb ------------------------------------------------------------

def test_no_basket_arb_when_yes_bids_above_100():
    """4 brackets at yes_bid=30c each = Σ 120c. Sell yes on all 4: collect
    120c, pay 100c to one winner. Gross edge = 20c. After 4 × 2c = 12c net."""
    arb = compute_event_arbitrage(_event(yes_asks=[35] * 4, yes_bids=[30] * 4))
    assert arb is not None
    assert arb.sum_yes_bids_cents == 120
    assert arb.no_basket_gross_edge_cents == 20
    assert arb.no_basket_net_edge_cents == 12
    assert arb.best_side == "no"
    assert arb.best_net_edge_cents == 12


def test_picks_higher_net_side_when_both_positive():
    """Yes-basket Σ 90 → gross +10; no-basket Σ 105 → gross +5.
    With 5×2c=10c fees: yes_net=0, no_net=-5. Both negative-ish, yes wins."""
    arb = compute_event_arbitrage(_event(yes_asks=[18] * 5, yes_bids=[21] * 5))
    assert arb is not None
    assert arb.yes_basket_net_edge_cents == 0
    assert arb.no_basket_net_edge_cents == -5
    assert arb.best_side == "yes"  # 0 > -5


# -- Missing-price handling ---------------------------------------------------

def test_missing_yes_ask_on_one_leg_invalidates_yes_basket():
    e = _event([15] * 5)
    e.markets[2].yes_ask = None
    arb = compute_event_arbitrage(e)
    assert arb is not None
    assert arb.sum_yes_asks_cents is None
    assert arb.yes_basket_net_edge_cents is None
    assert arb.n_priced_brackets == 4
    assert any("missing yes_ask" in n for n in arb.notes)


def test_missing_prices_on_all_legs_records_zero_signal():
    e = _event([15] * 5)
    for m in e.markets:
        m.yes_ask = None
        m.yes_bid = None
    arb = compute_event_arbitrage(e)
    assert arb is not None
    assert arb.best_side is None
    assert arb.best_net_edge_cents is None
    assert arb.is_actionable is False


def test_no_basket_can_still_compute_if_only_yes_ask_missing():
    """If only yes_ask is missing on some legs but yes_bid is present,
    the no-basket arb is still computable."""
    e = _event(yes_asks=[35] * 4, yes_bids=[30] * 4)
    e.markets[0].yes_ask = None
    arb = compute_event_arbitrage(e)
    assert arb is not None
    assert arb.sum_yes_asks_cents is None       # asks broken
    assert arb.sum_yes_bids_cents == 120        # bids intact
    assert arb.no_basket_net_edge_cents == 12   # arb still computable
    assert arb.best_side == "no"


# -- Eligibility --------------------------------------------------------------

def test_event_with_one_bracket_returns_none():
    e = KalshiEvent(
        event_ticker="E", series_ticker="", title="", sub_title="",
        markets=[_market("E-B0", yes_ask=50, yes_bid=49)],
    )
    assert compute_event_arbitrage(e) is None


def test_empty_event_returns_none():
    e = KalshiEvent(event_ticker="E", series_ticker="", title="", sub_title="",
                    markets=[])
    assert compute_event_arbitrage(e) is None


# -- Ranking ------------------------------------------------------------------

def test_ranking_filters_to_positive_net_edge_only():
    e_good = _event([15] * 5, event_ticker="GOOD")   # net +15
    e_break_even = _event([20] * 5, event_ticker="BREAK")  # net -10
    ranked = rank_arbitrage_opportunities(
        [e_good, e_break_even], fee_per_leg_cents=2, min_net_edge_cents=1,
    )
    assert len(ranked) == 1
    assert ranked[0].event_ticker == "GOOD"


def test_ranking_sorts_by_net_edge_descending():
    e1 = _event([15] * 5, event_ticker="E1")  # net +15
    e2 = _event([10] * 5, event_ticker="E2")  # net +40
    e3 = _event([18] * 5, event_ticker="E3")  # net 0 — filtered out
    ranked = rank_arbitrage_opportunities([e1, e2, e3])
    assert [a.event_ticker for a in ranked] == ["E2", "E1"]


def test_ranking_with_custom_fee_changes_threshold():
    """At fee=2c, net=+15 (above min). At fee=5c, net=0 (filtered)."""
    e = _event([15] * 5)
    assert len(rank_arbitrage_opportunities([e], fee_per_leg_cents=2)) == 1
    assert len(rank_arbitrage_opportunities([e], fee_per_leg_cents=5)) == 0


# -- Mutual-exclusivity gate (invariant against false positives) -------------

def test_ranking_skips_non_mex_events_by_default():
    """A non-whitelisted series (e.g. KXARTISTSTREAMSY) is skipped even when
    the raw Σ-deviation looks like a huge arb — those numbers come from
    non-disjoint bracket structures and aren't real edge."""
    fake_arb = _event(
        yes_asks=[80] * 30, yes_bids=[78] * 30,
        event_ticker="KXARTISTSTREAMSY-BEY26DEC31",
        series_ticker="KXARTISTSTREAMSY",
    )
    # Without gating, this would show Σ yes_bids = 2340c >> 100c — fake +1880c edge.
    ranked = rank_arbitrage_opportunities([fake_arb])
    assert ranked == []


def test_ranking_keeps_mex_events_with_real_edge():
    """A whitelisted series (weather) with the same numbers is treated as
    real candidate edge."""
    real_arb = _event(
        yes_asks=[80] * 30, yes_bids=[78] * 30,
        event_ticker="KXHIGHHOU-26MAY29",
        series_ticker="KXHIGHHOU",
    )
    ranked = rank_arbitrage_opportunities([real_arb])
    assert len(ranked) == 1


def test_require_mex_false_bypasses_gate_for_diagnostics():
    fake_arb = _event(
        yes_asks=[80] * 30, yes_bids=[78] * 30,
        event_ticker="KXARTISTSTREAMSY-BEY26DEC31",
        series_ticker="KXARTISTSTREAMSY",
    )
    ranked = rank_arbitrage_opportunities([fake_arb], require_mex=False)
    assert len(ranked) == 1
    # Big "arb" because Σ yes_bids = 2340, no_basket gross = 2240, net = 2180.
    assert ranked[0].best_net_edge_cents > 2000


def test_mex_detection_derives_series_from_event_ticker():
    """Even if series_ticker is empty, derive it from the event_ticker
    prefix so weather events still pass the gate."""
    from kalshi_scout.arbitrage import is_mutually_exclusive_event
    e = KalshiEvent(
        event_ticker="KXHIGHHOU-26MAY29", series_ticker="",
        title="", sub_title="", markets=[],
    )
    assert is_mutually_exclusive_event(e) is True
