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
    _parse_interval,
    compute_event_arbitrage,
    detect_numeric_partition,
    is_mutually_exclusive_event,
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
    # Without gating: Σ yes_bids = 2340c. no_basket gross = 2240c, fees = 30 × 2c = 60c,
    # net = +2180c — a fake huge "arb" from a non-mutually-exclusive event.
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
    e = KalshiEvent(
        event_ticker="KXHIGHHOU-26MAY29", series_ticker="",
        title="", sub_title="", markets=[],
    )
    assert is_mutually_exclusive_event(e) is True


# -- Algorithmic MEX detection -----------------------------------------------

def _typed_market(ticker: str, yes_sub_title: str,
                  yes_ask: int = 10, yes_bid: int = 9) -> KalshiMarket:
    """Variant of _market that lets the test set yes_sub_title — the field
    the algorithmic detector parses to extract numeric intervals."""
    return KalshiMarket(
        ticker=ticker, event_ticker="E", title="", yes_sub_title=yes_sub_title,
        status="open", close_time=None,
        yes_bid=yes_bid, yes_ask=yes_ask,
        no_bid=100 - yes_ask, no_ask=100 - yes_bid,
        last_price=None, volume=0, open_interest=0,
    )


def _typed_event(sub_titles: list[str], series_ticker: str = "KXUNKNOWN",
                 event_ticker: str = "KXUNKNOWN-26MAY29") -> KalshiEvent:
    markets = [
        _typed_market(f"{event_ticker}-B{i}", s) for i, s in enumerate(sub_titles)
    ]
    return KalshiEvent(
        event_ticker=event_ticker, series_ticker=series_ticker,
        title="", sub_title="", markets=markets,
    )


def test_detector_passes_clean_weather_partition():
    """5 weather-bucket markets tiling 60-65° with tail brackets at each end.
    `KXUNKNOWN` series isn't in the whitelist, so this only passes via the
    algorithmic detector."""
    event = _typed_event([
        "59° or below",      # LT tail
        "60° to 61°", "61° to 62°", "62° to 63°", "63° to 64°", "64° to 65°",
        "65° or above",      # GT tail
    ])
    result = detect_numeric_partition(event)
    assert result.is_mex is True
    assert result.n_parsed == result.n_markets
    # And the public gate also accepts it (since the series isn't whitelisted,
    # acceptance comes from the detector specifically).
    assert is_mutually_exclusive_event(event) is True


def test_detector_rejects_event_with_unparseable_sub_titles():
    """Artist-streams-style events have non-numeric labels like 'Beyoncé wins'.
    No interval parses → detector returns False → gate rejects."""
    event = _typed_event([
        "Beyoncé has most streams",
        "Taylor Swift has most streams",
        "Drake has most streams",
    ])
    result = detect_numeric_partition(event)
    assert result.is_mex is False
    assert result.n_parsed == 0
    assert "no parseable numeric range" in result.reason
    assert is_mutually_exclusive_event(event) is False


def test_detector_rejects_overlapping_ranges():
    """Tails-present event whose finite buckets overlap must be rejected
    — the no-arb math is invalid even if labels parse and tails exist."""
    event = _typed_event([
        "9 or below",
        "10 to 20", "15 to 25", "20 to 30",
        "30 or above",
    ])
    result = detect_numeric_partition(event)
    assert result.is_mex is False
    assert "overlap" in result.reason


def test_detector_rejects_large_gap():
    """Tails-present event with a missing middle bucket: finite intervals
    don't tile adjacently, so settlement in the gap has no winning bracket."""
    event = _typed_event([
        "59 or below",
        "60 to 65", "70 to 75",   # 5-unit gap between 65 and 70
        "75 or above",
    ])
    result = detect_numeric_partition(event)
    assert result.is_mex is False
    assert "gap" in result.reason


def test_detector_rejects_single_market_event():
    """A 1-market event isn't a partition by construction."""
    event = _typed_event(["60 to 61"])
    assert detect_numeric_partition(event).is_mex is False


def test_detector_accepts_mixed_inclusive_exclusive_boundaries():
    """Weather-style buckets share endpoints (`60-61` then `61-62`) with
    gap=0 between them. With tail brackets at both ends, this passes."""
    event = _typed_event([
        "59 or below",
        "60 to 61", "61 to 62", "62 to 63",
        "63 or above",
    ])
    assert detect_numeric_partition(event).is_mex is True


def test_detector_accepts_dollar_denominated_brackets():
    """Natural-gas-style $-denominated brackets, with tails covering prices
    below the lowest and above the highest finite bucket."""
    event = _typed_event([
        "$2.99 or below",
        "$3.00 to $3.25", "$3.25 to $3.50", "$3.50 to $3.75",
        "$3.75 or above",
    ])
    assert detect_numeric_partition(event).is_mex is True


def test_detector_rejects_finite_only_partition_without_tails():
    """Codex P1: finite-only `$3.00-$3.25, $3.25-$3.50, $3.50-$3.75` has
    no bracket for prices below $3.00 or above $3.75. A settlement at $4
    would lose every Yes leg, so the no-arb math is invalid."""
    event = _typed_event([
        "$3.00 to $3.25", "$3.25 to $3.50", "$3.50 to $3.75",
    ])
    result = detect_numeric_partition(event)
    assert result.is_mex is False
    assert "finite-only" in result.reason


def test_detector_rejects_partition_missing_only_high_tail():
    """A below-tail covers prices below the lowest finite bucket, but the
    top is still unbounded — a settlement above the last finite bucket
    has no winning bracket."""
    event = _typed_event([
        "2 or below", "2 to 3", "3 to 4",
    ])
    result = detect_numeric_partition(event)
    assert result.is_mex is False
    assert "above-tail" in result.reason


def test_detector_rejects_decimal_gap_below_old_blanket_tolerance():
    """Codex P2: `$3.00-$3.25` + `$4.70-$5.00` has a 1.45 gap that the old
    1.5-unit blanket tolerance accepted. With strict 0.01 epsilon between
    finite intervals, the missing $3.25-$4.70 span is now rejected."""
    event = _typed_event([
        "$2.99 or below",
        "$3.00 to $3.25", "$4.70 to $5.00",
        "$5.00 or above",
    ])
    result = detect_numeric_partition(event)
    assert result.is_mex is False
    assert "gap" in result.reason


def test_strict_mex_disables_algorithmic_fallback():
    """`--strict-mex` returns the V1.1 whitelist-only behavior."""
    event = _typed_event([
        "59 or below",
        "60 to 61", "61 to 62", "62 to 63",
        "63 or above",
    ], series_ticker="KXTOTALLYNEW")
    assert is_mutually_exclusive_event(event) is True
    assert is_mutually_exclusive_event(event, strict=True) is False


def test_ranker_strict_mex_rejects_algorithmically_detected_events():
    """End-to-end: an event the detector promotes shows up in `rank` with
    default args but disappears under `strict_mex=True`."""
    sub_titles = [
        "59 or below",
        "60 to 61", "61 to 62", "62 to 63", "63 to 64", "64 to 65",
        "65 or above",
    ]
    markets = [
        _typed_market(f"E-B{i}", s, yes_ask=12, yes_bid=11)
        for i, s in enumerate(sub_titles)
    ]
    event = KalshiEvent(
        event_ticker="KXTOTALLYNEW-26MAY29", series_ticker="KXTOTALLYNEW",
        title="", sub_title="", markets=markets,
    )
    permissive = rank_arbitrage_opportunities([event])
    strict = rank_arbitrage_opportunities([event], strict_mex=True)
    assert len(permissive) == 1
    assert len(strict) == 0


def test_detector_handles_above_below_tail_brackets():
    """Tail brackets extend coverage to ±inf so the contiguous-axis check
    passes for finite intervals between them."""
    event = _typed_event([
        "59° or below",
        "60° to 61°", "61° to 62°", "62° to 63°",
        "63° or above",
    ])
    assert detect_numeric_partition(event).is_mex is True


def test_detector_does_not_fall_back_to_title():
    """Copilot P2: the detector reads yes_sub_title only. An event whose
    bracket labels are non-numeric but whose titles contain numbers (e.g.
    'Will price be above $80?') must NOT be promoted."""
    markets = [
        KalshiMarket(
            ticker=f"E-B{i}", event_ticker="E",
            title=f"Will the price be above ${30 + i}?",  # numeric title
            yes_sub_title="Yes",                          # non-numeric label
            status="open", close_time=None,
            yes_bid=20, yes_ask=21, no_bid=79, no_ask=80,
            last_price=None, volume=0, open_interest=0,
        )
        for i in range(5)
    ]
    event = KalshiEvent(
        event_ticker="KXNUMERICTITLE", series_ticker="KXNUMERICTITLE",
        title="", sub_title="", markets=markets,
    )
    result = detect_numeric_partition(event)
    assert result.is_mex is False
    assert "no parseable numeric range" in result.reason


def test_parse_interval_handles_symbolic_operators():
    """Copilot P3: the regex word-boundary made `>=60` / `<60` / `=60`
    unreachable because there's no \\b before a non-word character at
    string start. The fix moves \\b inside the letter-only alternatives."""
    assert _parse_interval(">= 60") == (60.0, float("inf"))
    assert _parse_interval(">60") == (60.0, float("inf"))
    assert _parse_interval("<= 50") == (float("-inf"), 50.0)
    assert _parse_interval("<50") == (float("-inf"), 50.0)
    assert _parse_interval("= 75") == (75.0, 75.0)
