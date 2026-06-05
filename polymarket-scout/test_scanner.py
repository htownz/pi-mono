#!/usr/bin/env python3
"""Synthetic self-tests — prove the detectors fire (and, importantly, when they DON'T).

Run directly (stdlib only):   python test_scanner.py
Or via pytest:                pytest test_scanner.py
"""
from __future__ import annotations

import json

from pmscan.client import parse_event
from pmscan.models import BookLevel, Market, NegRiskEvent, OrderBook
from pmscan.scanner import group_negrisk, negrisk_snapshot, scan_market, scan_negrisk
from pmscan.temporal import detect_dips, group_by_event, robust_stats
from pmscan.parity import (
    ParityLink, VenueQuote, kalshi_venue_quote, pm_venue_quote, scan_parity, scan_parity_links,
)
from pmscan.parity_registry import REGISTRY, ParityCandidate, build_links
from pmscan.kalshi import market_to_quote


def _book(token_id: str, asks=(), bids=()) -> OrderBook:
    # Pass levels best-price-LAST to mimic the real CLOB ordering gotcha.
    return OrderBook(
        token_id=token_id,
        asks=[BookLevel(p, s) for p, s in asks],
        bids=[BookLevel(p, s) for p, s in bids],
    )


def _binary(yes_token="Y", no_token="N", **kw) -> Market:
    return Market(
        venue="polymarket", market_id="cond", question="Test market?", slug="test",
        outcomes=["Yes", "No"], token_ids=[yes_token, no_token], **kw,
    )


def _outcome(tid: str, q: str, *, neg_risk_request_id="EVT", neg_risk_other=False,
             volume_24hr=0.0) -> Market:
    return Market(
        venue="polymarket", market_id=q, question=q, slug=q,
        outcomes=["Yes", "No"], token_ids=[tid, tid + "_no"], neg_risk=True,
        neg_risk_request_id=neg_risk_request_id, neg_risk_other=neg_risk_other,
        volume_24hr=volume_24hr, group_title="Who wins?",
    )


# --------------------------------------------------------------------------- #
# Phase 1 — binary
# --------------------------------------------------------------------------- #
def test_binary_merge_edge():
    m = _binary()
    books = {
        "Y": _book("Y", asks=[(0.55, 999), (0.40, 80)]),   # best ask = 0.40, size 80
        "N": _book("N", asks=[(0.70, 999), (0.55, 120)]),  # best ask = 0.55, size 120
    }
    opp = scan_market(m, books, gas_usd=0.0)
    assert opp is not None and opp.side == "merge", opp
    assert abs(opp.price_sum - 0.95) < 1e-9
    assert abs(opp.edge_cents - 5.0) < 1e-9
    assert opp.capturable_sets == 80
    assert abs(opp.gross_profit_usd - 0.05 * 80) < 1e-9


def test_binary_split_edge():
    m = _binary()
    books = {
        "Y": _book("Y", bids=[(0.30, 999), (0.60, 50)]),   # best bid = 0.60, size 50
        "N": _book("N", bids=[(0.20, 999), (0.45, 70)]),   # best bid = 0.45, size 70
    }
    opp = scan_market(m, books, gas_usd=0.0)
    assert opp is not None and opp.side == "split", opp
    assert abs(opp.price_sum - 1.05) < 1e-9
    assert opp.capturable_sets == 50


def test_binary_no_edge_pinned_to_one():
    # The real-world tight case: ask-sum 1.001 / bid-sum 0.999. Must NOT fire.
    m = _binary()
    books = {
        "Y": _book("Y", asks=[(0.501, 500)], bids=[(0.499, 500)]),
        "N": _book("N", asks=[(0.500, 500)], bids=[(0.500, 500)]),
    }
    assert scan_market(m, books) is None


def test_binary_fee_eats_thin_edge():
    # 1c gross edge, but 1c/share/leg fee = 2c/set → net is negative (still detected, caller filters).
    m = _binary()
    books = {
        "Y": _book("Y", asks=[(0.49, 100)]),
        "N": _book("N", asks=[(0.50, 100)]),  # ask_sum 0.99 → 1c gross
    }
    opp = scan_market(m, books, fee_per_share=0.01, gas_usd=0.0)
    assert opp is not None and opp.side == "merge"
    assert abs(opp.edge_cents - 1.0) < 1e-9
    assert opp.net_profit_usd < 0, opp.net_profit_usd  # fee eats it


# --------------------------------------------------------------------------- #
# Phase 1b — NegRisk
# --------------------------------------------------------------------------- #
def _four_outcome_books(asks_bids):
    """asks_bids: list of (token, ask_price, ask_size, bid_price). Build YES books."""
    return {
        tok: _book(tok, asks=[(ap, asz)], bids=[(bp, asz)])
        for tok, ap, asz, bp in asks_bids
    }


def test_negrisk_grouping():
    ms = [_outcome("A", "a?"), _outcome("B", "b?"), _outcome("C", "c?", neg_risk_other=True)]
    events = group_negrisk(ms)
    assert len(events) == 1
    ev = events[0]
    assert ev.n == 3 and ev.has_other is True and ev.title == "Who wins?"


def test_negrisk_edge_verified():
    # Tight near-complete set: YES asks sum 0.99 → 1c edge; mids ≈ 0.986 ≥ 0.98 floor.
    outs = [_outcome(t, t + "?") for t in ("A", "B", "C", "D")]
    ev = NegRiskEvent(request_id="EVT", outcomes=outs, title="Who wins?")
    books = _four_outcome_books([
        ("A", 0.250, 100, 0.248),
        ("B", 0.250, 150, 0.248),
        ("C", 0.245, 200, 0.243),
        ("D", 0.245, 120, 0.243),
    ])
    opp = scan_negrisk(ev, books, gas_usd=0.0)
    assert opp is not None, "should detect the 1c basket edge"
    assert opp.legs == 4
    assert abs(opp.ask_sum - 0.99) < 1e-9
    assert abs(opp.edge_cents - 1.0) < 1e-6
    assert opp.capturable_sets == 100  # min leg size
    assert opp.exhaustive_verified is True, opp.uncertainty_reason
    assert abs(opp.gross_profit_usd - 0.01 * 100) < 1e-6
    # transparency fields for the time-series detector
    assert abs(opp.bid_sum - 0.982) < 1e-9
    assert abs(opp.spread - 0.008) < 1e-9
    assert abs(opp.implied_mass - 0.986) < 1e-9
    assert abs(opp.implied_other - 0.014) < 1e-9


def test_negrisk_other_bucket_flagged_uncertain():
    # Otherwise-verifiable prices, but one outcome carries Other/None → flag uncertain (not drop).
    outs = [_outcome(t, t + "?", neg_risk_other=(t == "D")) for t in ("A", "B", "C", "D")]
    ev = group_negrisk(outs)[0]
    books = _four_outcome_books([
        ("A", 0.250, 100, 0.248),
        ("B", 0.250, 150, 0.248),
        ("C", 0.245, 200, 0.243),
        ("D", 0.245, 120, 0.243),
    ])
    opp = scan_negrisk(ev, books, gas_usd=0.0)
    assert opp is not None
    assert opp.exhaustive_verified is False
    assert opp.uncertainty_reason == "explicit Other/None bucket present"  # Other alone, mass ok


def test_negrisk_low_mass_flagged_uncertain():
    # Big 10c gap → mids sum ≈ 0.88, below the 0.98 floor → set likely not exhaustive.
    # This is the structural case: the gap below $1 IS the missing mass (~12c on unlisted).
    outs = [_outcome(t, t + "?") for t in ("A", "B", "C", "D")]
    ev = NegRiskEvent(request_id="EVT", outcomes=outs)
    books = _four_outcome_books([
        ("A", 0.20, 100, 0.19),
        ("B", 0.25, 100, 0.24),
        ("C", 0.15, 100, 0.14),
        ("D", 0.30, 100, 0.29),
    ])
    opp = scan_negrisk(ev, books, gas_usd=0.0)
    assert opp is not None and abs(opp.edge_cents - 10.0) < 1e-6
    assert opp.exhaustive_verified is False
    assert "not exhaustive" in opp.uncertainty_reason
    # implied_other ≈ 1 - mass; mass = mids (0.195+0.245+0.145+0.295) = 0.88 → other ≈ 0.12
    assert abs(opp.implied_mass - 0.88) < 1e-9
    assert abs(opp.implied_other - 0.12) < 1e-9


def test_negrisk_spread_nan_when_a_leg_has_no_bid():
    # A one-sided leg (ask, no bid) → spread is undefined (NaN), but mass/edge still compute.
    import math
    outs = [_outcome(t, t + "?") for t in ("A", "B")]
    ev = NegRiskEvent(request_id="EVT", outcomes=outs)
    books = {
        "A": _book("A", asks=[(0.49, 100)], bids=[(0.48, 100)]),
        "B": _book("B", asks=[(0.49, 100)]),  # no bid on this leg
    }
    opp = scan_negrisk(ev, books, gas_usd=0.0)
    assert opp is not None
    assert math.isnan(opp.spread)
    assert abs(opp.bid_sum - 0.48) < 1e-9  # only the leg with a bid contributes


def test_negrisk_no_edge():
    outs = [_outcome(t, t + "?") for t in ("A", "B", "C")]
    ev = NegRiskEvent(request_id="EVT", outcomes=outs)
    books = _four_outcome_books([
        ("A", 0.40, 100, 0.39),
        ("B", 0.35, 100, 0.34),
        ("C", 0.30, 100, 0.29),  # ask_sum 1.05 → no buy-all-YES edge
    ])
    assert scan_negrisk(ev, books) is None


def test_negrisk_incomplete_legs_refused():
    # Missing one leg's book → refuse (None) rather than under-sum and overstate the edge.
    outs = [_outcome(t, t + "?") for t in ("A", "B", "C")]
    ev = NegRiskEvent(request_id="EVT", outcomes=outs)
    books = _four_outcome_books([
        ("A", 0.30, 100, 0.29),
        ("B", 0.30, 100, 0.29),
        # "C" intentionally absent
    ])
    assert scan_negrisk(ev, books) is None


def test_negrisk_truncated_fragment_flagged_uncertain():
    # Reproduces the live artifact: a big event (e.g. "Brazil Presidential Election")
    # truncated to its 2 longshot legs. YES asks ≈ 0 → a 99.6c "edge" that is NOT real.
    # The mass floor MUST quarantine it — a near-zero mass can never read as verified.
    outs = [_outcome(t, t + "?") for t in ("A", "B")]
    ev = NegRiskEvent(request_id="EVT", outcomes=outs, title="Brazil Presidential Election")
    books = _four_outcome_books([
        ("A", 0.002, 715903, 0.001),
        ("B", 0.002, 800000, 0.001),
    ])
    opp = scan_negrisk(ev, books, gas_usd=0.0)
    assert opp is not None, "a crossing IS present — we report it, but flagged"
    assert opp.legs == 2
    assert opp.edge_cents > 99.0
    assert opp.exhaustive_verified is False, "near-zero-mass fragment must never verify"
    assert "not exhaustive" in opp.uncertainty_reason


# --------------------------------------------------------------------------- #
# Phase 1b — complete-event grouping from /events records
# --------------------------------------------------------------------------- #
def _raw_child(cond: str, *, yes_tok: str, neg_other=False, vol=1000.0) -> dict:
    """A Gamma child-market dict as it appears nested under an /events record."""
    return {
        "conditionId": cond, "question": f"{cond}?", "slug": cond,
        "enableOrderBook": True, "active": True, "closed": False, "acceptingOrders": True,
        "clobTokenIds": json.dumps([yes_tok, yes_tok + "_no"]),
        "outcomes": json.dumps(["Yes", "No"]),
        "negRisk": True, "negRiskOther": neg_other, "volume24hr": vol,
        "orderPriceMinTickSize": 0.01,
    }


def test_parse_event_builds_complete_group():
    raw = {
        "id": "ev1", "title": "Who wins the election?", "slug": "who-wins", "negRisk": True,
        "negRiskMarketID": "NRM-1",
        "markets": [
            _raw_child("a", yes_tok="A"),
            _raw_child("b", yes_tok="B"),
            _raw_child("c", yes_tok="C", neg_other=True),
            {"active": False, "closed": True},  # a resolved/garbage child → dropped
        ],
    }
    ev = parse_event(raw)
    assert ev is not None
    assert ev.n == 3, "three usable outcomes; the closed child is dropped"
    assert ev.request_id == "NRM-1"
    assert ev.title == "Who wins the election?"
    assert ev.has_other is True
    assert [m.yes_token() for m in ev.outcomes] == ["A", "B", "C"]


def test_parse_event_skips_non_negrisk():
    assert parse_event({"id": "x", "negRisk": False, "markets": []}) is None
    # NegRisk but only one usable outcome → not a partition → skip.
    one = {"id": "y", "negRisk": True, "markets": [_raw_child("solo", yes_tok="S")]}
    assert parse_event(one) is None


# --------------------------------------------------------------------------- #
# Phase 1b — snapshots (temporal baseline feed)
# --------------------------------------------------------------------------- #
def test_negrisk_snapshot_records_noncrossing_basket():
    # ask_sum ≥ 1 (no edge) must STILL produce a snapshot — baselines need the normal level.
    outs = [_outcome(t, t + "?") for t in ("A", "B", "C")]
    ev = NegRiskEvent(request_id="EVT", outcomes=outs, title="Who wins?")
    books = _four_outcome_books([
        ("A", 0.40, 100, 0.39),
        ("B", 0.35, 100, 0.34),
        ("C", 0.30, 100, 0.29),  # ask_sum = 1.05, no crossing
    ])
    snap = negrisk_snapshot(ev, books)
    assert snap is not None and snap.legs == 3
    assert abs(snap.ask_sum - 1.05) < 1e-9
    assert abs(snap.bid_sum - 1.02) < 1e-9
    assert scan_negrisk(ev, books) is None  # ...while the opportunity scan correctly emits nothing


def test_negrisk_snapshot_refuses_incomplete():
    outs = [_outcome(t, t + "?") for t in ("A", "B")]
    ev = NegRiskEvent(request_id="EVT", outcomes=outs)
    books = {"A": _book("A", asks=[(0.5, 100)])}  # "B" missing
    assert negrisk_snapshot(ev, books) is None


# --------------------------------------------------------------------------- #
# Phase 1c — temporal dislocation detector
# --------------------------------------------------------------------------- #
def _snaps(request_id, asks, title="Evt"):
    return [{"ts": f"2026-06-03T00:{i:02d}:00+00:00", "request_id": request_id, "title": title,
             "legs": 5, "ask_sum": a, "bid_sum": a - 0.02, "implied_mass": a - 0.01,
             "has_other": False} for i, a in enumerate(asks)]


def test_robust_stats_outlier_resistant():
    # A few deep dips must NOT drag the baseline down (that's the whole point of median/MAD).
    med, sigma = robust_stats([0.99] * 18 + [0.90, 0.90])
    assert abs(med - 0.99) < 1e-9


def test_temporal_flat_structural_has_no_dip():
    # Phantom/structural event: ask_sum sits flat at a depressed level → NOT a dislocation.
    snaps = _snaps("FLAT", [0.962, 0.961, 0.962, 0.963, 0.961] * 4, title="Presidential 2028")
    assert detect_dips(snaps) == []


def test_temporal_transient_dislocation_detected():
    # Normally ≥ $1 (1.01), then a 3-cycle drop BELOW $1 to 0.969 (recovers) = a real,
    # capturable dislocation (basket briefly buyable under $1).
    asks = [1.01] * 10 + [0.97, 0.969, 0.97] + [1.01] * 6
    dips = detect_dips(_snaps("DISLOC", asks, title="Real Event"))
    assert len(dips) == 1, dips
    d = dips[0]
    assert d.n_points == 3
    assert abs(d.baseline_ask - 1.01) < 1e-9
    assert abs(d.min_ask - 0.969) < 1e-9
    assert abs(d.below_par - 0.031) < 1e-9   # $1 - 0.969 = the edge at the trough
    assert d.capturable is True
    assert d.duration_s > 0


def test_temporal_high_baseline_dip_is_not_capturable():
    # Illiquid basket normally at $2.10 dips to $1.67 — still >$1, NOT an edge. Filtered by
    # default; visible only with capturable_only=False.
    asks = [2.10] * 12 + [1.67, 1.68] + [2.10] * 6
    assert detect_dips(_snaps("HIGH", asks, title="Illiquid Futures")) == []
    raw = detect_dips(_snaps("HIGH", asks), capturable_only=False)
    assert len(raw) == 1 and raw[0].capturable is False and raw[0].below_par == 0.0


def test_temporal_structurally_cheap_basket_is_not_capturable():
    # Non-exhaustive fragment sits cheap forever (~0.64) and wobbles — baseline < $1, so it's
    # not a dislocation even though min_ask < 1. Filtered by default.
    asks = [0.64] * 12 + [0.55, 0.55] + [0.64] * 6
    assert detect_dips(_snaps("CHEAP", asks, title="Roland Garros frag")) == []
    raw = detect_dips(_snaps("CHEAP", asks), capturable_only=False)
    assert len(raw) == 1 and raw[0].capturable is False


def test_temporal_skips_events_with_too_little_history():
    # 6 points < default min_points (12) → no trustworthy baseline → skip.
    asks = [1.01, 1.01, 0.97, 1.01, 1.01, 1.01]
    assert detect_dips(_snaps("SHORT", asks)) == []
    # but with min_points lowered it fires
    assert len(detect_dips(_snaps("SHORT", asks), min_points=5)) == 1


# --------------------------------------------------------------------------- #
# Phase 2 (draft) — cross-venue parity
# --------------------------------------------------------------------------- #
def _vq(venue, key, **kw):
    return VenueQuote(venue=venue, market_key=key, label=key, **kw)


def test_parity_cross_venue_lock_detected():
    # PM YES ask 0.55 + Kalshi NO ask 0.40 = 0.95 → 5c locked edge, settlement verified.
    a = _vq("polymarket", "PM", yes_ask=0.55, no_ask=0.46, yes_ask_size=100, no_ask_size=80)
    b = _vq("kalshi", "KX", yes_ask=0.58, no_ask=0.40, yes_ask_size=50, no_ask_size=200)
    opp = scan_parity(ParityLink("Same outcome", a, b, settlement_verified=True))
    assert opp is not None
    assert opp.side == "A_yes+B_no"            # 0.55 + 0.40 beats 0.58 + 0.46
    assert abs(opp.cost_sum - 0.95) < 1e-9
    assert abs(opp.edge_cents - 5.0) < 1e-6
    assert opp.capturable_sets == 100          # min(PM yes size 100, Kalshi no size 200)
    assert opp.settlement_verified is True


def test_parity_picks_cheaper_construction():
    # Make B_yes + A_no the cheaper basket.
    a = _vq("polymarket", "PM", yes_ask=0.70, no_ask=0.30)
    b = _vq("kalshi", "KX", yes_ask=0.55, no_ask=0.55)
    opp = scan_parity(ParityLink("x", a, b, settlement_verified=True))
    assert opp is not None and opp.side == "B_yes+A_no"  # 0.55 + 0.30 = 0.85
    assert abs(opp.edge_cents - 15.0) < 1e-6


def test_parity_no_edge_when_baskets_at_or_above_par():
    a = _vq("polymarket", "PM", yes_ask=0.60, no_ask=0.45)
    b = _vq("kalshi", "KX", yes_ask=0.58, no_ask=0.47)   # cheapest basket = 0.58+0.45 = 1.03
    assert scan_parity(ParityLink("x", a, b, settlement_verified=True)) is None


def test_parity_unverified_settlement_still_emits_but_flagged():
    a = _vq("polymarket", "PM", yes_ask=0.50, no_ask=0.55)
    b = _vq("kalshi", "KX", yes_ask=0.55, no_ask=0.45)   # 0.50 + 0.45 = 0.95
    opp = scan_parity(ParityLink("maybe-same", a, b, settlement_verified=False, note="dates differ?"))
    assert opp is not None and opp.edge_cents > 0
    assert opp.settlement_verified is False
    assert opp.note == "dates differ?"


def test_parity_net_after_fees():
    a = _vq("polymarket", "PM", yes_ask=0.50, no_ask=0.55, yes_ask_size=100)
    b = _vq("kalshi", "KX", yes_ask=0.55, no_ask=0.45, no_ask_size=100)  # 0.50+0.45=0.95, 5c
    opp = scan_parity(ParityLink("x", a, b, settlement_verified=True), fee_per_leg=0.01, gas_usd=0.0)
    # gross = 0.05 * 100 = 5.00; fees = 2 * 0.01 * 100 = 2.00; net = 3.00
    assert abs(opp.net_profit_usd - 3.00) < 1e-6


def test_parity_kalshi_cents_adapter():
    q = kalshi_venue_quote("KXTICKER", label="Cand X", yes_bid_c=40, yes_ask_c=42,
                           no_bid_c=58, no_ask_c=60)
    assert q.venue == "kalshi" and abs(q.yes_ask - 0.42) < 1e-9 and abs(q.no_ask - 0.60) < 1e-9


def test_parity_pm_adapter_from_books():
    m = _binary(yes_token="Y", no_token="N")
    books = {
        "Y": _book("Y", asks=[(0.55, 100)], bids=[(0.53, 90)]),
        "N": _book("N", asks=[(0.46, 80)], bids=[(0.44, 70)]),
    }
    q = pm_venue_quote(m, books)
    assert q is not None and q.venue == "polymarket"
    assert abs(q.yes_ask - 0.55) < 1e-9 and abs(q.no_ask - 0.46) < 1e-9
    assert q.yes_ask_size == 100 and q.no_ask_size == 80
    # end-to-end through scan_parity_links against a Kalshi quote.
    kb = kalshi_venue_quote("KX", label="same", yes_bid_c=50, yes_ask_c=52, no_bid_c=46, no_ask_c=48)
    opps = scan_parity_links([ParityLink("same", q, kb, settlement_verified=True)])
    # A_yes+B_no = 0.55 + 0.48 = 1.03 (≥$1); B_yes+A_no = 0.52 + 0.46 = 0.98 (<$1) → the latter wins.
    assert len(opps) == 1
    assert opps[0].side == "B_yes+A_no"
    assert abs(opps[0].cost_sum - 0.98) < 1e-9
    assert abs(opps[0].edge_cents - 2.0) < 1e-6


def test_parity_registry_build_links_pairs_and_skips():
    cands = [
        ParityCandidate("Fed no change", "fed decision in june", "KXFED", settlement_verified=True),
        ParityCandidate("Absent", "no such pm market", "KXNONE"),
    ]
    pm = [VenueQuote("polymarket", "tok", "Fed Decision in June?", yes_ask=0.91, no_ask=0.10)]
    kal = {"KXFED": kalshi_venue_quote("KXFED", label="KXFED",
                                       yes_bid_c=88, yes_ask_c=90, no_bid_c=10, no_ask_c=12)}
    links, unmatched = build_links(cands, pm, kal)
    assert len(links) == 1 and links[0].name == "Fed no change"
    assert links[0].settlement_verified is True
    assert links[0].a.venue == "polymarket" and links[0].b.venue == "kalshi"
    assert unmatched == ["Absent [no PM match]"]   # reported with reason, not silently dropped


def test_parity_build_links_ambiguous_match_is_unmatched():
    # Substring "twins" hits TWO Polymarket markets → don't guess; report ambiguous.
    cands = [ParityCandidate("Twins game", "twins", "KX")]
    pm = [VenueQuote("polymarket", "a", "Royals vs. Twins", yes_ask=0.5, no_ask=0.5),
          VenueQuote("polymarket", "b", "Twins vs. Yankees", yes_ask=0.5, no_ask=0.5)]
    kal = {"KX": kalshi_venue_quote("KX", label="KX", yes_bid_c=40, yes_ask_c=42,
                                    no_bid_c=58, no_ask_c=60)}
    links, unmatched = build_links(cands, pm, kal)
    assert links == []
    assert len(unmatched) == 1 and "ambiguous" in unmatched[0]


def test_pm_venue_quote_resolves_yes_by_label_not_index():
    # outcomes reversed: index 0 is "No". yes_ask must come from the Yes token, not token_ids[0].
    m = Market(venue="polymarket", market_id="c", question="Spurs win?", slug="s",
               outcomes=["No", "Yes"], token_ids=["NO_TOK", "YES_TOK"])
    books = {"YES_TOK": _book("YES_TOK", asks=[(0.64, 100)], bids=[(0.62, 50)]),
             "NO_TOK": _book("NO_TOK", asks=[(0.38, 80)], bids=[(0.36, 40)])}
    q = pm_venue_quote(m, books)
    assert q is not None and q.market_key == "YES_TOK"
    assert abs(q.yes_ask - 0.64) < 1e-9 and abs(q.no_ask - 0.38) < 1e-9


def test_parity_registry_seed_is_wellformed():
    assert REGISTRY
    for c in REGISTRY:
        assert c.name and c.pm_match and c.kalshi_ticker
        # seed entries are templates — must never ship asserting verified settlement.
        assert c.settlement_verified is False


def test_kalshi_bridge_parses_new_dollars_schema():
    # 2026 schema: prices as dollar strings. Normalize to dollars in the VenueQuote.
    raw = {"ticker": "KXFED-26JUN-X", "title": "Fed: no change?",
           "yes_bid_dollars": "0.9100", "yes_ask_dollars": "0.9300",
           "no_bid_dollars": "0.0700", "no_ask_dollars": "0.0900"}
    q = market_to_quote(raw)
    assert q is not None and q.venue == "kalshi" and q.market_key == "KXFED-26JUN-X"
    assert abs(q.yes_ask - 0.93) < 1e-9 and abs(q.no_ask - 0.09) < 1e-9
    assert abs(q.yes_bid - 0.91) < 1e-9 and abs(q.no_bid - 0.07) < 1e-9


def test_kalshi_bridge_parses_legacy_cents_and_skips_tickerless():
    raw = {"ticker": "KX-LEGACY", "title": "t", "yes_bid": 40, "yes_ask": 42,
           "no_bid": 58, "no_ask": 60}
    q = market_to_quote(raw)
    assert abs(q.yes_ask - 0.42) < 1e-9 and abs(q.no_ask - 0.60) < 1e-9
    assert market_to_quote({"title": "no ticker"}) is None


def test_kalshi_bridge_skips_closed_market():
    # A settled/closed market still reports last prices — must NOT become a quote (phantom lock).
    for status in ("closed", "settled", "finalized", "determined"):
        raw = {"ticker": "KX-DONE", "title": "t", "status": status,
               "yes_ask_dollars": "0.0100", "no_ask_dollars": "0.0100"}  # 0.02 < $1 → fake lock
        assert market_to_quote(raw) is None, status
    # active / open / unspecified status still quotes
    assert market_to_quote({"ticker": "KX-LIVE", "status": "active",
                            "yes_ask_dollars": "0.40", "no_ask_dollars": "0.62"}) is not None


# --------------------------------------------------------------------------- #
def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
