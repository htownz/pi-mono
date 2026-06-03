#!/usr/bin/env python3
"""Synthetic self-tests — prove the detectors fire (and, importantly, when they DON'T).

Run directly (stdlib only):   python test_scanner.py
Or via pytest:                pytest test_scanner.py
"""
from __future__ import annotations

from pmscan.models import BookLevel, Market, NegRiskEvent, OrderBook
from pmscan.scanner import group_negrisk, scan_market, scan_negrisk


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
    # Realistic small gap: YES asks sum 0.985 → 1.5c edge; mids ≈ 0.975 → exhaustive plausible.
    outs = [_outcome(t, t + "?") for t in ("A", "B", "C", "D")]
    ev = NegRiskEvent(request_id="EVT", outcomes=outs, title="Who wins?")
    books = _four_outcome_books([
        ("A", 0.250, 100, 0.245),
        ("B", 0.250, 150, 0.245),
        ("C", 0.245, 200, 0.240),
        ("D", 0.240, 120, 0.235),
    ])
    opp = scan_negrisk(ev, books, gas_usd=0.0)
    assert opp is not None, "should detect the 1.5c basket edge"
    assert opp.legs == 4
    assert abs(opp.ask_sum - 0.985) < 1e-9
    assert abs(opp.edge_cents - 1.5) < 1e-6
    assert opp.capturable_sets == 100  # min leg size
    assert opp.exhaustive_verified is True, opp.uncertainty_reason
    assert abs(opp.gross_profit_usd - 0.015 * 100) < 1e-6


def test_negrisk_other_bucket_flagged_uncertain():
    # Same prices, but one outcome carries the Other/None flag → scan, flag uncertain (not drop).
    outs = [_outcome(t, t + "?", neg_risk_other=(t == "D")) for t in ("A", "B", "C", "D")]
    ev = group_negrisk(outs)[0]
    books = _four_outcome_books([
        ("A", 0.250, 100, 0.245),
        ("B", 0.250, 150, 0.245),
        ("C", 0.245, 200, 0.240),
        ("D", 0.240, 120, 0.235),
    ])
    opp = scan_negrisk(ev, books, gas_usd=0.0)
    assert opp is not None
    assert opp.exhaustive_verified is False
    assert "Other" in opp.uncertainty_reason


def test_negrisk_low_mass_flagged_uncertain():
    # Big 10c gap → mids sum ≈ 0.88, below the 0.90 floor → set likely not exhaustive.
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
