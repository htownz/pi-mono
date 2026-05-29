"""Regression tests for Kalshi's 2026 market API schema.

Verifies the kalshi.py client correctly translates _dollars / _fp string
fields into the internal int-cents representation.
"""

from kalshi_scout.kalshi import _dollars_to_cents, _market_from_dict


def test_dollars_to_cents_conversion():
    assert _dollars_to_cents("0.0900") == 9
    assert _dollars_to_cents("0.9200") == 92
    assert _dollars_to_cents("1.0000") == 100
    assert _dollars_to_cents("0.0000") == 0
    assert _dollars_to_cents(None) is None
    assert _dollars_to_cents("") is None
    assert _dollars_to_cents("garbage") is None


def test_market_from_dict_real_2026_response():
    """Real response shape from /markets/KXHIGHTHOU-26MAY29-B94.5 (May 2026)."""
    raw = {
        "ticker": "KXHIGHTHOU-26MAY29-B94.5",
        "event_ticker": "KXHIGHTHOU-26MAY29",
        "title": "Will the maximum temperature be 94-95° on May 29, 2026?",
        "yes_sub_title": "94° to 95°",
        "status": "active",
        "close_time": "2026-05-30T06:00:00Z",
        "yes_bid_dollars": "0.0800",
        "yes_ask_dollars": "0.0900",
        "no_bid_dollars": "0.9100",
        "no_ask_dollars": "0.9200",
        "last_price_dollars": "0.0900",
        "volume_fp": "1379.28",
        "open_interest_fp": "909.11",
        "rules_primary": "If the maximum temperature recorded at Houston for May 29, 2026...",
    }
    m = _market_from_dict(raw)
    assert m.ticker == "KXHIGHTHOU-26MAY29-B94.5"
    assert m.yes_bid == 8
    assert m.yes_ask == 9
    assert m.no_bid == 91
    assert m.no_ask == 92
    assert m.last_price == 9
    assert m.volume == 1379
    assert m.open_interest == 909
    # Raw dict preserved for the resolver to read rules_primary.
    assert "rules_primary" in m.raw
    assert "Houston" in m.raw["rules_primary"]


def test_market_from_dict_legacy_int_cents_fields():
    """Older clients/tests may still send int-cents fields. Both shapes work."""
    raw = {
        "ticker": "KXHIGHTHOU-26MAY29-B94.5",
        "event_ticker": "KXHIGHTHOU-26MAY29",
        "yes_bid": 8,
        "yes_ask": 9,
        "no_bid": 91,
        "no_ask": 92,
        "volume": 1379,
        "open_interest": 909,
    }
    m = _market_from_dict(raw)
    assert m.yes_bid == 8
    assert m.yes_ask == 9
    assert m.volume == 1379


def test_market_from_dict_missing_prices_gives_none():
    raw = {"ticker": "X", "event_ticker": "Y"}
    m = _market_from_dict(raw)
    assert m.yes_bid is None
    assert m.yes_ask is None
    assert m.volume == 0
    assert m.open_interest == 0
