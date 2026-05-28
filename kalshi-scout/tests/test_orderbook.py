"""Tests for V0.6 orderbook depth parsing + fill-quality math."""

from kalshi_scout.orderbook import (
    FillQuote,
    Orderbook,
    OrderbookLevel,
    parse_orderbook,
)


# -- Parsing -----------------------------------------------------------------

def test_parse_orderbook_handles_wrapped_response():
    raw = {
        "orderbook": {
            "yes": [[60, 100], [59, 200]],
            "no":  [[30, 150], [29, 250]],
        }
    }
    book = parse_orderbook(raw, market_ticker="X")
    assert book.market_ticker == "X"
    assert [(l.price_cents, l.size_contracts) for l in book.yes_bids] == [(60, 100), (59, 200)]
    assert [(l.price_cents, l.size_contracts) for l in book.no_bids] == [(30, 150), (29, 250)]


def test_parse_orderbook_unwrapped_form():
    raw = {"yes": [[55, 50]], "no": [[40, 75]]}
    book = parse_orderbook(raw)
    assert len(book.yes_bids) == 1
    assert len(book.no_bids) == 1


def test_parse_orderbook_filters_invalid_levels():
    raw = {
        "orderbook": {
            "yes": [[60, 100], [0, 50], [100, 50], [-5, 10], ["bad", "data"]],
            "no":  [],
        }
    }
    book = parse_orderbook(raw)
    # Only [60, 100] survives: 0 / 100 / negative / non-numeric all rejected.
    assert len(book.yes_bids) == 1
    assert book.yes_bids[0].price_cents == 60


def test_parse_orderbook_empty_returns_empty():
    book = parse_orderbook({})
    assert book.yes_bids == ()
    assert book.no_bids == ()


def test_parse_orderbook_sorts_bids_descending():
    raw = {"yes": [[50, 10], [60, 20], [55, 30]], "no": [[25, 5], [35, 15]]}
    book = parse_orderbook(raw)
    assert [l.price_cents for l in book.yes_bids] == [60, 55, 50]
    assert [l.price_cents for l in book.no_bids] == [35, 25]


# -- Derived asks ------------------------------------------------------------

def test_yes_ask_derived_from_no_bid_ascending():
    """Yes ask = 100 - No bid. Cheapest Yes ask first."""
    raw = {"yes": [], "no": [[30, 100], [29, 200]]}  # no_bids @ 30, 29
    book = parse_orderbook(raw)
    asks = book.yes_asks
    # 100-30=70, 100-29=71. Cheapest first.
    assert [l.price_cents for l in asks] == [70, 71]
    assert [l.size_contracts for l in asks] == [100, 200]


def test_top_ask_returns_cheapest():
    raw = {"yes": [[40, 10]], "no": [[35, 100], [30, 200]]}
    book = parse_orderbook(raw)
    # Yes asks: 100-35=65, 100-30=70. Top yes ask = 65.
    # No asks: 100-40=60. Top no ask = 60.
    assert book.top_ask("yes") == 65
    assert book.top_ask("no") == 60


def test_top_ask_none_when_no_liquidity():
    book = Orderbook(market_ticker="X", yes_bids=(), no_bids=())
    assert book.top_ask("yes") is None
    assert book.top_ask("no") is None


# -- Fillable at size --------------------------------------------------------

def test_fillable_at_size_walks_book_to_completion():
    """100 Yes contracts wanted; book has 60 at 65c + 80 at 67c.
    Avg fill = (60*65 + 40*67) / 100 = (3900 + 2680) / 100 = 65.8"""
    raw = {"yes": [], "no": [[35, 60], [33, 80]]}  # yes_asks 65@60, 67@80
    book = parse_orderbook(raw)
    quote = book.fillable_at_size("yes", 100)
    assert quote is not None
    assert quote.filled_size == 100
    assert quote.requested_size == 100
    assert quote.partial is False
    assert abs(quote.avg_price_cents - 65.8) < 0.01
    assert quote.worst_price_cents == 67


def test_fillable_at_size_partial_fill():
    """200 contracts wanted, only 50 available -> partial."""
    raw = {"yes": [], "no": [[40, 50]]}  # yes_asks 60@50
    book = parse_orderbook(raw)
    quote = book.fillable_at_size("yes", 200)
    assert quote is not None
    assert quote.filled_size == 50
    assert quote.partial is True
    assert quote.avg_price_cents == 60.0


def test_fillable_at_size_none_when_no_liquidity():
    book = Orderbook(market_ticker="X", yes_bids=(), no_bids=())
    assert book.fillable_at_size("yes", 100) is None
    assert book.fillable_at_size("no", 100) is None


def test_fillable_at_size_single_level_exact():
    raw = {"yes": [], "no": [[30, 100]]}  # yes_asks 70@100
    book = parse_orderbook(raw)
    quote = book.fillable_at_size("yes", 100)
    assert quote is not None
    assert quote.filled_size == 100
    assert quote.avg_price_cents == 70.0
    assert quote.partial is False


# -- Edge calculation --------------------------------------------------------

def test_fill_quote_edge_yes_side():
    """Yes fill at avg 70c against fair 0.85 -> edge 0.15."""
    quote = FillQuote(
        side="yes", requested_size=100, filled_size=100,
        avg_price_cents=70.0, worst_price_cents=72, partial=False,
    )
    assert abs(quote.edge_against(0.85) - 0.15) < 1e-9


def test_fill_quote_edge_no_side():
    """No fill at avg 30c against fair 0.10 -> edge (1-0.10)-0.30 = 0.60."""
    quote = FillQuote(
        side="no", requested_size=50, filled_size=50,
        avg_price_cents=30.0, worst_price_cents=30, partial=False,
    )
    assert abs(quote.edge_against(0.10) - 0.60) < 1e-9


def test_fill_quote_negative_edge_when_overpaying():
    """Buying Yes at 90c when fair is 0.50 -> -0.40 edge."""
    quote = FillQuote(
        side="yes", requested_size=10, filled_size=10,
        avg_price_cents=90.0, worst_price_cents=90, partial=False,
    )
    assert quote.edge_against(0.50) < 0
