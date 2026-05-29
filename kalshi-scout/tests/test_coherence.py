"""Unit tests for the cross-bracket coherence pass (invariant I7)."""

from datetime import date

from kalshi_scout.coherence import enforce_coherence
from kalshi_scout.models import (
    Bracket,
    BracketKind,
    ContractEvaluation,
    ContractState,
    KalshiMarket,
    Metric,
    ParsedContract,
)


def _make_eval(
    ticker: str,
    state: ContractState,
    yes_ask: int | None = None,
    bracket: Bracket | None = None,
    fair_lo: float = 0.5,
    fair_hi: float = 0.5,
) -> ContractEvaluation:
    contract = ParsedContract(
        market_ticker=ticker,
        event_ticker="KXLOWHOUSTON-26MAY28",
        city_slug="HOUSTON",
        metric=Metric.LOW,
        market_date=date(2026, 5, 28),
        bracket=bracket or Bracket(BracketKind.BETWEEN, lo=70.0, hi=71.0),
    )
    market = KalshiMarket(
        ticker=ticker,
        event_ticker="KXLOWHOUSTON-26MAY28",
        title="",
        yes_sub_title="",
        status="open",
        close_time=None,
        yes_bid=None,
        yes_ask=yes_ask,
        no_bid=None,
        no_ask=(100 - yes_ask) if yes_ask is not None else None,
        last_price=None,
        volume=0,
        open_interest=0,
    )
    return ContractEvaluation(
        contract=contract,
        market=market,
        state=state,
        reason="",
        fair_prob_low=fair_lo,
        fair_prob_high=fair_hi,
        yes_ask_cents=yes_ask,
        no_ask_cents=(100 - yes_ask) if yes_ask is not None else None,
        edge_yes=None,
        edge_no=None,
        grade="B",
        notes=[],
    )


def test_locked_yes_demotes_all_siblings_to_dead_no():
    evals = [
        _make_eval("a", ContractState.FORECAST_DEPENDENT),
        _make_eval("b", ContractState.LOCKED_YES),
        _make_eval("c", ContractState.BRACKET_HIT_VULNERABLE),
        _make_eval("d", ContractState.NOT_REACHED),
    ]
    out = enforce_coherence(evals)
    assert out[0].state is ContractState.DEAD_NO
    assert out[1].state is ContractState.LOCKED_YES
    assert out[2].state is ContractState.DEAD_NO
    assert out[3].state is ContractState.DEAD_NO
    for e in (out[0], out[2], out[3]):
        assert "b" in e.reason


def test_no_locked_yes_leaves_states_alone():
    evals = [
        _make_eval("a", ContractState.NOT_REACHED),
        _make_eval("b", ContractState.FORECAST_DEPENDENT),
    ]
    out = enforce_coherence(evals)
    assert out[0].state is ContractState.NOT_REACHED
    assert out[1].state is ContractState.FORECAST_DEPENDENT


def test_multiple_locked_yes_flags_inconsistency():
    evals = [
        _make_eval("a", ContractState.LOCKED_YES),
        _make_eval("b", ContractState.LOCKED_YES),
    ]
    out = enforce_coherence(evals)
    # No demotion — both stay LOCKED_YES — but both get an annotation.
    assert out[0].state is ContractState.LOCKED_YES
    assert out[1].state is ContractState.LOCKED_YES
    assert any("settlement-source mismatch" in n for n in out[0].notes)


def test_overpriced_book_flagged():
    """Sum of yes-asks > 105 should trigger overpriced annotation."""
    evals = [
        _make_eval("a", ContractState.NOT_REACHED, yes_ask=40),
        _make_eval("b", ContractState.NOT_REACHED, yes_ask=40),
        _make_eval("c", ContractState.NOT_REACHED, yes_ask=40),
    ]
    out = enforce_coherence(evals)
    assert any("overpriced" in n for n in out[0].notes)


def test_underpriced_book_flagged():
    """Sum of yes-asks < 95 should trigger stale/underpriced annotation."""
    evals = [
        _make_eval("a", ContractState.NOT_REACHED, yes_ask=20),
        _make_eval("b", ContractState.NOT_REACHED, yes_ask=20),
        _make_eval("c", ContractState.NOT_REACHED, yes_ask=20),
    ]
    out = enforce_coherence(evals)
    assert any("underpriced" in n or "stale" in n for n in out[0].notes)


def test_empty_input_returns_empty():
    assert enforce_coherence([]) == []
