"""Grade contract evaluations from raw state + market price.

Grade ladder (deliberately conservative — we'd rather skip a marginal trade
than label noise as A):

    A+  settlement state already proves the answer, and the market is stale
        by >= 8c. Highest-confidence "settlement-recognition" trade.
    A   settlement state already proves the answer; market is stale by 3-8c.
    B+  bracket already hit and forecast says escape is improbable; price stale.
    B   forecast-dependent edge >= 12c with hourly forecast in agreement.
    C   forecast-dependent edge 5-12c.
    D   edge < 5c or spread/liquidity makes it unfillable.
    F   settlement source ambiguous or data missing — never trade.
"""

from __future__ import annotations

from typing import Optional

from kalshi_scout.models import (
    ContractEvaluation,
    ContractState,
    KalshiMarket,
    ParsedContract,
)


def _yes_ask_cents(market: KalshiMarket) -> Optional[int]:
    if market.yes_ask is not None and market.yes_ask > 0:
        return market.yes_ask
    # Derive from No side: yes_ask = 100 - no_bid (if no_bid present)
    if market.no_bid is not None and market.no_bid > 0:
        return 100 - market.no_bid
    return None


def _no_ask_cents(market: KalshiMarket) -> Optional[int]:
    if market.no_ask is not None and market.no_ask > 0:
        return market.no_ask
    if market.yes_bid is not None and market.yes_bid > 0:
        return 100 - market.yes_bid
    return None


def _spread_cents(market: KalshiMarket) -> Optional[int]:
    if market.yes_bid is None or market.yes_ask is None:
        return None
    if market.yes_bid <= 0 or market.yes_ask <= 0:
        return None
    return market.yes_ask - market.yes_bid


def grade(
    contract: ParsedContract,
    market: KalshiMarket,
    state: ContractState,
    reason: str,
    fair_lo: float,
    fair_hi: float,
) -> ContractEvaluation:
    yes_ask = _yes_ask_cents(market)
    no_ask = _no_ask_cents(market)
    fair_mid = (fair_lo + fair_hi) / 2.0
    edge_yes: Optional[float] = None
    edge_no: Optional[float] = None
    if yes_ask is not None:
        edge_yes = fair_mid - yes_ask / 100.0
    if no_ask is not None:
        edge_no = (1.0 - fair_mid) - no_ask / 100.0

    notes: list[str] = []
    spread = _spread_cents(market)
    if spread is not None and spread >= 10:
        notes.append(f"wide spread ({spread}c)")
    if market.volume == 0:
        notes.append("zero volume today")
    if market.open_interest < 50:
        notes.append(f"low open interest ({market.open_interest})")

    g = _grade_value(state, edge_yes, edge_no, spread, notes)

    return ContractEvaluation(
        contract=contract,
        market=market,
        state=state,
        reason=reason,
        fair_prob_low=fair_lo,
        fair_prob_high=fair_hi,
        yes_ask_cents=yes_ask,
        no_ask_cents=no_ask,
        edge_yes=edge_yes,
        edge_no=edge_no,
        grade=g,
        notes=notes,
    )


def _grade_value(
    state: ContractState,
    edge_yes: Optional[float],
    edge_no: Optional[float],
    spread: Optional[int],
    notes: list[str],
) -> str:
    """Apply the ladder. Best edge across Yes/No sides is what matters."""
    best_edge = max(filter(lambda x: x is not None, [edge_yes, edge_no]), default=None)
    wide_spread = spread is not None and spread >= 10

    if state is ContractState.LOCKED_YES:
        # Yes is the right side. Larger edge means more stale.
        if edge_yes is None:
            return "F"
        if edge_yes >= 0.08:
            return "A+" if not wide_spread else "A"
        if edge_yes >= 0.03:
            return "A" if not wide_spread else "B+"
        return "B"

    if state is ContractState.DEAD_NO:
        if edge_no is None:
            return "F"
        if edge_no >= 0.08:
            return "A+" if not wide_spread else "A"
        if edge_no >= 0.03:
            return "A" if not wide_spread else "B+"
        return "B"

    if state is ContractState.BRACKET_HIT_VULNERABLE:
        if best_edge is None:
            return "D"
        if best_edge >= 0.12:
            return "B+" if not wide_spread else "B"
        if best_edge >= 0.05:
            return "B"
        return "C"

    # NOT_REACHED / FORECAST_DEPENDENT
    if best_edge is None:
        return "D"
    if best_edge >= 0.12:
        return "B"
    if best_edge >= 0.05:
        return "C"
    return "D"


_GRADE_RANK = {"A+": 0, "A": 1, "B+": 2, "B": 3, "C": 4, "D": 5, "F": 6}


def sort_key(eval_: ContractEvaluation) -> tuple:
    """Sort key for ranking: grade ascending, then best edge descending."""
    best_edge = max(filter(lambda x: x is not None, [eval_.edge_yes, eval_.edge_no]), default=0.0)
    return (_GRADE_RANK.get(eval_.grade, 99), -best_edge)
