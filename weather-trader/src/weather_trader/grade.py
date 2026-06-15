"""Turn a forecast distribution + market quote into an edge and a grade.

Fair value of a Yes contract = the forecast's probability the bracket settles
true (`mid`). Edge is fair value minus what it costs to take the position:

    edge_yes = mid           - yes_ask/100
    edge_no  = (1 - mid)     - no_ask/100

The grade reflects edge x forecast confidence x fillability. Because the
forecaster folds in observed-so-far truth, a fully-observed day produces a
`locked` distribution (mid = 0 or 1) — the highest-confidence signal, graded
A+/A when the market price still leaves money on the table.
"""

from __future__ import annotations

from typing import Optional

from weather_trader.forecast import ForecastDistribution
from weather_trader.models import BracketKind, Contract, Evaluation, KalshiMarket

GRADE_ORDER = ["A+", "A", "B+", "B", "C", "D", "F"]

# -- Ladder thresholds ------------------------------------------------------------
WIDE_SPREAD_CENTS = 10          # >= this yes_ask-yes_bid spread is "wide"
LOCK_HIGH_EDGE = 0.08           # locked + this edge -> A+
LOCK_LOW_EDGE = 0.03            # locked + this edge -> A
EDGE_BPLUS = 0.15               # forecast-dependent edge tiers
EDGE_B = 0.12
EDGE_C = 0.05
NARROW_BAND_F = 4.0             # q90-q10 spread (°F) below this = high agreement
BROAD_BAND_F = 9.0              # above this = low agreement

# Humility guard: an edge this large against a live quote, when it is NOT
# forced by observed data, is almost always the model being wrong/stale —
# most often a near-settled same-day market whose outcome is effectively known
# and that the forecast hasn't collapsed onto. Liquid markets don't leave 40c+
# on the table. Such edges are capped out of tradable grades. Observation-
# forced (path-locked) edges are real and pass through untouched.
SUSPECT_EDGE = 0.40


def _up(grade: str) -> str:
    i = GRADE_ORDER.index(grade)
    return GRADE_ORDER[max(0, i - 1)]


def _down(grade: str) -> str:
    i = GRADE_ORDER.index(grade)
    return GRADE_ORDER[min(len(GRADE_ORDER) - 1, i + 1)]


def _grade(
    *,
    locked: bool,
    best_edge: Optional[float],
    band_width_f: Optional[float],
    spread_cents: Optional[int],
    volume: int,
) -> str:
    if best_edge is None:
        return "F"
    wide = spread_cents is not None and spread_cents >= WIDE_SPREAD_CENTS
    illiquid = volume == 0

    if locked:
        if best_edge >= LOCK_HIGH_EDGE:
            return "A" if (wide or illiquid) else "A+"
        if best_edge >= LOCK_LOW_EDGE:
            return "B+" if (wide or illiquid) else "A"
        if best_edge > 0:
            return "B"
        return "D"

    if best_edge <= 0:
        return "D"
    if best_edge >= EDGE_BPLUS:
        g = "B+"
    elif best_edge >= EDGE_B:
        g = "B"
    elif best_edge >= EDGE_C:
        g = "C"
    else:
        g = "D"

    narrow = band_width_f is not None and band_width_f <= NARROW_BAND_F
    broad = band_width_f is not None and band_width_f >= BROAD_BAND_F
    if broad:
        g = _down(g)
    elif narrow and g in ("B", "C"):
        g = _up(g)
    if wide or illiquid:
        g = _down(g)
    return g


def _observed_settles(contract: Contract, observed: Optional[float]) -> bool:
    """True if the observed extremum ALONE already determines the outcome.

    Path dependence: a daily high can only rise from the running max, a low can
    only fall from the running min. So some outcomes are locked mid-day before
    the window closes (e.g. high already ≥ 80 settles "80°+" YES). These are
    real, observation-forced edges — the humility guard must not touch them.
    """
    if observed is None:
        return False
    b = contract.bracket
    k = b.kind
    if contract.metric.is_high:
        mx = observed
        if k is BracketKind.GTE:
            return mx >= b.lo          # already at/above -> YES
        if k is BracketKind.GT:
            return mx > b.lo
        if k is BracketKind.LTE:
            return mx > b.hi           # already exceeded -> NO
        if k is BracketKind.LT:
            return mx >= b.hi
        if k is BracketKind.BETWEEN:
            return mx > b.hi           # above the bracket -> NO
        if k is BracketKind.EQ:
            return mx > b.lo           # overshot -> can't equal -> NO
        return False
    mn = observed
    if k is BracketKind.LTE:
        return mn <= b.hi              # already at/below -> YES
    if k is BracketKind.LT:
        return mn < b.hi
    if k is BracketKind.GTE:
        return mn < b.lo               # already below -> NO
    if k is BracketKind.GT:
        return mn <= b.lo
    if k is BracketKind.BETWEEN:
        return mn < b.lo               # below the bracket -> NO
    if k is BracketKind.EQ:
        return mn < b.lo               # undershot -> can't equal -> NO
    return False


def evaluate(contract: Contract, market: KalshiMarket, dist: ForecastDistribution) -> Evaluation:
    """Grade a single contract against its forecast distribution."""
    notes: list[str] = []

    if not dist.usable:
        notes.append("no usable forecast (no obs/forecast data)")
        return Evaluation(
            contract=contract, market=market,
            fair_prob_low=0.0, fair_prob_high=1.0, fair_prob_mid=0.5,
            forecast_mean_f=None, band_width_f=None, locked=False,
            yes_ask_cents=market.yes_ask, no_ask_cents=market.no_ask,
            edge_yes=None, edge_no=None, grade="F", notes=notes,
        )

    lo, mid, hi = dist.prob_bracket(contract.bracket)
    edge_yes = (mid - market.yes_ask / 100.0) if market.yes_ask is not None else None
    edge_no = ((1.0 - mid) - market.no_ask / 100.0) if market.no_ask is not None else None
    band = dist.band_width_f()
    spread = (
        market.yes_ask - market.yes_bid
        if (market.yes_ask is not None and market.yes_bid is not None)
        else None
    )

    eval_ = Evaluation(
        contract=contract, market=market,
        fair_prob_low=lo, fair_prob_high=hi, fair_prob_mid=mid,
        forecast_mean_f=dist.mean(), band_width_f=band, locked=dist.locked,
        yes_ask_cents=market.yes_ask, no_ask_cents=market.no_ask,
        edge_yes=edge_yes, edge_no=edge_no, grade="F", notes=notes,
    )
    eval_.grade = _grade(
        locked=dist.locked, best_edge=eval_.best_edge,
        band_width_f=band, spread_cents=spread, volume=market.volume,
    )

    # Humility guard: a giant edge that is neither settled-by-day-close nor
    # forced by the observed extremum is almost certainly the forecast
    # disagreeing with a near-settled market — model error, not alpha. Cap it
    # out of tradable grades; observation-forced edges pass through.
    if (
        eval_.best_edge is not None
        and eval_.best_edge >= SUSPECT_EDGE
        and not dist.locked
        and not _observed_settles(contract, dist.observed_extremum_f)
    ):
        if GRADE_ORDER.index(eval_.grade) < GRADE_ORDER.index("D"):
            eval_.grade = "D"
        notes.append(
            f"⚠ implausible edge {eval_.best_edge * 100:+.0f}c not backed by observed "
            "data — likely model error / near-settled market, not a signal"
        )

    if dist.locked:
        notes.append("locked: outcome determined by observed data")
    if eval_.best_side is not None and eval_.best_edge is not None:
        notes.append(f"best: {eval_.best_side} edge {eval_.best_edge * 100:+.1f}c")
    if band is not None:
        notes.append(f"band(q10-q90)={band:.1f}°F")
    if dist.notes:
        notes.append("; ".join(dist.notes))
    return eval_


def sort_key(e: Evaluation):
    """Rank best-first: by grade, then by best edge descending."""
    gi = GRADE_ORDER.index(e.grade) if e.grade in GRADE_ORDER else len(GRADE_ORDER)
    edge = e.best_edge if e.best_edge is not None else -1.0
    return (gi, -edge)
