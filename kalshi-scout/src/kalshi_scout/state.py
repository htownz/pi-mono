"""State machine: given a parsed contract and the current station state,
decide where the contract sits and what the fair probability range is.

The cleanest A/A+ grades come from settlement-conclusive states (LOCKED_YES,
DEAD_NO). For FORECAST_DEPENDENT and BRACKET_HIT_VULNERABLE states, the
ranker still emits fair-probability ranges, but the grade tops out at B/B+
unless the forecast residual is very small.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from kalshi_scout.models import (
    Bracket,
    BracketKind,
    ContractState,
    Metric,
    ParsedContract,
    Station,
    StationReading,
    StationState,
)
from kalshi_scout.nws import HourlyPoint, NwsClient, market_day_window

if TYPE_CHECKING:
    from kalshi_scout.config import RankerConfig


def build_station_state(
    nws: NwsClient,
    station: Station,
    market_date,
    now_utc: Optional[datetime] = None,
) -> StationState:
    """Pull NWS observations + latest CLI and assemble a StationState.

    Crucially, we discard CLI values whose `report_date` does not match the
    market date. Stale CLIs (e.g. yesterday's report still on file at 1 AM)
    are a known trap.
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    window_start_local, window_end_local = market_day_window(market_date, station.tz)

    # Window into UTC for API calls; clamp end to "now" so we don't ask the
    # API for future timestamps.
    end_utc = min(window_end_local.astimezone(timezone.utc), now_utc)
    start_utc = window_start_local.astimezone(timezone.utc)

    obs: list[StationReading] = []
    if start_utc < end_utc:
        try:
            obs = nws.observations(station.icao, start=start_utc, end=end_utc)
        except Exception:
            obs = []

    running_max: Optional[float] = max((r.temperature_f for r in obs), default=None)
    running_min: Optional[float] = min((r.temperature_f for r in obs), default=None)

    latest = obs[-1] if obs else None
    if latest is None:
        try:
            latest = nws.latest_observation(station.icao)
        except Exception:
            latest = None

    cli_report_date = None
    cli_max_f: Optional[float] = None
    cli_min_f: Optional[float] = None
    try:
        cli = nws.latest_cli(station.cli_product)
    except Exception:
        cli = None
    if cli is not None and cli.report_date == market_date:
        cli_report_date = cli.report_date
        cli_max_f = cli.max_f
        cli_min_f = cli.min_f
        # If the CLI matches the market date, treat its values as more
        # authoritative than the running 5-min observations (it's the
        # official non-preliminary settlement value per Kalshi rules).
        if cli_max_f is not None:
            running_max = cli_max_f if running_max is None else max(running_max, cli_max_f)
        if cli_min_f is not None:
            running_min = cli_min_f if running_min is None else min(running_min, cli_min_f)

    return StationState(
        station=station,
        market_date=market_date,
        window_start=window_start_local,
        window_end=window_end_local,
        running_max_f=running_max,
        running_min_f=running_min,
        latest=latest,
        cli_report_date=cli_report_date,
        cli_max_f=cli_max_f,
        cli_min_f=cli_min_f,
        observations=obs,
    )


def _high_state(bracket: Bracket, running_max: Optional[float]) -> tuple[ContractState, str]:
    """Classify a HIGH-temp contract given the running max so far.

    Highs can only go up during the day's heating phase; once `running_max`
    crosses an "above"-side threshold, that side is settlement-locked. The
    "below"-side cannot be settlement-locked from the high alone (the day's
    final max could still climb), but once running_max exceeds the upper
    threshold, the below-side is dead.
    """
    if running_max is None:
        return ContractState.FORECAST_DEPENDENT, "no station obs yet"

    k = bracket.kind
    if k is BracketKind.GTE:
        assert bracket.lo is not None
        if running_max >= bracket.lo:
            return ContractState.LOCKED_YES, f"observed max {running_max:g} ≥ {bracket.lo:g}"
        return ContractState.NOT_REACHED, f"observed max {running_max:g} < {bracket.lo:g}, needs more heating"

    if k is BracketKind.GT:
        assert bracket.lo is not None
        if running_max > bracket.lo:
            return ContractState.LOCKED_YES, f"observed max {running_max:g} > {bracket.lo:g}"
        return ContractState.NOT_REACHED, f"observed max {running_max:g} ≤ {bracket.lo:g}, strictly above strike"

    if k is BracketKind.LTE:
        assert bracket.hi is not None
        if running_max > bracket.hi:
            return ContractState.DEAD_NO, f"observed max {running_max:g} > {bracket.hi:g}, cannot recover"
        return ContractState.FORECAST_DEPENDENT, f"observed max {running_max:g} ≤ {bracket.hi:g}, but day not over"

    if k is BracketKind.LT:
        assert bracket.hi is not None
        if running_max >= bracket.hi:
            return ContractState.DEAD_NO, f"observed max {running_max:g} ≥ {bracket.hi:g}, strictly-below side dead"
        return ContractState.FORECAST_DEPENDENT, f"observed max {running_max:g} < {bracket.hi:g}, but day not over"

    if k is BracketKind.EQ:
        # "Exactly X" on a HIGH: dead once max > X; otherwise still in play.
        assert bracket.lo is not None
        if running_max > bracket.lo:
            return ContractState.DEAD_NO, f"observed max {running_max:g} > {bracket.lo:g}, cannot settle exactly"
        return ContractState.FORECAST_DEPENDENT, f"observed max {running_max:g} ≤ {bracket.lo:g}, still possible to settle exactly"

    # BETWEEN
    assert bracket.lo is not None and bracket.hi is not None
    if running_max > bracket.hi:
        return ContractState.DEAD_NO, f"observed max {running_max:g} > bracket {bracket.lo:g}–{bracket.hi:g}, blown through"
    if running_max >= bracket.lo:
        return (
            ContractState.BRACKET_HIT_VULNERABLE,
            f"observed max {running_max:g} inside {bracket.lo:g}–{bracket.hi:g}; remaining risk: reaching {bracket.hi + 1:g}+",
        )
    return ContractState.NOT_REACHED, f"observed max {running_max:g} below bracket; bracket not yet reached"


def _low_state(bracket: Bracket, running_min: Optional[float]) -> tuple[ContractState, str]:
    """Classify a LOW-temp contract given the running min so far.

    Mirror of `_high_state`: a minimum can't un-happen, so once running_min
    crosses a "below"-side threshold, that side is settlement-locked.
    """
    if running_min is None:
        return ContractState.FORECAST_DEPENDENT, "no station obs yet"

    k = bracket.kind
    if k is BracketKind.LTE:
        assert bracket.hi is not None
        if running_min <= bracket.hi:
            return ContractState.LOCKED_YES, f"observed min {running_min:g} ≤ {bracket.hi:g}"
        return ContractState.FORECAST_DEPENDENT, f"observed min {running_min:g} > {bracket.hi:g}, still cooling potential"

    if k is BracketKind.LT:
        assert bracket.hi is not None
        if running_min < bracket.hi:
            return ContractState.LOCKED_YES, f"observed min {running_min:g} < {bracket.hi:g}"
        return ContractState.FORECAST_DEPENDENT, f"observed min {running_min:g} ≥ {bracket.hi:g}, needs more cooling"

    if k is BracketKind.GTE:
        assert bracket.lo is not None
        if running_min < bracket.lo:
            return ContractState.DEAD_NO, f"observed min {running_min:g} < {bracket.lo:g}, cannot recover"
        return ContractState.FORECAST_DEPENDENT, f"observed min {running_min:g} ≥ {bracket.lo:g} so far; more cooling risk"

    if k is BracketKind.GT:
        assert bracket.lo is not None
        if running_min <= bracket.lo:
            return ContractState.DEAD_NO, f"observed min {running_min:g} ≤ {bracket.lo:g}, strictly-above side dead"
        return ContractState.FORECAST_DEPENDENT, f"observed min {running_min:g} > {bracket.lo:g} so far; more cooling risk"

    if k is BracketKind.EQ:
        assert bracket.lo is not None
        if running_min < bracket.lo:
            return ContractState.DEAD_NO, f"observed min {running_min:g} < {bracket.lo:g}, cannot settle exactly"
        return ContractState.FORECAST_DEPENDENT, f"observed min {running_min:g} ≥ {bracket.lo:g}, still possible to settle exactly"

    # BETWEEN
    assert bracket.lo is not None and bracket.hi is not None
    if running_min < bracket.lo:
        return ContractState.DEAD_NO, f"observed min {running_min:g} < bracket {bracket.lo:g}–{bracket.hi:g}, dropped below"
    if running_min <= bracket.hi:
        return (
            ContractState.BRACKET_HIT_VULNERABLE,
            f"observed min {running_min:g} inside {bracket.lo:g}–{bracket.hi:g}; remaining risk: dropping to {bracket.lo - 1:g} or lower",
        )
    return ContractState.NOT_REACHED, f"observed min {running_min:g} above bracket; needs more cooling"


def classify(contract: ParsedContract, state: StationState) -> tuple[ContractState, str]:
    if contract.metric is Metric.HIGH:
        return _high_state(contract.bracket, state.running_max_f)
    return _low_state(contract.bracket, state.running_min_f)


# -- Forecast-dependent fair probability ----------------------------------------

def _remaining_extrema_from_forecast(
    metric: Metric,
    forecast: list[HourlyPoint],
    state: StationState,
    now_utc: datetime,
) -> tuple[Optional[float], Optional[float]]:
    """Given the hourly forecast, estimate the [low, high] of the remaining
    extreme inside the market day.

    For HIGH: returns the (min, max) of forecast hourly temps from now to
    window_end — the daily max if not yet observed is bounded between these.

    For LOW: same bounds, applied to the cooling side.
    """
    end_utc = state.window_end.astimezone(timezone.utc)
    in_window = [
        p for p in forecast if now_utc <= p.start <= end_utc
    ]
    if not in_window:
        return None, None
    temps = [p.temperature_f for p in in_window]
    return min(temps), max(temps)


def fair_probability(
    contract: ParsedContract,
    station_state: StationState,
    state: ContractState,
    forecast: Optional[list[HourlyPoint]],
    now_utc: Optional[datetime] = None,
    forecast_residual_f: float = 2.0,
    regime: Optional[str] = None,
    config: Optional["RankerConfig"] = None,
) -> tuple[float, float]:
    """Return a (low, high) fair-probability range for the Yes side.

    Conclusive states (LOCKED_YES / DEAD_NO) map to deterministic ranges
    with a tiny epsilon for settlement-source risk — no regime shift is
    applied to these.

    For non-deterministic states (BRACKET_HIT_VULNERABLE / FORECAST_DEPENDENT
    / NOT_REACHED), the result is shifted by the regime-specific delta from
    `config.regime_shift_for(regime, metric, bracket_kind)`. The shift is
    additive and clamped so both bounds stay in [0, 1].
    """
    eps = 0.02
    # Deterministic states get no regime shift — the snapshot store reflects
    # settlement-conclusive evidence, not a prediction.
    if state is ContractState.LOCKED_YES:
        return 1.0 - eps, 1.0
    if state is ContractState.DEAD_NO:
        return 0.0, eps

    def _shifted(lo: float, hi: float) -> tuple[float, float]:
        """Apply regime shift to a non-deterministic fair-prob output."""
        if config is None or regime is None:
            return lo, hi
        shift = config.regime_shift_for(
            regime=regime,
            metric=contract.metric.value,
            bracket_kind=contract.bracket.kind.value,
        )
        if shift == 0.0:
            return lo, hi
        return max(0.0, min(1.0, lo + shift)), max(0.0, min(1.0, hi + shift))

    now_utc = now_utc or datetime.now(timezone.utc)
    bracket = contract.bracket

    if state is ContractState.BRACKET_HIT_VULNERABLE:
        # Already in the bracket (only BETWEEN kinds reach this state); need
        # to estimate prob the path escapes by day's end.
        if forecast is None:
            return _shifted(0.45, 0.85)
        f_lo, f_hi = _remaining_extrema_from_forecast(contract.metric, forecast, station_state, now_utc)
        if f_lo is None or f_hi is None:
            return _shifted(0.45, 0.85)
        if contract.metric is Metric.HIGH:
            assert bracket.hi is not None
            escape_above = bracket.hi + 1
            margin = f_hi + forecast_residual_f - escape_above
            # margin > 0 means forecast can plausibly escape; degrade prob
            p = 0.92 if margin < -3 else 0.75 if margin < -1 else 0.55 if margin < 1 else 0.35
            return _shifted(max(0.0, p - 0.08), min(1.0, p + 0.08))
        else:
            assert bracket.lo is not None
            escape_below = bracket.lo - 1
            margin = escape_below - (f_lo - forecast_residual_f)
            p = 0.92 if margin < -3 else 0.75 if margin < -1 else 0.55 if margin < 1 else 0.35
            return _shifted(max(0.0, p - 0.08), min(1.0, p + 0.08))

    # NOT_REACHED or FORECAST_DEPENDENT
    if forecast is None:
        return _shifted(0.25, 0.75)
    f_lo, f_hi = _remaining_extrema_from_forecast(contract.metric, forecast, station_state, now_utc)
    if f_lo is None or f_hi is None:
        return _shifted(0.25, 0.75)

    if contract.metric is Metric.HIGH:
        # Projected day-high is max(running_max, max(forecast in window))
        proj_lo = max(station_state.running_max_f or -999, f_hi) - forecast_residual_f
        proj_hi = max(station_state.running_max_f or -999, f_hi) + forecast_residual_f
        lo, hi = _bracket_overlap_prob(bracket, proj_lo, proj_hi)
        return _shifted(lo, hi)

    # LOW
    proj_lo = min(station_state.running_min_f or 999, f_lo) - forecast_residual_f
    proj_hi = min(station_state.running_min_f or 999, f_lo) + forecast_residual_f
    lo, hi = _bracket_overlap_prob(bracket, proj_lo, proj_hi)
    return _shifted(lo, hi)


def _bracket_overlap_prob(bracket: Bracket, proj_lo: float, proj_hi: float) -> tuple[float, float]:
    """Crude fair-prob estimator: how much of the projected uncertainty band
    falls inside the bracket. Returns (low, high) bounds for the estimate.
    """
    if proj_hi <= proj_lo:
        proj_hi = proj_lo + 1.0
    span = proj_hi - proj_lo
    k = bracket.kind
    if k in (BracketKind.GTE, BracketKind.GT):
        assert bracket.lo is not None
        overlap = max(0.0, proj_hi - bracket.lo)
        center = min(1.0, overlap / span)
    elif k in (BracketKind.LTE, BracketKind.LT):
        assert bracket.hi is not None
        overlap = max(0.0, bracket.hi - proj_lo)
        center = min(1.0, overlap / span)
    elif k is BracketKind.EQ:
        # Very narrow target — probability concentrated near a single value.
        assert bracket.lo is not None
        target = bracket.lo
        if proj_lo <= target <= proj_hi:
            # Treat a 1°-wide window around target as the favorable region.
            center = min(1.0, 1.0 / span)
        else:
            center = 0.0
    else:
        assert bracket.lo is not None and bracket.hi is not None
        lo = max(proj_lo, bracket.lo)
        hi = min(proj_hi, bracket.hi)
        overlap = max(0.0, hi - lo)
        center = min(1.0, overlap / span)
    pad = 0.10
    return max(0.0, center - pad), min(1.0, center + pad)
