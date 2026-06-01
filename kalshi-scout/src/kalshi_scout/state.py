"""State machine: given a parsed contract and the current station state,
decide where the contract sits and what the fair probability range is.

The cleanest A/A+ grades come from settlement-conclusive states (LOCKED_YES,
DEAD_NO). For FORECAST_DEPENDENT and BRACKET_HIT_VULNERABLE states, the
ranker still emits fair-probability ranges, but the grade tops out at B/B+
unless the forecast residual is very small.
"""

from __future__ import annotations

import math
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

    Also queries `station.neighbors` (when defined) for the same window. The
    neighbor data is a cross-check signal and an ASOS-outage fallback only —
    no neighbor reading can lock a contract; only the primary's CLI does.
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

    # -- Neighbor cross-check --------------------------------------------------
    # Per-neighbor failures are tolerated individually; one offline ASOS in
    # the neighbor set must not collapse the primary's path.
    neighbor_max: Optional[float] = None
    neighbor_min: Optional[float] = None
    neighbor_sample_count = 0
    neighbor_icaos_queried: list[str] = []
    if station.neighbors and start_utc < end_utc:
        per_station_max: list[float] = []
        per_station_min: list[float] = []
        for neighbor_icao in station.neighbors:
            try:
                n_obs = nws.observations(neighbor_icao, start=start_utc, end=end_utc)
            except Exception:
                continue
            if not n_obs:
                continue
            neighbor_icaos_queried.append(neighbor_icao)
            neighbor_sample_count += len(n_obs)
            per_station_max.append(max(r.temperature_f for r in n_obs))
            per_station_min.append(min(r.temperature_f for r in n_obs))
        # Aggregate across neighbors: max-of-maxes / min-of-mins gives the
        # most aggressive bound (a neighbor saw it hotter than the primary).
        # The median would be a "consensus" but max/min is what matters for
        # the lock-side risk question. Use the median when computing
        # divergence signal — that's a future enhancement.
        if per_station_max:
            neighbor_max = max(per_station_max)
        if per_station_min:
            neighbor_min = min(per_station_min)

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
        neighbor_running_max_f=neighbor_max,
        neighbor_running_min_f=neighbor_min,
        neighbor_sample_count=neighbor_sample_count,
        neighbor_icaos=tuple(neighbor_icaos_queried),
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


def _lead_hours_to_extremum(
    metric: Metric,
    forecast: list[HourlyPoint],
    state: StationState,
    now_utc: datetime,
) -> Optional[float]:
    """Return hours from `now_utc` to the forecast point that defines the
    projected extremum (max temp for HIGH metric, min for LOW), inside the
    market-day window. Returns None when no forecast point covers the window
    — caller must fall back to the lead-time-agnostic default.

    The lead time of the *extremum* (not the average forecast point) is what
    drives forecast skill on that side of the bracket — a 4pm-peak forecast
    at noon has 4h skill, regardless of the rest of the day's points.
    """
    end_utc = state.window_end.astimezone(timezone.utc)
    in_window = [p for p in forecast if now_utc <= p.start <= end_utc]
    if not in_window:
        return None
    if metric is Metric.HIGH:
        target = max(in_window, key=lambda p: p.temperature_f)
    else:
        target = min(in_window, key=lambda p: p.temperature_f)
    delta = target.start - now_utc
    return max(0.0, delta.total_seconds() / 3600.0)


def project_extremum(
    metric: Metric,
    forecast: Optional[list[HourlyPoint]],
    station_state: StationState,
    now_utc: Optional[datetime] = None,
) -> Optional[float]:
    """Best point estimate of today's daily extremum (max for HIGH, min for LOW).

    Combines what's already observed (`running_max/min_f`) with what the
    forecast says is still ahead. Used by `fair_probability` to size the
    uncertainty band, and stored on each snapshot so the calibration tuner
    can compare projection-vs-realized after settlement.

    Returns None only when neither observation nor forecast offers any
    coverage of the market window.
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    observed = station_state.running_max_f if metric is Metric.HIGH else station_state.running_min_f
    if forecast is None:
        return observed
    f_lo, f_hi = _remaining_extrema_from_forecast(metric, forecast, station_state, now_utc)
    if f_lo is None or f_hi is None:
        return observed
    if metric is Metric.HIGH:
        return max(observed, f_hi) if observed is not None else f_hi
    return min(observed, f_lo) if observed is not None else f_lo


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

    # Compute lead time to the forecast extremum so the residual lookup can
    # use a tighter band for near-settlement trades and a wider one for
    # speculative early-day positions. Falls back to lead-time-agnostic when
    # the forecast covers nothing inside the window (handled below).
    lead_hours: Optional[float] = None
    if forecast is not None:
        lead_hours = _lead_hours_to_extremum(
            contract.metric, forecast, station_state, now_utc
        )

    # Override the default residual with a per-(station, metric) calibrated
    # value when one is available; otherwise use the lead-time tier default
    # (config._residual_for_lead) when lead_hours is known. The calibration
    # gate (sample size) is enforced in config.forecast_residual_for.
    if config is not None and station_state.station is not None:
        forecast_residual_f = config.forecast_residual_for(
            station_icao=station_state.station.icao,
            metric=contract.metric.value,
            lead_hours=lead_hours,
        )
    elif lead_hours is not None:
        # No config object at all — still apply tier defaults so the
        # uncalibrated path benefits from lead-time awareness. Import
        # lazily to avoid a hard dependency in the no-config path.
        from kalshi_scout.config import _residual_for_lead
        forecast_residual_f = _residual_for_lead(lead_hours)

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

    # Use the neighbor fallback when the primary's ASOS is silent — keeps
    # us from collapsing to the no-data prior on transient outages. Never
    # used to lock a contract (LOCKED_YES/DEAD_NO short-circuited above).
    if contract.metric is Metric.HIGH:
        observed = station_state.effective_running_max_f
        proj_lo = max(observed if observed is not None else -999, f_hi) - forecast_residual_f
        proj_hi = max(observed if observed is not None else -999, f_hi) + forecast_residual_f
        lo, hi = _bracket_overlap_prob(bracket, proj_lo, proj_hi)
        return _shifted(lo, hi)

    # LOW
    observed = station_state.effective_running_min_f
    proj_lo = min(observed if observed is not None else 999, f_lo) - forecast_residual_f
    proj_hi = min(observed if observed is not None else 999, f_lo) + forecast_residual_f
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


# -- Ensemble-based fair probability (opt-in, Tier 1B) -----------------------

def fair_probability_from_ensemble(
    contract: ParsedContract,
    station_state: StationState,
    state: ContractState,
    ensemble,                       # list[EnsembleHourlyPoint] — avoid hard import
    now_utc: Optional[datetime] = None,
    regime: Optional[str] = None,
    config: Optional["RankerConfig"] = None,
    min_members: int = 10,
) -> Optional[tuple[float, float]]:
    """Compute fair_prob by counting ensemble members that settle YES.

    Returns the (lo, hi) range using a 95% Wilson-style binomial interval
    around the empirical fraction, so a small ensemble produces a wider
    band. Returns `None` when there's no useful signal — caller must fall
    back to the NWS-only `fair_probability` path. The None cases:

      - LOCKED_YES / DEAD_NO: deterministic; ensemble adds no value, return
        None so the caller uses the eps-locked range.
      - No ensemble points inside the market window.
      - Fewer than `min_members` per point (ensemble too thin to trust).

    Computation:
      For each member, find that member's remaining-window max (HIGH) or
      min (LOW). Combine with observed running extremum to get the per-
      member projected daily extremum. Check whether the bracket would
      contain that value at settlement. The fraction yes ≈ fair_prob.
    """
    # Deterministic states — let the caller handle these with the eps-locked
    # range. Ensemble adds no value once settlement is conclusive.
    if state is ContractState.LOCKED_YES or state is ContractState.DEAD_NO:
        return None
    if not ensemble:
        return None

    now_utc = now_utc or datetime.now(timezone.utc)
    end_utc = station_state.window_end.astimezone(timezone.utc)
    in_window = [p for p in ensemble if now_utc <= p.start <= end_utc]
    if not in_window:
        return None

    # Number of members in the first in-window point. Open-Meteo returns the
    # same count per hour; the parser already drops hours with no valid
    # members. Use min across points so we never index out of bounds on a
    # ragged series.
    n_members = min(len(p.members_f) for p in in_window)
    if n_members < min_members:
        return None

    yes_count = 0
    for m in range(n_members):
        # Per-member remaining extremum inside the window.
        member_max = max(p.members_f[m] for p in in_window)
        member_min = min(p.members_f[m] for p in in_window)
        if contract.metric is Metric.HIGH:
            observed = station_state.effective_running_max_f
            projection = max(observed if observed is not None else -999.0, member_max)
        else:
            observed = station_state.effective_running_min_f
            projection = min(observed if observed is not None else 999.0, member_min)
        if contract.bracket.contains(projection):
            yes_count += 1

    p = yes_count / n_members
    # Wilson 95% CI half-width (rather than naive ±2 SE): well-behaved at
    # p=0 and p=1, doesn't pad the band below 0 / above 1.
    z = 1.96
    denom = 1.0 + z * z / n_members
    center = (p + z * z / (2 * n_members)) / denom
    half = (z / denom) * math.sqrt(
        max(0.0, p * (1 - p) / n_members + z * z / (4 * n_members * n_members))
    )
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)

    # Same regime shift as fair_probability (config.regime_shift_for).
    if config is not None and regime is not None:
        shift = config.regime_shift_for(
            regime=regime,
            metric=contract.metric.value,
            bracket_kind=contract.bracket.kind.value,
        )
        if shift != 0.0:
            lo = max(0.0, min(1.0, lo + shift))
            hi = max(0.0, min(1.0, hi + shift))
    return lo, hi
