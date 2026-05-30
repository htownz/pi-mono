"""Derive a RankerConfig from stored snapshots + settlements.

Two derivations:

  1. Per-tier edge cutoffs:
     For each (state, grade tier), look at the stored snapshots that had
     a known settlement. If N >= MIN_N_PER_TIER and realized hit_rate is
     reasonable, suggest a new cutoff equal to the median |edge| of
     winning trades — that's the level above which historical alerts
     converted to profitable trades.

  2. Per-(regime, metric, bracket-kind) fair-prob shift:
     For each combination with N >= MIN_N_PER_REGIME settled snapshots,
     compute the average bias between the engine's stored fair_prob
     midpoint and the realized outcome (1 for Yes, 0 for No). Positive
     bias means the engine consistently under-predicted Yes; the shift
     adjusts future fair_prob outputs to match historical reality.

Invariant I9: tiers/regimes with N below threshold default to the
existing magic numbers; the report flags them as `applied=False` so the
operator can see exactly which knobs got moved and which didn't.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from statistics import median
from typing import Optional

from kalshi_scout.config import (
    DEFAULT_BRACKET_HIT,
    DEFAULT_DEAD_NO,
    DEFAULT_FORECAST_DEPENDENT,
    DEFAULT_LOCKED_YES,
    ForecastResidual,
    MIN_N_PER_REGIME,
    MIN_N_PER_RESIDUAL,
    MIN_N_PER_TIER,
    RankerConfig,
    RegimeShift,
    StateThresholds,
    regime_key,
    residual_key,
)
from kalshi_scout.models import ContractState
from kalshi_scout.store import SnapshotRow, SnapshotStore, backtest


@dataclass(frozen=True)
class TierTuning:
    """One row of the tuning report — per (state, grade tier)."""
    state: str
    grade: str
    n_settled: int
    n_winners: int
    suggested_cutoff: float
    default_cutoff: float
    applied: bool        # False when N < MIN_N_PER_TIER; default kept
    note: str


@dataclass(frozen=True)
class RegimeTuning:
    """One row of the tuning report — per (regime, metric, bracket-kind)."""
    regime: str
    metric: str
    bracket_kind: str
    n_settled: int
    avg_bias: float       # mean(realized_outcome - fair_prob_mid)
    applied: bool
    note: str


@dataclass(frozen=True)
class ResidualTuning:
    """One row of the tuning report — per (station_icao, metric)."""
    station_icao: str
    metric: str
    n_settled: int
    median_residual_f: float    # median |projected - realized|, in °F
    applied: bool
    note: str


@dataclass(frozen=True)
class TuningReport:
    """The full tuning audit. Returned alongside the derived RankerConfig
    so the operator can see exactly what changed and why."""
    tiers: list[TierTuning]
    regimes: list[RegimeTuning]
    residuals: list[ResidualTuning] = field(default_factory=list)


# -- Threshold derivation ----------------------------------------------------

_STATE_DEFAULTS = {
    ContractState.LOCKED_YES.value: DEFAULT_LOCKED_YES,
    ContractState.DEAD_NO.value: DEFAULT_DEAD_NO,
    ContractState.BRACKET_HIT_VULNERABLE.value: DEFAULT_BRACKET_HIT,
    ContractState.FORECAST_DEPENDENT.value: DEFAULT_FORECAST_DEPENDENT,
    ContractState.NOT_REACHED.value: DEFAULT_FORECAST_DEPENDENT,
}

# Each state has two cutoffs (high/low). The grade labels at those cutoffs
# differ per state — see `config.StateThresholds` docstring.
_STATE_GRADE_LABELS = {
    ContractState.LOCKED_YES.value: {"high_cutoff": "A+", "low_cutoff": "A"},
    ContractState.DEAD_NO.value: {"high_cutoff": "A+", "low_cutoff": "A"},
    ContractState.BRACKET_HIT_VULNERABLE.value: {"high_cutoff": "B+", "low_cutoff": "B"},
    ContractState.FORECAST_DEPENDENT.value: {"high_cutoff": "B", "low_cutoff": "C"},
    ContractState.NOT_REACHED.value: {"high_cutoff": "B", "low_cutoff": "C"},
}


def _edge_for_grade(row: SnapshotRow) -> Optional[float]:
    """The edge that put this snapshot into its grade tier.

    For LOCKED_YES rows, edge_yes is the operative side; for DEAD_NO,
    edge_no. For mixed states, take whichever side has the larger edge.
    """
    state = row.state
    if state == ContractState.LOCKED_YES.value:
        return row.edge_yes
    if state == ContractState.DEAD_NO.value:
        return row.edge_no
    candidates = [e for e in (row.edge_yes, row.edge_no) if e is not None]
    return max(candidates) if candidates else None


def derive_tier_thresholds(
    store: SnapshotStore,
    since: Optional[datetime] = None,
) -> tuple[dict[str, StateThresholds], list[TierTuning]]:
    """For each (state, grade) bucket, suggest a new edge cutoff based on
    realized history. Returns (state->StateThresholds, [TierTuning] report).

    Conservative algorithm:
      - Pull every settled snapshot in this bucket.
      - If N >= MIN_N_PER_TIER:
          suggested_cutoff = median of |edge| across winning trades
      - Else: keep default cutoff (applied=False).
    """
    # Pull all settled snapshots in one pass via backtest()
    settled_rows = backtest(store, min_grade="D", since=since)
    settled_index = {b.snapshot_id: b for b in settled_rows}

    all_snapshots = store.query_snapshots(min_grade="D", since=since)

    report: list[TierTuning] = []
    derived: dict[str, dict[str, float]] = {
        state: {"high_cutoff": default.high_cutoff, "low_cutoff": default.low_cutoff}
        for state, default in _STATE_DEFAULTS.items()
    }

    for state, default in _STATE_DEFAULTS.items():
        labels = _STATE_GRADE_LABELS[state]
        for cutoff_field, grade_label in labels.items():
            default_cutoff = getattr(default, cutoff_field)

            # Filter snapshots: state matches, grade matches, and settlement present.
            bucket = [
                row for row in all_snapshots
                if row.state == state
                and row.grade == grade_label
                and row.id in settled_index
            ]
            n = len(bucket)
            if n == 0:
                report.append(TierTuning(
                    state=state, grade=grade_label, n_settled=0, n_winners=0,
                    suggested_cutoff=default_cutoff, default_cutoff=default_cutoff,
                    applied=False, note="no settled samples",
                ))
                continue

            winners = [row for row in bucket if settled_index[row.id].won]
            winning_edges = [
                abs(e) for row in winners if (e := _edge_for_grade(row)) is not None
            ]
            n_winners = len(winners)

            if n < MIN_N_PER_TIER:
                report.append(TierTuning(
                    state=state, grade=grade_label, n_settled=n,
                    n_winners=n_winners,
                    suggested_cutoff=default_cutoff, default_cutoff=default_cutoff,
                    applied=False,
                    note=f"N={n} below threshold {MIN_N_PER_TIER}; default kept",
                ))
                continue

            if not winning_edges:
                report.append(TierTuning(
                    state=state, grade=grade_label, n_settled=n,
                    n_winners=n_winners,
                    suggested_cutoff=default_cutoff, default_cutoff=default_cutoff,
                    applied=False,
                    note=f"N={n} but zero winners; default kept",
                ))
                continue

            suggested = float(median(winning_edges))
            derived[state][cutoff_field] = suggested
            report.append(TierTuning(
                state=state, grade=grade_label, n_settled=n,
                n_winners=n_winners,
                suggested_cutoff=suggested, default_cutoff=default_cutoff,
                applied=True,
                note=f"N={n}, winners={n_winners}, median edge {suggested:.3f}",
            ))

    thresholds: dict[str, StateThresholds] = {
        state: StateThresholds(**vals) for state, vals in derived.items()
    }
    return thresholds, report


# -- Regime shift derivation -------------------------------------------------

def derive_regime_shifts(
    store: SnapshotStore,
    since: Optional[datetime] = None,
) -> tuple[dict[str, RegimeShift], list[RegimeTuning]]:
    """For each (regime, metric, bracket-kind) triple with enough samples,
    compute the average bias between stored fair_prob and realized outcome.

    A positive bias means historical fair_prob *under-predicted* Yes — the
    engine should bump fair_prob up for that triple in the future.
    """
    settled_rows = backtest(store, min_grade="D", since=since)
    settled_index = {b.snapshot_id: b for b in settled_rows}
    all_snapshots = store.query_snapshots(min_grade="D", since=since)

    # Group settled snapshots by (regime, metric, bracket_kind).
    groups: dict[tuple[str, str, str], list[tuple[float, bool]]] = {}
    for row in all_snapshots:
        if row.id not in settled_index:
            continue
        regime = row.regime or "unknown"
        key = (regime, row.metric, row.bracket_kind)
        fair_mid = (row.fair_prob_low + row.fair_prob_high) / 2.0
        resolved_yes = settled_index[row.id].resolved_yes
        groups.setdefault(key, []).append((fair_mid, resolved_yes))

    shifts: dict[str, RegimeShift] = {}
    report: list[RegimeTuning] = []
    for (regime, metric, kind), samples in groups.items():
        n = len(samples)
        biases = [(1.0 if won else 0.0) - fair_mid for fair_mid, won in samples]
        avg_bias = sum(biases) / n
        if n < MIN_N_PER_REGIME:
            shifts[regime_key(regime, metric, kind)] = RegimeShift.of(
                delta=avg_bias, n=n, applied=False
            )
            report.append(RegimeTuning(
                regime=regime, metric=metric, bracket_kind=kind,
                n_settled=n, avg_bias=avg_bias, applied=False,
                note=f"N={n} below threshold {MIN_N_PER_REGIME}; shift not applied",
            ))
        else:
            shifts[regime_key(regime, metric, kind)] = RegimeShift.of(
                delta=avg_bias, n=n, applied=True
            )
            report.append(RegimeTuning(
                regime=regime, metric=metric, bracket_kind=kind,
                n_settled=n, avg_bias=avg_bias, applied=True,
                note=f"N={n}, applied shift {avg_bias:+.3f}",
            ))

    return shifts, report


# -- Forecast-residual derivation --------------------------------------------

def derive_forecast_residuals(
    store: SnapshotStore,
    since: Optional[datetime] = None,
) -> tuple[dict[str, ForecastResidual], list[ResidualTuning]]:
    """For each (station_icao, metric), compute the median absolute residual
    between the engine's projected daily extremum and the realized CLI value.

    The result feeds `RankerConfig.forecast_residual_for(station, metric)`,
    which `fair_probability` uses to size the projected uncertainty band.

    Per (station, metric) because microclimate matters: KSFO's marine layer
    has different forecast skill than KIAH's humid stagnation, and morning
    lows have different skill than afternoon highs.

    Gates on `MIN_N_PER_RESIDUAL` settled days. Below threshold, the entry
    is stored with applied=False and the engine keeps the 2.0°F default.
    """
    settled_rows = backtest(store, min_grade="D", since=since)
    settled_index = {b.snapshot_id: b for b in settled_rows}
    all_snapshots = store.query_snapshots(min_grade="D", since=since)

    # Group |residual| by (station_icao, metric). One day's max contract
    # contributes one residual; same-day siblings would double-count, so
    # dedupe by (station, metric, market_date).
    seen_days: set[tuple[str, str, str]] = set()
    groups: dict[tuple[str, str], list[float]] = {}
    for row in all_snapshots:
        if row.projected_extremum_f is None or row.station_icao is None:
            continue
        if row.id not in settled_index:
            continue
        # Need the realized CLI value, not just win/loss. Look up the settlement.
        # Settlements are keyed by market_ticker; one settlement per market.
        settlement = store.get_settlement(row.market_ticker)
        if settlement is None or settlement.cli_value_f is None:
            continue
        day_key = (row.station_icao, row.metric, row.market_date.isoformat())
        if day_key in seen_days:
            continue
        seen_days.add(day_key)
        residual = abs(row.projected_extremum_f - settlement.cli_value_f)
        groups.setdefault((row.station_icao, row.metric), []).append(residual)

    residuals: dict[str, ForecastResidual] = {}
    report: list[ResidualTuning] = []
    for (station_icao, metric), values in groups.items():
        n = len(values)
        med = float(median(values))
        applied = n >= MIN_N_PER_RESIDUAL
        residuals[residual_key(station_icao, metric)] = ForecastResidual.of(
            residual_f=med, n=n, applied=applied,
        )
        if applied:
            note = f"N={n}, median |residual| {med:.2f}°F"
        else:
            note = f"N={n} below threshold {MIN_N_PER_RESIDUAL}; default kept"
        report.append(ResidualTuning(
            station_icao=station_icao, metric=metric,
            n_settled=n, median_residual_f=med, applied=applied, note=note,
        ))
    return residuals, report


# -- Top-level entrypoint ----------------------------------------------------

def derive_config(
    store: SnapshotStore,
    since: Optional[datetime] = None,
) -> tuple[RankerConfig, TuningReport]:
    """Build a RankerConfig + auditable TuningReport from stored history."""
    thresholds, tier_report = derive_tier_thresholds(store, since=since)
    shifts, regime_report = derive_regime_shifts(store, since=since)
    residuals, residual_report = derive_forecast_residuals(store, since=since)
    total_snaps = len(store.query_snapshots(min_grade="D", since=since))
    config = RankerConfig(
        generated_at=datetime.now().astimezone(),
        based_on_snapshots=total_snaps,
        locked_yes=thresholds[ContractState.LOCKED_YES.value],
        dead_no=thresholds[ContractState.DEAD_NO.value],
        bracket_hit=thresholds[ContractState.BRACKET_HIT_VULNERABLE.value],
        forecast_dependent=thresholds[ContractState.FORECAST_DEPENDENT.value],
        regime_shifts=shifts,
        forecast_residuals=residuals,
    )
    return config, TuningReport(
        tiers=tier_report, regimes=regime_report, residuals=residual_report,
    )
