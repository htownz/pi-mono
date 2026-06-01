"""Runtime configuration for the ranker + state engine.

`RankerConfig` is the calibration artifact V0.9 produces and the engine
consumes. It controls:

  - Edge cutoffs per (state, grade tier) in the ranker grade ladder
  - Additive fair-probability shifts per (regime, metric, bracket-kind)
    triple, applied only to non-deterministic states
    (FORECAST_DEPENDENT / BRACKET_HIT_VULNERABLE)

Defaults preserve V0.3-V0.8 behavior exactly. Loading a config is opt-in
via --config in the CLI; without it, the magic numbers below are used.

Invariant I9 enforcement lives in `tuning.py`: a config is only emitted
with per-tier or per-regime adjustments when the underlying sample size
clears a threshold. Below that, the tier/regime stays at the default and
the tuning report marks it `applied=False`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# -- Defaults (the V0.3-V0.8 magic numbers, named) --------------------------

@dataclass(frozen=True)
class StateThresholds:
    """Two edge cutoffs that bin a snapshot into one of three grade tiers.

    The ranker maps states to specific grade labels at each cutoff:

      state                    high_cutoff -> tier   low_cutoff -> tier   below
      ---------------------    -------------------   ------------------   -------
      LOCKED_YES               A+                    A                    B
      DEAD_NO                  A+                    A                    B
      BRACKET_HIT_VULNERABLE   B+                    B                    C
      FORECAST_DEPENDENT       B                     C                    D
      NOT_REACHED              B                     C                    D

    For LOCKED_YES / DEAD_NO the relevant single-side edge is compared
    (edge_yes for LOCKED_YES, edge_no for DEAD_NO). For the other states,
    max(edge_yes, edge_no) is used.
    """
    high_cutoff: float
    low_cutoff: float


# Default values match the V0.3-V0.8 magic numbers in ranker.py.
DEFAULT_LOCKED_YES = StateThresholds(high_cutoff=0.08, low_cutoff=0.03)
DEFAULT_DEAD_NO = StateThresholds(high_cutoff=0.08, low_cutoff=0.03)
DEFAULT_BRACKET_HIT = StateThresholds(high_cutoff=0.12, low_cutoff=0.05)
DEFAULT_FORECAST_DEPENDENT = StateThresholds(high_cutoff=0.12, low_cutoff=0.05)


@dataclass(frozen=True)
class RegimeShift:
    """Additive fair-probability shift for a (regime, metric, bracket-kind) triple.

    `delta` is signed: +0.05 means bump fair-prob midpoint by 5 percentage
    points (favorable to Yes). Clamped to [-0.20, +0.20] at construction so
    a noisy calibration doesn't move the engine by half-the-spectrum.

    `applied` is False when the underlying sample size was below the tuner's
    threshold; in that case `delta` is always 0.0 and the engine sees no shift.
    """
    delta: float
    n_samples: int
    applied: bool

    @staticmethod
    def of(delta: float, n: int, applied: bool) -> "RegimeShift":
        clamped = max(-0.20, min(0.20, delta))
        return RegimeShift(delta=clamped if applied else 0.0, n_samples=n, applied=applied)


def _regime_key(regime: str, metric: str, bracket_kind: str) -> str:
    return f"{regime}|{metric}|{bracket_kind}"


#: Default uncertainty band half-width for the daily-extremum projection,
#: in °F. The 2.0°F number is the historical V0.3-V0.9 magic constant from
#: state.py; per-station calibration overrides it via ForecastResidual.
#: Used as the lead-time-agnostic fallback when no `lead_hours` is supplied
#: to `forecast_residual_for`.
DEFAULT_FORECAST_RESIDUAL_F = 2.0


#: Lead-time-tiered residual defaults. NWS hourly forecast skill degrades
#: monotonically with horizon — a 2h-out temp forecast typically beats the
#: 12h-out one by a wide margin, but the V0.3-V0.9 model treated them
#: identically (flat 2.0°F). These tiers replace the flat default whenever a
#: lead time is known, tightening the band on near-settlement trades and
#: widening it on speculative early-day positions.
#:
#: Numbers are rough medians of typical NAM/HRRR temp errors at each horizon
#: from public skill stats; per-(station, metric) calibration in tuning.py
#: still wins when sufficient samples exist.
#:
#: Format: ((max_hours_inclusive, residual_f), ...) — first match wins, in
#: order. The final entry's `max_hours_inclusive` is unused (catch-all).
DEFAULT_RESIDUAL_TIERS: tuple[tuple[float, float], ...] = (
    (2.0, 0.8),       # 0-2h out: near settlement, very tight
    (6.0, 1.5),       # 2-6h out: short-term, still good
    (12.0, 2.5),      # 6-12h out: mid-day → afternoon, fair
    (24.0, 3.5),      # 12-24h out: overnight → next-day, loose
    (float("inf"), 4.5),  # 24h+ out: speculative, very loose
)


def _residual_for_lead(lead_hours: float) -> float:
    """Look up the default residual for a forecast lead time, in °F.

    `lead_hours` is the time from now to the forecast point that drives the
    projected extremum. Negative or zero values clamp to the tightest tier.
    """
    if lead_hours < 0:
        lead_hours = 0.0
    for max_h, res in DEFAULT_RESIDUAL_TIERS:
        if lead_hours <= max_h:
            return res
    # Unreachable: the last tier has max_h=inf
    return DEFAULT_RESIDUAL_TIERS[-1][1]


@dataclass(frozen=True)
class ForecastResidual:
    """Per-(station, metric) typical absolute error between the engine's
    projected daily extremum and the realized CLI value, in °F.

    fair_probability uses this to size the projected uncertainty band
    instead of the hard-coded 2.0°F default. Tighter residuals collapse
    wide forecast_dependent fair-prob bands into actionable ones.

    `applied=False` (sample size below threshold) forces the lookup to
    return DEFAULT_FORECAST_RESIDUAL_F so engine output is unchanged.
    """
    residual_f: float
    n_samples: int
    applied: bool

    @staticmethod
    def of(residual_f: float, n: int, applied: bool) -> "ForecastResidual":
        # 0.5°F is the practical floor on NWS hourly skill — tighter than
        # that and we're fitting noise. 10°F means the forecast is useless
        # and we shouldn't be tightening at all.
        clamped = max(0.5, min(10.0, residual_f))
        return ForecastResidual(
            residual_f=clamped if applied else DEFAULT_FORECAST_RESIDUAL_F,
            n_samples=n,
            applied=applied,
        )


def _residual_key(station_icao: str, metric: str) -> str:
    return f"{station_icao}|{metric}"


@dataclass
class RankerConfig:
    """The full set of tunable knobs. Generated by `tuning.derive_config()`."""
    generated_at: datetime
    based_on_snapshots: int
    locked_yes: StateThresholds = field(default_factory=lambda: DEFAULT_LOCKED_YES)
    dead_no: StateThresholds = field(default_factory=lambda: DEFAULT_DEAD_NO)
    bracket_hit: StateThresholds = field(default_factory=lambda: DEFAULT_BRACKET_HIT)
    forecast_dependent: StateThresholds = field(default_factory=lambda: DEFAULT_FORECAST_DEPENDENT)
    regime_shifts: dict[str, RegimeShift] = field(default_factory=dict)
    forecast_residuals: dict[str, ForecastResidual] = field(default_factory=dict)
    #: Opt-in: use Open-Meteo's ensemble forecast to compute fair_prob by
    #: counting members above/below the bracket threshold, instead of the
    #: NWS-only Gaussian-band overlap. The settlement source is unchanged
    #: (still the primary station's CLI). When True and ensemble data is
    #: available, the ensemble fair_prob is used; on failure or insufficient
    #: members, the engine silently falls back to the NWS-only path.
    use_ensemble: bool = False

    @classmethod
    def default(cls) -> "RankerConfig":
        """Returns a config with default thresholds + no regime shifts.

        Equivalent to passing `config=None` everywhere — the ranker and
        state engine produce identical output.
        """
        return cls(
            generated_at=datetime.now(timezone.utc),
            based_on_snapshots=0,
        )

    def thresholds_for(self, state_value: str) -> StateThresholds:
        """Lookup helper used by ranker.py."""
        return {
            "locked_yes": self.locked_yes,
            "dead_no": self.dead_no,
            "bracket_hit_vulnerable": self.bracket_hit,
            "not_reached": self.forecast_dependent,
            "forecast_dependent": self.forecast_dependent,
        }.get(state_value, self.forecast_dependent)

    def regime_shift_for(self, regime: str, metric: str, bracket_kind: str) -> float:
        """Return the signed shift to apply to fair_prob midpoint.

        Zero when:
          - the regime/metric/bracket combo isn't present in the config
          - the calibration's sample size was below threshold (RegimeShift.applied=False)
        """
        shift = self.regime_shifts.get(_regime_key(regime, metric, bracket_kind))
        if shift is None or not shift.applied:
            return 0.0
        return shift.delta

    def forecast_residual_for(
        self,
        station_icao: str,
        metric: str,
        lead_hours: Optional[float] = None,
    ) -> float:
        """Return the °F half-width to use for the projected uncertainty band.

        Resolution order:
          1. Calibrated `(station, metric)` value from backtest, if `applied`.
          2. Lead-time-tiered default via `DEFAULT_RESIDUAL_TIERS`, if
             `lead_hours` is supplied.
          3. The flat 2.0°F default, for backward compatibility when no lead
             time is known.

        The calibrated value still wins even when `lead_hours` is provided —
        per-station backtest evidence beats a generic tier default. A future
        PR can extend the calibrator to be tier-aware, but the gains from
        lead-time-tiering hit hardest in the uncalibrated case (most new
        stations) where the flat 2.0°F was always crude.
        """
        res = self.forecast_residuals.get(_residual_key(station_icao, metric))
        if res is not None and res.applied:
            return res.residual_f
        if lead_hours is not None:
            return _residual_for_lead(lead_hours)
        return DEFAULT_FORECAST_RESIDUAL_F

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at.astimezone(timezone.utc).isoformat(),
            "based_on_snapshots": self.based_on_snapshots,
            "locked_yes": asdict(self.locked_yes),
            "dead_no": asdict(self.dead_no),
            "bracket_hit": asdict(self.bracket_hit),
            "forecast_dependent": asdict(self.forecast_dependent),
            "regime_shifts": {
                k: asdict(v) for k, v in self.regime_shifts.items()
            },
            "forecast_residuals": {
                k: asdict(v) for k, v in self.forecast_residuals.items()
            },
            "use_ensemble": self.use_ensemble,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RankerConfig":
        def _state(key: str, fallback: StateThresholds) -> StateThresholds:
            x = d.get(key)
            return StateThresholds(**x) if x else fallback

        regime_shifts = {}
        for k, v in (d.get("regime_shifts") or {}).items():
            regime_shifts[k] = RegimeShift(
                delta=v.get("delta", 0.0),
                n_samples=v.get("n_samples", 0),
                applied=v.get("applied", False),
            )
        forecast_residuals = {}
        for k, v in (d.get("forecast_residuals") or {}).items():
            # Route through `.of()` so the clamping + applied-flag invariant
            # is enforced even for hand-edited config files; otherwise an
            # operator could load a 99°F residual or store a non-default
            # value on an unapplied entry and the engine would trust it.
            forecast_residuals[k] = ForecastResidual.of(
                residual_f=v.get("residual_f", DEFAULT_FORECAST_RESIDUAL_F),
                n=v.get("n_samples", 0),
                applied=v.get("applied", False),
            )
        generated = d.get("generated_at")
        if generated:
            if generated.endswith("Z"):
                generated = generated[:-1] + "+00:00"
            generated_at = datetime.fromisoformat(generated).astimezone(timezone.utc)
        else:
            generated_at = datetime.now(timezone.utc)
        return cls(
            generated_at=generated_at,
            based_on_snapshots=d.get("based_on_snapshots", 0),
            locked_yes=_state("locked_yes", DEFAULT_LOCKED_YES),
            dead_no=_state("dead_no", DEFAULT_DEAD_NO),
            bracket_hit=_state("bracket_hit", DEFAULT_BRACKET_HIT),
            forecast_dependent=_state("forecast_dependent", DEFAULT_FORECAST_DEPENDENT),
            regime_shifts=regime_shifts,
            forecast_residuals=forecast_residuals,
            use_ensemble=bool(d.get("use_ensemble", False)),
        )

    def save_json(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load_json(cls, path: Path | str) -> "RankerConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


# -- Sample-size thresholds for tuning (invariant I9) -----------------------

#: Minimum samples per (state, grade tier) before we change its edge cutoff.
MIN_N_PER_TIER = 30

#: Minimum samples per (regime, metric, bracket_kind) before we apply a shift.
MIN_N_PER_REGIME = 20

#: Minimum settled days per (station, metric) before we replace the default
#: forecast residual with the calibrated one.
MIN_N_PER_RESIDUAL = 15


def regime_key(regime: str, metric: str, bracket_kind: str) -> str:
    """Public accessor for the canonical regime-shift key shape."""
    return _regime_key(regime, metric, bracket_kind)


def residual_key(station_icao: str, metric: str) -> str:
    """Public accessor for the canonical forecast-residual key shape."""
    return _residual_key(station_icao, metric)
