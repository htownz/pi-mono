"""Tests for V0.9 RankerConfig: JSON round-trip, defaults, regime lookup."""

from datetime import datetime, timezone
from pathlib import Path

from kalshi_scout.config import (
    DEFAULT_BRACKET_HIT,
    DEFAULT_FORECAST_DEPENDENT,
    DEFAULT_FORECAST_RESIDUAL_F,
    DEFAULT_LOCKED_YES,
    ForecastResidual,
    RankerConfig,
    RegimeShift,
    StateThresholds,
    regime_key,
    residual_key,
)


def test_default_config_matches_pre_v09_magic_numbers():
    """Sanity: defaults exactly match the V0.3-V0.8 cutoffs in ranker.py."""
    cfg = RankerConfig.default()
    assert cfg.locked_yes.high_cutoff == 0.08
    assert cfg.locked_yes.low_cutoff == 0.03
    assert cfg.bracket_hit.high_cutoff == 0.12
    assert cfg.bracket_hit.low_cutoff == 0.05
    assert cfg.forecast_dependent.high_cutoff == 0.12
    assert cfg.forecast_dependent.low_cutoff == 0.05


def test_regime_shift_zero_when_not_applied():
    """A shift built with applied=False always returns delta=0."""
    shift = RegimeShift.of(delta=0.10, n=5, applied=False)
    assert shift.delta == 0.0


def test_regime_shift_clamped_to_20_percent():
    """A delta outside [-0.20, +0.20] is clamped at construction."""
    shift = RegimeShift.of(delta=0.50, n=100, applied=True)
    assert shift.delta == 0.20
    shift = RegimeShift.of(delta=-0.99, n=100, applied=True)
    assert shift.delta == -0.20


def test_regime_shift_for_returns_zero_when_missing():
    cfg = RankerConfig.default()
    assert cfg.regime_shift_for("rain_cooled", "high", "lte") == 0.0


def test_regime_shift_for_returns_zero_when_unapplied():
    cfg = RankerConfig.default()
    cfg.regime_shifts[regime_key("rain_cooled", "high", "lte")] = RegimeShift.of(
        delta=0.10, n=5, applied=False
    )
    assert cfg.regime_shift_for("rain_cooled", "high", "lte") == 0.0


def test_regime_shift_for_returns_delta_when_applied():
    cfg = RankerConfig.default()
    cfg.regime_shifts[regime_key("rain_cooled", "high", "lte")] = RegimeShift.of(
        delta=0.05, n=50, applied=True
    )
    assert cfg.regime_shift_for("rain_cooled", "high", "lte") == 0.05


def test_thresholds_for_dispatches_by_state():
    cfg = RankerConfig.default()
    assert cfg.thresholds_for("locked_yes") == DEFAULT_LOCKED_YES
    assert cfg.thresholds_for("bracket_hit_vulnerable") == DEFAULT_BRACKET_HIT
    assert cfg.thresholds_for("forecast_dependent") == DEFAULT_FORECAST_DEPENDENT
    # Unknown state falls back to forecast-dependent defaults.
    assert cfg.thresholds_for("nonsense") == DEFAULT_FORECAST_DEPENDENT


def test_json_round_trip(tmp_path: Path):
    cfg = RankerConfig(
        generated_at=datetime(2026, 5, 30, 12, 0, tzinfo=timezone.utc),
        based_on_snapshots=1234,
        locked_yes=StateThresholds(high_cutoff=0.05, low_cutoff=0.02),
        regime_shifts={
            regime_key("rain_cooled", "high", "lte"): RegimeShift.of(0.04, 50, True),
            regime_key("marine_layer", "high", "between"): RegimeShift.of(-0.03, 12, False),
        },
    )
    path = tmp_path / "cfg.json"
    cfg.save_json(path)
    loaded = RankerConfig.load_json(path)

    assert loaded.based_on_snapshots == 1234
    assert loaded.locked_yes.high_cutoff == 0.05
    # Applied shift round-trips with delta intact.
    assert loaded.regime_shift_for("rain_cooled", "high", "lte") == 0.04
    # Unapplied shift round-trips as zero (its delta was zeroed by RegimeShift.of).
    assert loaded.regime_shift_for("marine_layer", "high", "between") == 0.0


# -- ForecastResidual --------------------------------------------------------

def test_forecast_residual_for_returns_default_when_missing():
    """No calibrated entry for this (station, metric) → fall back to 2.0°F."""
    cfg = RankerConfig.default()
    assert cfg.forecast_residual_for("KSFO", "high") == DEFAULT_FORECAST_RESIDUAL_F


def test_forecast_residual_for_returns_default_when_unapplied():
    """Entry exists but sample size is below threshold → ignore it."""
    cfg = RankerConfig.default()
    cfg.forecast_residuals[residual_key("KSFO", "high")] = ForecastResidual.of(
        residual_f=1.2, n=5, applied=False,
    )
    assert cfg.forecast_residual_for("KSFO", "high") == DEFAULT_FORECAST_RESIDUAL_F


def test_forecast_residual_for_returns_calibrated_when_applied():
    cfg = RankerConfig.default()
    cfg.forecast_residuals[residual_key("KSFO", "high")] = ForecastResidual.of(
        residual_f=1.2, n=50, applied=True,
    )
    assert cfg.forecast_residual_for("KSFO", "high") == 1.2


def test_forecast_residual_clamps_floor_and_ceiling():
    """0.3°F is below the practical NWS floor; 99°F means the forecast is
    useless. Both get clamped to keep downstream math sane."""
    too_tight = ForecastResidual.of(residual_f=0.3, n=50, applied=True)
    assert too_tight.residual_f == 0.5
    too_loose = ForecastResidual.of(residual_f=99.0, n=50, applied=True)
    assert too_loose.residual_f == 10.0


def test_forecast_residual_unapplied_stores_default_value():
    """When applied=False, the stored residual_f is the default — so any
    later read that bypasses the .applied guard still sees a sane number."""
    res = ForecastResidual.of(residual_f=1.5, n=3, applied=False)
    assert res.residual_f == DEFAULT_FORECAST_RESIDUAL_F


def test_forecast_residual_from_dict_enforces_clamping_and_applied_invariant():
    """Regression for the Copilot finding on PR #4: a hand-edited config
    that loads a 99°F residual or stores a non-default value on an
    unapplied entry must NOT bypass `ForecastResidual.of()` guards.
    `from_dict` routes through `.of()` so the invariants hold."""
    cfg = RankerConfig.from_dict({
        "forecast_residuals": {
            "KSFO|high": {"residual_f": 99.0, "n_samples": 50, "applied": True},
            "KIAH|low":  {"residual_f": 1.2,  "n_samples": 5,  "applied": False},
        },
    })
    # Out-of-range residual clamped to 10°F ceiling.
    assert cfg.forecast_residuals["KSFO|high"].residual_f == 10.0
    # Unapplied entry forced back to default — `applied=False` invariant.
    assert cfg.forecast_residuals["KIAH|low"].residual_f == DEFAULT_FORECAST_RESIDUAL_F


def test_forecast_residual_for_uses_lead_time_tiers_when_no_calibration():
    """No calibrated entry + lead_hours supplied → tier default applies.

    Tier table: (≤2h, 0.8), (≤6h, 1.5), (≤12h, 2.5), (≤24h, 3.5), (∞, 4.5).
    """
    cfg = RankerConfig.default()
    assert cfg.forecast_residual_for("KSFO", "high", lead_hours=0.5) == 0.8
    assert cfg.forecast_residual_for("KSFO", "high", lead_hours=2.0) == 0.8
    assert cfg.forecast_residual_for("KSFO", "high", lead_hours=4.0) == 1.5
    assert cfg.forecast_residual_for("KSFO", "high", lead_hours=10.0) == 2.5
    assert cfg.forecast_residual_for("KSFO", "high", lead_hours=20.0) == 3.5
    assert cfg.forecast_residual_for("KSFO", "high", lead_hours=48.0) == 4.5


def test_forecast_residual_for_calibrated_wins_over_tier_default():
    """A calibrated (applied) entry beats the tier default even with lead_hours."""
    cfg = RankerConfig.default()
    cfg.forecast_residuals[residual_key("KSFO", "high")] = ForecastResidual.of(
        residual_f=1.2, n=50, applied=True,
    )
    # Without lead_hours: calibrated value.
    assert cfg.forecast_residual_for("KSFO", "high") == 1.2
    # With lead_hours that would normally pick 4.5°F tier — calibrated still wins.
    assert cfg.forecast_residual_for("KSFO", "high", lead_hours=48.0) == 1.2


def test_forecast_residual_for_lead_hours_none_preserves_legacy_default():
    """The lead-time tier system is purely additive — when `lead_hours` is
    None, the engine sees exactly the historical 2.0°F default (or the
    calibrated value when present). No grade-line drift on existing callers
    that haven't opted in."""
    cfg = RankerConfig.default()
    assert cfg.forecast_residual_for("KSFO", "high") == DEFAULT_FORECAST_RESIDUAL_F
    assert cfg.forecast_residual_for("KSFO", "high", lead_hours=None) == DEFAULT_FORECAST_RESIDUAL_F


def test_forecast_residuals_round_trip_through_json(tmp_path: Path):
    cfg = RankerConfig(
        generated_at=datetime(2026, 5, 30, 12, 0, tzinfo=timezone.utc),
        based_on_snapshots=100,
        forecast_residuals={
            residual_key("KSFO", "high"): ForecastResidual.of(1.2, 50, True),
            residual_key("KIAH", "low"): ForecastResidual.of(3.4, 8, False),
        },
    )
    path = tmp_path / "cfg.json"
    cfg.save_json(path)
    loaded = RankerConfig.load_json(path)

    assert loaded.forecast_residual_for("KSFO", "high") == 1.2
    # Below-threshold entry round-trips but still returns default.
    assert loaded.forecast_residual_for("KIAH", "low") == DEFAULT_FORECAST_RESIDUAL_F
    # The raw entry is preserved so an operator can inspect the audit trail.
    assert loaded.forecast_residuals[residual_key("KIAH", "low")].n_samples == 8
    assert loaded.forecast_residuals[residual_key("KIAH", "low")].applied is False
