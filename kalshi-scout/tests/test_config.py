"""Tests for V0.9 RankerConfig: JSON round-trip, defaults, regime lookup."""

from datetime import datetime, timezone
from pathlib import Path

from kalshi_scout.config import (
    DEFAULT_BRACKET_HIT,
    DEFAULT_FORECAST_DEPENDENT,
    DEFAULT_LOCKED_YES,
    RankerConfig,
    RegimeShift,
    StateThresholds,
    regime_key,
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
