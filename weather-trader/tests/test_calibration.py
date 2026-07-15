import json
from datetime import date, datetime, timezone

from weather_trader.calibration import (
    Calibration,
    derive_calibration,
    load_residuals,
)
from weather_trader.forecast import build_forecast_distribution
from weather_trader.models import Bracket, BracketKind, Metric, Station
from weather_trader.openmeteo import EnsembleHourlyPoint

STATION = Station("KTST", "Test", "TEST", "UTC", 0.0, 0.0)


def _res(station, metric, residual):
    return {"station": station, "metric": metric, "residual_f": residual,
            "predicted_q50_f": 80.0, "actual_f": 80.0 + residual}


def test_bias_is_mean_residual_when_enough_samples():
    rows = [_res("KNYC", "high", 2.0) for _ in range(5)]
    cal = derive_calibration(rows, min_samples=5)
    assert cal.bias_for("KNYC", "high") == 2.0       # mean(+2 x5)
    assert cal.bias_for("knyc", "high") == 2.0        # case-insensitive
    e = cal._entry("KNYC", "high")
    assert e.applied and e.n == 5 and e.sigma_f == 0.0


def test_bias_gated_below_min_samples():
    rows = [_res("KNYC", "high", 2.0) for _ in range(3)]
    cal = derive_calibration(rows, min_samples=5)
    assert cal.bias_for("KNYC", "high") == 0.0        # not applied
    assert cal._entry("KNYC", "high").applied is False


def test_bias_is_clamped():
    rows = [_res("KLAX", "low", 20.0) for _ in range(6)]
    cal = derive_calibration(rows, min_samples=5, clamp_f=8.0)
    assert cal.bias_for("KLAX", "low") == 8.0
    assert cal._entry("KLAX", "low").clamped is True


def test_unknown_station_returns_zero_bias():
    cal = derive_calibration([_res("KNYC", "high", 2.0) for _ in range(5)])
    assert cal.bias_for("KMIA", "high") == 0.0
    assert cal.sigma_for("KMIA", "high") is None


def test_sigma_from_residual_spread():
    rows = [_res("KORD", "high", r) for r in (-2.0, -1.0, 0.0, 1.0, 2.0)]
    cal = derive_calibration(rows, min_samples=5)
    assert cal.bias_for("KORD", "high") == 0.0        # symmetric -> no bias
    sigma = cal.sigma_for("KORD", "high")
    assert sigma is not None and sigma > 1.0          # but real spread


def test_residuals_deduped_by_full_key():
    # Re-running backfill over overlapping windows produces duplicate rows; the
    # same (station, metric, market_date, ts_forecast) must count once.
    base = {"station": "KNYC", "metric": "high", "market_date": "2026-06-16", "residual_f": 2.0}
    rows = [
        {**base, "ts_forecast": "t1"},
        {**base, "ts_forecast": "t1"},   # exact duplicate -> dropped
        {**base, "ts_forecast": "t2"},   # distinct forecast -> kept
    ]
    cal = derive_calibration(rows, min_samples=1)
    assert cal._entry("KNYC", "high").n == 2


def test_save_load_roundtrip(tmp_path):
    rows = [_res("KNYC", "high", 2.0) for _ in range(5)]
    cal = derive_calibration(rows, min_samples=5)
    path = tmp_path / "cal.json"
    cal.save_json(str(path))
    loaded = Calibration.load_json(str(path))
    assert loaded.bias_for("KNYC", "high") == 2.0
    assert loaded.min_samples == 5


def test_load_residuals_reads_jsonl(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text("\n".join(json.dumps(_res("KNYC", "high", 1.0)) for _ in range(3)) + "\n")
    rows = load_residuals(str(path))
    assert len(rows) == 3 and rows[0]["station"] == "KNYC"


def test_calibrated_bias_shifts_forecast_probability():
    # Without bias, members peak at 79 -> "80°+" is impossible (prob 0).
    ens = [EnsembleHourlyPoint(datetime(2026, 6, 16, 14, tzinfo=timezone.utc), tuple([79.0] * 20))]
    cal = derive_calibration([_res("KTST", "high", 2.0) for _ in range(5)], min_samples=5)
    bias = cal.bias_for(STATION.icao, "high")          # +2.0
    dist = build_forecast_distribution(
        metric=Metric.HIGH, market_date=date(2026, 6, 16), station=STATION,
        observations=[], nws_hourly=[], ensemble=ens,
        bias_f=bias, now_utc=datetime(2026, 6, 16, 12, tzinfo=timezone.utc),
    )
    # +2°F bias lifts members to 81 -> now clears the 80 strike.
    _, mid, _ = dist.prob_bracket(Bracket(BracketKind.GTE, lo=80, hi=None))
    assert mid == 1.0
