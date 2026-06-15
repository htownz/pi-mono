from datetime import date, datetime, timezone

from weather_trader.forecast import build_forecast_distribution
from weather_trader.models import Bracket, BracketKind, Metric, Station
from weather_trader.nws import HourlyPoint, Observation
from weather_trader.openmeteo import EnsembleHourlyPoint

STATION = Station("KTST", "Test", "TEST", "UTC", 0.0, 0.0)
MD = date(2026, 6, 16)  # window in UTC: 2026-06-16 00:00 .. 23:59:59


def _utc(h: int, day: int = 16) -> datetime:
    return datetime(2026, 6, day, h, 0, tzinfo=timezone.utc)


def _gte(t):
    return Bracket(BracketKind.GTE, lo=t, hi=None)


def _between(lo, hi):
    return Bracket(BracketKind.BETWEEN, lo=lo, hi=hi)


def test_locked_when_day_fully_observed():
    obs = [Observation(_utc(15), 85.0)]
    dist = build_forecast_distribution(
        metric=Metric.HIGH, market_date=MD, station=STATION,
        observations=obs, nws_hourly=[], ensemble=[], now_utc=_utc(6, day=17),
    )
    assert dist.locked is True
    assert dist.observed_extremum_f == 85.0
    assert dist.prob_bracket(_gte(80)) == (1.0, 1.0, 1.0)
    assert dist.prob_bracket(_gte(90)) == (0.0, 0.0, 0.0)
    assert dist.prob_bracket(_between(84, 86)) == (1.0, 1.0, 1.0)
    assert dist.band_width_f() == 0.0


def test_path_dependence_locks_bracket_before_close():
    # It's midday, we've already hit 82°F. Any "80°+" bracket is settled YES
    # regardless of the (cooler) remaining-day forecast.
    obs = [Observation(_utc(10), 82.0)]
    ens = [EnsembleHourlyPoint(_utc(14), tuple([79.0] * 20)),
           EnsembleHourlyPoint(_utc(18), tuple([78.0] * 20))]
    dist = build_forecast_distribution(
        metric=Metric.HIGH, market_date=MD, station=STATION,
        observations=obs, nws_hourly=[], ensemble=ens, now_utc=_utc(12),
    )
    assert dist.locked is False  # day not over...
    _, mid, _ = dist.prob_bracket(_gte(80))
    assert mid == 1.0           # ...but the bracket is already decided


def test_ensemble_member_counting():
    # 20 members: half peak at 79, half at 81 over the remaining window.
    ens = [
        EnsembleHourlyPoint(_utc(14), tuple([79.0] * 10 + [81.0] * 10)),
        EnsembleHourlyPoint(_utc(18), tuple([78.0] * 10 + [80.0] * 10)),
    ]
    dist = build_forecast_distribution(
        metric=Metric.HIGH, market_date=MD, station=STATION,
        observations=[], nws_hourly=[], ensemble=ens, now_utc=_utc(12),
    )
    lo, mid, hi = dist.prob_bracket(_gte(80))
    assert mid == 0.5           # exactly half the members reach 80+
    assert lo <= mid <= hi
    assert lo < 0.5 < hi        # Wilson band straddles the point estimate


def test_bias_shifts_distribution():
    ens = [
        EnsembleHourlyPoint(_utc(14), tuple([79.0] * 10 + [81.0] * 10)),
        EnsembleHourlyPoint(_utc(18), tuple([78.0] * 10 + [80.0] * 10)),
    ]
    # +2°F bias pushes every member to 81/83 -> all clear the 80 strike.
    dist = build_forecast_distribution(
        metric=Metric.HIGH, market_date=MD, station=STATION,
        observations=[], nws_hourly=[], ensemble=ens, now_utc=_utc(12), bias_f=2.0,
    )
    _, mid, _ = dist.prob_bracket(_gte(80))
    assert mid == 1.0


def test_synthetic_spread_when_no_ensemble_is_not_overconfident():
    # NWS-only deterministic point at exactly the strike -> must NOT collapse to
    # a 0/1; the synthesized spread should leave real uncertainty.
    nws = [HourlyPoint(_utc(14), 80.0), HourlyPoint(_utc(18), 79.0)]
    dist = build_forecast_distribution(
        metric=Metric.HIGH, market_date=MD, station=STATION,
        observations=[], nws_hourly=nws, ensemble=[], now_utc=_utc(12),
    )
    lo, mid, hi = dist.prob_bracket(_gte(80))
    assert 0.25 < mid < 0.75
    assert dist.band_width_f() > 2.0   # genuine spread, not a point mass
    assert lo < hi


def test_no_data_is_unusable():
    dist = build_forecast_distribution(
        metric=Metric.HIGH, market_date=MD, station=STATION,
        observations=[], nws_hourly=[], ensemble=[], now_utc=_utc(12),
    )
    assert dist.usable is False
    assert dist.mean() is None


def test_low_metric_uses_min_and_floor():
    # Lows: observed min is a ceiling the daily low can only go below.
    obs = [Observation(_utc(8), 60.0)]
    ens = [EnsembleHourlyPoint(_utc(14), tuple([70.0] * 20))]  # warmer later, irrelevant to the low
    dist = build_forecast_distribution(
        metric=Metric.LOW, market_date=MD, station=STATION,
        observations=obs, nws_hourly=[], ensemble=ens, now_utc=_utc(12),
    )
    # daily low <= 60 already, so "60° or below" (LTE 60) is locked YES.
    lte60 = Bracket(BracketKind.LTE, lo=None, hi=60)
    _, mid, _ = dist.prob_bracket(lte60)
    assert mid == 1.0


def test_quantiles_and_mean_ordered():
    ens = [EnsembleHourlyPoint(_utc(14), tuple(float(v) for v in range(70, 90)))]
    dist = build_forecast_distribution(
        metric=Metric.HIGH, market_date=MD, station=STATION,
        observations=[], nws_hourly=[], ensemble=ens, now_utc=_utc(12),
    )
    assert dist.quantile(0.1) <= dist.quantile(0.5) <= dist.quantile(0.9)
    assert 70 <= dist.mean() <= 90
