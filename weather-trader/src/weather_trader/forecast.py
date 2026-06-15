"""The forecasting core: a scenario ensemble of the day's extreme temperature.

A *scenario* is one plausible realized daily high (or low). The bot's fair
probability for any bracket is the weighted fraction of scenarios that land
inside it. Each scenario combines three things:

  1. Observed-so-far truth (path dependence). A daily high can't un-happen, so
     every scenario is floored at the running max observed inside the local
     market day (ceiling at the running min, for lows). This is what makes the
     distribution collapse to a single point once the day is fully observed.
  2. The remaining-day forecast. For the hours left we draw scenarios from the
     Open-Meteo ensemble (one per member -> the calibrated spread) and anchor
     them with the NWS deterministic hourly forecast (added as weighted
     scenarios). When the ensemble is missing/thin we synthesize a Gaussian
     spread around the deterministic anchor so we never emit an overconfident
     0/1 from a single point estimate.
  3. A per-station additive bias correction (`bias_f`) applied to the
     *forecast* (never to observed truth). Ships at 0.0; a learned correction
     model drops in here later without touching the pricing/grading layers.

`bias_f` corrects forecast error: scenario = combine(observed_truth,
forecast_remaining + bias_f).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

from weather_trader.models import Bracket, Metric, Station, market_day_window
from weather_trader.nws import HourlyPoint, NwsClient, Observation, observed_extremum
from weather_trader.openmeteo import EnsembleHourlyPoint, OpenMeteoClient

#: Below this many usable ensemble members we synthesize a Gaussian spread
#: instead of trusting the raw member count.
MIN_ENSEMBLE_MEMBERS = 10

#: Std-dev (°F) of the synthesized remaining-day extremum spread when the
#: ensemble is unavailable. A reasonable day-ahead high/low error; tunable per
#: station later via the same calibration loop that learns `bias_f`.
DEFAULT_FORECAST_SIGMA_F = 3.0

#: Number of synthetic scenarios drawn when synthesizing a spread.
DEFAULT_SYNTH_N = 41

#: Fraction of total scenario weight given to the NWS deterministic anchor.
#: 0.2 -> NWS nudges the distribution without dominating the ensemble.
DEFAULT_NWS_WEIGHT_FRAC = 0.2

_NORMAL = statistics.NormalDist()


@dataclass(frozen=True)
class Scenario:
    """One plausible realized daily extremum (°F) and its weight."""
    value_f: float
    source: str             # "observed" | "ensemble" | "nws" | "synthetic"
    weight: float = 1.0


@dataclass
class ForecastDistribution:
    """The bot's belief about a station's daily HIGH or LOW for a market date."""
    metric: Metric
    market_date: date
    station: Station
    scenarios: list[Scenario]
    observed_extremum_f: Optional[float]
    locked: bool                    # day fully observed -> outcome determined
    bias_f: float
    n_members: int                  # effective independent sample count (for CI)
    notes: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return bool(self.scenarios)

    def _total_weight(self) -> float:
        return sum(s.weight for s in self.scenarios)

    def mean(self) -> Optional[float]:
        if not self.scenarios:
            return None
        tw = self._total_weight()
        return sum(s.value_f * s.weight for s in self.scenarios) / tw

    def quantile(self, q: float) -> Optional[float]:
        if not self.scenarios:
            return None
        pairs = sorted((s.value_f, s.weight) for s in self.scenarios)
        total = sum(w for _, w in pairs)
        target = q * total
        cum = 0.0
        for v, w in pairs:
            cum += w
            if cum >= target:
                return v
        return pairs[-1][0]

    def band_width_f(self) -> Optional[float]:
        """q90 - q10: a spread proxy. 0 when locked (single scenario)."""
        if not self.scenarios:
            return None
        lo, hi = self.quantile(0.1), self.quantile(0.9)
        if lo is None or hi is None:
            return None
        return hi - lo

    def prob_bracket(self, bracket: Bracket) -> tuple[float, float, float]:
        """Return (low, mid, high) fair probability for the bracket.

        `mid` is the weighted empirical fraction of scenarios inside the
        bracket; `low`/`high` are a Wilson 95% interval around it (collapsed to
        the point when the day is locked), clamped to contain `mid`.
        """
        if not self.scenarios:
            return (0.0, 0.5, 1.0)
        tw = self._total_weight()
        yes_w = sum(s.weight for s in self.scenarios if bracket.contains(s.value_f))
        p = yes_w / tw
        if self.locked:
            return (p, p, p)
        lo, hi = _wilson(p, max(self.n_members, 1))
        # Guarantee low <= mid <= high so the reported band always brackets the
        # point estimate (Wilson's center is pulled toward 0.5 near the edges).
        return (min(p, lo), p, max(p, hi))

    def summary(self) -> dict:
        return {
            "metric": self.metric.value,
            "market_date": self.market_date.isoformat(),
            "station": self.station.icao,
            "observed_extremum_f": self.observed_extremum_f,
            "locked": self.locked,
            "bias_f": self.bias_f,
            "n_members": self.n_members,
            "mean_f": self.mean(),
            "q10_f": self.quantile(0.1),
            "q50_f": self.quantile(0.5),
            "q90_f": self.quantile(0.9),
            "band_width_f": self.band_width_f(),
            "notes": list(self.notes),
        }


def _wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n)))
    return (max(0.0, center - half), min(1.0, center + half))


def _combine(observed: Optional[float], forecast_remaining: Optional[float], is_high: bool) -> Optional[float]:
    """Daily extremum = best of observed truth and the (biased) remaining forecast."""
    vals = [v for v in (observed, forecast_remaining) if v is not None]
    if not vals:
        return None
    return max(vals) if is_high else min(vals)


def _extremum(values, is_high: bool) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return max(vals) if is_high else min(vals)


def _stdnormal_grid(n: int) -> list[float]:
    """n symmetric standard-normal z-scores at the (i+0.5)/n quantiles."""
    return [_NORMAL.inv_cdf((i + 0.5) / n) for i in range(n)]


def build_forecast_distribution(
    *,
    metric: Metric,
    market_date: date,
    station: Station,
    observations: list[Observation],
    nws_hourly: list[HourlyPoint],
    ensemble: list[EnsembleHourlyPoint],
    bias_f: float = 0.0,
    now_utc: Optional[datetime] = None,
    nws_weight_frac: float = DEFAULT_NWS_WEIGHT_FRAC,
    forecast_sigma_f: float = DEFAULT_FORECAST_SIGMA_F,
    min_ensemble_members: int = MIN_ENSEMBLE_MEMBERS,
    synth_n: int = DEFAULT_SYNTH_N,
) -> ForecastDistribution:
    """Build the scenario ensemble from already-fetched inputs. Pure + offline-testable."""
    is_high = metric.is_high
    now_utc = now_utc or datetime.now(timezone.utc)
    ws_local, we_local = market_day_window(market_date, station.tz)
    ws = ws_local.astimezone(timezone.utc)
    we = we_local.astimezone(timezone.utc)
    notes: list[str] = []

    in_window_obs = [o for o in observations if ws <= o.observed_at <= we]
    observed = observed_extremum(in_window_obs, is_high)
    if observed is not None:
        notes.append(f"observed {'max' if is_high else 'min'}={observed:g}°F ({len(in_window_obs)} obs)")

    # Locked: the local market day has fully elapsed and we have an observation.
    if now_utc >= we and observed is not None:
        notes.append("locked: market day fully observed")
        return ForecastDistribution(
            metric, market_date, station,
            [Scenario(observed, "observed", 1.0)],
            observed, True, bias_f, 0, notes,
        )

    rstart = max(now_utc, ws)
    fut_ens = [p for p in ensemble if rstart <= p.start <= we]
    fut_nws = [p for p in nws_hourly if rstart <= p.start <= we]

    scenarios: list[Scenario] = []
    n_members = 0

    usable_members = min((len(p.members_f) for p in fut_ens), default=0)
    if usable_members >= min_ensemble_members:
        for m in range(usable_members):
            rem = _extremum((p.members_f[m] for p in fut_ens), is_high)
            assert rem is not None
            val = _combine(observed, rem + bias_f, is_high)
            assert val is not None
            scenarios.append(Scenario(val, "ensemble", 1.0))
        n_members = usable_members
        notes.append(f"ensemble: {usable_members} members")
    else:
        if fut_ens:
            notes.append(f"ensemble thin ({usable_members}<{min_ensemble_members}); synthesizing spread")
        # Center the synthetic spread on the best available remaining-day anchor.
        if fut_nws:
            center = _extremum((p.temperature_f for p in fut_nws), is_high)
            notes.append("synthetic spread centered on NWS forecast")
        elif fut_ens:
            center = _extremum((p.mean_f for p in fut_ens), is_high)
            notes.append("synthetic spread centered on ensemble mean")
        elif observed is not None:
            # No remaining forecast but day not over: only certainty is the
            # one-sided room left above the observed high (below the low).
            center = observed
            notes.append("no remaining forecast; one-sided spread above observed")
        else:
            notes.append("no forecast or observation available")
            return ForecastDistribution(metric, market_date, station, [], observed, False, bias_f, 0, notes)
        assert center is not None
        center_biased = center + bias_f
        for z in _stdnormal_grid(synth_n):
            rem = center_biased + forecast_sigma_f * z
            val = _combine(observed, rem, is_high)
            assert val is not None
            scenarios.append(Scenario(val, "synthetic", 1.0))
        n_members = synth_n

    # NWS deterministic anchor: a weighted scenario nudging toward the official
    # forecast without overriding the ensemble's spread.
    if fut_nws and scenarios:
        rem_nws = _extremum((p.temperature_f for p in fut_nws), is_high)
        if rem_nws is not None:
            nws_val = _combine(observed, rem_nws + bias_f, is_high)
            if nws_val is not None:
                base_weight = sum(s.weight for s in scenarios)
                w_nws = max(1.0, nws_weight_frac / (1.0 - nws_weight_frac) * base_weight)
                scenarios.append(Scenario(nws_val, "nws", w_nws))
                notes.append("nws anchor added")

    return ForecastDistribution(
        metric, market_date, station, scenarios, observed, False, bias_f, n_members, notes,
    )


def forecast_for_station(
    nws: NwsClient,
    om: Optional[OpenMeteoClient],
    station: Station,
    metric: Metric,
    market_date: date,
    *,
    now_utc: Optional[datetime] = None,
    bias_f: float = 0.0,
    forecast_sigma_f: Optional[float] = None,
) -> ForecastDistribution:
    """Network orchestration: fetch inputs for a station/metric/date, then build.

    Each fetch is guarded — a failure degrades the distribution (e.g. no
    ensemble -> synthesized spread) rather than raising, so the bot always
    produces a forecast it can grade or flag as unusable.
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    ws_local, we_local = market_day_window(market_date, station.tz)
    ws = ws_local.astimezone(timezone.utc)
    we = we_local.astimezone(timezone.utc)

    observations: list[Observation] = []
    if now_utc >= ws:
        # The local market day has started; pull observations up to now.
        try:
            observations = nws.observations(station.icao, start=ws, end=min(now_utc, we))
        except Exception:
            observations = []
    try:
        nws_hourly = nws.hourly_forecast(station)
    except Exception:
        nws_hourly = []
    ensemble: list[EnsembleHourlyPoint] = []
    if om is not None:
        try:
            ensemble = om.ensemble_hourly_temperature(station.latitude, station.longitude, station.tz)
        except Exception:
            ensemble = []

    return build_forecast_distribution(
        metric=metric, market_date=market_date, station=station,
        observations=observations, nws_hourly=nws_hourly, ensemble=ensemble,
        bias_f=bias_f, now_utc=now_utc,
        forecast_sigma_f=forecast_sigma_f if forecast_sigma_f is not None else DEFAULT_FORECAST_SIGMA_F,
    )
