"""Forecast logging + residual backfill — the data loop for the learned model.

Every graded contract can be appended to a JSONL forecast log along with the
distribution that priced it. After a market day settles, `backfill_residuals`
fetches the realized daily high/low from NWS and joins it to the logged
forecasts, emitting `(predicted, actual, residual)` rows.

Those residual rows are the training set for the future learned correction
model: a per-station `bias_f` is, to first order, just `mean(actual -
predicted)` over that station's history. Until then the bot runs at bias_f=0
(pure blend) and simply accumulates the evidence.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from weather_trader.forecast import ForecastDistribution
from weather_trader.models import Evaluation, Metric, market_day_window
from weather_trader.nws import NwsClient, observed_extremum


class ForecastLog:
    """Append-only JSONL log of graded forecasts."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def append_evaluation(
        self, e: Evaluation, dist: ForecastDistribution, now_utc: Optional[datetime] = None
    ) -> None:
        now_utc = now_utc or datetime.now(timezone.utc)
        row = {
            "ts_utc": now_utc.isoformat(),
            "ticker": e.market.ticker,
            "event": e.market.event_ticker,
            "city": e.contract.city_slug,
            "station": dist.station.icao,
            "tz": dist.station.tz,
            "metric": e.contract.metric.value,
            "market_date": e.contract.market_date.isoformat(),
            "bracket": e.contract.bracket.label(),
            "grade": e.grade,
            "side": e.best_side,
            "edge_cents": round(e.best_edge * 100, 1) if e.best_edge is not None else None,
            "fair_low": round(e.fair_prob_low, 3),
            "fair_mid": round(e.fair_prob_mid, 3),
            "fair_high": round(e.fair_prob_high, 3),
            "yes_ask": e.yes_ask_cents,
            "no_ask": e.no_ask_cents,
            "volume": e.market.volume,
            "predicted_mean_f": dist.mean(),
            "predicted_q50_f": dist.quantile(0.5),
            "band_width_f": dist.band_width_f(),
            "observed_extremum_f": dist.observed_extremum_f,
            "locked": dist.locked,
            "bias_f": dist.bias_f,
            "n_members": dist.n_members,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        out: list[dict] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out


def _realized_extremum(
    nws: NwsClient, icao: str, tz: str, metric: Metric, market_date: date
) -> Optional[float]:
    ws_local, we_local = market_day_window(market_date, tz)
    obs = nws.observations(
        icao,
        start=ws_local.astimezone(timezone.utc),
        end=we_local.astimezone(timezone.utc),
    )
    return observed_extremum(obs, metric.is_high)


def backfill_residuals(
    log_path: str,
    target_date: date,
    nws: NwsClient,
    out_path: Optional[str] = None,
) -> list[dict]:
    """Join logged forecasts for `target_date` to realized NWS extrema.

    Dedups forecasts by (station, metric, ts) so multiple bracket rows from one
    scan collapse to one residual per distribution. Returns the residual rows
    and, if `out_path` is given, appends them as JSONL.
    """
    rows = ForecastLog(log_path).read()
    iso = target_date.isoformat()
    todays = [r for r in rows if r.get("market_date") == iso]

    seen: set[tuple] = set()
    forecasts: list[dict] = []
    for r in todays:
        key = (r.get("station"), r.get("metric"), r.get("ts_utc"))
        if None in key or key in seen:
            continue
        seen.add(key)
        forecasts.append(r)

    actual_cache: dict[tuple[str, str], Optional[float]] = {}
    residuals: list[dict] = []
    for r in forecasts:
        station = r["station"]
        metric = Metric(r["metric"])
        ck = (station, r["metric"])
        if ck not in actual_cache:
            actual_cache[ck] = _realized_extremum(nws, station, r["tz"], metric, target_date)
        actual = actual_cache[ck]
        predicted = r.get("predicted_q50_f")
        if actual is None or predicted is None:
            continue
        residuals.append({
            "station": station,
            "metric": r["metric"],
            "market_date": iso,
            "ts_forecast": r["ts_utc"],
            "predicted_q50_f": predicted,
            "predicted_mean_f": r.get("predicted_mean_f"),
            "band_width_f": r.get("band_width_f"),
            "n_members": r.get("n_members"),
            "actual_f": actual,
            "residual_f": round(actual - predicted, 2),
        })

    if out_path and residuals:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            for row in residuals:
                fh.write(json.dumps(row) + "\n")

    return residuals
