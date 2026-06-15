"""Learned per-station bias correction — the "model later" half of the hybrid.

Accumulated forecast->actual residuals (produced by `store.backfill_residuals`)
are turned into a per-(station, metric) additive correction. To first order the
optimal additive bias is `mean(actual - predicted)`: adding it to the forecast
minimizes the squared error of the corrected prediction. We also record the
residual spread (`sigma_f`) so the synthetic-spread fallback can use a
station-calibrated width instead of the global default.

How the forecaster consumes it:
  - `bias_f` shifts EVERY scenario (ensemble and synthetic paths alike), so it
    always applies.
  - `sigma_f` only widens/narrows the SYNTHETIC fallback spread; a real
    ensemble already carries its own empirical spread, so we never override it.

Corrections are sample-size gated (`min_samples`) and clamped (`clamp_f`) so a
couple of noisy days can't swing the model. The unit of correction is
(station, metric); lead-time / regime splits are a deliberate future refinement.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_MIN_SAMPLES = 5
DEFAULT_CLAMP_F = 8.0


@dataclass
class StationMetricBias:
    """Calibration for one (station, metric): the applied correction + diagnostics."""
    station: str
    metric: str
    n: int
    mean_residual_f: float
    median_residual_f: float
    mae_f: float
    sigma_f: Optional[float]
    bias_f: float              # the applied correction (clamped); 0.0 when not applied
    applied: bool
    clamped: bool
    note: str = ""


@dataclass
class Calibration:
    """A bundle of per-(station, metric) corrections, loadable by the forecaster."""
    generated_at: str
    based_on_residuals: int
    min_samples: int
    clamp_f: float
    entries: dict[str, dict[str, StationMetricBias]] = field(default_factory=dict)

    def _entry(self, icao: str, metric: str) -> Optional[StationMetricBias]:
        return self.entries.get(icao.upper(), {}).get(metric)

    def bias_for(self, icao: str, metric: str) -> float:
        """Applied additive bias for a station/metric, or 0.0 when not calibrated."""
        e = self._entry(icao, metric)
        return e.bias_f if (e is not None and e.applied) else 0.0

    def sigma_for(self, icao: str, metric: str) -> Optional[float]:
        """Calibrated synthetic-spread sigma, or None to use the forecaster default."""
        e = self._entry(icao, metric)
        if e is not None and e.applied and e.sigma_f is not None:
            return e.sigma_f
        return None

    def iter_entries(self):
        for by_metric in self.entries.values():
            for e in by_metric.values():
                yield e

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "based_on_residuals": self.based_on_residuals,
            "min_samples": self.min_samples,
            "clamp_f": self.clamp_f,
            "stations": {
                icao: {metric: asdict(e) for metric, e in by_metric.items()}
                for icao, by_metric in self.entries.items()
            },
        }

    def save_json(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str) -> "Calibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        entries: dict[str, dict[str, StationMetricBias]] = {}
        for icao, by_metric in (data.get("stations") or {}).items():
            entries[icao.upper()] = {
                metric: StationMetricBias(**e) for metric, e in by_metric.items()
            }
        return cls(
            generated_at=data.get("generated_at", ""),
            based_on_residuals=int(data.get("based_on_residuals", 0)),
            min_samples=int(data.get("min_samples", DEFAULT_MIN_SAMPLES)),
            clamp_f=float(data.get("clamp_f", DEFAULT_CLAMP_F)),
            entries=entries,
        )

    @classmethod
    def empty(cls) -> "Calibration":
        return cls("", 0, DEFAULT_MIN_SAMPLES, DEFAULT_CLAMP_F, {})


def load_residuals(path: str) -> list[dict]:
    """Read a residuals JSONL file (from `backfill --out`). Skips bad lines."""
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _dedupe(residual_rows: list[dict]) -> list[dict]:
    """Drop duplicate forecast/actual pairs by (station, metric, market_date,
    ts_forecast). Rows missing any key part are kept as-is (never deduped)."""
    seen: set[tuple] = set()
    out: list[dict] = []
    for r in residual_rows:
        key = (r.get("station"), r.get("metric"), r.get("market_date"), r.get("ts_forecast"))
        if None in key:
            out.append(r)
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def derive_calibration(
    residual_rows: list[dict],
    min_samples: int = DEFAULT_MIN_SAMPLES,
    clamp_f: float = DEFAULT_CLAMP_F,
) -> Calibration:
    """Group residuals by (station, metric) and derive per-group corrections.

    bias_f = clamp(mean(residual_f)), applied only when n >= min_samples.
    residual_f is `actual - predicted_q50`, so adding bias_f to the forecast
    pushes the prediction toward the realized value.

    Residuals are deduped by (station, metric, market_date, ts_forecast) so a
    daily `cycle` that re-backfills overlapping date windows can't inflate the
    sample. Rows missing any of those keys (e.g. hand-built) are never deduped.
    """
    residual_rows = _dedupe(residual_rows)
    groups: dict[tuple[str, str], list[float]] = {}
    for r in residual_rows:
        st, me, res = r.get("station"), r.get("metric"), r.get("residual_f")
        if st is None or me is None or res is None:
            continue
        try:
            groups.setdefault((str(st).upper(), str(me)), []).append(float(res))
        except (TypeError, ValueError):
            continue

    entries: dict[str, dict[str, StationMetricBias]] = {}
    for (st, me), vals in sorted(groups.items()):
        n = len(vals)
        mean_r = statistics.fmean(vals)
        median_r = statistics.median(vals)
        mae = statistics.fmean(abs(v) for v in vals)
        sigma = statistics.pstdev(vals) if n >= 2 else None
        applied = n >= min_samples
        clamped = False
        bias = 0.0
        if applied:
            bias = mean_r
            if bias > clamp_f:
                bias, clamped = clamp_f, True
            elif bias < -clamp_f:
                bias, clamped = -clamp_f, True
            note = "applied" + (" (clamped)" if clamped else "")
        else:
            note = f"n<{min_samples}: not applied"
        entries.setdefault(st, {})[me] = StationMetricBias(
            station=st, metric=me, n=n,
            mean_residual_f=round(mean_r, 3), median_residual_f=round(median_r, 3),
            mae_f=round(mae, 3), sigma_f=round(sigma, 3) if sigma is not None else None,
            bias_f=round(bias, 3), applied=applied, clamped=clamped, note=note,
        )

    return Calibration(
        generated_at=datetime.now(timezone.utc).isoformat(),
        based_on_residuals=sum(len(v) for v in groups.values()),
        min_samples=min_samples, clamp_f=clamp_f, entries=entries,
    )
