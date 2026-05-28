"""Calibration report — realized statistics per grade tier.

Reads the snapshot store + settlements, joins them, and produces hit-rate
/ average-P&L / sample-N stats for each grade. This is the diagnostic that
tells you whether the magic-number cutoffs in `ranker.py` are actually
calibrated to reality.

In this slice the output is **observability only**. We deliberately do NOT
auto-shift `ranker.py` thresholds — invariant I9 says signals must be
backtest-supported, and a single calibration run on a thin sample isn't
that. V0.9 closes the loop by feeding a stable calibration into the
ranker; V0.8 ships the measurement framework.

Reported per grade:
  n              snapshot count with a known settlement
  n_unique       distinct markets (a single market can appear in many scans)
  hit_rate       wins / n
  avg_pnl_c      mean per-contract P&L in cents
  total_pnl_c    sum across all contracts
  median_edge    median |edge| stored at snapshot time (sanity check that
                 the grade tier's nominal threshold matches reality)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from statistics import median
from typing import Optional

from kalshi_scout.store import SnapshotStore, backtest


@dataclass(frozen=True)
class GradeStats:
    grade: str
    n: int
    n_unique_markets: int
    wins: int
    hit_rate: float
    avg_pnl_c: float
    total_pnl_c: int
    median_edge: Optional[float]


@dataclass(frozen=True)
class CalibrationReport:
    since: Optional[datetime]
    total_snapshots: int
    settled_snapshots: int
    stats_by_grade: dict[str, GradeStats]

    def has_any_data(self) -> bool:
        return any(s.n > 0 for s in self.stats_by_grade.values())


GRADE_TIERS = ["A+", "A", "B+", "B", "C", "D"]


def calibrate(
    store: SnapshotStore,
    since: Optional[datetime] = None,
) -> CalibrationReport:
    """Compute per-grade realized statistics from the store.

    Joins snapshots ↔ settlements via the existing `backtest()` helper, then
    aggregates by grade. Snapshots without a known settlement are counted
    in `total_snapshots` but not in any grade's stats.
    """
    # Pull all snapshots first for the total count.
    all_snaps = store.query_snapshots(since=since, min_grade="D")
    total = len(all_snaps)

    stats_by_grade: dict[str, GradeStats] = {}
    for tier in GRADE_TIERS:
        rows = store.query_snapshots(min_grade=tier, since=since)
        # query_snapshots(min_grade=X) returns X-and-better; filter to exactly X
        rows = [r for r in rows if r.grade == tier]
        # Collect edges (best of edge_yes/edge_no, in absolute value).
        edges: list[float] = []
        for r in rows:
            candidates = [e for e in (r.edge_yes, r.edge_no) if e is not None]
            if candidates:
                edges.append(max(candidates))

        # Run backtest for this tier to get realized P&L on settled samples.
        # backtest() filters by grade>=tier, so we again filter to exact tier.
        backtest_rows = [
            b for b in backtest(store, min_grade=tier, since=since)
            if b.grade == tier
        ]
        n = len(backtest_rows)
        wins = sum(1 for b in backtest_rows if b.won)
        total_pnl = sum(b.pnl_cents for b in backtest_rows)
        unique_markets = len({b.market_ticker for b in backtest_rows})

        stats_by_grade[tier] = GradeStats(
            grade=tier,
            n=n,
            n_unique_markets=unique_markets,
            wins=wins,
            hit_rate=(wins / n) if n else 0.0,
            avg_pnl_c=(total_pnl / n) if n else 0.0,
            total_pnl_c=total_pnl,
            median_edge=median(edges) if edges else None,
        )

    settled_n = sum(s.n for s in stats_by_grade.values())
    return CalibrationReport(
        since=since,
        total_snapshots=total,
        settled_snapshots=settled_n,
        stats_by_grade=stats_by_grade,
    )


def report_to_dict(report: CalibrationReport) -> dict:
    """Serializable form, suitable for JSON output."""
    return {
        "since": report.since.isoformat() if report.since else None,
        "total_snapshots": report.total_snapshots,
        "settled_snapshots": report.settled_snapshots,
        "by_grade": {
            tier: {
                "n": s.n,
                "n_unique_markets": s.n_unique_markets,
                "wins": s.wins,
                "hit_rate": round(s.hit_rate, 3),
                "avg_pnl_c": round(s.avg_pnl_c, 2),
                "total_pnl_c": s.total_pnl_c,
                "median_edge": round(s.median_edge, 3) if s.median_edge is not None else None,
            }
            for tier, s in report.stats_by_grade.items()
        },
    }
