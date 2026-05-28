"""FastAPI dashboard for the snapshot store.

Read-only — never writes. Pages:

  /                  alerts feed + current state per market + risk summary
  /calibration       calibration table from stored history
  /risk              full risk report (collisions, buckets)
  /api/alerts        JSON: recent grade-improvement transitions
  /api/snapshots     JSON: most recent snapshot per market
  /api/calibration   JSON: calibration report
  /api/risk          JSON: risk report

Run with: `kalshi-scout dashboard --store scout.db --host 0.0.0.0 --port 8080`

The dashboard auto-refreshes every 30 seconds via a <meta> tag so a tab
left open in a browser will always be current within half a minute.
"""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from kalshi_scout.calibrate import calibrate as run_calibrate, report_to_dict
from kalshi_scout.risk import aggregate_risk
from kalshi_scout.store import SnapshotStore


_AUTO_REFRESH = '<meta http-equiv="refresh" content="30">'
_STYLE = """
body { font-family: -apple-system, system-ui, sans-serif; max-width: 1200px;
       margin: 2rem auto; padding: 0 1rem; color: #222; }
h1, h2 { font-weight: 600; }
nav a { margin-right: 1rem; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { padding: 0.4rem 0.8rem; border-bottom: 1px solid #eee; text-align: left;
         font-size: 0.9rem; }
th { background: #f6f8fa; }
.grade { font-weight: 700; }
.grade.A\\+, .grade.A { color: #0a8c0a; }
.grade.B\\+, .grade.B { color: #b58900; }
.grade.C, .grade.D { color: #888; }
.grade.F { color: #c0392b; }
.pnl-pos { color: #0a8c0a; }
.pnl-neg { color: #c0392b; }
.warn { background: #fff3cd; padding: 0.5rem; border-radius: 4px; }
.dim { color: #888; font-size: 0.85rem; }
"""


def create_app(store_path: Path | str) -> FastAPI:
    """Build the dashboard app. Store path is captured in closure so the
    app can be served by stock uvicorn without any further configuration."""
    store_path = Path(store_path)
    app = FastAPI(title="kalshi-scout dashboard")

    def _store() -> SnapshotStore:
        return SnapshotStore(store_path)

    # -- HTML pages ---------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        with _store() as store:
            # Most recent snapshot per market (top 50 grade-A-or-better)
            snaps = store.query_snapshots(min_grade="C", limit=200)
            # De-dupe to most recent per ticker
            seen: dict[str, object] = {}
            current: list = []
            for s in snaps:
                if s.market_ticker in seen:
                    continue
                seen[s.market_ticker] = s
                current.append(s)
            risk = aggregate_risk(store)
            recent_high_grade = [s for s in current if s.grade in ("A+", "A", "B+", "B")][:25]

        return _render(
            "kalshi-scout: dashboard",
            f"""
            <h1>kalshi-scout</h1>
            <nav>
                <a href="/">overview</a>
                <a href="/calibration">calibration</a>
                <a href="/risk">risk</a>
                <a href="/api/snapshots">snapshots api</a>
            </nav>
            <h2>Current alerts (grade B+ or better)</h2>
            {_table_recent(recent_high_grade)}
            <h2>Risk summary</h2>
            {_risk_summary(risk)}
            <p class="dim">Generated at {datetime.now(timezone.utc).isoformat(timespec="seconds")}; auto-refreshes every 30s.</p>
            """,
        )

    @app.get("/calibration", response_class=HTMLResponse)
    def calibration() -> str:
        with _store() as store:
            report = run_calibrate(store)
        return _render(
            "kalshi-scout: calibration",
            f"""
            <h1>Calibration</h1>
            <nav><a href="/">← overview</a></nav>
            <p>{report.settled_snapshots} settled of {report.total_snapshots} total snapshots</p>
            {_table_calibration(report)}
            """,
        )

    @app.get("/risk", response_class=HTMLResponse)
    def risk_page() -> str:
        with _store() as store:
            risk = aggregate_risk(store)
        return _render(
            "kalshi-scout: risk",
            f"""
            <h1>Open position risk</h1>
            <nav><a href="/">← overview</a></nav>
            {_risk_summary(risk)}
            {_risk_collisions(risk)}
            {_risk_buckets(risk)}
            """,
        )

    # -- JSON API -----------------------------------------------------------

    @app.get("/api/snapshots")
    def api_snapshots(limit: int = 100, min_grade: str = "C") -> JSONResponse:
        with _store() as store:
            rows = store.query_snapshots(min_grade=min_grade, limit=limit)
        return JSONResponse([_snap_to_dict(r) for r in rows])

    @app.get("/api/calibration")
    def api_calibration() -> JSONResponse:
        with _store() as store:
            report = run_calibrate(store)
        return JSONResponse(report_to_dict(report))

    @app.get("/api/risk")
    def api_risk() -> JSONResponse:
        with _store() as store:
            r = aggregate_risk(store)
        return JSONResponse({
            "total_open_positions": r.total_open_positions,
            "total_open_contracts": r.total_open_contracts,
            "total_max_loss_cents": r.total_max_loss_cents,
            "by_city": {k: _bucket_to_dict(v) for k, v in r.by_city.items()},
            "by_market_date": {k: _bucket_to_dict(v) for k, v in r.by_market_date.items()},
            "by_regime": {k: _bucket_to_dict(v) for k, v in r.by_regime.items()},
            "by_event": {k: _bucket_to_dict(v) for k, v in r.by_event.items()},
            "event_collisions": [
                {
                    "event_ticker": c.event_ticker,
                    "n_yes_positions": len(c.yes_positions),
                    "total_max_loss_cents": c.total_max_loss_cents,
                    "guaranteed_loss_cents": c.guaranteed_loss_cents,
                }
                for c in r.event_collisions
            ],
        })

    @app.get("/api/health")
    def api_health() -> JSONResponse:
        with _store() as store:
            n = len(store.query_snapshots(limit=1))
        return JSONResponse({"ok": True, "has_data": n > 0})

    return app


# -- Rendering helpers -------------------------------------------------------

def _render(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en"><head>
{_AUTO_REFRESH}
<title>{escape(title)}</title>
<style>{_STYLE}</style>
</head><body>
{body}
</body></html>
"""


def _table_recent(snaps: list) -> str:
    if not snaps:
        return '<p class="dim">No high-grade snapshots yet. Run <code>kalshi-scout serve --store ...</code> to start collecting.</p>'
    rows = "".join(
        f'<tr><td>{escape(s.market_ticker)}</td>'
        f'<td><span class="grade {s.grade.replace("+", "+")}">{s.grade}</span></td>'
        f'<td>{escape(s.state)}</td>'
        f'<td>{s.yes_ask if s.yes_ask else "—"}</td>'
        f'<td>{s.fair_prob_low * 100:.0f}–{s.fair_prob_high * 100:.0f}%</td>'
        f'<td>{escape(s.regime or "—")}</td>'
        f'<td>{s.scanned_at_utc.strftime("%m-%d %H:%M")}</td></tr>'
        for s in snaps
    )
    return f"""
    <table>
      <thead><tr>
        <th>Market</th><th>Grade</th><th>State</th><th>Yes ask</th>
        <th>Fair %</th><th>Regime</th><th>Last scan</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """


def _table_calibration(report) -> str:
    rows = ""
    for tier in ("A+", "A", "B+", "B", "C", "D"):
        s = report.stats_by_grade[tier]
        if s.n == 0:
            rows += f'<tr><td>{tier}</td><td>0</td><td colspan="5" class="dim">no settled samples</td></tr>'
            continue
        pnl_class = "pnl-pos" if s.total_pnl_c > 0 else ("pnl-neg" if s.total_pnl_c < 0 else "")
        rows += (
            f'<tr><td>{tier}</td>'
            f'<td>{s.n}</td><td>{s.n_unique_markets}</td>'
            f'<td>{s.wins} ({s.hit_rate * 100:.1f}%)</td>'
            f'<td>{s.avg_pnl_c:+.1f}c</td>'
            f'<td class="{pnl_class}">{s.total_pnl_c:+d}c</td>'
            f'<td>{s.median_edge:+.3f}</td></tr>'
        )
    return f"""
    <table>
      <thead><tr><th>Grade</th><th>N</th><th>Markets</th><th>Wins</th>
        <th>Avg P&L</th><th>Total P&L</th><th>Median edge</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """


def _risk_summary(risk) -> str:
    if risk.total_open_positions == 0:
        return '<p class="dim">No open positions tracked. Use <code>kalshi-scout positions add</code> to record one.</p>'
    collision_note = ""
    if risk.event_collisions:
        n = len(risk.event_collisions)
        loss = sum(c.guaranteed_loss_cents for c in risk.event_collisions)
        collision_note = f'<div class="warn">⚠ {n} event collision(s): ${loss / 100:.2f} of guaranteed loss already locked in.</div>'
    return f"""
    {collision_note}
    <p>
      <strong>{risk.total_open_positions}</strong> open positions
      / <strong>{risk.total_open_contracts}</strong> contracts
      / <strong>${risk.total_max_loss_dollars:.2f}</strong> max loss
    </p>
    """


def _risk_collisions(risk) -> str:
    if not risk.event_collisions:
        return ""
    rows = "".join(
        f'<tr><td>{escape(c.event_ticker)}</td>'
        f'<td>{len(c.yes_positions)}</td>'
        f'<td>${c.total_max_loss_cents / 100:.2f}</td>'
        f'<td class="pnl-neg">${c.guaranteed_loss_cents / 100:.2f}</td></tr>'
        for c in risk.event_collisions
    )
    return f"""
    <h2>Event collisions</h2>
    <p class="dim">Holding Yes on multiple brackets of the same event guarantees a loss on all but one.</p>
    <table>
      <thead><tr><th>Event</th><th>Yes positions</th><th>Total cost basis</th><th>Guaranteed loss</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """


def _risk_buckets(risk) -> str:
    sections = ""
    for title, bucket in (
        ("By city", risk.by_city),
        ("By market date", risk.by_market_date),
        ("By regime", risk.by_regime),
    ):
        items = sorted(bucket.items(), key=lambda kv: -kv[1].total_max_loss_cents)
        if not items:
            continue
        rows = "".join(
            f'<tr><td>{escape(k)}</td><td>{b.n_positions}</td>'
            f'<td>{b.total_contracts}</td>'
            f'<td>${b.total_max_loss_dollars:.2f}</td></tr>'
            for k, b in items
        )
        sections += f"""
        <h2>{title}</h2>
        <table>
          <thead><tr><th>Key</th><th>Positions</th><th>Contracts</th><th>Max loss</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """
    return sections


def _snap_to_dict(s) -> dict:
    return {
        "id": s.id,
        "market_ticker": s.market_ticker,
        "event_ticker": s.event_ticker,
        "scanned_at_utc": s.scanned_at_utc.isoformat(),
        "state": s.state,
        "grade": s.grade,
        "regime": s.regime,
        "yes_ask": s.yes_ask,
        "no_ask": s.no_ask,
        "fair_prob": [round(s.fair_prob_low, 3), round(s.fair_prob_high, 3)],
        "edge_yes": s.edge_yes,
        "edge_no": s.edge_no,
    }


def _bucket_to_dict(b) -> dict:
    return {
        "n_positions": b.n_positions,
        "total_contracts": b.total_contracts,
        "total_max_loss_cents": b.total_max_loss_cents,
        "market_tickers": b.market_tickers,
    }
