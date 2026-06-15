"""weather-trader CLI: cities, doctor, forecast, scan, backfill.

Rich tables for humans; `--json` for machine-readable output.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from typing import Optional

import click
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from weather_trader.alerts import AlertDispatcher, JsonlSink, StdoutSink
from weather_trader.calibration import Calibration, derive_calibration, load_residuals
from weather_trader.forecast import ForecastDistribution, forecast_for_station
from weather_trader.grade import GRADE_ORDER, evaluate, sort_key
from weather_trader.kalshi import KalshiClient, iter_temperature_events
from weather_trader.models import Evaluation, KalshiEvent, Metric, Station
from weather_trader.nws import NWS_BASE_URL, NwsClient
from weather_trader.openmeteo import OPENMETEO_ENSEMBLE_URL, OpenMeteoClient
from weather_trader.parser import parse_market
from weather_trader.stations import all_cities, get_station
from weather_trader.store import ForecastLog, backfill_residuals
from weather_trader.kalshi import DEFAULT_BASE_URL as KALSHI_BASE_URL

console = Console()

# A never-resolved station, used only to build F-grade evals for unknown cities.
_PLACEHOLDER_STATION = Station("????", "unknown", "UNKNOWN", "UTC", 0.0, 0.0)


# -- Orchestration ----------------------------------------------------------------

def _evaluate_event(
    nws: NwsClient,
    om: Optional[OpenMeteoClient],
    event: KalshiEvent,
    now_utc: Optional[datetime] = None,
    calibration: Optional[Calibration] = None,
) -> tuple[list[Evaluation], Optional[ForecastDistribution]]:
    """Evaluate every parseable contract in an event against one shared forecast.

    All markets in a Kalshi temperature event share city/metric/date, so the
    forecast distribution is built once and every bracket is priced off it.
    """
    parsed = [(c, m) for m in event.markets for c in (parse_market(m),) if c is not None]
    if not parsed:
        return [], None

    first_contract = parsed[0][0]
    station = get_station(first_contract.city_slug)
    if station is None:
        evals = []
        for contract, market in parsed:
            e = evaluate(
                contract, market,
                ForecastDistribution(contract.metric, contract.market_date,
                                     _PLACEHOLDER_STATION, [], None, False, 0.0, 0, []),
            )
            e.notes.append(f"unknown station for city {contract.city_slug}")
            evals.append(e)
        return evals, None

    metric = first_contract.metric
    bias_f = calibration.bias_for(station.icao, metric.value) if calibration else 0.0
    sigma_f = calibration.sigma_for(station.icao, metric.value) if calibration else None
    dist = forecast_for_station(
        nws, om, station, metric, first_contract.market_date,
        now_utc=now_utc, bias_f=bias_f, forecast_sigma_f=sigma_f,
    )
    evals = [evaluate(contract, market, dist) for contract, market in parsed]
    evals.sort(key=sort_key)
    return evals, dist


def _eval_to_dict(e: Evaluation) -> dict:
    return {
        "ticker": e.market.ticker,
        "event": e.market.event_ticker,
        "city": e.contract.city_slug,
        "metric": e.contract.metric.value,
        "market_date": e.contract.market_date.isoformat(),
        "bracket": e.contract.bracket.label(),
        "grade": e.grade,
        "locked": e.locked,
        "fair_prob": [round(e.fair_prob_low, 3), round(e.fair_prob_mid, 3), round(e.fair_prob_high, 3)],
        "forecast_mean_f": round(e.forecast_mean_f, 1) if e.forecast_mean_f is not None else None,
        "band_width_f": round(e.band_width_f, 1) if e.band_width_f is not None else None,
        "yes_ask": e.yes_ask_cents,
        "no_ask": e.no_ask_cents,
        "edge_yes": round(e.edge_yes, 3) if e.edge_yes is not None else None,
        "edge_no": round(e.edge_no, 3) if e.edge_no is not None else None,
        "best_side": e.best_side,
        "volume": e.market.volume,
        "notes": e.notes,
    }


_GRADE_COLOR = {
    "A+": "bold green", "A": "green", "B+": "yellow", "B": "yellow",
    "C": "white", "D": "dim", "F": "red",
}


def _print_evals_table(evals: list[Evaluation], title: str) -> None:
    table = Table(title=title, header_style="bold cyan")
    table.add_column("Grade", justify="center")
    table.add_column("Bracket", no_wrap=True)
    table.add_column("Fair %", justify="right")
    table.add_column("Mean °F", justify="right")
    table.add_column("Yes", justify="right")
    table.add_column("Edge Y", justify="right")
    table.add_column("No", justify="right")
    table.add_column("Edge N", justify="right")
    table.add_column("Vol", justify="right")
    for e in evals:
        color = _GRADE_COLOR.get(e.grade, "white")
        fair = f"{e.fair_prob_low * 100:.0f}–{e.fair_prob_high * 100:.0f}%"
        mean = "—" if e.forecast_mean_f is None else f"{e.forecast_mean_f:.1f}"
        ya = "—" if e.yes_ask_cents is None else f"{e.yes_ask_cents}c"
        na = "—" if e.no_ask_cents is None else f"{e.no_ask_cents}c"
        ey = "—" if e.edge_yes is None else f"{e.edge_yes * 100:+.1f}c"
        en = "—" if e.edge_no is None else f"{e.edge_no * 100:+.1f}c"
        table.add_row(
            f"[{color}]{e.grade}[/{color}]", e.contract.bracket.label(),
            fair, mean, ya, ey, na, en, str(e.market.volume),
        )
    console.print(table)


def _print_forecast_panel(dist: ForecastDistribution) -> None:
    s = dist.summary()
    obs = "—" if s["observed_extremum_f"] is None else f"{s['observed_extremum_f']:g}°F"
    body = (
        f"station:   {dist.station.icao} ({dist.station.name})\n"
        f"metric:    daily {dist.metric.value}    date: {dist.market_date.isoformat()}\n"
        f"observed:  {obs}    locked: {dist.locked}\n"
        f"mean:      {_g(s['mean_f'])}°F    median: {_g(s['q50_f'])}°F\n"
        f"q10–q90:   {_g(s['q10_f'])}–{_g(s['q90_f'])}°F    band: {_g(s['band_width_f'])}°F\n"
        f"bias_f:    {dist.bias_f:+.2f}°F    members: {dist.n_members}\n"
        f"notes:     {'; '.join(dist.notes) if dist.notes else '—'}"
    )
    console.print(Panel(body, title="Forecast distribution", border_style="green"))


def _g(v) -> str:
    return "—" if v is None else f"{v:g}"


# -- Commands ---------------------------------------------------------------------

@click.group()
def main() -> None:
    """weather-trader: forecast-driven trading for Kalshi temperature markets."""


@main.command()
def cities() -> None:
    """List the cities/stations the bot can forecast."""
    for slug in all_cities():
        st = get_station(slug)
        assert st is not None
        console.print(f"  {slug:<14} -> {st.icao}  ({st.name}, {st.tz})")


def _probe(url: str, headers: dict, params: Optional[dict] = None) -> tuple[bool, str]:
    try:
        resp = httpx.get(url, headers=headers, params=params, timeout=10.0)
    except httpx.HTTPError as exc:
        return False, f"connection error: {type(exc).__name__}"
    if resp.headers.get("x-deny-reason") == "host_not_allowed":
        return False, "blocked by network egress policy (host_not_allowed)"
    return True, f"HTTP {resp.status_code}"


@main.command()
def doctor() -> None:
    """Check outbound reachability to the three required hosts.

    Exits non-zero if Kalshi or NWS is blocked — so you can tell an empty scan
    apart from a network wall.
    """
    checks = [
        ("kalshi (required)", f"{KALSHI_BASE_URL}/markets",
         {"User-Agent": "weather-trader/0.1", "Accept": "application/json"}, {"limit": 1}),
        ("nws (required)", f"{NWS_BASE_URL}/",
         {"User-Agent": "weather-trader/0.1 (ben.melson@gmail.com)"}, None),
        ("open-meteo (recommended)", OPENMETEO_ENSEMBLE_URL,
         {"User-Agent": "weather-trader/0.1"},
         {"latitude": 40.78, "longitude": -73.97, "hourly": "temperature_2m", "forecast_days": 1}),
    ]
    table = Table(title="egress check", header_style="bold cyan")
    table.add_column("host")
    table.add_column("ok", justify="center")
    table.add_column("detail")
    required_blocked = False
    for name, url, headers, params in checks:
        ok, detail = _probe(url, headers, params)
        if not ok and "required" in name:
            required_blocked = True
        table.add_row(name, "[green]✓[/green]" if ok else "[red]✗[/red]", detail)
    console.print(table)
    if required_blocked:
        console.print("[red]A required host is blocked. Add it to the environment's egress allowlist.[/red]")
        sys.exit(1)
    console.print("[green]all required hosts reachable[/green]")


@main.command()
@click.argument("event_ticker", required=False)
@click.option("--city", help="City slug for a market-less forecast (with --metric/--date).")
@click.option("--metric", type=click.Choice(["high", "low"]), help="With --city.")
@click.option("--date", "date_str", help="YYYY-MM-DD, with --city/--metric.")
@click.option("--no-ensemble", is_flag=True, help="Skip Open-Meteo; NWS + synthetic spread only.")
@click.option("--calibration", "calibration_path", type=click.Path(exists=True), default=None,
              help="Apply a learned bias model (from `calibrate --out`).")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of tables.")
def forecast(event_ticker: Optional[str], city: Optional[str], metric: Optional[str],
             date_str: Optional[str], no_ensemble: bool, calibration_path: Optional[str],
             as_json: bool) -> None:
    """Build and print the forecast distribution for an event (or a city/metric/date).

    \b
    weather-trader forecast KXHIGHNYC-26JUN16
    weather-trader forecast --city NYC --metric high --date 2026-06-16
    """
    now_utc = datetime.now(timezone.utc)
    calibration = Calibration.load_json(calibration_path) if calibration_path else None

    # Market-less mode: forecast a station directly.
    if event_ticker is None:
        if not (city and metric and date_str):
            raise click.UsageError("pass an EVENT_TICKER, or --city + --metric + --date")
        station = get_station(city)
        if station is None:
            raise click.BadParameter(f"unknown city slug {city!r} (see `weather-trader cities`)")
        md = _parse_date(date_str)
        bias_f = calibration.bias_for(station.icao, metric) if calibration else 0.0
        sigma_f = calibration.sigma_for(station.icao, metric) if calibration else None
        with NwsClient() as nws:
            om = None if no_ensemble else OpenMeteoClient()
            try:
                dist = forecast_for_station(nws, om, station, Metric(metric), md, now_utc=now_utc,
                                            bias_f=bias_f, forecast_sigma_f=sigma_f)
            finally:
                if om is not None:
                    om.close()
        if as_json:
            click.echo(json.dumps(dist.summary(), indent=2, default=str))
        else:
            _print_forecast_panel(dist)
        return

    # Event mode: price every bracket in the event.
    with KalshiClient() as kc, NwsClient() as nws:
        om = None if no_ensemble else OpenMeteoClient()
        try:
            event = KalshiEvent(
                event_ticker=event_ticker, series_ticker="", title="", sub_title="",
                markets=list(kc.iter_markets(event_ticker=event_ticker)),
            )
            evals, dist = _evaluate_event(nws, om, event, now_utc=now_utc, calibration=calibration)
        finally:
            if om is not None:
                om.close()

    if not evals:
        console.print(f"[red]No parseable temperature contracts for {event_ticker}[/red]")
        sys.exit(1)

    if as_json:
        click.echo(json.dumps({
            "event": event_ticker,
            "forecast": dist.summary() if dist else None,
            "contracts": [_eval_to_dict(e) for e in evals],
        }, indent=2, default=str))
        return

    if dist is not None:
        _print_forecast_panel(dist)
    _print_evals_table(evals, event_ticker)


@main.command()
@click.option("--city", help="Filter to one city slug (e.g. HOUSTON, NYC).")
@click.option("--limit", type=int, default=None, help="Stop after N events.")
@click.option("--min-grade", default="C", help=f"Skip results worse than this grade {GRADE_ORDER}.")
@click.option("--no-ensemble", is_flag=True, help="Skip Open-Meteo; NWS + synthetic spread only.")
@click.option("--calibration", "calibration_path", type=click.Path(exists=True), default=None,
              help="Apply a learned bias model (from `calibrate --out`).")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of tables.")
@click.option("--log", "log_path", default=None, help="Append every graded forecast to a JSONL log.")
@click.option("--notify", "notify_specs", multiple=True,
              help="Alert sink: 'stdout' or 'jsonl:/path.jsonl'. May repeat.")
@click.option("--notify-min-grade", default="B", help="Fire alerts at this grade or better.")
def scan(city: Optional[str], limit: Optional[int], min_grade: str, no_ensemble: bool,
         calibration_path: Optional[str], as_json: bool, log_path: Optional[str],
         notify_specs: tuple[str, ...], notify_min_grade: str) -> None:
    """Crawl all open Kalshi temperature markets, forecast + grade every contract."""
    if min_grade not in GRADE_ORDER:
        raise click.BadParameter(f"min-grade must be one of {GRADE_ORDER}")
    cutoff = GRADE_ORDER.index(min_grade)
    calibration = Calibration.load_json(calibration_path) if calibration_path else None
    if calibration is not None:
        applied = sum(1 for e in calibration.iter_entries() if e.applied)
        console.print(f"[dim]loaded calibration ({applied} station/metric corrections applied)[/dim]")

    sinks = []
    for spec in notify_specs:
        if spec == "stdout":
            sinks.append(StdoutSink())
        elif spec.startswith("jsonl:"):
            sinks.append(JsonlSink(spec[len("jsonl:"):]))
        else:
            raise click.BadParameter(f"--notify spec {spec!r}; use 'stdout' or 'jsonl:PATH'")
    dispatcher = AlertDispatcher(sinks, min_grade=notify_min_grade) if sinks else None
    flog = ForecastLog(log_path) if log_path else None

    now_utc = datetime.now(timezone.utc)
    shown: list[list[Evaluation]] = []
    all_evals: list[Evaluation] = []
    count = 0
    with KalshiClient() as kc, NwsClient() as nws:
        om = None if no_ensemble else OpenMeteoClient()
        try:
            for event in iter_temperature_events(kc):
                if city and city.upper() not in event.event_ticker.upper():
                    continue
                evals, dist = _evaluate_event(nws, om, event, now_utc=now_utc, calibration=calibration)
                if not evals:
                    continue
                all_evals.extend(evals)
                if flog is not None and dist is not None:
                    for e in evals:
                        flog.append_evaluation(e, dist, now_utc=now_utc)
                keep = [e for e in evals if GRADE_ORDER.index(e.grade) <= cutoff]
                if keep:
                    shown.append(keep)
                    count += 1
                if limit is not None and count >= limit:
                    break
        finally:
            if om is not None:
                om.close()

    fired = dispatcher.dispatch(all_evals, now_utc=now_utc) if dispatcher else []

    if as_json:
        out = [{
            "event": evals[0].market.event_ticker,
            "contracts": [_eval_to_dict(e) for e in evals],
        } for evals in shown]
        click.echo(json.dumps(out, indent=2, default=str))
        return

    if flog is not None:
        console.print(f"[dim]logged {len(all_evals)} forecasts to {log_path}[/dim]")
    if fired:
        console.print(f"[bold green]fired {len(fired)} alert(s)[/bold green]")
    if not shown:
        console.print("[yellow]No contracts matched. Try --min-grade D, or run `doctor` to check egress.[/yellow]")
        return
    for evals in shown:
        _print_evals_table(evals, f"{evals[0].market.event_ticker}")


@main.command()
@click.option("--log", "log_path", required=True, help="Forecast log to read (from `scan --log`).")
@click.option("--date", "date_str", required=True, help="Settled market date (YYYY-MM-DD).")
@click.option("--out", "out_path", default=None, help="Append residual rows to this JSONL.")
@click.option("--json", "as_json", is_flag=True, help="Emit residuals as JSON.")
def backfill(log_path: str, date_str: str, out_path: Optional[str], as_json: bool) -> None:
    """Join logged forecasts to realized NWS extrema -> residual training rows."""
    md = _parse_date(date_str)
    with NwsClient() as nws:
        residuals = backfill_residuals(log_path, md, nws, out_path=out_path)
    if as_json:
        click.echo(json.dumps(residuals, indent=2, default=str))
        return
    if not residuals:
        console.print("[yellow]no residuals (no matching forecasts, or actuals unavailable)[/yellow]")
        return
    table = Table(title=f"residuals for {date_str}", header_style="bold cyan")
    for col in ("station", "metric", "predicted °F", "actual °F", "residual °F", "members"):
        table.add_column(col, justify="right" if "°F" in col or col == "members" else "left")
    for r in residuals:
        table.add_row(
            r["station"], r["metric"], _g(r["predicted_q50_f"]),
            _g(r["actual_f"]), f"{r['residual_f']:+g}", str(r.get("n_members", "—")),
        )
    console.print(table)
    if out_path:
        console.print(f"[dim]appended {len(residuals)} residual rows to {out_path}[/dim]")


@main.command()
@click.option("--residuals", "residuals_path", required=True, type=click.Path(exists=True),
              help="Residuals JSONL to learn from (from `backfill --out`).")
@click.option("--out", "out_path", default=None, help="Write the calibration JSON here.")
@click.option("--min-samples", type=int, default=5, help="Min residuals per station/metric to apply a bias.")
@click.option("--clamp", "clamp_f", type=float, default=8.0, help="Max |bias_f| in °F.")
@click.option("--json", "as_json", is_flag=True, help="Emit the calibration as JSON.")
def calibrate(residuals_path: str, out_path: Optional[str], min_samples: int,
              clamp_f: float, as_json: bool) -> None:
    """Derive per-station bias corrections from accumulated forecast residuals.

    The closed loop: `scan --log` records forecasts, `backfill --out` joins them
    to realized highs/lows, and `calibrate` turns that history into a bias model
    that `scan --calibration` / `forecast --calibration` apply going forward.
    """
    rows = load_residuals(residuals_path)
    calibration = derive_calibration(rows, min_samples=min_samples, clamp_f=clamp_f)
    if out_path:
        calibration.save_json(out_path)

    if as_json:
        click.echo(json.dumps(calibration.to_dict(), indent=2, default=str))
        return

    if not list(calibration.iter_entries()):
        console.print("[yellow]no residuals to calibrate from[/yellow]")
        return
    table = Table(title=f"calibration ({calibration.based_on_residuals} residuals, "
                        f"min_samples={min_samples}, clamp=±{clamp_f:g}°F)", header_style="bold cyan")
    for col in ("station", "metric", "n", "mean res °F", "bias_f °F", "sigma °F", "note"):
        table.add_column(col, justify="right" if "°F" in col or col == "n" else "left")
    for e in sorted(calibration.iter_entries(), key=lambda x: (x.station, x.metric)):
        color = "green" if e.applied else "dim"
        table.add_row(
            e.station, e.metric, str(e.n), f"{e.mean_residual_f:+g}",
            f"[{color}]{e.bias_f:+g}[/{color}]",
            _g(e.sigma_f), e.note,
        )
    console.print(table)
    if out_path:
        console.print(f"[dim]wrote calibration to {out_path}[/dim]")


def _parse_date(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError as exc:
        raise click.BadParameter(f"date must be YYYY-MM-DD: {s!r}") from exc


if __name__ == "__main__":
    main()
