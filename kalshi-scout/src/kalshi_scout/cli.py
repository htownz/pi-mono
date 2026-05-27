"""kalshi-scout CLI: `scan`, `evaluate`, `watch`.

Output: rich tables for humans; `--json` for machine-readable.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime, timezone
from typing import Iterable, Optional

import click
from rich.console import Console
from rich.table import Table

from kalshi_scout.kalshi import KalshiClient, iter_temperature_events
from kalshi_scout.models import (
    ContractEvaluation,
    ContractState,
    KalshiEvent,
    KalshiMarket,
    ParsedContract,
    Station,
    StationState,
)
from kalshi_scout.nws import NwsClient
from kalshi_scout.parser import parse_market
from kalshi_scout.ranker import grade, sort_key
from kalshi_scout.state import build_station_state, classify, fair_probability
from kalshi_scout.stations import all_cities, get_station

console = Console()


def _evaluate_event(
    nws: NwsClient,
    event: KalshiEvent,
    now_utc: Optional[datetime] = None,
) -> list[ContractEvaluation]:
    """Evaluate every market in a single event. Returns evaluations sorted by grade."""
    parsed: list[tuple[ParsedContract, KalshiMarket]] = []
    for market in event.markets:
        p = parse_market(market)
        if p is None:
            continue
        parsed.append((p, market))

    if not parsed:
        return []

    # All markets in one event share the same city/date/metric, so build the
    # station state once.
    first_contract, _ = parsed[0]
    station = get_station(first_contract.city_slug)
    if station is None:
        return [
            grade(
                contract=p,
                market=m,
                state=ContractState.FORECAST_DEPENDENT,
                reason=f"no station registered for city {first_contract.city_slug}",
                fair_lo=0.25,
                fair_hi=0.75,
            )
            for p, m in parsed
        ]

    station_state = build_station_state(nws, station, first_contract.market_date, now_utc=now_utc)
    try:
        forecast = nws.hourly_forecast(station)
    except Exception:
        forecast = []

    evals: list[ContractEvaluation] = []
    for contract, market in parsed:
        state, reason = classify(contract, station_state)
        fair_lo, fair_hi = fair_probability(
            contract,
            station_state,
            state,
            forecast or None,
            now_utc=now_utc,
        )
        if not station_state.cli_matches_market_date:
            # The official CLI for this date hasn't been issued yet — make
            # that visible so a trader doesn't misread our running max/min as
            # already-settled.
            pass
        eval_ = grade(contract, market, state, reason, fair_lo, fair_hi)
        if not station_state.cli_matches_market_date:
            eval_.notes.append("no matching CLI yet (preliminary obs only)")
        evals.append(eval_)

    evals.sort(key=sort_key)
    return evals


def _eval_to_dict(e: ContractEvaluation, station: Optional[Station]) -> dict:
    return {
        "ticker": e.market.ticker,
        "event": e.market.event_ticker,
        "title": e.market.title,
        "yes_sub_title": e.market.yes_sub_title,
        "city": e.contract.city_slug,
        "station": station.icao if station else None,
        "metric": e.contract.metric.value,
        "market_date": e.contract.market_date.isoformat(),
        "bracket": e.contract.bracket.label(),
        "state": e.state.value,
        "reason": e.reason,
        "fair_prob": [round(e.fair_prob_low, 3), round(e.fair_prob_high, 3)],
        "yes_ask": e.yes_ask_cents,
        "no_ask": e.no_ask_cents,
        "edge_yes": round(e.edge_yes, 3) if e.edge_yes is not None else None,
        "edge_no": round(e.edge_no, 3) if e.edge_no is not None else None,
        "grade": e.grade,
        "volume": e.market.volume,
        "open_interest": e.market.open_interest,
        "notes": e.notes,
    }


def _print_table(evals: list[ContractEvaluation], title: str) -> None:
    table = Table(title=title, header_style="bold cyan")
    table.add_column("Grade", justify="center")
    table.add_column("Bracket", no_wrap=True)
    table.add_column("State", no_wrap=True)
    table.add_column("Fair %", justify="right")
    table.add_column("Yes ask", justify="right")
    table.add_column("Edge Y", justify="right")
    table.add_column("No ask", justify="right")
    table.add_column("Edge N", justify="right")
    table.add_column("Vol", justify="right")
    table.add_column("Notes")
    for e in evals:
        fair = f"{e.fair_prob_low * 100:.0f}–{e.fair_prob_high * 100:.0f}%"
        ya = "—" if e.yes_ask_cents is None else f"{e.yes_ask_cents}c"
        na = "—" if e.no_ask_cents is None else f"{e.no_ask_cents}c"
        ey = "—" if e.edge_yes is None else f"{e.edge_yes * 100:+.1f}c"
        en = "—" if e.edge_no is None else f"{e.edge_no * 100:+.1f}c"
        grade_color = {
            "A+": "bold green",
            "A": "green",
            "B+": "yellow",
            "B": "yellow",
            "C": "white",
            "D": "dim",
            "F": "red",
        }.get(e.grade, "white")
        table.add_row(
            f"[{grade_color}]{e.grade}[/{grade_color}]",
            e.contract.bracket.label(),
            e.state.value,
            fair,
            ya,
            ey,
            na,
            en,
            str(e.market.volume),
            ", ".join(e.notes) if e.notes else "",
        )
    console.print(table)


# -- Command group ---------------------------------------------------------------

@click.group()
def main() -> None:
    """Kalshi temperature-market intelligence scanner."""


@main.command()
@click.option("--city", help="Filter to one Kalshi city slug (e.g. HOUSTON, NYC).")
@click.option("--limit", type=int, default=None, help="Stop after N events.")
@click.option("--min-grade", default="C", help="Skip results worse than this grade (A+/A/B+/B/C/D/F).")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of tables.")
def scan(city: Optional[str], limit: Optional[int], min_grade: str, as_json: bool) -> None:
    """Crawl all open Kalshi temperature events and rank every contract.

    This is the V0.3 universe scanner. It pulls every open event under known
    temperature series prefixes, runs each through the parser + state engine,
    and emits a ranked opportunity board.
    """
    grade_order = ["A+", "A", "B+", "B", "C", "D", "F"]
    if min_grade not in grade_order:
        raise click.BadParameter(f"min-grade must be one of {grade_order}")
    cutoff = grade_order.index(min_grade)

    all_evals: list[tuple[KalshiEvent, list[ContractEvaluation]]] = []
    with KalshiClient() as kclient, NwsClient() as nclient:
        count = 0
        for event in iter_temperature_events(kclient):
            if city and city.upper() not in event.event_ticker.upper():
                continue
            evals = _evaluate_event(nclient, event)
            if not evals:
                continue
            evals = [e for e in evals if grade_order.index(e.grade) <= cutoff]
            if not evals:
                continue
            all_evals.append((event, evals))
            count += 1
            if limit is not None and count >= limit:
                break

    if as_json:
        out = []
        for event, evals in all_evals:
            station = get_station(evals[0].contract.city_slug) if evals else None
            out.append({
                "event": event.event_ticker,
                "title": event.title,
                "contracts": [_eval_to_dict(e, station) for e in evals],
            })
        click.echo(json.dumps(out, indent=2))
        return

    if not all_evals:
        console.print("[yellow]No events matched. Try --min-grade D or remove --city.[/yellow]")
        return

    for event, evals in all_evals:
        _print_table(evals, f"{event.event_ticker} — {event.title}")


@main.command()
@click.argument("event_or_market")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of tables.")
def evaluate(event_or_market: str, as_json: bool) -> None:
    """Evaluate a single Kalshi event or market ticker.

    Accepts either an event ticker (e.g. KXLOWHOUSTON-26MAY28) which evaluates
    all contracts in the event, or a single market ticker.
    """
    with KalshiClient() as kclient, NwsClient() as nclient:
        if "-T" in event_or_market or "-B" in event_or_market.split("-")[-1].upper():
            market = kclient.get_market(event_or_market)
            event = KalshiEvent(
                event_ticker=market.event_ticker,
                series_ticker="",
                title=market.title,
                sub_title="",
                markets=[market],
            )
        else:
            event = KalshiEvent(
                event_ticker=event_or_market,
                series_ticker="",
                title="",
                sub_title="",
                markets=list(kclient.iter_markets(event_ticker=event_or_market)),
            )
        evals = _evaluate_event(nclient, event)

    if as_json:
        station = get_station(evals[0].contract.city_slug) if evals else None
        click.echo(json.dumps(
            {
                "event": event.event_ticker,
                "contracts": [_eval_to_dict(e, station) for e in evals],
            },
            indent=2,
        ))
        return

    if not evals:
        console.print(f"[red]No parsable contracts for {event_or_market}[/red]")
        sys.exit(1)
    _print_table(evals, f"{event.event_ticker}")


@main.command()
@click.argument("event_ticker")
@click.option("--interval", type=int, default=300, help="Seconds between polls.")
@click.option("--min-grade", default="B", help="Only emit alerts at or above this grade.")
def watch(event_ticker: str, interval: int, min_grade: str) -> None:
    """Re-evaluate an event on a loop and print state changes.

    Tonight's KHOU low: `kalshi-scout watch KXLOWHOUSTON-26MAY28 --interval 300`
    """
    grade_order = ["A+", "A", "B+", "B", "C", "D", "F"]
    cutoff = grade_order.index(min_grade) if min_grade in grade_order else grade_order.index("B")
    last_state: dict[str, str] = {}
    while True:
        now = datetime.now(timezone.utc)
        console.rule(f"{now.isoformat(timespec='seconds')} — polling {event_ticker}")
        try:
            with KalshiClient() as kclient, NwsClient() as nclient:
                event = KalshiEvent(
                    event_ticker=event_ticker,
                    series_ticker="",
                    title="",
                    sub_title="",
                    markets=list(kclient.iter_markets(event_ticker=event_ticker)),
                )
                evals = _evaluate_event(nclient, event)
        except Exception as exc:
            console.print(f"[red]poll failed: {exc}[/red]")
            time.sleep(interval)
            continue

        changed: list[ContractEvaluation] = []
        for e in evals:
            if grade_order.index(e.grade) > cutoff:
                continue
            prev = last_state.get(e.market.ticker)
            if prev != e.state.value:
                changed.append(e)
            last_state[e.market.ticker] = e.state.value

        if changed:
            _print_table(changed, f"State changes (grade ≤ {min_grade})")
        else:
            console.print(f"[dim]no actionable changes (watching {len(evals)} contracts)[/dim]")

        time.sleep(interval)


@main.command()
def cities() -> None:
    """List the city slugs the scout knows how to settle against."""
    for slug in all_cities():
        station = get_station(slug)
        assert station is not None
        console.print(f"  {slug:<14} -> {station.icao}  ({station.name}, {station.tz})")


if __name__ == "__main__":
    main()
