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

from kalshi_scout.coherence import enforce_coherence
from kalshi_scout.kalshi import KalshiClient, iter_temperature_events
from kalshi_scout.models import (
    Bracket,
    BracketKind,
    ContractEvaluation,
    ContractState,
    KalshiEvent,
    KalshiMarket,
    Metric,
    ParsedContract,
    Settlement,
    SettlementProvenance,
    Station,
    StationState,
)
from kalshi_scout.nws import NwsClient
from kalshi_scout.parser import parse_market
from kalshi_scout.ranker import grade, sort_key
from kalshi_scout.resolver import resolve_settlement
from kalshi_scout.state import build_station_state, classify, fair_probability
from kalshi_scout.stations import all_cities, get_station
from kalshi_scout.store import (
    SnapshotStore,
    backtest as run_backtest,
    replay as replay_snapshot,
    settlement_from_cli,
)

console = Console()


def _evaluate_event(
    nws: NwsClient,
    event: KalshiEvent,
    now_utc: Optional[datetime] = None,
    station_state_sink: Optional[dict[str, dict]] = None,
) -> list[ContractEvaluation]:
    """Evaluate every market in a single event. Returns evaluations sorted by grade.

    When `station_state_sink` is provided, the per-ticker StationState
    (running_max/min, CLI report data, station identity, provenance) is
    written into it. Callers use that to populate the snapshot store so
    snapshots are replayable per AGENTS.md invariant D1.
    """
    parsed: list[tuple[ParsedContract, KalshiMarket]] = []
    for market in event.markets:
        p = parse_market(market)
        if p is None:
            continue
        parsed.append((p, market))

    if not parsed:
        return []

    # All markets in one event share the same city/date/metric. The resolver
    # consults the rules text for each market individually (markets *can* in
    # principle pin different stations, though same-event ones usually don't).
    # We use the first market's settlement to build StationState, but every
    # market is graded against its own resolved station — if any differs,
    # that market is independently re-evaluated.
    first_contract, first_market = parsed[0]
    first_settlement = resolve_settlement(first_market, first_contract)

    if first_settlement.station is None:
        # Invariant I4: refuse to grade without a verified settlement source.
        return [
            _make_unverified_eval(p, m, first_settlement)
            for p, m in parsed
        ]

    station = first_settlement.station
    station_state = build_station_state(nws, station, first_contract.market_date, now_utc=now_utc)
    try:
        forecast = nws.hourly_forecast(station)
    except Exception:
        forecast = []

    evals: list[ContractEvaluation] = []
    for contract, market in parsed:
        settlement = resolve_settlement(market, contract)
        if settlement.station is None:
            evals.append(_make_unverified_eval(contract, market, settlement))
            continue
        if settlement.station.icao != station.icao:
            # Per-market station override — build its own StationState.
            local_state = build_station_state(nws, settlement.station, contract.market_date, now_utc=now_utc)
            local_forecast: list = []
            try:
                local_forecast = nws.hourly_forecast(settlement.station)
            except Exception:
                local_forecast = []
        else:
            local_state = station_state
            local_forecast = forecast

        state, reason = classify(contract, local_state)
        fair_lo, fair_hi = fair_probability(
            contract,
            local_state,
            state,
            local_forecast or None,
            now_utc=now_utc,
        )
        eval_ = grade(contract, market, state, reason, fair_lo, fair_hi)
        if not local_state.cli_matches_market_date:
            eval_.notes.append("no matching CLI yet (preliminary obs only)")
        if settlement.provenance is SettlementProvenance.REGISTRY:
            eval_.notes.append("settlement: registry fallback (not pinned in rules text)")
        else:
            eval_.notes.append(f"settlement: {settlement.station.icao} via resolver")
        evals.append(eval_)

        if station_state_sink is not None:
            station_state_sink[market.ticker] = {
                "station_icao": settlement.station.icao if settlement.station else None,
                "cli_product": settlement.station.cli_product if settlement.station else None,
                "source_provenance": settlement.provenance.value,
                "running_max_f": local_state.running_max_f,
                "running_min_f": local_state.running_min_f,
                "cli_report_date": local_state.cli_report_date,
                "cli_max_f": local_state.cli_max_f,
                "cli_min_f": local_state.cli_min_f,
            }

    # Cross-bracket coherence pass (invariant I7).
    evals = enforce_coherence(evals)
    evals.sort(key=sort_key)
    return evals


def _make_unverified_eval(
    contract: ParsedContract,
    market: KalshiMarket,
    settlement: Settlement,
) -> ContractEvaluation:
    """Build an F-graded evaluation for a market with no verifiable source.

    Per invariant I4, this is how we refuse to trade rather than guessing.
    """
    eval_ = grade(
        contract=contract,
        market=market,
        state=ContractState.FORECAST_DEPENDENT,
        reason=f"unverified settlement source ({settlement.area_description or 'unknown area'})",
        fair_lo=0.25,
        fair_hi=0.75,
    )
    eval_.grade = "F"
    eval_.notes.append("invariant I4: settlement source not verified")
    for note in settlement.notes:
        eval_.notes.append(f"resolver: {note}")
    return eval_


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
@click.option("--store", "store_path", type=click.Path(), default=None,
              help="Persist every evaluation to a SQLite snapshot store (path).")
def scan(city: Optional[str], limit: Optional[int], min_grade: str,
         as_json: bool, store_path: Optional[str]) -> None:
    """Crawl all open Kalshi temperature events and rank every contract.

    This is the universe scanner. It pulls every open event under known
    temperature series prefixes, runs each through the parser + state engine,
    and emits a ranked opportunity board.

    Pass --store to persist every evaluation as a snapshot row for later
    backtest / replay (invariants D1/D2).
    """
    grade_order = ["A+", "A", "B+", "B", "C", "D", "F"]
    if min_grade not in grade_order:
        raise click.BadParameter(f"min-grade must be one of {grade_order}")
    cutoff = grade_order.index(min_grade)

    store = SnapshotStore(store_path) if store_path else None
    scan_id: Optional[str] = None
    scanned_at: Optional[datetime] = None

    all_evals: list[tuple[KalshiEvent, list[ContractEvaluation]]] = []
    persistable: list[ContractEvaluation] = []
    sink: dict[str, dict] = {}
    with KalshiClient() as kclient, NwsClient() as nclient:
        count = 0
        scanned_at = datetime.now(timezone.utc)
        for event in iter_temperature_events(kclient):
            if city and city.upper() not in event.event_ticker.upper():
                continue
            evals = _evaluate_event(nclient, event, station_state_sink=sink)
            if not evals:
                continue
            persistable.extend(evals)
            evals = [e for e in evals if grade_order.index(e.grade) <= cutoff]
            if not evals:
                continue
            all_evals.append((event, evals))
            count += 1
            if limit is not None and count >= limit:
                break

    if store is not None and persistable:
        scan_id = store.record_scan(
            evaluations=persistable,
            scanned_at=scanned_at,
            station_state_map=sink,
        )
        console.print(
            f"[dim]wrote {len(persistable)} snapshots to {store_path} (scan_id={scan_id})[/dim]"
        )
        store.close()

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
@click.option("--store", "store_path", type=click.Path(), default=None,
              help="Persist this evaluation to a SQLite snapshot store.")
def evaluate(event_or_market: str, as_json: bool, store_path: Optional[str]) -> None:
    """Evaluate a single Kalshi event or market ticker.

    Accepts either an event ticker (e.g. KXLOWHOUSTON-26MAY28) which evaluates
    all contracts in the event, or a single market ticker.
    """
    sink: dict[str, dict] = {}
    scanned_at = datetime.now(timezone.utc)
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
        evals = _evaluate_event(nclient, event, station_state_sink=sink)

    if store_path and evals:
        with SnapshotStore(store_path) as store:
            scan_id = store.record_scan(
                evaluations=evals,
                scanned_at=scanned_at,
                station_state_map=sink,
            )
        console.print(
            f"[dim]wrote {len(evals)} snapshots to {store_path} (scan_id={scan_id})[/dim]"
        )

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


# -- V0.7 snapshot / settlement / backtest / replay --------------------------

@main.command()
@click.option("--store", "store_path", required=True, type=click.Path())
@click.option("--market-date", type=click.DateTime(formats=["%Y-%m-%d"]),
              help="Filter to a single market date (YYYY-MM-DD).")
@click.option("--min-grade", default=None, help="Show only snapshots at this grade or better.")
@click.option("--limit", type=int, default=50)
def snapshots(store_path: str, market_date: Optional[datetime],
              min_grade: Optional[str], limit: int) -> None:
    """Query the snapshot store. Useful for inspecting what scan recorded."""
    md = market_date.date() if market_date else None
    with SnapshotStore(store_path) as store:
        rows = store.query_snapshots(market_date=md, min_grade=min_grade, limit=limit)
    if not rows:
        console.print("[yellow]no snapshots match[/yellow]")
        return
    table = Table(header_style="bold cyan")
    table.add_column("id", justify="right")
    table.add_column("when")
    table.add_column("ticker")
    table.add_column("grade", justify="center")
    table.add_column("state")
    table.add_column("yes_ask", justify="right")
    table.add_column("fair %", justify="right")
    table.add_column("running max/min", justify="right")
    for r in rows:
        rm = f"{r.running_max_f:g}" if r.running_max_f is not None else "—"
        rmin = f"{r.running_min_f:g}" if r.running_min_f is not None else "—"
        ya = f"{r.yes_ask}c" if r.yes_ask else "—"
        fair = f"{r.fair_prob_low * 100:.0f}-{r.fair_prob_high * 100:.0f}%"
        table.add_row(
            str(r.id),
            r.scanned_at_utc.strftime("%m-%d %H:%M"),
            r.market_ticker,
            r.grade,
            r.state,
            ya,
            fair,
            f"{rm} / {rmin}",
        )
    console.print(table)


@main.command(name="backfill-settlements")
@click.option("--store", "store_path", required=True, type=click.Path())
@click.option("--date", "date_str", required=True,
              help="Market date to settle (YYYY-MM-DD). Pulls the matching CLI report per station.")
@click.option("--dry-run", is_flag=True, help="Compute outcomes but don't write to the store.")
def backfill_settlements(store_path: str, date_str: str, dry_run: bool) -> None:
    """Fetch the official CLI report for each market date and write realized outcomes.

    For every distinct (event_ticker, station, market_date) in stored
    snapshots for the given date, fetches the station's CLI, applies each
    contract's bracket.contains() to the CLI value, and writes one
    SettlementRow per market.
    """
    target_date = date.fromisoformat(date_str)
    with SnapshotStore(store_path) as store, NwsClient() as nclient:
        snaps = store.query_snapshots(market_date=target_date)
        if not snaps:
            console.print(f"[yellow]no snapshots for {target_date}[/yellow]")
            return

        # Group by (station, market_date) so we fetch each CLI exactly once.
        by_station: dict[tuple[str, str], list] = {}
        for s in snaps:
            if not s.cli_product or not s.station_icao:
                continue
            key = (s.cli_product, s.station_icao)
            by_station.setdefault(key, []).append(s)

        cli_cache: dict[str, Optional[float]] = {}  # (cli_product, metric) -> value
        written = 0
        for (cli_product, station_icao), group in by_station.items():
            try:
                cli = nclient.latest_cli(cli_product)
            except Exception as exc:
                console.print(f"[red]CLI fetch failed for {cli_product}: {exc}[/red]")
                continue
            if cli is None or cli.report_date != target_date:
                console.print(
                    f"[yellow]{cli_product}: no matching CLI for {target_date} "
                    f"(latest={cli.report_date if cli else 'none'})[/yellow]"
                )
                continue

            for s in group:
                metric = Metric(s.metric)
                value = cli.max_f if metric is Metric.HIGH else cli.min_f
                if value is None:
                    continue
                bracket = Bracket(
                    kind=BracketKind(s.bracket_kind),
                    lo=s.bracket_lo, hi=s.bracket_hi,
                )
                settlement = settlement_from_cli(
                    market_ticker=s.market_ticker,
                    event_ticker=s.event_ticker,
                    market_date=target_date,
                    city_slug=s.city_slug,
                    metric=metric,
                    bracket=bracket,
                    station_icao=station_icao,
                    cli_product=cli_product,
                    cli_report_date=cli.report_date,
                    cli_value_f=value,
                )
                if dry_run:
                    side = "YES" if settlement.resolved_yes else "NO"
                    console.print(
                        f"  {s.market_ticker:<40} {side}  "
                        f"(cli={value:g} bracket={bracket.label()})"
                    )
                else:
                    store.record_settlement(settlement)
                    written += 1
        console.print(
            f"[green]{'(dry-run) ' if dry_run else ''}"
            f"wrote {written} settlements[/green]"
        )


@main.command()
@click.option("--store", "store_path", required=True, type=click.Path())
@click.option("--min-grade", default="A", help="Backtest only snapshots at this grade or better.")
@click.option("--since", default=None,
              help="Only include snapshots scanned after this date (YYYY-MM-DD).")
def backtest(store_path: str, min_grade: str, since: Optional[str]) -> None:
    """Compute realized P&L for stored snapshots that have settlements.

    For each snapshot at the chosen grade with a known settlement, take the
    natural side (Yes if LOCKED_YES or fair>=0.5, else No) at the recorded
    ask price; payout is 100c if our side won, else 0. P&L = payout - price.
    """
    since_dt = (
        datetime.fromisoformat(since).replace(tzinfo=timezone.utc) if since else None
    )
    with SnapshotStore(store_path) as store:
        results = run_backtest(store, min_grade=min_grade, since=since_dt)
    if not results:
        console.print("[yellow]no backtestable snapshots found[/yellow]")
        return

    total_pnl = sum(r.pnl_cents for r in results)
    wins = sum(1 for r in results if r.won)
    hit_rate = wins / len(results) * 100
    avg_pnl = total_pnl / len(results)

    table = Table(header_style="bold cyan", title=f"Backtest (grade ≥ {min_grade})")
    table.add_column("ticker")
    table.add_column("when", justify="right")
    table.add_column("grade", justify="center")
    table.add_column("side")
    table.add_column("paid", justify="right")
    table.add_column("won", justify="center")
    table.add_column("P&L", justify="right")
    for r in results[:50]:
        color = "green" if r.pnl_cents > 0 else "red"
        table.add_row(
            r.market_ticker, r.market_date.isoformat(), r.grade,
            r.side.upper(), f"{r.price_paid_cents}c",
            "✓" if r.won else "✗",
            f"[{color}]{r.pnl_cents:+d}c[/{color}]",
        )
    console.print(table)
    console.print(
        f"[bold]N={len(results)}  hit_rate={hit_rate:.1f}%  "
        f"avg_pnl={avg_pnl:+.1f}c  total_pnl={total_pnl:+d}c[/bold]"
    )


@main.command()
@click.option("--store", "store_path", required=True, type=click.Path())
@click.argument("snapshot_id", type=int)
def replay(store_path: str, snapshot_id: int) -> None:
    """Re-derive state + grade from a stored snapshot's inputs.

    This is invariant D1 (AGENTS.md) in code form: a snapshot that can't be
    replayed deterministically is a snapshot the engine shouldn't have
    alerted on. Exits non-zero on drift so this can be wired into CI.
    """
    with SnapshotStore(store_path) as store:
        result = replay_snapshot(store, snapshot_id)
    color = "green" if result.matches else "red"
    console.print(
        f"snapshot {snapshot_id}: "
        f"state {result.stored_state} -> {result.replayed_state}; "
        f"grade {result.stored_grade} -> {result.replayed_grade} "
        f"[{color}]{'MATCH' if result.matches else 'DRIFT'}[/{color}]"
    )
    if result.drift_reason:
        console.print(f"  reason: {result.drift_reason}")
    if not result.matches:
        sys.exit(1)


if __name__ == "__main__":
    main()
