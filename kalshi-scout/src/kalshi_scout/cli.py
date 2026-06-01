"""kalshi-scout CLI: `scan`, `evaluate`, `watch`.

Output: rich tables for humans; `--json` for machine-readable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import time
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kalshi_scout.audit import AuditSummary, read_audit_log, summarize as summarize_audit
from kalshi_scout.arbitrage import (
    MUTUALLY_EXCLUSIVE_SERIES,
    _parse_interval,
    detect_numeric_partition,
    is_mutually_exclusive_event,
    rank_arbitrage_opportunities,
)
from kalshi_scout.coherence import enforce_coherence
from kalshi_scout.calibrate import calibrate as run_calibrate, report_to_dict
from kalshi_scout.config import RankerConfig
from kalshi_scout.kalshi import KalshiClient, iter_all_open_events, iter_temperature_events
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
from kalshi_scout.notify import (
    AlertDispatcher,
    AlertSink,
    DiscordSink,
    JsonlSink,
    NtfySink,
    StdoutSink,
    WebhookSink,
)
from kalshi_scout.nws import NwsClient
from kalshi_scout.openmeteo import OpenMeteoClient
from kalshi_scout.orderbook import parse_orderbook
from kalshi_scout.risk import aggregate_risk
from kalshi_scout.parser import parse_market
from kalshi_scout.ranker import grade, sort_key
from kalshi_scout.regime import classify_regime
from kalshi_scout.resolver import resolve_settlement
from kalshi_scout.trading import (
    AutoTrader,
    KalshiTradingClient,
    KillSwitch,
    PositionMonitor,
    RiskGuard,
    RiskLimits,
    auto_close_settled_positions,
)
from kalshi_scout.tuning import derive_config
from kalshi_scout.state import (
    build_station_state,
    classify,
    fair_probability,
    fair_probability_from_ensemble,
    project_extremum,
)
from kalshi_scout.stations import all_cities, get_station
from kalshi_scout.store import (
    SnapshotStore,
    backtest as run_backtest,
    replay as replay_snapshot,
    settlement_from_cli,
)

console = Console()


def _print_tuning_report(report, written_to: str) -> None:
    """Render a TuningReport (from tuning.derive_config) for the operator."""
    console.print(f"[bold]Tuned RankerConfig written to {written_to}[/bold]")
    tier_table = Table(title="Tier thresholds", header_style="bold cyan")
    tier_table.add_column("State")
    tier_table.add_column("Grade", justify="center")
    tier_table.add_column("N settled", justify="right")
    tier_table.add_column("Default", justify="right")
    tier_table.add_column("Suggested", justify="right")
    tier_table.add_column("Applied", justify="center")
    tier_table.add_column("Note")
    for t in report.tiers:
        applied_str = "[green]yes[/green]" if t.applied else "[dim]no[/dim]"
        tier_table.add_row(
            t.state, t.grade, str(t.n_settled),
            f"{t.default_cutoff:.3f}", f"{t.suggested_cutoff:.3f}",
            applied_str, t.note,
        )
    console.print(tier_table)

    if report.regimes:
        reg_table = Table(title="Regime shifts", header_style="bold cyan")
        reg_table.add_column("Regime")
        reg_table.add_column("Metric")
        reg_table.add_column("Bracket")
        reg_table.add_column("N", justify="right")
        reg_table.add_column("Avg bias", justify="right")
        reg_table.add_column("Applied", justify="center")
        reg_table.add_column("Note")
        for r in report.regimes:
            applied_str = "[green]yes[/green]" if r.applied else "[dim]no[/dim]"
            reg_table.add_row(
                r.regime, r.metric, r.bracket_kind, str(r.n_settled),
                f"{r.avg_bias:+.3f}", applied_str, r.note,
            )
        console.print(reg_table)
    else:
        console.print("[dim]no regime-shift candidates in history[/dim]")

    if report.residuals:
        res_table = Table(title="Forecast residuals (°F)", header_style="bold cyan")
        res_table.add_column("Station")
        res_table.add_column("Metric")
        res_table.add_column("N days", justify="right")
        res_table.add_column("Median |residual|", justify="right")
        res_table.add_column("Applied", justify="center")
        res_table.add_column("Note")
        for r in report.residuals:
            applied_str = "[green]yes[/green]" if r.applied else "[dim]no[/dim]"
            res_table.add_row(
                r.station_icao, r.metric, str(r.n_settled),
                f"{r.median_residual_f:.2f}", applied_str, r.note,
            )
        console.print(res_table)
    else:
        console.print(
            "[dim]no forecast-residual candidates yet "
            "(needs settled snapshots with projections)[/dim]"
        )


def _build_sinks(specs: tuple[str, ...]) -> list[AlertSink]:
    """Parse --notify spec strings into sink instances.

    Accepted forms:
      stdout                          -> StdoutSink
      jsonl:/abs/or/rel/path.jsonl    -> JsonlSink(path)
      webhook:https://example.com     -> WebhookSink(url)         (raw alert JSON)
      ntfy:<topic-or-url>             -> NtfySink                 (push to phone)
      discord:https://discord.com/... -> DiscordSink              (rich embed)
    """
    sinks: list[AlertSink] = []
    for spec in specs:
        if spec == "stdout":
            sinks.append(StdoutSink())
        elif spec.startswith("jsonl:"):
            sinks.append(JsonlSink(spec[len("jsonl:"):]))
        elif spec.startswith("webhook:"):
            sinks.append(WebhookSink(spec[len("webhook:"):]))
        elif spec.startswith("ntfy:"):
            sinks.append(NtfySink(spec[len("ntfy:"):]))
        elif spec.startswith("discord:"):
            sinks.append(DiscordSink(spec[len("discord:"):]))
        else:
            raise click.BadParameter(
                f"--notify spec '{spec}' not recognized; "
                "use stdout, jsonl:PATH, webhook:URL, ntfy:TOPIC, or discord:URL"
            )
    return sinks


def _evaluate_event(
    nws: NwsClient,
    event: KalshiEvent,
    now_utc: Optional[datetime] = None,
    station_state_sink: Optional[dict[str, dict]] = None,
    config: Optional[RankerConfig] = None,
    om_client: Optional["OpenMeteoClient"] = None,
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

    # V0.5: classify the regime once per station. Notes-only signal — does
    # not modify fair_prob or grade per invariant I9 (no signal moves the
    # ladder without backtest evidence). Future slice can multiply the
    # forecast residual buffer by a regime-specific factor.
    regime_reading = classify_regime(
        station=station,
        forecast=forecast or None,
        recent_obs=station_state.observations,
        now_utc=now_utc,
    )

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
        projected_f = project_extremum(
            contract.metric, local_forecast or None, local_state, now_utc=now_utc,
        )

        # Ensemble path (Tier 1B, opt-in via config.use_ensemble). When
        # om_client is wired AND the config flag is on, try ensemble first.
        # On None return — deterministic state, empty/thin ensemble, out-of-
        # window — silently fall back to the NWS-only path. Failures NEVER
        # bubble up; the engine must always grade.
        fair_lo = fair_hi = None
        ensemble_used = False
        if (
            om_client is not None
            and config is not None
            and config.use_ensemble
            and settlement.station is not None
        ):
            try:
                ens = om_client.ensemble_hourly_temperature(
                    latitude=settlement.station.latitude,
                    longitude=settlement.station.longitude,
                    tz=settlement.station.tz,
                )
            except Exception:
                ens = []
            if ens:
                ens_result = fair_probability_from_ensemble(
                    contract, local_state, state, ens,
                    now_utc=now_utc, regime=regime_reading.regime.value,
                    config=config,
                )
                if ens_result is not None:
                    fair_lo, fair_hi = ens_result
                    ensemble_used = True

        if fair_lo is None:
            fair_lo, fair_hi = fair_probability(
                contract,
                local_state,
                state,
                local_forecast or None,
                now_utc=now_utc,
                regime=regime_reading.regime.value,
                config=config,
            )
        eval_ = grade(contract, market, state, reason, fair_lo, fair_hi, config=config)
        if ensemble_used:
            eval_.notes.append("fair_prob: open-meteo ensemble")
        if not local_state.cli_matches_market_date:
            eval_.notes.append("no matching CLI yet (preliminary obs only)")
        if settlement.provenance is SettlementProvenance.REGISTRY:
            eval_.notes.append("settlement: registry fallback (not pinned in rules text)")
        else:
            eval_.notes.append(f"settlement: {settlement.station.icao} via resolver")
        # Regime annotation — same string for every contract in this event.
        eval_.notes.append(
            f"regime: {regime_reading.regime.value}"
            + (f" ({regime_reading.reasoning[0]})" if regime_reading.reasoning else "")
        )
        evals.append(eval_)

        if station_state_sink is not None:
            station_state_sink[market.ticker] = {
                "station_icao": settlement.station.icao if settlement.station else None,
                "cli_product": settlement.station.cli_product if settlement.station else None,
                "source_provenance": settlement.provenance.value,
                "regime": regime_reading.regime.value,
                "running_max_f": local_state.running_max_f,
                "running_min_f": local_state.running_min_f,
                "projected_extremum_f": projected_f,
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
@click.option("--notify", "notify_specs", multiple=True,
              help="Alert sink spec: 'stdout', 'jsonl:/path.jsonl', 'webhook:URL', "
                   "'ntfy:TOPIC' (or 'ntfy:https://self-hosted/topic'), or 'discord:WEBHOOK_URL'. "
                   "May be passed multiple times. Requires --store.")
@click.option("--notify-min-grade", default="A",
              help="Only fire alerts at this grade or better (default A).")
@click.option("--config", "config_path", type=click.Path(exists=True), default=None,
              help="Load a calibrated RankerConfig (from `calibrate --apply`).")
def scan(city: Optional[str], limit: Optional[int], min_grade: str,
         as_json: bool, store_path: Optional[str],
         notify_specs: tuple[str, ...], notify_min_grade: str,
         config_path: Optional[str]) -> None:
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

    if notify_specs and not store_path:
        raise click.BadParameter("--notify requires --store (alerts read prior grades from snapshots)")

    ranker_config = RankerConfig.load_json(config_path) if config_path else None
    if ranker_config is not None:
        console.print(
            f"[dim]loaded ranker config from {config_path} "
            f"(generated_at={ranker_config.generated_at.isoformat()}, "
            f"based_on={ranker_config.based_on_snapshots} snapshots)[/dim]"
        )

    store = SnapshotStore(store_path) if store_path else None
    dispatcher: Optional[AlertDispatcher] = None
    if notify_specs:
        sinks = _build_sinks(notify_specs)
        dispatcher = AlertDispatcher(sinks=sinks, store=store, min_grade=notify_min_grade)
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
            evals = _evaluate_event(
                nclient, event,
                station_state_sink=sink,
                config=ranker_config,
            )
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

    # Dispatch alerts BEFORE recording the scan so prior-grade lookup
    # excludes the snapshots from this run (invariant I8: alerts are
    # state-transition functions of the store).
    fired_alerts: list = []
    if dispatcher is not None and persistable:
        fired_alerts = dispatcher.dispatch(persistable, now_utc=scanned_at)

    if store is not None and persistable:
        scan_id = store.record_scan(
            evaluations=persistable,
            scanned_at=scanned_at,
            station_state_map=sink,
        )
        console.print(
            f"[dim]wrote {len(persistable)} snapshots to {store_path} (scan_id={scan_id})[/dim]"
        )
        if fired_alerts:
            console.print(f"[bold green]fired {len(fired_alerts)} alert(s)[/bold green]")
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
@click.option("--depth", type=int, default=0,
              help="Fetch orderbook per contract and show fillable price at "
                   "this contract size (e.g. --depth 100). 0 disables.")
@click.option("--config", "config_path", type=click.Path(exists=True), default=None,
              help="Load a calibrated RankerConfig (from `calibrate --apply`).")
def evaluate(event_or_market: str, as_json: bool,
             store_path: Optional[str], depth: int,
             config_path: Optional[str]) -> None:
    """Evaluate a single Kalshi event or market ticker.

    Accepts either an event ticker (e.g. KXLOWHOUSTON-26MAY28) which evaluates
    all contracts in the event, or a single market ticker.

    With --depth N, fetches each contract's orderbook and computes the average
    fill price for N contracts on the natural trade side (Yes if state is
    LOCKED_YES or fair>=0.5, else No). Useful for confirming a stale-price
    A+/A edge is actually fillable.
    """
    sink: dict[str, dict] = {}
    scanned_at = datetime.now(timezone.utc)
    ranker_config = RankerConfig.load_json(config_path) if config_path else None
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
        evals = _evaluate_event(
            nclient, event,
            station_state_sink=sink,
            config=ranker_config,
        )

        # V0.6: optional orderbook depth fetch. We do this per-contract here
        # rather than inside _evaluate_event because depth requires an extra
        # API call per market — opt-in only.
        if depth > 0:
            for e in evals:
                try:
                    raw_book = kclient.get_orderbook(e.market.ticker)
                except Exception as exc:
                    e.notes.append(f"depth: fetch failed ({exc})")
                    continue
                book = parse_orderbook(raw_book, market_ticker=e.market.ticker)
                fair_mid = (e.fair_prob_low + e.fair_prob_high) / 2.0
                side = "yes" if (e.state == ContractState.LOCKED_YES or fair_mid >= 0.5) else "no"
                quote = book.fillable_at_size(side, depth)
                if quote is None:
                    e.notes.append(f"depth: no {side} liquidity")
                    continue
                edge = quote.edge_against(fair_mid)
                partial = " (partial)" if quote.partial else ""
                e.notes.append(
                    f"depth: {quote.filled_size}/{depth} {side} @ avg "
                    f"{quote.avg_price_cents:.1f}c (worst {quote.worst_price_cents}c){partial}; "
                    f"edge {edge * 100:+.1f}c"
                )

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
@click.argument("market_ticker")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of panels.")
@click.option("--config", "config_path", type=click.Path(exists=True), default=None,
              help="Load a calibrated RankerConfig (from `calibrate --apply`).")
def explain(market_ticker: str, as_json: bool, config_path: Optional[str]) -> None:
    """Trace one market through the full evaluation pipeline.

    Prints every intermediate: market quote, parsed contract, resolved
    settlement source, station state (running max/min, observation count,
    CLI report), in-window forecast points, regime classification, state
    machine output, fair-probability derivation, grade derivation and notes.

    Use when an alert fires (or fails to fire) and you want to see exactly
    which pipeline step produced the outcome.
    """
    now_utc = datetime.now(timezone.utc)
    ranker_config = RankerConfig.load_json(config_path) if config_path else RankerConfig.default()

    with KalshiClient() as kclient, NwsClient() as nclient:
        market = kclient.get_market(market_ticker)
        contract = parse_market(market)
        settlement = resolve_settlement(market, contract) if contract else None

        station_state: Optional[StationState] = None
        forecast: list = []
        regime_reading = None
        state_value = None
        state_reason = ""
        fair_lo = fair_hi = None
        eval_: Optional[ContractEvaluation] = None

        if contract is not None and settlement is not None and settlement.station is not None:
            station_state = build_station_state(
                nclient, settlement.station, contract.market_date, now_utc=now_utc,
            )
            try:
                forecast = nclient.hourly_forecast(settlement.station)
            except Exception:
                forecast = []
            regime_reading = classify_regime(
                station=settlement.station,
                forecast=forecast or None,
                recent_obs=station_state.observations,
                now_utc=now_utc,
            )
            state_value, state_reason = classify(contract, station_state)
            fair_lo, fair_hi = fair_probability(
                contract, station_state, state_value,
                forecast or None, now_utc=now_utc,
                regime=regime_reading.regime.value,
                config=ranker_config,
            )
            eval_ = grade(
                contract, market, state_value, state_reason,
                fair_lo, fair_hi, config=ranker_config,
            )
        elif contract is not None and settlement is not None:
            # Parseable contract but resolver couldn't pin a station — invariant I4
            # forces an F. `scan`/`evaluate` surface this via _make_unverified_eval;
            # `explain` must do the same so its grade output agrees.
            eval_ = _make_unverified_eval(contract, market, settlement)

    if as_json:
        click.echo(json.dumps(
            _explain_to_dict(
                market, contract, settlement, station_state,
                forecast, regime_reading, state_value, state_reason,
                fair_lo, fair_hi, eval_, ranker_config, now_utc,
            ),
            indent=2,
            default=str,
        ))
        return

    _print_explain(
        market, contract, settlement, station_state,
        forecast, regime_reading, state_value, state_reason,
        fair_lo, fair_hi, eval_, ranker_config, now_utc,
    )


def _in_window_forecast(forecast: list, station_state: Optional[StationState],
                        now_utc: datetime) -> list:
    if not forecast or station_state is None:
        return []
    end_utc = station_state.window_end.astimezone(timezone.utc)
    return [p for p in forecast if now_utc <= p.start <= end_utc]


def _derivation_for_eval(eval_: ContractEvaluation, state_value,
                         spread_cents, config: RankerConfig) -> str:
    """Wrapper that special-cases the I4 unverified-source path before falling
    back to the ladder-rung derivation. _make_unverified_eval forces grade=F
    independent of the edge math, so the ladder explanation would be misleading.
    """
    if eval_.grade == "F" and any("invariant I4" in n for n in eval_.notes):
        return "F: invariant I4 — settlement source not verified"
    return _grade_derivation(state_value, eval_.edge_yes, eval_.edge_no,
                             spread_cents, config)


def _grade_derivation(state_value, edge_yes, edge_no, spread_cents,
                      config: RankerConfig) -> str:
    """Reproduce the ladder rung that ranker._grade_value chose, in words."""
    if state_value is None:
        return "no state — pipeline aborted before grading"
    edges = [e for e in (edge_yes, edge_no) if e is not None]
    best_edge = max(edges) if edges else None
    wide = spread_cents is not None and spread_cents >= 10
    t = config.thresholds_for(state_value.value)
    if state_value is ContractState.LOCKED_YES:
        if edge_yes is None:
            return "F: LOCKED_YES but no yes_ask available"
        if edge_yes >= t.high_cutoff:
            return (f"edge_yes {edge_yes:+.3f} ≥ high cutoff {t.high_cutoff:.2f}"
                    + (" (wide spread → A)" if wide else " → A+"))
        if edge_yes >= t.low_cutoff:
            return (f"edge_yes {edge_yes:+.3f} ≥ low cutoff {t.low_cutoff:.2f}"
                    + (" (wide spread → B+)" if wide else " → A"))
        return f"edge_yes {edge_yes:+.3f} below cutoff {t.low_cutoff:.2f} → B"
    if state_value is ContractState.DEAD_NO:
        if edge_no is None:
            return "F: DEAD_NO but no no_ask available"
        if edge_no >= t.high_cutoff:
            return (f"edge_no {edge_no:+.3f} ≥ high cutoff {t.high_cutoff:.2f}"
                    + (" (wide spread → A)" if wide else " → A+"))
        if edge_no >= t.low_cutoff:
            return (f"edge_no {edge_no:+.3f} ≥ low cutoff {t.low_cutoff:.2f}"
                    + (" (wide spread → B+)" if wide else " → A"))
        return f"edge_no {edge_no:+.3f} below cutoff {t.low_cutoff:.2f} → B"
    if state_value is ContractState.BRACKET_HIT_VULNERABLE:
        if best_edge is None:
            return "D: bracket-hit but no fillable side"
        if best_edge >= t.high_cutoff:
            return (f"best_edge {best_edge:+.3f} ≥ high cutoff {t.high_cutoff:.2f}"
                    + (" (wide spread → B)" if wide else " → B+"))
        if best_edge >= t.low_cutoff:
            return f"best_edge {best_edge:+.3f} ≥ low cutoff {t.low_cutoff:.2f} → B"
        return f"best_edge {best_edge:+.3f} below cutoff {t.low_cutoff:.2f} → C"
    # NOT_REACHED / FORECAST_DEPENDENT
    if best_edge is None:
        return "D: no fillable side"
    if best_edge >= t.high_cutoff:
        return f"best_edge {best_edge:+.3f} ≥ high cutoff {t.high_cutoff:.2f} → B"
    if best_edge >= t.low_cutoff:
        return f"best_edge {best_edge:+.3f} ≥ low cutoff {t.low_cutoff:.2f} → C"
    return f"best_edge {best_edge:+.3f} below cutoff {t.low_cutoff:.2f} → D"


def _explain_to_dict(
    market, contract, settlement, station_state, forecast, regime_reading,
    state_value, state_reason, fair_lo, fair_hi, eval_, config, now_utc,
) -> dict:
    """Structured form of the explain output. Used by --json mode."""
    in_window = _in_window_forecast(forecast, station_state, now_utc)
    return {
        "scanned_at_utc": now_utc.isoformat(),
        "market": {
            "ticker": market.ticker,
            "event_ticker": market.event_ticker,
            "title": market.title,
            "yes_sub_title": market.yes_sub_title,
            "status": market.status,
            "close_time": market.close_time.isoformat() if market.close_time else None,
            "yes_bid": market.yes_bid, "yes_ask": market.yes_ask,
            "no_bid": market.no_bid, "no_ask": market.no_ask,
            "last_price": market.last_price,
            "volume": market.volume, "open_interest": market.open_interest,
        },
        "parsed_contract": (
            None if contract is None else {
                "city_slug": contract.city_slug,
                "metric": contract.metric.value,
                "market_date": contract.market_date.isoformat(),
                "bracket": {
                    "kind": contract.bracket.kind.value,
                    "lo": contract.bracket.lo, "hi": contract.bracket.hi,
                    "label": contract.bracket.label(),
                },
            }
        ),
        "settlement": (
            None if settlement is None else {
                "station_icao": settlement.station.icao if settlement.station else None,
                "station_name": settlement.station.name if settlement.station else None,
                "tz": settlement.station.tz if settlement.station else None,
                "cli_product": settlement.station.cli_product if settlement.station else None,
                "source_agency": settlement.source_agency,
                "area_description": settlement.area_description,
                "provenance": settlement.provenance.value,
                "notes": list(settlement.notes),
            }
        ),
        "station_state": (
            None if station_state is None else {
                "window_start_local": station_state.window_start.isoformat(),
                "window_end_local": station_state.window_end.isoformat(),
                "running_max_f": station_state.running_max_f,
                "running_min_f": station_state.running_min_f,
                "n_observations": len(station_state.observations),
                "latest_observed_at": (
                    station_state.latest.observed_at.isoformat() if station_state.latest else None
                ),
                "latest_temperature_f": (
                    station_state.latest.temperature_f if station_state.latest else None
                ),
                "cli_report_date": (
                    station_state.cli_report_date.isoformat() if station_state.cli_report_date else None
                ),
                "cli_max_f": station_state.cli_max_f,
                "cli_min_f": station_state.cli_min_f,
                "cli_matches_market_date": station_state.cli_matches_market_date,
            }
        ),
        "forecast_in_window": [
            {
                "start": p.start.isoformat(), "temperature_f": p.temperature_f,
                "sky_cover_pct": p.sky_cover_pct,
                "probability_of_precip": p.probability_of_precip,
                "wind_speed_mph": p.wind_speed_mph,
            }
            for p in in_window
        ],
        "regime": (
            None if regime_reading is None else {
                "regime": regime_reading.regime.value,
                "confidence": regime_reading.confidence,
                "reasoning": list(regime_reading.reasoning),
            }
        ),
        "state_machine": (
            None if state_value is None else {
                "state": state_value.value, "reason": state_reason,
            }
        ),
        "fair_probability": (
            None if fair_lo is None else {
                "low": round(fair_lo, 3), "high": round(fair_hi, 3),
                "mid": round((fair_lo + fair_hi) / 2.0, 3),
                "regime_shift": (
                    config.regime_shift_for(
                        regime_reading.regime.value,
                        contract.metric.value,
                        contract.bracket.kind.value,
                    ) if (regime_reading and contract) else 0.0
                ),
            }
        ),
        "grade": (
            None if eval_ is None else {
                "grade": eval_.grade,
                "yes_ask_cents": eval_.yes_ask_cents,
                "no_ask_cents": eval_.no_ask_cents,
                "edge_yes": eval_.edge_yes,
                "edge_no": eval_.edge_no,
                "derivation": _derivation_for_eval(
                    eval_, state_value,
                    (market.yes_ask - market.yes_bid)
                    if (market.yes_ask and market.yes_bid) else None,
                    config,
                ),
                "notes": list(eval_.notes),
            }
        ),
    }


def _print_explain(
    market, contract, settlement, station_state, forecast, regime_reading,
    state_value, state_reason, fair_lo, fair_hi, eval_, config, now_utc,
) -> None:
    """Rich panel rendering of the full pipeline trace."""
    sub = market.yes_sub_title or "—"
    close = market.close_time.isoformat() if market.close_time else "—"
    market_block = (
        f"[bold]{market.ticker}[/bold]\n"
        f"event:   {market.event_ticker}\n"
        f"title:   {market.title}\n"
        f"yes_sub: {sub}\n"
        f"status:  {market.status}    close: {close}"
    )
    console.print(Panel(market_block, title="Market", border_style="cyan"))

    quote = (
        f"yes_bid/ask: {market.yes_bid}c / {market.yes_ask}c    "
        f"no_bid/ask: {market.no_bid}c / {market.no_ask}c\n"
        f"last:        {market.last_price}c\n"
        f"volume:      {market.volume}    open_interest: {market.open_interest}"
    )
    console.print(Panel(quote, title="Quote", border_style="cyan"))

    if contract is None:
        console.print(Panel(
            "[red]parse_market returned None — market ticker doesn't match "
            "a parseable temperature contract.[/red]\n"
            "Pipeline aborted. Check rules text / city_slug mapping.",
            title="Parsed Contract", border_style="red",
        ))
        return
    parsed_block = (
        f"city:    {contract.city_slug}\n"
        f"metric:  {contract.metric.value}\n"
        f"date:    {contract.market_date.isoformat()}\n"
        f"bracket: {contract.bracket.kind.value}  "
        f"lo={contract.bracket.lo}  hi={contract.bracket.hi}  "
        f"({contract.bracket.label()})"
    )
    console.print(Panel(parsed_block, title="Parsed Contract", border_style="cyan"))

    if settlement is None or settlement.station is None:
        prov = settlement.provenance.value if settlement else "n/a"
        notes_join = ("\n" + "\n".join(settlement.notes)) if settlement and settlement.notes else ""
        console.print(Panel(
            f"[red]Settlement unverified (provenance={prov}).[/red]\n"
            "No station resolved — market grades F per invariant I4." + notes_join,
            title="Settlement", border_style="red",
        ))
        if eval_ is not None:
            notes_line = (
                "notes:\n  - " + "\n  - ".join(eval_.notes)
                if eval_.notes else "notes: —"
            )
            console.print(Panel(
                f"[bold]grade: {eval_.grade}[/bold]\n"
                "derivation: F: invariant I4 — settlement source not verified\n"
                f"{notes_line}",
                title="Grade", border_style="bold red",
            ))
        return
    s = settlement.station
    settle_block = (
        f"station: {s.icao}  ({s.name}, {s.tz})\n"
        f"cli:     {s.cli_product}\n"
        f"agency:  {settlement.source_agency}\n"
        f"area:    {settlement.area_description or '—'}\n"
        f"provenance: {settlement.provenance.value}"
    )
    if settlement.notes:
        settle_block += "\nnotes: " + "; ".join(settlement.notes)
    console.print(Panel(settle_block, title="Settlement", border_style="cyan"))

    if station_state is None:
        console.print(Panel(
            "[yellow]No station_state — NWS fetch failed.[/yellow]",
            title="Station State", border_style="yellow",
        ))
        return
    rm = f"{station_state.running_max_f:g}°F" if station_state.running_max_f is not None else "—"
    rmin = f"{station_state.running_min_f:g}°F" if station_state.running_min_f is not None else "—"
    latest = "—"
    if station_state.latest:
        latest = (f"{station_state.latest.temperature_f:g}°F at "
                  f"{station_state.latest.observed_at.isoformat(timespec='minutes')}")
    cli_line = "—"
    if station_state.cli_report_date:
        match = "matches market date" if station_state.cli_matches_market_date else "STALE — different date"
        cli_line = (f"{station_state.cli_report_date.isoformat()}  "
                    f"max={station_state.cli_max_f}  min={station_state.cli_min_f}  ({match})")
    state_block = (
        f"window:    {station_state.window_start.isoformat()}  →  "
        f"{station_state.window_end.isoformat()}\n"
        f"running:   max {rm}    min {rmin}\n"
        f"obs count: {len(station_state.observations)}\n"
        f"latest:    {latest}\n"
        f"CLI:       {cli_line}"
    )
    console.print(Panel(state_block, title="Station State", border_style="cyan"))

    in_window = _in_window_forecast(forecast, station_state, now_utc)
    if not in_window:
        console.print(Panel("[dim]no forecast points remaining in market window[/dim]",
                            title="Forecast (in-window)", border_style="cyan"))
    else:
        f_table = Table(header_style="bold cyan", show_edge=False)
        f_table.add_column("when (UTC)")
        f_table.add_column("temp", justify="right")
        f_table.add_column("sky %", justify="right")
        f_table.add_column("precip %", justify="right")
        f_table.add_column("wind", justify="right")
        # Cap at first 12 to keep output tight.
        for p in in_window[:12]:
            f_table.add_row(
                p.start.isoformat(timespec="minutes"),
                f"{p.temperature_f:g}°F",
                f"{p.sky_cover_pct:g}" if p.sky_cover_pct is not None else "—",
                f"{p.probability_of_precip:g}" if p.probability_of_precip is not None else "—",
                f"{p.wind_speed_mph:g}" if p.wind_speed_mph is not None else "—",
            )
        tail = f"  (+{len(in_window) - 12} more)" if len(in_window) > 12 else ""
        console.print(Panel(f_table, title=f"Forecast (in-window){tail}", border_style="cyan"))

    if regime_reading is not None:
        regime_block = (
            f"regime:     {regime_reading.regime.value}\n"
            f"confidence: {regime_reading.confidence:.2f}\n"
            "reasoning:  " + ("; ".join(regime_reading.reasoning) or "—")
        )
        console.print(Panel(regime_block, title="Regime", border_style="cyan"))

    if state_value is not None:
        console.print(Panel(
            f"state:  [bold]{state_value.value}[/bold]\nreason: {state_reason}",
            title="State Machine", border_style="green",
        ))

    if fair_lo is not None:
        shift = config.regime_shift_for(
            regime_reading.regime.value if regime_reading else "unknown",
            contract.metric.value, contract.bracket.kind.value,
        )
        mid = (fair_lo + fair_hi) / 2.0
        shift_line = (f"\nregime shift applied: {shift:+.3f}"
                      if shift != 0.0 else
                      "\nregime shift applied: 0 (no calibrated shift for this regime/metric/kind)")
        console.print(Panel(
            f"fair_prob: [{fair_lo:.3f}, {fair_hi:.3f}]  (mid {mid:.3f})" + shift_line,
            title="Fair Probability", border_style="green",
        ))

    if eval_ is not None:
        ey = f"{eval_.edge_yes:+.3f}" if eval_.edge_yes is not None else "—"
        en = f"{eval_.edge_no:+.3f}" if eval_.edge_no is not None else "—"
        spread = (market.yes_ask - market.yes_bid) if (market.yes_ask and market.yes_bid) else None
        derivation = _derivation_for_eval(eval_, state_value, spread, config)
        notes_line = ("notes:\n  - " + "\n  - ".join(eval_.notes)) if eval_.notes else "notes: —"
        grade_block = (
            f"[bold]grade: {eval_.grade}[/bold]\n"
            f"yes_ask: {eval_.yes_ask_cents}c  edge_yes: {ey}\n"
            f"no_ask:  {eval_.no_ask_cents}c  edge_no:  {en}\n"
            f"derivation: {derivation}\n"
            f"{notes_line}"
        )
        console.print(Panel(grade_block, title="Grade", border_style="bold green"))


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

_TIME_FORMATS = ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M",
                 "%Y-%m-%d"]


@main.command()
@click.option("--store", "store_path", required=True, type=click.Path())
@click.option("--ticker", "market_ticker", default=None,
              help="Filter to one market ticker's full history.")
@click.option("--event", "event_ticker", default=None,
              help="Filter to all markets in one event over time.")
@click.option("--market-date", type=click.DateTime(formats=["%Y-%m-%d"]),
              help="Filter to a single market date (YYYY-MM-DD).")
@click.option("--min-grade", default=None, help="Show only snapshots at this grade or better.")
@click.option("--since", "since_str", default=None,
              help="Lower bound on scanned_at_utc (e.g. 2026-05-29T12:00 or 2026-05-29).")
@click.option("--until", "until_str", default=None,
              help="Upper bound on scanned_at_utc.")
@click.option("--limit", type=int, default=50)
def snapshots(store_path: str, market_ticker: Optional[str],
              event_ticker: Optional[str], market_date: Optional[datetime],
              min_grade: Optional[str], since_str: Optional[str],
              until_str: Optional[str], limit: int) -> None:
    """Query the snapshot store. Useful for inspecting what scan recorded.

    `--ticker` pulls a single market's full price/state history — useful when
    a flagged contract has aged out of the default top-N view. Combine with
    `--since` / `--until` to scope a time window.
    """
    md = market_date.date() if market_date else None
    since = _parse_utc(since_str) if since_str else None
    until = _parse_utc(until_str) if until_str else None
    with SnapshotStore(store_path) as store:
        rows = store.query_snapshots(
            market_ticker=market_ticker, event_ticker=event_ticker,
            market_date=md, min_grade=min_grade,
            since=since, until=until, limit=limit,
        )
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


def _parse_utc(s: str) -> datetime:
    """Parse a CLI time string (date or datetime) as UTC."""
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise click.BadParameter(f"unrecognized time format: {s!r}")


@main.command()
@click.option("--store", "store_path", required=True, type=click.Path())
@click.option("--older-than-days", type=int, required=True,
              help="Delete snapshots scanned more than N days ago.")
@click.option("--keep-grade", default=None,
              help="Preserve rows at this grade or better (e.g. --keep-grade A "
                   "keeps A+/A history forever).")
@click.option("--dry-run", is_flag=True,
              help="Report how many rows would be deleted without touching the store.")
def prune(store_path: str, older_than_days: int,
          keep_grade: Optional[str], dry_run: bool) -> None:
    """Evict old snapshot rows. Use to bound the store's growth.

    Settled history that drives the calibration loop should be preserved with
    `--keep-grade A` — the realized P/L for A+/A picks is what tunes the next
    iteration of cutoffs. Lower-grade noise can be pruned aggressively.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
    grade_order = ["A+", "A", "B+", "B", "C", "D", "F"]
    keep_tuple: Optional[tuple[str, ...]] = None
    if keep_grade:
        if keep_grade not in grade_order:
            raise click.BadParameter(f"--keep-grade must be one of {grade_order}")
        keep_tuple = tuple(grade_order[: grade_order.index(keep_grade) + 1])

    with SnapshotStore(store_path) as store:
        if dry_run:
            n_eligible = store.count_snapshots(before=cutoff, keep_grades=keep_tuple)
            kept = ""
            if keep_tuple:
                kept = f"  (preserving grades {', '.join(keep_tuple)})"
            console.print(
                f"[dim]dry-run: {n_eligible} snapshots older than "
                f"{cutoff.isoformat(timespec='minutes')} would be deleted{kept}[/dim]"
            )
            return
        deleted = store.prune_snapshots(before=cutoff, keep_grades=keep_tuple)
        console.print(
            f"[green]pruned {deleted} snapshots older than "
            f"{cutoff.isoformat(timespec='minutes')}[/green]"
        )


@main.command(name="backfill-settlements")
@click.option("--store", "store_path", required=True, type=click.Path())
@click.option("--date", "date_str", required=True,
              help="Market date to settle (YYYY-MM-DD). Pulls the matching CLI report per station.")
@click.option("--dry-run", is_flag=True, help="Compute outcomes but don't write to the store.")
@click.option("--auto-close", is_flag=True,
              help="After recording settlements, close any open positions on "
                   "those markets with the realized exit price (100 for "
                   "winners, 0 for losers). Captures realized P&L without "
                   "needing a separate `positions close` per ticker.")
def backfill_settlements(store_path: str, date_str: str, dry_run: bool,
                         auto_close: bool) -> None:
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

        if auto_close and not dry_run:
            closed = auto_close_settled_positions(store, on_settled_date=target_date)
            if closed:
                console.print(
                    f"[green]auto-closed {len(closed)} position(s) with realized P&L:[/green]"
                )
                for pid, ticker, exit_price in closed:
                    verdict = "WON" if exit_price == 100 else "LOST"
                    console.print(
                        f"  position {pid}: {ticker} → {verdict} @ {exit_price}c"
                    )
            else:
                console.print("[dim]no open positions matched these settlements[/dim]")


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
@click.option("--audit-log", "audit_log_path", type=click.Path(), default=None,
              help="Path to the auto-trader's JSONL audit log. "
                   "Defaults to <store-dir>/auto-trade.jsonl.")
@click.option("--since", "since_str", default=None,
              help="Lower bound on entry timestamps (YYYY-MM-DD or "
                   "YYYY-MM-DDTHH:MM). Default: 24 hours ago.")
@click.option("--until", "until_str", default=None,
              help="Upper bound (same format as --since).")
@click.option("--ticker", "market_ticker", default=None,
              help="Filter to one market ticker.")
@click.option("--json", "as_json", is_flag=True,
              help="Emit JSON instead of formatted panels.")
def audit(store_path: str, audit_log_path: Optional[str],
          since_str: Optional[str], until_str: Optional[str],
          market_ticker: Optional[str], as_json: bool) -> None:
    """Summarize the auto-trader's audit JSONL by day.

    Shows attempts placed vs refused, refusal breakdown by category,
    realized cost deployed, and the last few placed / refused entries
    per day. Use this to answer "what did the bot do today?" after a
    `serve --auto-trade --paper` soak.
    """
    audit_path = (
        Path(audit_log_path) if audit_log_path
        else Path(store_path).resolve().parent / "auto-trade.jsonl"
    )
    since = _parse_utc(since_str) if since_str else (
        datetime.now(timezone.utc) - timedelta(hours=24)
    )
    until = _parse_utc(until_str) if until_str else None

    entries = list(read_audit_log(audit_path))
    summary = summarize_audit(
        entries, since=since, until=until, ticker=market_ticker, recent_n=5,
    )

    if as_json:
        out = summary.to_dict()
        out["audit_log"] = str(audit_path)
        out["since"] = since.isoformat() if since else None
        out["until"] = until.isoformat() if until else None
        click.echo(json.dumps(out, indent=2))
        return

    if not summary.days:
        console.print(
            f"[yellow]no audit entries in {audit_path} "
            f"(since {since.isoformat(timespec='minutes') if since else 'forever'}"
            f"{f' filtered to {market_ticker}' if market_ticker else ''})[/yellow]"
        )
        return

    console.print(
        f"[bold]auto-trade audit[/bold]  source={audit_path}  "
        f"window={since.isoformat(timespec='minutes') if since else 'forever'} → "
        f"{until.isoformat(timespec='minutes') if until else 'now'}"
        f"{f'  ticker={market_ticker}' if market_ticker else ''}"
    )
    for day in summary.days:
        body_lines = [
            f"[bold]Attempts: {day.total_attempts}[/bold]  "
            f"placed=[green]{day.placed}[/green]  "
            f"refused=[red]{day.refused}[/red]",
        ]
        if day.placed:
            body_lines.append(
                f"  filled-full={day.placed_live_filled_full}  "
                f"filled-partial={day.placed_live_partial}  "
                f"paper={day.placed_paper}"
            )
            body_lines.append(
                f"  total cost: ${day.total_cost_cents / 100:.2f}"
            )
        if day.refusal_breakdown:
            body_lines.append("[bold]Refusal reasons:[/bold]")
            for reason, count in day.refusal_breakdown.most_common():
                body_lines.append(f"  {reason:<26}  {count}")
        if day.by_grade:
            grades = "  ".join(
                f"{g}={n}" for g, n in sorted(day.by_grade.items())
            )
            body_lines.append(f"[bold]By grade:[/bold]  {grades}")
        if day.recent_placed:
            body_lines.append(f"[bold]Recent placed (last {len(day.recent_placed)}):[/bold]")
            for e in day.recent_placed:
                tag = "(paper)" if e.paper else (
                    f"order={e.order_id}" if e.order_id else "—"
                )
                body_lines.append(
                    f"  {e.fired_at_utc.strftime('%H:%M:%S')} "
                    f"{e.market_ticker:<40} {e.side} "
                    f"{e.size_contracts}@{e.price_cents}c  {tag}"
                )
        if day.recent_refused:
            body_lines.append(f"[bold]Recent refused (last {len(day.recent_refused)}):[/bold]")
            for e in day.recent_refused:
                body_lines.append(
                    f"  {e.fired_at_utc.strftime('%H:%M:%S')} "
                    f"{e.market_ticker:<40} {e.side} — {e.reason}"
                )
        console.print(Panel(
            "\n".join(body_lines),
            title=f"Day {day.day.isoformat()}",
            border_style="cyan",
        ))


@main.command()
@click.option("--store", "store_path", required=True, type=click.Path())
@click.option("--api-key-id", envvar="KALSHI_API_KEY_ID", default=None,
              help="Kalshi API key ID. Required unless --paper.")
@click.option("--api-key-path", envvar="KALSHI_API_KEY_PATH", default=None,
              type=click.Path(exists=True),
              help="Path to Kalshi API private key PEM. Required unless --paper.")
@click.option("--paper", is_flag=True,
              help="Skip the live Kalshi auth round-trip (use to validate "
                   "kill-switch / store / NWS plumbing without a key).")
@click.option("--kill-file", type=click.Path(), default=None,
              help="Kill-switch path to verify writability. "
                   "Default <store-dir>/scout.kill.")
@click.option("--audit-log", type=click.Path(), default=None,
              help="Audit log path to verify writability. "
                   "Default <store-dir>/auto-trade.jsonl.")
def doctor(store_path: str, api_key_id: Optional[str],
           api_key_path: Optional[str], paper: bool,
           kill_file: Optional[str], audit_log: Optional[str]) -> None:
    """One-shot go-live validator. Runs every check needed before turning
    on `serve --auto-trade` live. Exits non-zero on any FAIL so this can
    be wired into deploy scripts.

    Without --paper, this is the FIRST real validation of a new Kalshi key:
    it performs an authenticated balance round-trip, which proves the PEM
    signs correctly and Kalshi accepts the signature. The balance is shown
    so the operator confirms they're hitting the right account.
    """
    store_dir = Path(store_path).resolve().parent
    kill_path = Path(kill_file) if kill_file else store_dir / "scout.kill"
    audit_path = Path(audit_log) if audit_log else store_dir / "auto-trade.jsonl"

    results: list[tuple[str, bool, str]] = []   # (check, passed, detail)

    # 1. Snapshot store.
    try:
        with SnapshotStore(store_path) as store:
            n = store.count_snapshots()
        results.append(("snapshot store readable", True, f"{n} snapshots"))
    except Exception as exc:
        results.append(("snapshot store readable", False, str(exc)))

    # 2. Kill-switch path writable.
    try:
        kill_path.parent.mkdir(parents=True, exist_ok=True)
        probe = kill_path.parent / ".kalshi-scout-doctor-probe"
        probe.write_text("")
        probe.unlink()
        results.append(("kill-switch path writable", True, str(kill_path)))
    except Exception as exc:
        results.append(("kill-switch path writable", False, str(exc)))

    # 3. Audit log path writable.
    try:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        probe = audit_path.parent / ".kalshi-scout-doctor-probe-audit"
        probe.write_text("")
        probe.unlink()
        results.append(("audit log path writable", True, str(audit_path)))
    except Exception as exc:
        results.append(("audit log path writable", False, str(exc)))

    # 4. NWS observations (known-good station — KHOU).
    try:
        with NwsClient() as nclient:
            latest = nclient.latest_observation("KHOU")
        if latest is None:
            results.append(("NWS observations reachable", False, "KHOU returned no obs"))
        else:
            results.append((
                "NWS observations reachable", True,
                f"KHOU latest: {latest.temperature_f:g}°F at "
                f"{latest.observed_at.isoformat(timespec='minutes')}",
            ))
    except Exception as exc:
        results.append(("NWS observations reachable", False, str(exc)))

    # 5. NWS CLI report.
    try:
        with NwsClient() as nclient:
            cli = nclient.latest_cli("CLIHOU")
        results.append((
            "NWS CLI reachable", cli is not None,
            f"CLIHOU report_date={cli.report_date}" if cli else "no CLI returned",
        ))
    except Exception as exc:
        results.append(("NWS CLI reachable", False, str(exc)))

    # 6. Kalshi auth round-trip (skipped in paper mode).
    if paper:
        results.append((
            "Kalshi auth round-trip", True,
            "[dim]skipped (--paper)[/dim]",
        ))
    elif not api_key_id or not api_key_path:
        results.append((
            "Kalshi auth round-trip", False,
            "missing --api-key-id / --api-key-path (or env vars). Use --paper to skip.",
        ))
    else:
        try:
            client = KalshiTradingClient(
                key_id=api_key_id, private_key_path=Path(api_key_path),
            )
            balance = client.get_balance_cents()
            client.close()
            results.append((
                "Kalshi auth round-trip", True,
                f"balance ${balance / 100:.2f}",
            ))
        except Exception as exc:
            results.append(("Kalshi auth round-trip", False, str(exc)))

    table = Table(title="kalshi-scout doctor", header_style="bold cyan")
    table.add_column("Check")
    table.add_column("Status", justify="center")
    table.add_column("Detail")
    all_pass = True
    for check, passed, detail in results:
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        table.add_row(check, status, detail)
        if not passed:
            all_pass = False
    console.print(table)

    if all_pass:
        console.print("[bold green]All checks passed.[/bold green] Ready to trade.")
    else:
        console.print(
            "[bold red]One or more checks failed. Fix before turning on "
            "`serve --auto-trade` (live).[/bold red]"
        )
        sys.exit(1)


@main.command()
@click.option("--store", "store_path", required=True, type=click.Path())
@click.option("--since", default=None,
              help="Only include snapshots scanned after this date (YYYY-MM-DD).")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of a table.")
@click.option("--apply", "apply_path", type=click.Path(), default=None,
              help="Derive a RankerConfig from history and write it to PATH. "
                   "Tiers/regimes below the per-bucket sample-size threshold "
                   "fall back to defaults (invariant I9). Without --apply this "
                   "command is observability-only.")
def calibrate(store_path: str, since: Optional[str], as_json: bool,
              apply_path: Optional[str]) -> None:
    """Realized hit-rate / P&L per grade tier from stored history.

    Without --apply: prints the observability report and exits.
    With --apply PATH: also derives a RankerConfig from history and writes
    it to PATH. The written config respects invariant I9 — any tier with
    N < MIN_N_PER_TIER or regime with N < MIN_N_PER_REGIME keeps the
    default value; only buckets with enough samples are tuned.
    """
    since_dt = (
        datetime.fromisoformat(since).replace(tzinfo=timezone.utc) if since else None
    )
    with SnapshotStore(store_path) as store:
        report = run_calibrate(store, since=since_dt)
        if apply_path:
            cfg, tuning_report = derive_config(store, since=since_dt)
            cfg.save_json(apply_path)
            _print_tuning_report(tuning_report, apply_path)

    if as_json:
        click.echo(json.dumps(report_to_dict(report), indent=2))
        return

    if not report.has_any_data():
        console.print(
            f"[yellow]no settled snapshots yet "
            f"({report.total_snapshots} total snapshots stored, "
            f"{report.settled_snapshots} settled)[/yellow]"
        )
        console.print(
            "[dim]Run `kalshi-scout backfill-settlements --date YYYY-MM-DD` after CLI publishes."
            "[/dim]"
        )
        return

    table = Table(
        title=f"Calibration ({report.settled_snapshots} settled of "
              f"{report.total_snapshots} total)",
        header_style="bold cyan",
    )
    table.add_column("Grade", justify="center")
    table.add_column("N", justify="right")
    table.add_column("Markets", justify="right")
    table.add_column("Wins", justify="right")
    table.add_column("Hit rate", justify="right")
    table.add_column("Avg P&L", justify="right")
    table.add_column("Total P&L", justify="right")
    table.add_column("Median edge", justify="right")
    for tier in ["A+", "A", "B+", "B", "C", "D"]:
        s = report.stats_by_grade[tier]
        if s.n == 0:
            table.add_row(tier, "0", "—", "—", "—", "—", "—",
                          f"{s.median_edge:+.2f}" if s.median_edge is not None else "—")
            continue
        pnl_color = "green" if s.total_pnl_c > 0 else ("red" if s.total_pnl_c < 0 else "white")
        table.add_row(
            tier,
            str(s.n), str(s.n_unique_markets), str(s.wins),
            f"{s.hit_rate * 100:.1f}%",
            f"{s.avg_pnl_c:+.1f}c",
            f"[{pnl_color}]{s.total_pnl_c:+d}c[/{pnl_color}]",
            f"{s.median_edge:+.2f}" if s.median_edge is not None else "—",
        )
    console.print(table)


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


# -- V1.0 operational commands ----------------------------------------------

@main.command()
@click.option("--store", "store_path", required=True, type=click.Path())
@click.option("--interval", type=int, default=300,
              help="Seconds between scans.")
@click.option("--min-grade", default="C",
              help="Skip results worse than this grade.")
@click.option("--notify", "notify_specs", multiple=True,
              help="Alert sink spec: 'stdout', 'jsonl:PATH', 'webhook:URL', "
                   "'ntfy:TOPIC' (or 'ntfy:https://self-hosted/topic'), "
                   "or 'discord:WEBHOOK_URL'.")
@click.option("--notify-min-grade", default="A",
              help="Only fire alerts at this grade or better.")
@click.option("--config", "config_path", type=click.Path(exists=True), default=None,
              help="Load a calibrated RankerConfig.")
@click.option("--once", is_flag=True,
              help="Run a single scan and exit (useful for cron).")
# -- auto-trade -----------------------------------------------------------
@click.option("--auto-trade", is_flag=True,
              help="Auto-place orders on each fired alert via the Kalshi "
                   "trading API. Requires --api-key-id and --api-key-path.")
@click.option("--api-key-id", envvar="KALSHI_API_KEY_ID", default=None,
              help="Kalshi API key ID (or set KALSHI_API_KEY_ID env var).")
@click.option("--api-key-path", envvar="KALSHI_API_KEY_PATH", default=None,
              type=click.Path(exists=True),
              help="Path to Kalshi API private key PEM file "
                   "(or set KALSHI_API_KEY_PATH env var).")
@click.option("--paper", is_flag=True,
              help="Run the full auto-trade pipeline (risk-check, audit, "
                   "position record) but skip the actual API call. Use this "
                   "for a multi-day soak test before going live.")
@click.option("--default-size", type=int, default=1,
              help="Contracts per auto-trade order. Default 1 (max-conservative).")
@click.option("--max-position-size", type=int, default=5,
              help="Hard cap on contracts per order. Default 5.")
@click.option("--max-position-cost", type=int, default=500,
              help="Hard cap on cents deployed per order. Default 500c ($5).")
@click.option("--max-daily-loss", type=int, default=5000,
              help="Kill threshold for realized losses today (UTC). "
                   "Default 5000c ($50).")
@click.option("--max-concentration-per-event", type=int, default=2500,
              help="Hard cap on cumulative open cost per event_ticker. "
                   "Default 2500c ($25).")
@click.option("--min-edge", "min_edge_cents_opt", type=int, default=5,
              help="Refuse orders whose snapshot edge is below this. Default 5c.")
@click.option("--rounding-buffer", type=float, default=0.5,
              help="Refuse dead_no/locked_yes orders whose running extremum "
                   "is within this many °F of the bracket boundary. Default "
                   "0.5°F (matches CLI report rounding). Set 0 to disable.")
@click.option("--kill-file", type=click.Path(),
              default=None,
              help="Touch this file to halt all trading. Default "
                   "<store-dir>/scout.kill.")
@click.option("--audit-log", type=click.Path(),
              default=None,
              help="JSONL file capturing every trade attempt. Default "
                   "<store-dir>/auto-trade.jsonl.")
@click.option("--refresh-quote", is_flag=True,
              help="Before each auto-trade placement, re-fetch the live "
                   "Kalshi quote and use that ask instead of the stored "
                   "snapshot's. Catches stale-snapshot drift on books "
                   "that moved since the last scan.")
@click.option("--use-ensemble", is_flag=True,
              help="Use Open-Meteo's free ensemble forecast to compute "
                   "fair_prob by counting members above/below the bracket, "
                   "instead of the NWS-only Gaussian-band model. The "
                   "settlement source is unchanged (still the primary "
                   "station's CLI). On any ensemble failure or thin "
                   "response, the engine silently falls back to the "
                   "NWS-only path.")
@click.option("--monitor-positions", is_flag=True,
              help="After each scan, walk open positions and apply two "
                   "exit triggers: (1) take-profit when the bid for our "
                   "side reaches --take-profit-bid (default 95c), (2) "
                   "cut-loss when the snapshot's state flipped against "
                   "our side (NO position now in LOCKED_YES, etc). Paper "
                   "mode closes locally via close_position; live mode "
                   "logs 'live_skipped' (sell-order placement deferred). "
                   "Off by default — explicit opt-in for behavior change.")
@click.option("--take-profit-bid", type=int, default=95,
              help="Bid threshold (cents) at which the monitor closes a "
                   "winning position to lock in gains. Default 95c. Only "
                   "matters with --monitor-positions.")
@click.option("--no-cut-loss-on-state-flip", is_flag=True,
              help="Disable the state-flip cut-loss branch of the position "
                   "monitor. Take-profit is still active. Useful for "
                   "operators who want to hold to expiration on adverse "
                   "moves (hope it recovers) but still capture take-profit.")
@click.option("--exit-audit-log", type=click.Path(),
              default=None,
              help="JSONL file for exit attempts. Default "
                   "<store-dir>/auto-exit.jsonl. Sibling of --audit-log "
                   "(which covers entries); kept separate so the existing "
                   "`audit` command and dashboard panel stay aligned.")
def serve(store_path: str, interval: int, min_grade: str,
          notify_specs: tuple[str, ...], notify_min_grade: str,
          config_path: Optional[str], once: bool,
          auto_trade: bool, api_key_id: Optional[str],
          api_key_path: Optional[str], paper: bool,
          default_size: int, max_position_size: int,
          max_position_cost: int, max_daily_loss: int,
          max_concentration_per_event: int, min_edge_cents_opt: int,
          rounding_buffer: float, kill_file: Optional[str],
          audit_log: Optional[str], refresh_quote: bool,
          use_ensemble: bool,
          monitor_positions: bool, take_profit_bid: int,
          no_cut_loss_on_state_flip: bool,
          exit_audit_log: Optional[str]) -> None:
    """Run the universe scanner on a loop.

    Equivalent to running `scan --store ... --notify ...` in a `while sleep`
    loop, but with proper signal handling, structured logs to stderr, and
    automatic config / sink wiring.

    For cron: pass --once to run a single scan and exit.
    """
    import logging
    import signal as _signal

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    log = logging.getLogger("kalshi-scout")

    grade_order = ["A+", "A", "B+", "B", "C", "D", "F"]
    if min_grade not in grade_order:
        raise click.BadParameter(f"min-grade must be one of {grade_order}")
    cutoff = grade_order.index(min_grade)
    ranker_config = RankerConfig.load_json(config_path) if config_path else None
    # --use-ensemble is a CLI-level override on top of any config.use_ensemble
    # already set in the loaded JSON. Synthesize a default-config if no
    # config was loaded so the flag still takes effect.
    if use_ensemble:
        if ranker_config is None:
            ranker_config = RankerConfig.default()
        ranker_config.use_ensemble = True

    stop = {"flag": False}
    def _handle_signal(signum, _frame):
        log.info(f"received signal {signum}; finishing current scan then exiting")
        stop["flag"] = True
    _signal.signal(_signal.SIGINT, _handle_signal)
    _signal.signal(_signal.SIGTERM, _handle_signal)

    sinks = _build_sinks(notify_specs) if notify_specs else []

    # -- auto-trade setup ------------------------------------------------
    trading_client: Optional[KalshiTradingClient] = None
    risk_limits = RiskLimits(
        max_position_size_contracts=max_position_size,
        max_position_cost_cents=max_position_cost,
        max_daily_loss_cents=max_daily_loss,
        max_concentration_per_event_cents=max_concentration_per_event,
        min_edge_cents=min_edge_cents_opt,
        rounding_risk_buffer_f=rounding_buffer,
    )
    store_dir = Path(store_path).resolve().parent
    kill_path = Path(kill_file) if kill_file else store_dir / "scout.kill"
    audit_path = Path(audit_log) if audit_log else store_dir / "auto-trade.jsonl"
    exit_audit_path = (
        Path(exit_audit_log) if exit_audit_log else store_dir / "auto-exit.jsonl"
    )
    if auto_trade and not paper:
        if not api_key_id or not api_key_path:
            raise click.BadParameter(
                "--auto-trade (live mode) requires --api-key-id and --api-key-path "
                "(or KALSHI_API_KEY_ID / KALSHI_API_KEY_PATH env vars). "
                "Use --paper to run the auto-trade pipeline without calling the API."
            )
        trading_client = KalshiTradingClient(
            key_id=api_key_id, private_key_path=Path(api_key_path),
        )
    if auto_trade:
        log.info(
            f"auto-trade {'PAPER' if paper else 'LIVE'} enabled: "
            f"size={default_size} max_pos=${max_position_cost / 100:.2f} "
            f"max_event=${max_concentration_per_event / 100:.2f} "
            f"max_daily_loss=${max_daily_loss / 100:.2f} "
            f"min_edge={min_edge_cents_opt}c "
            f"rounding_buffer={rounding_buffer}°F "
            f"kill_file={kill_path}"
        )
    if ranker_config and ranker_config.use_ensemble:
        log.info("ensemble fair_prob enabled (open-meteo); NWS-only fallback on failure")
    if monitor_positions:
        log.info(
            f"position monitor enabled: take_profit_bid={take_profit_bid}c "
            f"cut_loss_on_state_flip={not no_cut_loss_on_state_flip} "
            f"exit_audit={exit_audit_path}"
        )

    iteration = 0
    while not stop["flag"]:
        iteration += 1
        scan_started = datetime.now(timezone.utc)
        log.info(f"scan iteration {iteration} starting")
        try:
            sink_map: dict[str, dict] = {}
            persistable: list[ContractEvaluation] = []
            # Fresh ensemble client per-iteration so its per-(lat, lon)
            # cache reflects only the current scan — we want a new ensemble
            # fetch on each iteration, just not duplicate calls across markets
            # that share a station within the same iteration.
            from contextlib import nullcontext
            _use_ens = bool(ranker_config and ranker_config.use_ensemble)
            _om_ctx = OpenMeteoClient() if _use_ens else nullcontext()
            with KalshiClient() as kclient, NwsClient() as nclient, _om_ctx as om:
                for event in iter_temperature_events(kclient):
                    evals = _evaluate_event(
                        nclient, event,
                        station_state_sink=sink_map,
                        config=ranker_config,
                        om_client=om if _use_ens else None,
                    )
                    if not evals:
                        continue
                    persistable.extend(evals)

            store = SnapshotStore(store_path)
            try:
                fired: list = []
                # Run the dispatcher when EITHER notification sinks or
                # auto-trade is on. Without this, `--auto-trade` with no
                # `--notify` would silently skip every alert because the
                # `fired` list never gets populated. The dispatcher's
                # alert-transition logic runs independently of sinks; an
                # empty sinks list just means no notification emission,
                # but the Alert objects still come back for auto-trade.
                if sinks or auto_trade:
                    dispatcher = AlertDispatcher(
                        sinks=sinks, store=store, min_grade=notify_min_grade
                    )
                    fired = dispatcher.dispatch(persistable, now_utc=scan_started)
                if persistable:
                    store.record_scan(
                        evaluations=persistable,
                        scanned_at=scan_started,
                        station_state_map=sink_map,
                    )

                # Auto-trade runs AFTER record_scan so the fresh snapshot is
                # already in the store for the AutoTrader to look up.
                n_placed = n_refused = 0
                if auto_trade and fired:
                    guard = RiskGuard(risk_limits, store, KillSwitch(kill_path))
                    # Read-only KalshiClient is shared across the trader's
                    # refresh-quote calls within this scan; cheap to
                    # re-create per iteration since httpx connection pools
                    # are managed inside the client.
                    rt_kalshi_client = KalshiClient() if refresh_quote else None
                    trader = AutoTrader(
                        client=trading_client, guard=guard, store=store,
                        default_size=default_size, paper=paper,
                        audit_log_path=audit_path,
                        kalshi_client=rt_kalshi_client,
                    )
                    for alert in fired:
                        snaps = store.query_snapshots(
                            market_ticker=alert.market_ticker, limit=1,
                        )
                        if not snaps:
                            log.warning(
                                f"auto-trade: no snapshot for {alert.market_ticker} — skipping"
                            )
                            continue
                        attempt = trader.maybe_trade(
                            alert, snaps[0], refresh_quote=refresh_quote,
                        )
                        if attempt.placed:
                            n_placed += 1
                            log.info(
                                f"auto-trade PLACED: {attempt.market_ticker} "
                                f"{attempt.side} {attempt.size_contracts}@{attempt.price_cents}c "
                                f"{'(paper)' if attempt.paper else f'order_id={attempt.order_id}'}"
                            )
                        else:
                            n_refused += 1
                            log.info(
                                f"auto-trade REFUSED: {attempt.market_ticker} — {attempt.reason}"
                            )
                    if rt_kalshi_client is not None:
                        rt_kalshi_client.close()

                # Position monitor: walk open positions and apply
                # take-profit / cut-loss triggers using the snapshots that
                # were just persisted. Snapshot-driven, so no extra Kalshi
                # calls. Opt-in via --monitor-positions; default off.
                n_exits = 0
                n_exit_examined = 0
                if monitor_positions:
                    monitor = PositionMonitor(
                        store=store,
                        take_profit_bid_cents=take_profit_bid,
                        cut_loss_on_state_flip=not no_cut_loss_on_state_flip,
                        paper=paper,
                        audit_log_path=exit_audit_path,
                    )
                    n_exits, n_exit_examined = monitor.run(now_utc=scan_started)

                trade_summary = (
                    f", auto-trade {n_placed} placed / {n_refused} refused"
                    if auto_trade else ""
                )
                exit_summary = (
                    f", monitor {n_exits} closed / {n_exit_examined} examined"
                    if monitor_positions else ""
                )
                log.info(
                    f"scan {iteration}: persisted {len(persistable)} snapshots, "
                    f"fired {len(fired)} alerts{trade_summary}{exit_summary}"
                )
            finally:
                store.close()
        except Exception as exc:
            log.exception(f"scan iteration {iteration} failed: {exc}")

        if once or stop["flag"]:
            break
        log.info(f"sleeping {interval}s until next scan")
        # Sleep in 1-second slices so SIGTERM is responsive.
        for _ in range(interval):
            if stop["flag"]:
                break
            time.sleep(1)


@main.command()
@click.option("--store", "store_path", required=True, type=click.Path())
@click.option("--host", default="127.0.0.1")
@click.option("--port", type=int, default=8080)
@click.option("--audit-log", "audit_log_path", type=click.Path(), default=None,
              help="Path to the auto-trader's JSONL audit log for the "
                   "/auto-trade panel. Default <store-dir>/auto-trade.jsonl.")
def dashboard(store_path: str, host: str, port: int,
              audit_log_path: Optional[str]) -> None:
    """Start the FastAPI dashboard reading scout.db.

    Default binds to 127.0.0.1 so it isn't exposed without an explicit
    --host 0.0.0.0. The dashboard is read-only; never writes to the store.
    """
    import uvicorn
    from kalshi_scout.server import create_app
    app = create_app(store_path, audit_log_path=audit_log_path)
    console.print(
        f"[bold green]kalshi-scout dashboard[/bold green] -> http://{host}:{port}"
    )
    uvicorn.run(app, host=host, port=port, log_level="info")


@main.group()
def positions() -> None:
    """Manage manually-tracked open positions (for risk aggregation)."""


@positions.command("add")
@click.option("--store", "store_path", required=True, type=click.Path())
@click.argument("market_ticker")
@click.option("--side", required=True, type=click.Choice(["yes", "no"]))
@click.option("--size", required=True, type=int, help="Number of contracts.")
@click.option("--price", required=True, type=int, help="Average fill price in cents (1..99).")
@click.option("--event", "event_ticker", default=None,
              help="Event ticker (auto-derived from market_ticker if omitted).")
@click.option("--note", default="", help="Free-text note.")
def positions_add(store_path: str, market_ticker: str, side: str,
                  size: int, price: int, event_ticker: Optional[str],
                  note: str) -> None:
    """Record a new open position."""
    derived_event = event_ticker or market_ticker.rsplit("-", 1)[0]
    with SnapshotStore(store_path) as store:
        pid = store.add_position(
            market_ticker=market_ticker,
            event_ticker=derived_event,
            side=side, size_contracts=size, avg_price_cents=price,
            notes=note,
        )
    console.print(f"[green]added position id={pid}[/green]: {market_ticker} {side} {size}@{price}c")


@positions.command("close")
@click.option("--store", "store_path", required=True, type=click.Path())
@click.argument("position_id", type=int)
@click.option("--at-price", "at_price_cents", type=int, default=None,
              help="Per-contract exit price in cents. Use 100 for a winning "
                   "settlement, 0 for a losing one, or the mid-trade close "
                   "price you actually took. Enables realized-P&L on listing.")
def positions_close(store_path: str, position_id: int,
                    at_price_cents: Optional[int]) -> None:
    """Mark a position closed. Pass `--at-price N` to capture the exit
    value — needed for realized-P&L reporting on `positions list`."""
    with SnapshotStore(store_path) as store:
        ok = store.close_position(position_id, at_price_cents=at_price_cents)
    if ok:
        price_note = (
            f" @ {at_price_cents}c" if at_price_cents is not None
            else " (no exit price recorded — P&L unavailable)"
        )
        console.print(f"[green]closed position {position_id}[/green]{price_note}")
    else:
        console.print(f"[yellow]position {position_id} not found or already closed[/yellow]")


@positions.command("list")
@click.option("--store", "store_path", required=True, type=click.Path())
@click.option("--all", "show_all", is_flag=True, help="Include closed positions.")
def positions_list(store_path: str, show_all: bool) -> None:
    """List positions (default: open only). Closed rows show realized P&L
    when an exit price was captured at close time."""
    with SnapshotStore(store_path) as store:
        rows = store.query_positions(open_only=not show_all)
    if not rows:
        console.print("[dim]no positions[/dim]")
        return
    table = Table(header_style="bold cyan")
    table.add_column("ID", justify="right")
    table.add_column("Market")
    table.add_column("Side")
    table.add_column("Size", justify="right")
    table.add_column("Entry", justify="right")
    table.add_column("Exit", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("Opened")
    table.add_column("Closed")
    for r in rows:
        exit_str = f"{r.closed_at_price_cents}c" if r.closed_at_price_cents is not None else "—"
        if r.realized_pnl_cents is None:
            pnl_str = "—"
        else:
            pnl_dollars = r.realized_pnl_cents / 100.0
            # Skip Rich markup for break-even so the row inherits the user's
            # terminal foreground (white-on-white is invisible on light themes).
            if r.realized_pnl_cents > 0:
                pnl_str = f"[green]${pnl_dollars:+.2f}[/green]"
            elif r.realized_pnl_cents < 0:
                pnl_str = f"[red]${pnl_dollars:+.2f}[/red]"
            else:
                pnl_str = f"${pnl_dollars:+.2f}"
        table.add_row(
            str(r.id), r.market_ticker, r.side, str(r.size_contracts),
            f"{r.avg_price_cents}c", exit_str,
            f"${r.cost_basis_cents / 100:.2f}",
            pnl_str,
            r.opened_at_utc.strftime("%m-%d %H:%M"),
            r.closed_at_utc.strftime("%m-%d %H:%M") if r.closed_at_utc else "—",
        )
    console.print(table)


def _derive_take_side(snap) -> tuple[str, Optional[int]]:
    """Pick the actionable (side, ask_cents) from a snapshot.

    LOCKED_YES → yes side, yes_ask price.
    DEAD_NO    → no side, no_ask price.
    Other     → whichever edge is larger.
    """
    if snap.state == "locked_yes":
        return "yes", snap.yes_ask
    if snap.state == "dead_no":
        return "no", snap.no_ask
    yes_e = snap.edge_yes if snap.edge_yes is not None else float("-inf")
    no_e = snap.edge_no if snap.edge_no is not None else float("-inf")
    if no_e > yes_e:
        return "no", snap.no_ask
    return "yes", snap.yes_ask


@main.command()
@click.option("--store", "store_path", required=True, type=click.Path())
@click.argument("market_ticker")
@click.option("--size", required=True, type=int, help="Number of contracts.")
@click.option("--side", type=click.Choice(["yes", "no"]), default=None,
              help="Override the side derived from the latest snapshot.")
@click.option("--price", type=int, default=None,
              help="Override the per-contract fill price (cents). Defaults to "
                   "the snapshot's yes_ask/no_ask for the chosen side.")
@click.option("--note", default="", help="Free-text note.")
@click.option("--dry-run", is_flag=True,
              help="Show what would be recorded without writing to the store.")
def take(store_path: str, market_ticker: str, size: int,
         side: Optional[str], price: Optional[int],
         note: str, dry_run: bool) -> None:
    """Record a position you just took manually, auto-filling side+price
    from the latest snapshot.

    Workflow: alert fires -> open Kalshi web UI -> place the trade ->
    `kalshi-scout take TICKER --size N`. The scout looks up the latest
    snapshot for the ticker, derives the actionable side from its
    state/edge, and uses its stored ask as the fill price. Override
    either with `--side` / `--price` if you took it at a different
    price than the snapshot recorded.
    """
    # Validate size up-front so --dry-run also catches typos like `--size 0`
    # instead of computing nonsensical costs and only failing at write time.
    if size <= 0:
        raise click.BadParameter(f"--size must be > 0, got {size}")

    with SnapshotStore(store_path) as store:
        snaps = store.query_snapshots(market_ticker=market_ticker, limit=1)
        if not snaps:
            console.print(
                f"[red]no snapshot found for {market_ticker}. "
                f"Run `kalshi-scout evaluate {market_ticker} --store {store_path}` "
                "first, or pass --side and --price explicitly via `positions add`.[/red]"
            )
            sys.exit(1)
        snap = snaps[0]
        derived_side, derived_ask = _derive_take_side(snap)
        side = side or derived_side
        if price is None:
            if derived_side == side and derived_ask is not None:
                price = derived_ask
            elif side == "yes":
                price = snap.yes_ask
            else:
                price = snap.no_ask
        if price is None or price <= 0 or price >= 100:
            # add_position requires 0 < price < 100 — surface the same
            # friendly error here so a settled-LOCKED_YES snap (ask=100) or
            # missing-ask doesn't crash with a raw ValueError stack trace.
            console.print(
                f"[red]no usable {side}_ask in the latest snapshot "
                f"(got {price}c). Pass --price explicitly (must be 1..99).[/red]"
            )
            sys.exit(1)

        event_ticker = market_ticker.rsplit("-", 1)[0]
        # Clamp future-dated snapshots to "0m ago" instead of reporting a
        # misleading negative minute count from clock skew.
        snap_age = max(
            datetime.now(timezone.utc) - snap.scanned_at_utc,
            timedelta(0),
        )
        override_note = ""
        if side != derived_side:
            override_note = (
                f"side overridden: took {side}, scout would have taken {derived_side}"
            )
        snap_note = (
            f"scout: grade={snap.grade} state={snap.state} "
            f"fair={snap.fair_prob_low * 100:.0f}-{snap.fair_prob_high * 100:.0f}% "
            f"(snapshot {snap.scanned_at_utc.strftime('%m-%d %H:%M')}, "
            f"{int(snap_age.total_seconds() // 60)}m ago)"
        )
        final_note = "; ".join(n for n in (note, override_note, snap_note) if n)

        if dry_run:
            console.print(Panel(
                f"[bold]would record:[/bold]\n"
                f"  ticker: {market_ticker}\n"
                f"  event:  {event_ticker}\n"
                f"  side:   {side}\n"
                f"  size:   {size}\n"
                f"  price:  {price}c\n"
                f"  cost:   ${size * price / 100:.2f}\n"
                f"  note:   {final_note}",
                title="take --dry-run", border_style="yellow",
            ))
            return

        pid = store.add_position(
            market_ticker=market_ticker, event_ticker=event_ticker,
            side=side, size_contracts=size, avg_price_cents=price,
            notes=final_note,
        )
    console.print(
        f"[green]took position id={pid}[/green]: "
        f"{market_ticker} {side} {size}@{price}c "
        f"(cost ${size * price / 100:.2f})"
    )


@main.command()
@click.argument("market_ticker")
@click.option("--store", "store_path", required=True, type=click.Path())
@click.option("--size", required=True, type=int, help="Number of contracts.")
@click.option("--api-key-id", envvar="KALSHI_API_KEY_ID", default=None,
              help="Kalshi API key ID (or KALSHI_API_KEY_ID env var). "
                   "Not required with --paper.")
@click.option("--api-key-path", envvar="KALSHI_API_KEY_PATH", default=None,
              type=click.Path(exists=True),
              help="Path to Kalshi API private key PEM file "
                   "(or KALSHI_API_KEY_PATH env var). Not required with --paper.")
@click.option("--side", type=click.Choice(["yes", "no"]), default=None,
              help="Override the side derived from the latest snapshot.")
@click.option("--price", type=int, default=None,
              help="Override the snapshot's ask. Must be 1..99.")
@click.option("--paper", is_flag=True,
              help="Run the risk-guard + audit pipeline without calling the API.")
@click.option("--yes", "skip_confirm", is_flag=True,
              help="Skip the interactive y/N confirmation. Required for "
                   "non-interactive shells.")
@click.option("--max-position-size", type=int, default=5)
@click.option("--max-position-cost", type=int, default=500)
@click.option("--max-daily-loss", type=int, default=5000)
@click.option("--max-concentration-per-event", type=int, default=2500)
@click.option("--min-edge", "min_edge_cents_opt", type=int, default=5)
@click.option("--rounding-buffer", type=float, default=0.5)
@click.option("--kill-file", type=click.Path(), default=None)
@click.option("--audit-log", type=click.Path(), default=None)
@click.option("--refresh-quote", is_flag=True,
              help="Pull the live Kalshi quote right before placement and "
                   "use it as the ask instead of the snapshot's. Guards "
                   "against acting on a 5+-minute-old snapshot when the "
                   "book has moved. `--price` (if passed) still wins.")
def fire(market_ticker: str, store_path: str, size: int,
         api_key_id: Optional[str], api_key_path: Optional[str],
         side: Optional[str], price: Optional[int],
         paper: bool, skip_confirm: bool,
         max_position_size: int, max_position_cost: int,
         max_daily_loss: int, max_concentration_per_event: int,
         min_edge_cents_opt: int, rounding_buffer: float,
         kill_file: Optional[str], audit_log: Optional[str],
         refresh_quote: bool) -> None:
    """Single-shot manual order through the same risk-guard + audit pipeline
    as `serve --auto-trade`.

    Use this to validate the auth path with Kalshi (1-contract paper order
    first, then 1-contract live) before turning on the full auto-trader.
    Also handy for overriding the bot when you want to take a position the
    risk guards would normally refuse — bump --max-* knobs explicitly.
    """
    if not paper and (not api_key_id or not api_key_path):
        raise click.BadParameter(
            "live `fire` requires --api-key-id and --api-key-path "
            "(or KALSHI_API_KEY_ID / KALSHI_API_KEY_PATH env vars). "
            "Use --paper to dry-run."
        )
    if size <= 0:
        raise click.BadParameter(f"--size must be > 0, got {size}")

    store_dir = Path(store_path).resolve().parent
    kill_path = Path(kill_file) if kill_file else store_dir / "scout.kill"
    audit_path = Path(audit_log) if audit_log else store_dir / "auto-trade.jsonl"
    limits = RiskLimits(
        max_position_size_contracts=max_position_size,
        max_position_cost_cents=max_position_cost,
        max_daily_loss_cents=max_daily_loss,
        max_concentration_per_event_cents=max_concentration_per_event,
        min_edge_cents=min_edge_cents_opt,
        rounding_risk_buffer_f=rounding_buffer,
    )

    with SnapshotStore(store_path) as store:
        snaps = store.query_snapshots(market_ticker=market_ticker, limit=1)
        if not snaps:
            console.print(
                f"[red]no snapshot found for {market_ticker} — run "
                f"`kalshi-scout evaluate {market_ticker} --store {store_path}` first.[/red]"
            )
            sys.exit(1)
        snap = snaps[0]

        # Synthesize an Alert (the AutoTrader expects one for the audit trail).
        from kalshi_scout.notify import Alert
        alert = Alert(
            fired_at_utc=datetime.now(timezone.utc),
            market_ticker=snap.market_ticker,
            event_ticker=snap.event_ticker,
            city_slug=snap.city_slug, market_date=snap.market_date.isoformat(),
            bracket=f"{snap.bracket_kind} lo={snap.bracket_lo} hi={snap.bracket_hi}",
            metric=snap.metric, state=snap.state,
            reason=f"manual fire by operator (grade {snap.grade})",
            grade=snap.grade, previous_grade=None,
            yes_ask_cents=snap.yes_ask, no_ask_cents=snap.no_ask,
            edge_yes=snap.edge_yes, edge_no=snap.edge_no,
            fair_prob_low=snap.fair_prob_low, fair_prob_high=snap.fair_prob_high,
            notes=[],
        )

        # Derive side/price (allowing overrides).
        derived_side, derived_ask = AutoTrader._derive_side_and_price(snap)
        if side is None:
            side = derived_side
        if price is None:
            price = derived_ask if (side == derived_side and derived_ask is not None) else (
                snap.yes_ask if side == "yes" else snap.no_ask
            )
        if price is None or price <= 0 or price >= 100:
            console.print(
                f"[red]no usable {side}_ask in snapshot (got {price}c). "
                "Pass --price explicitly (1..99).[/red]"
            )
            sys.exit(1)

        # Confirmation prompt — last chance to bail.
        cost_dollars = size * price / 100.0
        mode = "[yellow]PAPER[/yellow]" if paper else "[bold red]LIVE[/bold red]"
        console.print(Panel(
            f"  ticker:  [bold]{market_ticker}[/bold]\n"
            f"  side:    {side}\n"
            f"  size:    {size} contracts\n"
            f"  price:   {price}c\n"
            f"  cost:    ${cost_dollars:.2f}\n"
            f"  mode:    {mode}\n"
            f"  grade:   {snap.grade}    state: {snap.state}",
            title=f"fire {market_ticker}", border_style="yellow",
        ))
        if not skip_confirm:
            if not click.confirm("Place this order?", default=False):
                console.print("[dim]aborted by operator[/dim]")
                return

        trading_client: Optional[KalshiTradingClient] = None
        if not paper:
            trading_client = KalshiTradingClient(
                key_id=api_key_id, private_key_path=Path(api_key_path),
            )
        rt_kalshi_client = KalshiClient() if refresh_quote else None
        try:
            guard = RiskGuard(limits, store, KillSwitch(kill_path))
            trader = AutoTrader(
                client=trading_client, guard=guard, store=store,
                default_size=size, paper=paper, audit_log_path=audit_path,
                kalshi_client=rt_kalshi_client,
            )
            # Pass the operator's explicit side/price through so the order
            # placed matches the one shown in the confirmation panel.
            # Without these the AutoTrader would re-derive from the
            # snapshot, ignoring --side / --price overrides.
            attempt = trader.maybe_trade(
                alert, snap, size=size,
                side_override=side, price_override=price,
                refresh_quote=refresh_quote,
            )
        finally:
            if trading_client is not None:
                trading_client.close()
            if rt_kalshi_client is not None:
                rt_kalshi_client.close()

    if attempt.placed:
        order_str = f" order_id={attempt.order_id}" if attempt.order_id else ""
        console.print(
            f"[green]PLACED[/green]: {attempt.market_ticker} {attempt.side} "
            f"{attempt.size_contracts}@{attempt.price_cents}c "
            f"{'(paper)' if attempt.paper else ''}{order_str} "
            f"position_id={attempt.position_id}"
        )
    else:
        console.print(f"[red]REFUSED[/red]: {attempt.reason}")
        sys.exit(1)


@main.command()
@click.option("--store", "store_path", required=True, type=click.Path())
@click.option("--json", "as_json", is_flag=True)
def risk(store_path: str, as_json: bool) -> None:
    """Pre-flight risk aggregation across open positions.

    Buckets exposure by city / market_date / regime / event, and flags
    event collisions (Yes positions across multiple brackets of the same
    event, which guarantee partial loss).
    """
    with SnapshotStore(store_path) as store:
        report = aggregate_risk(store)

    if as_json:
        click.echo(json.dumps({
            "total_open_positions": report.total_open_positions,
            "total_open_contracts": report.total_open_contracts,
            "total_max_loss_cents": report.total_max_loss_cents,
            "collisions": [
                {"event": c.event_ticker,
                 "yes_positions": len(c.yes_positions),
                 "guaranteed_loss_cents": c.guaranteed_loss_cents}
                for c in report.event_collisions
            ],
        }, indent=2))
        return

    if report.total_open_positions == 0:
        console.print("[dim]no open positions tracked[/dim]")
        return

    console.print(
        f"[bold]{report.total_open_positions}[/bold] open positions / "
        f"[bold]{report.total_open_contracts}[/bold] contracts / "
        f"[bold]${report.total_max_loss_dollars:.2f}[/bold] max loss"
    )

    if report.event_collisions:
        coll_table = Table(title="⚠ Event collisions", header_style="bold red")
        coll_table.add_column("Event")
        coll_table.add_column("Yes positions", justify="right")
        coll_table.add_column("Cost basis", justify="right")
        coll_table.add_column("Guaranteed loss", justify="right")
        for c in report.event_collisions:
            coll_table.add_row(
                c.event_ticker, str(len(c.yes_positions)),
                f"${c.total_max_loss_cents / 100:.2f}",
                f"[red]${c.guaranteed_loss_cents / 100:.2f}[/red]",
            )
        console.print(coll_table)

    for title, bucket in (
        ("By city", report.by_city),
        ("By market date", report.by_market_date),
        ("By regime", report.by_regime),
    ):
        t = Table(title=title, header_style="bold cyan")
        t.add_column("Key")
        t.add_column("Positions", justify="right")
        t.add_column("Contracts", justify="right")
        t.add_column("Max loss", justify="right")
        for k, b in sorted(bucket.items(), key=lambda kv: -kv[1].total_max_loss_cents):
            t.add_row(
                k, str(b.n_positions), str(b.total_contracts),
                f"${b.total_max_loss_dollars:.2f}",
            )
        console.print(t)


@main.command()
@click.option("--fee-per-leg", type=int, default=2,
              help="Estimated Kalshi fee per leg in cents (default 2c).")
@click.option("--min-edge", type=int, default=1,
              help="Minimum net edge in cents after fees (default 1).")
@click.option("--min-brackets", type=int, default=3,
              help="Skip events with fewer than this many brackets (default 3).")
@click.option("--limit", type=int, default=50,
              help="Show top N opportunities.")
@click.option("--max-markets", type=int, default=None,
              help="Stop crawling after this many markets (for fast partial scans).")
@click.option("--all-events", "all_events", is_flag=True,
              help="Diagnostic: include events whose brackets aren't verified "
                   "mutually exclusive. Most hits will be FALSE POSITIVES. "
                   "Do not trade off this output.")
@click.option("--strict-mex", is_flag=True,
              help="Whitelist-only MEX gating (the pre-detector V1.1 behavior). "
                   "Disables the algorithmic numeric-partition fallback.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of a table.")
def arbitrage(fee_per_leg: int, min_edge: int, min_brackets: int,
              limit: int, max_markets: Optional[int],
              all_events: bool, strict_mex: bool, as_json: bool) -> None:
    """Cross-bracket arbitrage scan across ALL open Kalshi events.

    Category-agnostic — works on weather, sports, politics, econ data, etc.
    For each event with mutually-exclusive brackets, computes:

      Yes-basket arb: profit = 100 - Σ yes_asks - N × fee_per_leg
      No-basket arb:  profit = Σ yes_bids - 100 - N × fee_per_leg

    Ranks events by net edge (after fees) and shows the top N. This is a
    one-shot scan; it does not persist to the snapshot store. Wire into
    cron / `serve` once the math is validated against real prices.
    """
    # When --json is on, route human diagnostics (progress, warnings) to
    # stderr so stdout stays a parseable JSON document for `| jq` consumers.
    diag = Console(stderr=True) if as_json else console

    diag.print(
        f"[dim]scanning all open Kalshi events "
        f"(fee={fee_per_leg}c/leg, min_edge={min_edge}c, min_brackets={min_brackets})...[/dim]"
    )
    def _progress(markets_seen: int, events_seen: int) -> None:
        diag.print(f"[dim]  {markets_seen} markets, {events_seen} events grouped...[/dim]")

    with KalshiClient() as kclient:
        events = list(iter_all_open_events(
            kclient, min_brackets=min_brackets,
            on_progress=_progress, max_markets=max_markets,
        ))

    diag.print(f"[dim]{len(events)} multi-bracket events meet the bracket threshold[/dim]")

    if all_events:
        diag.print(
            "[bold red]WARNING: --all-events ON — most hits below are FALSE "
            "POSITIVES from non-mutually-exclusive events. Do NOT trade.[/bold red]"
        )
    arbs = rank_arbitrage_opportunities(
        events, fee_per_leg_cents=fee_per_leg, min_net_edge_cents=min_edge,
        require_mex=not all_events, strict_mex=strict_mex,
    )

    if as_json:
        click.echo(json.dumps(
            [
                {
                    "event_ticker": a.event_ticker,
                    "n_brackets": a.n_brackets,
                    "n_priced": a.n_priced_brackets,
                    "sum_yes_asks_cents": a.sum_yes_asks_cents,
                    "sum_yes_bids_cents": a.sum_yes_bids_cents,
                    "best_side": a.best_side,
                    "best_net_edge_cents": a.best_net_edge_cents,
                    "fee_per_leg_cents": a.fee_per_leg_cents,
                    "market_tickers": list(a.market_tickers),
                    "notes": list(a.notes),
                }
                for a in arbs[:limit]
            ],
            indent=2,
        ))
        return

    if not arbs:
        console.print("[yellow]no arbitrage opportunities above threshold[/yellow]")
        console.print("[dim]try lowering --min-edge or --fee-per-leg[/dim]")
        return

    table = Table(
        title=f"Cross-bracket arbitrage (top {min(limit, len(arbs))} of {len(arbs)}; "
              f"edge ≥ {min_edge}c after {fee_per_leg}c/leg fees)",
        header_style="bold cyan",
    )
    table.add_column("Event")
    table.add_column("N", justify="right")
    table.add_column("Priced", justify="right")
    table.add_column("Σ yes_ask", justify="right")
    table.add_column("Σ yes_bid", justify="right")
    table.add_column("Side", justify="center")
    table.add_column("Net edge", justify="right")
    for a in arbs[:limit]:
        sum_asks_s = f"{a.sum_yes_asks_cents}c" if a.sum_yes_asks_cents is not None else "—"
        sum_bids_s = f"{a.sum_yes_bids_cents}c" if a.sum_yes_bids_cents is not None else "—"
        edge_color = "green" if (a.best_net_edge_cents or 0) >= 3 else "yellow"
        table.add_row(
            a.event_ticker,
            str(a.n_brackets),
            f"{a.n_priced_brackets}/{a.n_brackets}",
            sum_asks_s,
            sum_bids_s,
            (a.best_side or "—").upper(),
            f"[bold {edge_color}]+{a.best_net_edge_cents}c[/bold {edge_color}]",
        )
    console.print(table)
    console.print(
        f"[dim]To act: buy 1 contract on every bracket of the chosen event "
        f"on the SIDE column. Profit = {{Net edge}} × basket_size.[/dim]"
    )


@main.command(name="mex-check")
@click.argument("event_ticker")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of a panel.")
def mex_check(event_ticker: str, as_json: bool) -> None:
    """Inspect one event's MEX (mutually-exclusive) eligibility.

    Reports whether the event passes the curated `MUTUALLY_EXCLUSIVE_SERIES`
    whitelist, what the algorithmic numeric-partition detector decides and
    why, and the parsed yes_sub_title interval for each market.

    Use when deciding whether to trust a new series' arbitrage signal —
    or to debug why a known-MEX event is being rejected.
    """
    with KalshiClient() as kclient:
        markets = list(kclient.iter_markets(event_ticker=event_ticker))
    if not markets:
        console.print(f"[red]no markets found for event {event_ticker}[/red]")
        sys.exit(1)
    event = KalshiEvent(
        event_ticker=event_ticker, series_ticker="",
        title="", sub_title="", markets=markets,
    )
    series = (event.event_ticker.split("-", 1)[0] if event.event_ticker else "").upper()
    in_whitelist = series in MUTUALLY_EXCLUSIVE_SERIES
    detection = detect_numeric_partition(event)
    accepted = is_mutually_exclusive_event(event)

    parsed_intervals = [_parse_interval(m.yes_sub_title) for m in markets]

    if as_json:
        click.echo(json.dumps({
            "event_ticker": event_ticker,
            "series": series,
            "n_markets": len(markets),
            "in_whitelist": in_whitelist,
            "detector": {
                "is_mex": detection.is_mex,
                "reason": detection.reason,
                "n_parsed": detection.n_parsed,
            },
            "accepted_by_default_gate": accepted,
            "accepted_by_strict_gate": in_whitelist,
            "markets": [
                {
                    "ticker": m.ticker,
                    "yes_sub_title": m.yes_sub_title,
                    "parsed_interval": _interval_for_json(iv),
                }
                for m, iv in zip(markets, parsed_intervals)
            ],
        }, indent=2))
        return

    verdict = "[green]ACCEPTED[/green]" if accepted else "[red]REJECTED[/red]"
    body = (
        f"[bold]{event_ticker}[/bold]   series={series}   n_markets={len(markets)}\n"
        f"\n"
        f"whitelist match:  {'[green]yes[/green]' if in_whitelist else '[dim]no[/dim]'}\n"
        f"detector verdict: "
        f"{'[green]is_mex=True[/green]' if detection.is_mex else '[red]is_mex=False[/red]'}"
        f"  ({detection.n_parsed}/{detection.n_markets} parsed)\n"
        f"reason:           {detection.reason}\n"
        f"\n"
        f"default gate:     {verdict}\n"
        f"strict gate:      {'[green]ACCEPTED[/green]' if in_whitelist else '[red]REJECTED[/red]'}"
    )
    console.print(Panel(body, title="MEX check", border_style="cyan"))

    sample_table = Table(title="Markets (first 10)", header_style="bold cyan")
    sample_table.add_column("Ticker")
    sample_table.add_column("yes_sub_title")
    sample_table.add_column("Parsed interval")
    for m, iv in zip(markets[:10], parsed_intervals[:10]):
        sample_table.add_row(
            m.ticker,
            m.yes_sub_title or "[dim]—[/dim]",
            _interval_for_display(iv),
        )
    console.print(sample_table)
    if len(markets) > 10:
        console.print(f"[dim]  (+{len(markets) - 10} more markets)[/dim]")


def _interval_for_json(iv: Optional[tuple[float, float]]) -> Optional[dict]:
    """JSON-safe encoding of a parsed interval. Infinite endpoints become
    the strings "-inf" / "+inf" since json.dumps refuses non-finite floats."""
    if iv is None:
        return None
    return {
        "lo": _endpoint_for_json(iv[0]),
        "hi": _endpoint_for_json(iv[1]),
    }


def _endpoint_for_json(v: float):
    if v == float("-inf"):
        return "-inf"
    if v == float("inf"):
        return "+inf"
    return v


def _interval_for_display(iv: Optional[tuple[float, float]]) -> str:
    if iv is None:
        return "[red]unparseable[/red]"
    lo = "−∞" if iv[0] == float("-inf") else f"{iv[0]:g}"
    hi = "+∞" if iv[1] == float("inf") else f"{iv[1]:g}"
    if iv[0] == iv[1]:
        return f"= {lo}"
    return f"[{lo}, {hi}]"


if __name__ == "__main__":
    main()
