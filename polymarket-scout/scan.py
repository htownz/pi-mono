#!/usr/bin/env python3
"""pmscan CLI — read-only Polymarket sum-to-one scanner (detection + logging only).

Binary (Phase 1):   python scan.py --once
NegRisk (Phase 1b): python scan.py --negrisk --once

No keys, no wallet, no signing, no orders. Anywhere. Ever (in this phase).
"""
from __future__ import annotations

import argparse
import sys
import time

from pmscan.client import ClobClient, GammaClient, parse_event, parse_market
from pmscan.models import Market, NegRiskEvent
from pmscan.scanner import group_negrisk, scan_market, scan_negrisk


def _discover(max_markets: int, min_volume: float) -> list[Market]:
    gamma = GammaClient()
    markets: list[Market] = []
    for raw in gamma.iter_active_markets(max_markets=max_markets):
        m = parse_market(raw)
        if m is None or m.volume_24hr < min_volume:
            continue
        markets.append(m)
    return markets


def _discover_events(max_events: int, min_volume: float) -> list[NegRiskEvent]:
    """Pull complete NegRisk events (all outcomes) from Gamma /events.

    min_volume filters on the event's summed outcome volume so a sparse, low-interest
    event doesn't dilute the scan.
    """
    gamma = GammaClient()
    events: list[NegRiskEvent] = []
    for raw in gamma.iter_active_events(max_events=max_events):
        ev = parse_event(raw)
        if ev is None:
            continue
        if sum(m.volume_24hr for m in ev.outcomes) < min_volume:
            continue
        events.append(ev)
    return events


def _log(line: str, out_path: str | None) -> None:
    if out_path:
        with open(out_path, "a") as f:
            f.write(line + "\n")


# --------------------------------------------------------------------------- #
def run_binary(args) -> int:
    markets = [m for m in _discover(args.max_markets, args.min_volume) if not m.neg_risk]
    token_ids = [t for m in markets for t in m.token_ids]
    books = ClobClient().get_books(token_ids)

    hits = 0
    for m in markets:
        opp = scan_market(m, books, fee_per_share=args.fee, gas_usd=args.gas)
        if opp is None:
            continue
        if opp.edge_cents < args.min_edge or opp.capturable_sets < args.min_sets:
            continue
        hits += 1
        print(f"[{opp.side:5}] {opp.edge_cents:5.2f}c x{opp.capturable_sets:>8.0f} sets  "
              f"net=${opp.net_profit_usd:>9.2f}  {opp.question[:70]}")
        _log(opp.to_json(), args.out)

    print(f"-- binary: scanned {len(markets)} markets, {hits} crossing edge(s) "
          f">= {args.min_edge}c / {args.min_sets} sets", file=sys.stderr)
    return hits


def run_negrisk(args) -> int:
    if args.complete_events:
        # Authoritative: each event carries its full outcome set (no truncation artifacts).
        events = _discover_events(args.max_events, args.min_volume)
        source = f"{len(events)} complete event(s) via /events"
    else:
        # Fallback: regroup a volume-truncated /markets sample. Fast, but baskets may be
        # partial — only trust results whose implied mass lands near 1.0.
        markets = _discover(args.max_markets, args.min_volume)
        events = group_negrisk(markets)
        source = (f"{len(events)} group(s) from a {args.max_markets}-market sample "
                  f"(TRUNCATED — partial baskets expected; prefer --complete-events)")
        print(f"-- negrisk: WARNING grouping from sample, not /events — "
              f"sub-$1 baskets here are likely fragments, not edges.", file=sys.stderr)
    # one batched book fetch for every outcome's YES token across all events
    yes_tokens = [t for ev in events for m in ev.outcomes if (t := m.yes_token())]
    books = ClobClient().get_books(yes_tokens)

    hits = 0
    skipped_incomplete = 0
    for ev in events:
        opp = scan_negrisk(ev, books, fee_per_share=args.fee, gas_usd=args.gas)
        if opp is None:
            # crossing absent or a leg's book was missing — distinguish for visibility
            if any(m.yes_token() not in books for m in ev.outcomes):
                skipped_incomplete += 1
            continue
        if opp.edge_cents < args.min_edge or opp.capturable_sets < args.min_sets:
            continue
        hits += 1
        tag = "OK " if opp.exhaustive_verified else "??!"
        print(f"[negrisk {tag}] {opp.edge_cents:5.2f}c  N={opp.legs:>2}  "
              f"ask_sum={opp.ask_sum:.3f} mass={opp.implied_mass:.3f} "
              f"other={opp.implied_other * 100:4.1f}c  "
              f"x{opp.capturable_sets:>7.0f} sets  net=${opp.net_profit_usd:>9.2f}  "
              f"{(opp.title or '')[:42]}")
        if not opp.exhaustive_verified:
            print(f"            ↳ uncertain: {opp.uncertainty_reason}")
        _log(opp.to_json(), args.out)

    print(f"-- negrisk: {source}, {hits} crossing edge(s), "
          f"{skipped_incomplete} skipped (incomplete books)", file=sys.stderr)
    return hits


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Read-only Polymarket sum-to-one scanner.")
    p.add_argument("--negrisk", action="store_true",
                   help="Phase 1b: scan NegRisk multi-outcome events (buy-all-YES).")
    p.add_argument("--complete-events", action=argparse.BooleanOptionalAction, default=True,
                   help="NegRisk: group full outcome sets via Gamma /events (default). "
                        "Use --no-complete-events to regroup a truncated /markets sample.")
    p.add_argument("--once", action="store_true", help="single pass then exit (default).")
    p.add_argument("--interval", type=float, default=None,
                   help="loop every N seconds (persistence measurement).")
    p.add_argument("--max-markets", type=int, default=800)
    p.add_argument("--max-events", type=int, default=1000,
                   help="NegRisk --complete-events: max events to pull from /events.")
    p.add_argument("--min-volume", type=float, default=0.0, help="24h USD volume filter.")
    p.add_argument("--min-edge", type=float, default=None,
                   help="min gross per-set edge, cents (loop default 1.0, single-pass 0).")
    p.add_argument("--min-sets", type=float, default=None,
                   help="min capturable top-of-book sets (loop default 5, single-pass 0).")
    p.add_argument("--fee", type=float, default=0.0, help="per-share per-leg fee (USD).")
    p.add_argument("--gas", type=float, default=0.01, help="round-trip gas (USD).")
    p.add_argument("--out", type=str, default=None, help="append matched opportunities as JSONL.")
    args = p.parse_args(argv)

    # Loop (persistence) mode defaults to noise filters so 0-set / sub-cent crossings don't
    # flood the log; a single pass shows everything. Explicit flags always win.
    loop = bool(args.interval and not args.once)
    if args.min_edge is None:
        args.min_edge = 1.0 if loop else 0.0
    if args.min_sets is None:
        args.min_sets = 5.0 if loop else 0.0

    runner = run_negrisk if args.negrisk else run_binary

    if args.interval and not args.once:
        try:
            while True:
                runner(args)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nstopped.", file=sys.stderr)
            return 0
    else:
        runner(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
