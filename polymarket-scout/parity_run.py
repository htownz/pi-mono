#!/usr/bin/env python3
"""Cross-venue parity runner (DRAFT). DETECTION ONLY — read-only, no wallet, no orders.

Pulls the Polymarket side live (Gamma + CLOB) and pairs it, via the hand-curated registry,
against Kalshi quotes. Both venues can be pulled live; Kalshi prices may also be supplied (or
overridden) by hand for tickers you can't resolve automatically.

  1. Edit pmscan/parity_registry.py so each candidate's pm_match + kalshi_ticker point at the
     SAME real-world outcome; set settlement_verified=True only once you've confirmed both
     venues resolve it identically (source, date, bucket).
  2. Run it:
       python parity_run.py --kalshi-live                       # auto-fetch both sides live
       python parity_run.py --kalshi-quotes kalshi_quotes.json  # hand-entered Kalshi prices (cents)
       python parity_run.py --kalshi-live --kalshi-quotes k.json # live + manual overrides
     (manual JSON shape: {"TICKER": {"yes_bid": 91, "yes_ask": 93, "no_bid": 7, "no_ask": 9}})
"""
from __future__ import annotations

import argparse
import json
import sys

from pmscan.client import ClobClient, GammaClient, parse_market
from pmscan.kalshi import KalshiQuotes
from pmscan.models import Market, OrderBook
from pmscan.parity import VenueQuote, kalshi_venue_quote, scan_parity_links
from pmscan.parity_registry import REGISTRY, build_links


def _load_kalshi_quotes(path: str) -> dict[str, VenueQuote]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, VenueQuote] = {}
    for ticker, q in raw.items():
        if not isinstance(q, dict) or not any(k in q for k in ("yes_ask", "no_ask", "yes_bid", "no_bid")):
            continue  # skip comments / non-quote keys
        out[ticker] = kalshi_venue_quote(
            ticker, label=ticker,
            yes_bid_c=q.get("yes_bid"), yes_ask_c=q.get("yes_ask"),
            no_bid_c=q.get("no_bid"), no_ask_c=q.get("no_ask"),
            yes_ask_size=q.get("yes_ask_size"), no_ask_size=q.get("no_ask_size"),
        )
    return out


def _pm_markets_for_registry(max_markets: int) -> tuple[list[Market], dict[str, OrderBook]]:
    """Fetch Polymarket markets, keep only those a registry candidate points at (matched on
    question OR slug), and fetch their books in one batch. NegRisk binary children are kept —
    most cross-venue candidates (team-winner / champion markets) are exactly those."""
    needles = [c.pm_match.lower() for c in REGISTRY]
    matched: list[Market] = []
    for raw in GammaClient().iter_active_markets(max_markets=max_markets):
        m = parse_market(raw)
        if m is None:
            continue
        hay = f"{m.question} {m.slug or ''}".lower()
        if any(n in hay for n in needles):
            matched.append(m)
    token_ids = [t for m in matched for t in m.token_ids]
    books = ClobClient().get_books(token_ids) if token_ids else {}
    return matched, books


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Cross-venue (Polymarket+Kalshi) parity runner.")
    p.add_argument("--kalshi-live", action="store_true",
                   help="auto-fetch Kalshi quotes for every registry ticker via the public API.")
    p.add_argument("--kalshi-quotes", default=None,
                   help="JSON of Kalshi quotes in cents (manual; supplements/overrides --kalshi-live).")
    p.add_argument("--max-markets", type=int, default=1500, help="Polymarket markets to scan for matches.")
    p.add_argument("--fee", type=float, default=0.0, help="per-leg fee (USD).")
    p.add_argument("--gas", type=float, default=0.0, help="per-lock settlement/gas (USD).")
    p.add_argument("--out", type=str, default=None, help="append crossing locks as JSONL.")
    args = p.parse_args(argv)

    if not args.kalshi_live and not args.kalshi_quotes:
        p.error("provide a Kalshi source: --kalshi-live and/or --kalshi-quotes <file>")

    kalshi_quotes: dict[str, VenueQuote] = {}
    if args.kalshi_live:
        tickers = sorted({c.kalshi_ticker for c in REGISTRY})
        kalshi_quotes.update(KalshiQuotes().get_quotes(tickers))
        print(f"-- parity: fetched {len(kalshi_quotes)}/{len(tickers)} Kalshi quotes live",
              file=sys.stderr)
    if args.kalshi_quotes:
        kalshi_quotes.update(_load_kalshi_quotes(args.kalshi_quotes))  # manual wins
    pm_markets, books = _pm_markets_for_registry(args.max_markets)
    links, unmatched = build_links(REGISTRY, pm_markets, books, kalshi_quotes)

    print(f"-- parity: {len(links)} linked outcome(s), {len(unmatched)} unmatched "
          f"(of {len(REGISTRY)} registry candidates)", file=sys.stderr)
    if unmatched:
        print(f"   unmatched (no live PM market and/or Kalshi quote): {', '.join(unmatched)}",
              file=sys.stderr)

    opps = scan_parity_links(links, fee_per_leg=args.fee, gas_usd=args.gas)
    if not opps:
        print("no cross-venue lock < $1 among the linked outcomes.", file=sys.stderr)
        return 0
    for o in opps:
        tag = "OK " if o.settlement_verified else "??!"
        sets = f"x{o.capturable_sets:>6.0f}" if o.capturable_sets is not None else "x   ?  "
        print(f"[parity {tag}] {o.edge_cents:5.2f}c  cost={o.cost_sum:.3f}  "
              f"buy YES@{o.yes_venue}={o.yes_ask:.3f} + NO@{o.no_venue}={o.no_ask:.3f}  "
              f"{sets} sets  {o.name[:46]}")
        if not o.settlement_verified:
            print(f"            ↳ UNVERIFIED settlement equivalence{(': ' + o.note) if o.note else ''}")
        if args.out:
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(o.to_json() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
