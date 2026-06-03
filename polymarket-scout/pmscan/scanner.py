"""Sum-to-one mispricing detectors. DETECTION ONLY — nothing here places an order.

Phase 1  — scan_market():    within a single binary market (YES+NO complete set).
Phase 1b — scan_negrisk():   across a NegRisk event's N mutually-exclusive outcomes.

Both are deliberately top-of-book and size-limited to min(leg sizes): we never walk deeper
levels, so a reported `capturable_sets` is the conservative floor a taker could lift at the
quoted price, not an optimistic depth-walked figure.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .models import Market, NegRiskEvent, NegRiskOpportunity, OrderBook, Opportunity


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Phase 1 — binary within-market (YES+NO) sum-to-one
# --------------------------------------------------------------------------- #
def scan_market(
    market: Market,
    books: dict[str, OrderBook],
    *,
    fee_per_share: float = 0.0,
    gas_usd: float = 0.01,
) -> Opportunity | None:
    """Detect a merge or split edge on one binary market.

    merge: best_ask(YES) + best_ask(NO) < $1  → buy a set, merge to $1.
    split: best_bid(YES) + best_bid(NO) > $1  → mint a set for $1, sell both.

    Returns the crossing Opportunity (which may net negative after fees/gas — the caller
    filters), or None when the book does not cross. Net models 2 legs/set of `fee_per_share`
    plus a flat round-trip `gas_usd`.
    """
    if not market.is_binary or len(market.token_ids) != 2:
        return None
    yes_book = books.get(market.token_ids[0])
    no_book = books.get(market.token_ids[1])
    if yes_book is None or no_book is None:
        return None

    ask_yes, ask_no = yes_book.best_ask(), no_book.best_ask()
    bid_yes, bid_no = yes_book.best_bid(), no_book.best_bid()

    ask_sum = (ask_yes.price + ask_no.price) if (ask_yes and ask_no) else None
    bid_sum = (bid_yes.price + bid_no.price) if (bid_yes and bid_no) else None

    side = yes_price = no_price = price_sum = None
    edge_per_set = 0.0
    capturable = 0.0

    if ask_sum is not None and ask_sum < 1.0:
        side, price_sum = "merge", ask_sum
        yes_price, no_price = ask_yes.price, ask_no.price
        edge_per_set = 1.0 - ask_sum
        capturable = min(ask_yes.size, ask_no.size)
    elif bid_sum is not None and bid_sum > 1.0:
        side, price_sum = "split", bid_sum
        yes_price, no_price = bid_yes.price, bid_no.price
        edge_per_set = bid_sum - 1.0
        capturable = min(bid_yes.size, bid_no.size)
    else:
        return None  # no crossing → no false positive

    gross = edge_per_set * capturable
    fees = 2.0 * fee_per_share * capturable          # YES + NO = 2 legs per set
    net = gross - fees - gas_usd
    return Opportunity(
        ts=_now(),
        venue=market.venue,
        market_id=market.market_id,
        question=market.question,
        slug=market.slug,
        side=side,
        yes_price=yes_price,
        no_price=no_price,
        price_sum=round(price_sum, 6),
        edge_cents=round(edge_per_set * 100.0, 4),
        capturable_sets=capturable,
        gross_profit_usd=round(gross, 6),
        net_profit_usd=round(net, 6),
        fees_enabled=market.fees_enabled,
        neg_risk=market.neg_risk,
        volume_24hr=market.volume_24hr,
    )


# --------------------------------------------------------------------------- #
# Phase 1b — NegRisk multi-outcome (buy-all-YES)
# --------------------------------------------------------------------------- #
def group_negrisk(markets: list[Market]) -> list[NegRiskEvent]:
    """Bucket NegRisk outcome markets into events by shared negRiskRequestID.

    Markets without a request id (or not flagged neg_risk) are ignored. An event is marked
    `has_other` if any of its outcome markets exposes the implicit Other/None bucket.
    """
    buckets: dict[str, NegRiskEvent] = {}
    for m in markets:
        if not m.neg_risk or not m.neg_risk_request_id:
            continue
        ev = buckets.get(m.neg_risk_request_id)
        if ev is None:
            ev = NegRiskEvent(request_id=m.neg_risk_request_id, title=m.group_title)
            buckets[m.neg_risk_request_id] = ev
        ev.outcomes.append(m)
        ev.has_other = ev.has_other or m.neg_risk_other
        if ev.title is None and m.group_title:
            ev.title = m.group_title
    return list(buckets.values())


def scan_negrisk(
    event: NegRiskEvent,
    books: dict[str, OrderBook],
    *,
    fee_per_share: float = 0.0,
    gas_usd: float = 0.01,
    mass_band: tuple[float, float] = (0.98, 1.02),
) -> NegRiskOpportunity | None:
    """Detect a buy-all-YES edge across a NegRisk event's outcomes.

    edge = $1 - Σ best_ask(YES_i). Exactly one outcome resolves $1, so a sub-$1 basket is a
    structural edge *iff* the set is exhaustive. We require ≥2 outcomes and a complete set of
    YES asks (a missing leg understates the sum → overstates the edge → we refuse to emit).

    IMPORTANT — what a static snapshot can and cannot tell you. With only top-of-book bid/ask,
    "missing probability mass" and "the edge" are the SAME quantity and cannot be separated:
    since mid_i = (bid_i+ask_i)/2 ≤ ask_i, we always have mass = Σ mid_i ≤ Σ ask_i = ask_sum,
    so 1 − mass ≥ 1 − ask_sum = edge. For fairly-priced two-sided books a *positive* edge
    therefore implies the listed set is non-exhaustive (the gap below $1 exceeds the spread
    only when probability is missing). A genuine arb comes from a transient ask *dislocation*;
    distinguishing that from structural missing-mass needs a TIME SERIES (see persistence
    logging), not a single frame.

    So the confidence flag here is a deliberately conservative proxy, not proof:
      - `has_other` (explicit Other/None bucket present)  → unverified;
      - implied mass Σ mid(YES_i) outside `mass_band` (default tight: [0.98, 1.02]) → the set
        is probably not a clean partition (too low ⇒ missing outcomes — which is also where
        the larger "edges" live; too high ⇒ overlap/dup).
    A verified flag means only "near-complete and small — still warrants structural/temporal
    confirmation," never "confirmed arbitrage." The edge is always reported (scan, flag as
    uncertain — never drop a candidate silently). bid_sum / spread / implied_other are exposed
    for the downstream time-series detector.

    Returns None when there is no crossing or the data is incomplete.
    """
    if event.n < 2:
        return None

    yes_books: list[tuple[Market, OrderBook]] = []
    for m in event.outcomes:
        tid = m.yes_token()
        bk = books.get(tid) if tid else None
        if bk is None or bk.best_ask() is None:
            return None  # incomplete leg data — refuse rather than under-sum the basket
        yes_books.append((m, bk))

    asks = [bk.best_ask() for _, bk in yes_books]
    ask_sum = sum(a.price for a in asks)
    edge_per_set = 1.0 - ask_sum
    if edge_per_set <= 0:
        return None  # basket already ≥ $1 — no buy-all-YES edge

    # Implied probability mass: for a clean exhaustive partition the YES mids should sum ≈ 1.
    mids = [bk.mid() for _, bk in yes_books]
    implied_mass = sum(md for md in mids if md is not None)
    implied_other = max(0.0, 1.0 - implied_mass)   # est. probability on unlisted/other outcomes

    # bid_sum / spread expose the book width for the downstream time-series detector. bid_sum
    # only counts legs that actually have a bid; spread is meaningful only when all legs do.
    bids = [bk.best_bid() for _, bk in yes_books]
    bid_sum = sum(b.price for b in bids if b is not None)
    spread = (ask_sum - bid_sum) if all(b is not None for b in bids) else float("nan")

    reasons: list[str] = []
    if event.has_other:
        reasons.append("explicit Other/None bucket present")
    lo, hi = mass_band
    if implied_mass < lo:
        reasons.append(f"implied mass {implied_mass:.3f} < {lo:.2f} (set likely not exhaustive; "
                       f"~{implied_other * 100:.1f}c of probability is on unlisted outcomes)")
    elif implied_mass > hi:
        reasons.append(f"implied mass {implied_mass:.3f} > {hi:.2f} (outcomes may overlap)")
    verified = not reasons

    capturable = min(a.size for a in asks)
    n = len(yes_books)
    gross = edge_per_set * capturable
    fees = n * fee_per_share * capturable            # one YES buy per outcome = N legs
    net = gross - fees - gas_usd

    fees_enabled = any(m.fees_enabled for m, _ in yes_books)
    vol = sum(m.volume_24hr for m, _ in yes_books)
    title = event.title or event.request_id

    return NegRiskOpportunity(
        ts=_now(),
        venue="polymarket",
        request_id=event.request_id,
        title=title,
        side="buy_all_yes",
        legs=n,
        outcomes=[m.question for m, _ in yes_books],
        ask_sum=round(ask_sum, 6),
        bid_sum=round(bid_sum, 6),
        spread=round(spread, 6),
        implied_mass=round(implied_mass, 6),
        implied_other=round(implied_other, 6),
        edge_cents=round(edge_per_set * 100.0, 4),
        capturable_sets=capturable,
        gross_profit_usd=round(gross, 6),
        net_profit_usd=round(net, 6),
        exhaustive_verified=verified,
        uncertainty_reason="; ".join(reasons),
        fees_enabled=fees_enabled,
        volume_24hr=vol,
    )
