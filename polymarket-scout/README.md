# pmscan — Polymarket sum-to-one scanner (Phase 0 + 1 + 1b + 1c, + Phase 2 parity draft)

Read-only Polymarket **scanner**: detection and logging only. It never authenticates, signs,
holds a wallet, or places an order. It is the Polymarket half of a planned unified
Kalshi + Polymarket edge-finding system (the Kalshi half lives in `../kalshi-scout`, whose
cross-bracket arbitrage math is the direct analogue of the NegRisk detector here).

## What it does

- **Phase 0 — data layer.** Pulls active markets from Polymarket's public **Gamma API**
  (`gamma-api.polymarket.com`) and live order books from the **CLOB API**
  (`clob.polymarket.com`), and normalizes them into venue-agnostic `Market` / `OrderBook`
  objects (`pmscan/models.py`). A future Kalshi adapter populates the same shapes, so the
  scanner logic never has to care which venue a market came from.
- **Phase 1 — binary sum-to-one detector** (`scan_market`). For each binary market it checks
  the two complementary edges enabled by Polymarket's atomic mint/merge:
  - **merge:** `best_ask(YES) + best_ask(NO) < $1` → buy a set, merge to $1.
  - **split:** `best_bid(YES) + best_bid(NO) > $1` → mint a set for $1, sell both.
  Profit is top-of-book size-limited to `min(size_yes, size_no)` (deliberately conservative —
  it does not walk deeper levels), then netted against modeled gas/fees.
- **Phase 1b — NegRisk multi-outcome detector** (`scan_negrisk`). A NegRisk event is a
  mutually-exclusive set where exactly one outcome resolves YES (e.g., "Which country wins the
  World Cup?"). Each outcome is its own binary market, linked by a shared `negRiskRequestID`.
  Since exactly one YES pays $1:

      buy-all-YES edge:  if  Σ best_ask(YES_i) < $1  → buy one YES per outcome,
                         exactly one resolves $1, profit = 1 − Σ.

  This is the clean, detection-safe check. The sell/convert side uses the on-chain NegRisk
  adapter `convertPositions` and is more involved — flagged here but **not** implemented in
  detection.

## Run

```bash
python scan.py --once                                  # binary, top 800 markets by 24h volume
python scan.py --once --max-markets 1500 --min-edge 1.0 --out opportunities.jsonl
python scan.py --negrisk --once                        # NegRisk, COMPLETE events via /events (default)
python scan.py --negrisk --interval 30 --out neg.jsonl --snapshot snap.jsonl  # watch + baseline log
python -m pmscan.temporal snap.jsonl --summary         # detect transient dips over the snapshot log
python test_scanner.py                                 # synthetic self-tests (36, all passing)
```

No install needed — standard library only, Python 3.10+.

Key flags: `--negrisk` (switch to multi-outcome mode), `--complete-events` (default; pull full
outcome sets from Gamma `/events` — see below), `--max-events` (cap for `/events`), `--min-edge`
(min gross per-set edge, cents), `--min-sets` (min capturable top-of-book sets), `--min-volume`
(24h USD filter), `--fee` (per-share per-leg, default 0), `--gas` (round-trip USD, default 0.01),
`--out` (append JSONL log), `--interval` (loop seconds), `--max-markets`.

### Why NegRisk grouping must come from `/events`

A NegRisk basket is only an edge if you have **every** outcome of the event. If you reconstruct
the group from a volume-ranked `/markets` sample (`--no-complete-events`), large events get
**truncated** — you capture only their most-liquid couple of legs and miss the frontrunners.
That produces baskets like *"Brazil Presidential Election, N=2, ask_sum=0.004"* — a 99.6¢
"edge" worth six figures on paper that is pure artifact (you priced two longshots, not the
event). `--complete-events` (the default) pulls each event with its full child-market set from
Gamma `/events`, so `N` is the true outcome count and the basket is actually complete. The
implied-mass check below is the backstop that catches any fragment that still slips through —
in the truncated case `mass ≈ 0`, which can never read as verified.

## The NegRisk exhaustiveness problem (and how we handle it)

The buy-all-YES identity `Σ ask(YES_i) < $1 ⇒ guaranteed edge` holds **only if the outcome
set is truly exhaustive** (exactly one of the listed outcomes must resolve YES). If the event
has an implicit "Other / None of the above" bucket, or the API simply doesn't list every
outcome, the naive sum understates the true price of a full hedge and the "edge" is a mirage.

You cannot prove exhaustiveness from the public API alone, so rather than silently trusting it
(false positives → phantom P/L) or silently dropping candidates (missed real edges), the
detector **scans and attaches a confidence flag**:

1. **Explicit Other bucket** — any outcome flagged `negRiskOther` ⇒ `exhaustive_verified=False`.
2. **Implied-probability-mass check.** For a clean partition the YES **mid** prices sum to ≈ $1
   (they are the market's probabilities). We compute `implied_mass = Σ mid(YES_i)` and
   `implied_other = max(0, 1 − mass)` (the probability the market puts on *unlisted* outcomes).
   Default acceptance band is the **tight** `[0.98, 1.02]`; outside it ⇒ `exhaustive_verified=
   False`. (See the hard limit below for why the band must be tight, not loose.)
3. **Incomplete legs** — if any outcome's YES book is missing, we **refuse** to emit (return
   `None`) rather than under-sum the basket and overstate the edge.

Verified hits print `[negrisk OK ]`; everything else prints `[negrisk ??!]` with a reason. Both
are logged with full transparency fields (`ask_sum, bid_sum, spread, implied_mass,
implied_other, edge_cents, exhaustive_verified`) for the time-series detector below.

### The hard limit: a static snapshot cannot separate "edge" from "missing mass"

This is the central finding from the first complete-event live run, and it shapes everything
after. With only top-of-book bid/ask, **the edge and the missing probability mass are the same
quantity.** Per leg `mid = (bid+ask)/2 ≤ ask`, so `mass = Σ mid ≤ Σ ask = ask_sum`, hence:

```
implied_other = 1 − mass  ≥  1 − ask_sum = edge        (always)
```

For fairly-priced two-sided books a **positive** buy-all-YES edge therefore *implies* the listed
set is non-exhaustive — the gap below $1 can exceed the spread only when probability is missing.
A genuine arb instead comes from a **transient ask dislocation**. You cannot tell the two apart
from one frame; the "risk-adjusted edge" `(1−ask_sum) − (1−mass) = mass − ask_sum` is `≤ 0` by
construction. So:

- `exhaustive_verified=True` here means only **"near-complete and small — still needs structural
  or temporal confirmation,"** never "confirmed arbitrage." (Tight mass band ⇒ only tiny edges
  qualify, which is the honest ceiling for a static check.)
- **The real discriminator is time.** A structural/phantom edge sits at a *stable* depressed
  `ask_sum` indefinitely; a real arb is a *brief dip* below the event's own rolling baseline.
  That is what the temporal detector below is for — it is the actual edge detector, not just a
  measuring tape.

## Phase 1c — temporal dislocation detector (`pmscan/temporal.py`)

Since a snapshot can't separate edge from missing-mass, we watch each event over **time**:

1. `scan.py --snapshot snap.jsonl` logs **every** event's basket level each cycle —
   *crossing or not*, `ask_sum` ≥ $1 included. (The crossing log `--out` can't feed this: it
   only records sub-$1 events, so it never captures the *normal* level a dip is measured against.)
2. `python -m pmscan.temporal snap.jsonl` builds a robust per-event baseline (median / MAD,
   so a real dip doesn't inflate its own benchmark) and flags **dips**: contiguous runs where
   `ask_sum` falls a robust z-score (`-k`, default 4) below baseline.
3. By default it reports only **capturable dislocations** — baskets normally priced **≥ $1**
   (no standing edge, i.e. efficient & exhaustive) that **transiently fell below $1**. This is
   the only economically real signal, and it excludes the two noise classes the raw relative
   view drowns in:
   - **illiquid high-baseline baskets** — a many-way futures market normally at `ask_sum` $2–5
     (wide spreads on every leg); a dip from 2.1→1.7 is still >$1, *not* an edge;
   - **structurally-cheap fragments** — non-exhaustive sets that sit below $1 *forever*
     (`Roland Garros` at 0.64); cheap is their normal, not a dislocation.
   Each dip reports **below-par edge** (`$1 − min_ask`, the actual buy-all-YES profit at the
   trough) and **duration** (`pts × poll interval`, the lifetime to act). `--all-dips` shows
   the raw relative view for diagnostics; `--summary` adds the per-event baseline table and the
   ≥$1 / <$1 split.

**Duration is the go/no-go**: a dip that survives only one cycle is a latency race we lose; one
that persists for minutes is potentially capturable. Run the snapshot log for hours, then let
the detector tell you whether any real dislocation persists at all.

> **First live read (~26 min, 469 events):** zero persistent capturable dislocations. The deep
> "dips" were all illiquid high-baseline baskets (still ≫ $1) or structural fragments; the only
> sub-$1 crossings were ≤ 2¢ and lasted 1–3 cycles (30–90s) — a latency race. Honest go/no-go
> so far: no capturable NegRisk edge at 30s cadence. Longer logs may surface news-driven ones.

## Live finding (Phase 1, first run, ~1,500 highest-volume markets)

**Zero crossable single-condition (binary) sum-to-one edges.** The tightest binary markets sit
at ask-sum `1.001` / bid-sum `0.999` — the complete set pinned to exactly $1.00 with a one-tick
spread straddling it. The detector correctly scores that as negative edge (no false positive;
`test_binary_no_edge_pinned_to_one`). This matches the research: bots keep binary
single-condition markets flat, and the documented ~$39.6M Polymarket arbitrage pool was mostly
**~$29M of NegRisk *multi-condition* rebalancing**, not binary YES+NO. **That ~$29M is exactly
what Phase 1b targets.**

### Phase 1b run 1 — the fragment lesson

The first NegRisk pass (grouping from a `/markets` sample) surfaced 5 "crossing edges," all
2-leg, all with `mass ≈ 0.003` — e.g. *"Brazil Presidential Election, net=$713k."* Every one
was correctly flagged `??!` by the mass check: they were **truncated fragments**, not edges.
Fix: `--complete-events` (now default) groups from `/events` so a basket reflects the whole
event.

### Phase 1b run 2 (complete events) — the structural lesson

With correct grouping, events came back whole (`Presidential Election Winner 2028` → **N=36**,
not N=2) and a handful of small `OK` hits appeared. But inspection killed them: the top one had
a 3.6c gap against `mass=0.944` — i.e. **5.6c of probability sits on unlisted candidates**
(write-ins). Buying all 36 YES for 96.4c does *not* guarantee $1. Subtract `1 − mass` from every
`OK` and they all go negative — because, as proven above, `1 − mass ≥ edge` always. So the loose
`0.90` floor was waving open-candidate elections through as `OK`. Tightening the band to
`[0.98, 1.02]` reclassifies them as `??!`, leaving only tiny near-complete sets — the honest
ceiling for a static check. **Net genuinely-confirmable edges from a snapshot: still zero**, and
now we understand *why* it must be: a real arb is a temporal dislocation, invisible to one frame.
Next: the time-series detector that flags `ask_sum` dipping below each event's rolling baseline.

## Strategy / ROI direction (where the edge actually is)

The honest read from Phase 1 is that latency-flat binary markets are competed out. The return
is in three places, in order of effort:

1. **NegRisk basket dislocations (this phase).** Multi-outcome events re-price unevenly during
   news; the buy-all-YES basket transiently dips below $1 before bots rebalance. A snapshot
   can't tell that transient dip from structural missing-mass, so this is now handled by the
   **temporal detector** (Phase 1c above): `--snapshot` logs every event's `ask_sum` each cycle
   and `pmscan.temporal` flags dips below each event's rolling baseline — a *relative*
   dislocation, not an absolute `< $1` test. The crossing loop also defaults to
   `--min-sets 5 --min-edge 1.0` so 0-size / sub-cent noise stays out of the `--out` log. The
   dip's *duration*, not a single frame, is the real go/no-go for capital.
2. **The convert/sell side** (flagged, not built). The on-chain NegRisk adapter
   `convertPositions` lets a holder of "NO on every outcome" redeem $1, opening the
   complementary `Σ bid(YES_i) > $1` short-the-basket edge. Detection-safe to *model*;
   execution needs the adapter and a wallet (a later, separate phase).
3. **Cross-venue parity (the unified-system payoff).** Once `kalshi-scout` and `pmscan` share
   the venue-agnostic `Market`/`OrderBook` shapes, the same mutually-exclusive event priced on
   both venues becomes a cross-venue arbitrage surface — Polymarket NegRisk basket vs. the
   matching Kalshi event bracket. That is the structural, non-latency edge this codebase is
   being shaped toward, and the first scaffold for it now exists (below).

## Phase 2 (DRAFT) — cross-venue parity (`pmscan/parity.py`)

The same real-world binary outcome priced on two venues is a *locked* arb that doesn't depend
on latency: buy YES on the cheaper venue and the complementary NO on the other for < $1
combined, and exactly one pays $1 regardless of how the event resolves.

```
construction 1:  buy YES@A + NO@B   cost = yes_ask_A + no_ask_B
construction 2:  buy YES@B + NO@A   cost = yes_ask_B + no_ask_A
edge = $1 − min(cost);   scan_parity() takes the cheaper basket
```

This is the cross-venue analogue of the within-market merge edge — but with one critical,
non-mechanical precondition: **the lock is real only if both venues settle the identical event
identically** (same resolution source, date, bucket boundaries). That settlement-equivalence is
the parity analogue of NegRisk exhaustiveness — unprovable from prices — so each `ParityLink`
carries a manually-asserted `settlement_verified` flag, and an opportunity is never trusted
without it (scan-and-flag, same ethos as everywhere else).

**What the scaffold has:** the venue-agnostic `VenueQuote`, the lock math, `ParityOpportunity`,
adapters (`pm_venue_quote` from Polymarket books, `kalshi_venue_quote` from Kalshi cents — no
`kalshi_scout` import; a thin bridge is the integration seam), and synthetic self-tests.

**What it deliberately does not have yet** (the genuinely hard part): an *automated* cross-venue
**matching** layer — which Polymarket token corresponds to which Kalshi ticker. That is an
entity-resolution problem of its own (fuzzy title/date/bucket alignment), left to a later phase.

**Testing it against reality** (`pmscan/parity_registry.py` + `pmscan/kalshi.py` + `parity_run.py`):
v1 matching is a **hand-curated registry** of `ParityCandidate`s — each declares a Polymarket
market (by question substring) and a Kalshi ticker for the same outcome, plus a
`settlement_verified` assertion. Both venues' quotes are now pulled **live**: Polymarket via
Gamma/CLOB, Kalshi via `pmscan/kalshi.py` — a stdlib read-only client over the public
`/trade-api/v2/markets` endpoints (no `kalshi_scout` import, no auth, no orders).

```bash
# 1. edit pmscan/parity_registry.py: point each candidate's pm_match + kalshi_ticker at the
#    SAME outcome; set settlement_verified=True only after you confirm identical resolution.
python parity_run.py --kalshi-live                       # auto-fetch both sides live
python parity_run.py --kalshi-quotes kalshi_quotes.json  # or hand-enter Kalshi prices (cents)
python parity_run.py --kalshi-live --kalshi-quotes k.json  # live + manual overrides
```

Verified locks print `[parity OK ]`; unverified ones `[parity ??!]` with the settlement caveat.
The seed registry entries are **templates** (`settlement_verified=False`, placeholder tickers) —
replace them with verified live identifiers first. **Detection-only, like everything else.**

## Project layout

```
polymarket-scout/
  scan.py              # CLI entry (detection + logging only): binary + --negrisk paths
  parity_run.py        # cross-venue parity runner: live Polymarket + live/manual Kalshi  (Phase 2)
  test_scanner.py      # synthetic self-tests (4 binary + 12 NegRisk + 6 temporal + 14 parity/kalshi)
  requirements.txt     # (stdlib only)
  pmscan/
    __init__.py        # package exports
    models.py          # Market / OrderBook / Opportunity / NegRiskEvent / -Opportunity / -Snapshot
    client.py          # Gamma + CLOB read-only clients, parse_market, parse_event (/events)
    scanner.py         # scan_market (1) + group_negrisk / scan_negrisk / negrisk_snapshot (1b)
    temporal.py        # rolling-baseline dip detector over the snapshot log          (Phase 1c)
    parity.py          # cross-venue (Polymarket + Kalshi) lock detector — DRAFT       (Phase 2)
    parity_registry.py # hand-curated ParityCandidate registry + matching             (Phase 2)
    kalshi.py          # stdlib read-only Kalshi market-data bridge (live quotes)      (Phase 2)
```

## Boundaries (by design)

- Read-only public endpoints. No keys, no wallet, no signing, no orders — anywhere, this phase.
- Top-of-book only; not a sizing/execution model.
- NegRisk **buy** side only in detection; the convert/sell side is flagged, not implemented.
- Edges must be proven to **persist** (via `--interval` logging) before anything resembling
  execution is even designed.
