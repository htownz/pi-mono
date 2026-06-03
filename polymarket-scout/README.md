# pmscan — Polymarket sum-to-one scanner (Phase 0 + 1 + 1b)

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
python scan.py --negrisk --once                        # NegRisk multi-outcome (Phase 1b)
python scan.py --negrisk --interval 30 --out neg.jsonl # watch loop (persistence measurement)
python test_scanner.py                                 # synthetic self-tests (10, all passing)
```

No install needed — standard library only, Python 3.10+.

Key flags: `--negrisk` (switch to multi-outcome mode), `--min-edge` (min gross per-set edge,
cents), `--min-sets` (min capturable top-of-book sets), `--min-volume` (24h USD filter),
`--fee` (per-share per-leg, default 0), `--gas` (round-trip USD, default 0.01), `--out`
(append JSONL log), `--interval` (loop seconds), `--max-markets`.

## The NegRisk exhaustiveness problem (and how we handle it)

The buy-all-YES identity `Σ ask(YES_i) < $1 ⇒ guaranteed edge` holds **only if the outcome
set is truly exhaustive** (exactly one of the listed outcomes must resolve YES). If the event
has an implicit "Other / None of the above" bucket, or the API simply doesn't list every
outcome, the naive sum understates the true price of a full hedge and the "edge" is a mirage.

You cannot prove exhaustiveness from the public API alone, so rather than silently trusting it
(false positives → phantom P/L) or silently dropping candidates (missed real edges), the
detector **scans and attaches a confidence flag**:

1. **Explicit Other bucket** — any outcome flagged `negRiskOther` ⇒ `exhaustive_verified=False`.
2. **Implied-probability-mass sanity check** *(innovation worth calling out).* For a clean
   partition the YES **mid** prices should sum to ≈ $1 (they are the market's probabilities and
   must total 1). We compute `implied_mass = Σ mid(YES_i)`:
   - `mass` well **below** 1 ⇒ outcomes are missing (the set isn't complete) — the classic
     cause of a fake "10c gap." Flagged uncertain.
   - `mass` well **above** 1 ⇒ outcomes overlap/duplicate — also not a clean partition.
   Default acceptance band is `[0.90, 1.08]`; outside it ⇒ `exhaustive_verified=False` with a
   reason. This single cheap check kills the most common phantom-edge pattern for free.
3. **Incomplete legs** — if any outcome's YES book is missing, we **refuse** to emit (return
   `None`) rather than under-sum the basket and overstate the edge.

Verified hits print `[negrisk OK ]`; unverified-but-real-if-complete hits print `[negrisk ??!]`
with the reason, and both are logged (the `exhaustive_verified` field is in the JSONL).

## Live finding (Phase 1, first run, ~1,500 highest-volume markets)

**Zero crossable single-condition (binary) sum-to-one edges.** The tightest binary markets sit
at ask-sum `1.001` / bid-sum `0.999` — the complete set pinned to exactly $1.00 with a one-tick
spread straddling it. The detector correctly scores that as negative edge (no false positive;
`test_binary_no_edge_pinned_to_one`). This matches the research: bots keep binary
single-condition markets flat, and the documented ~$39.6M Polymarket arbitrage pool was mostly
**~$29M of NegRisk *multi-condition* rebalancing**, not binary YES+NO. **That ~$29M is exactly
what Phase 1b targets.**

## Strategy / ROI direction (where the edge actually is)

The honest read from Phase 1 is that latency-flat binary markets are competed out. The return
is in three places, in order of effort:

1. **NegRisk basket dislocations (this phase).** Multi-outcome events re-price unevenly during
   news; the buy-all-YES basket transiently dips below $1 before bots rebalance. The
   `--interval` watch loop exists to **measure how long those dips survive at our latency** —
   that persistence number, not a single snapshot, is the real go/no-go for any capital.
2. **The convert/sell side** (flagged, not built). The on-chain NegRisk adapter
   `convertPositions` lets a holder of "NO on every outcome" redeem $1, opening the
   complementary `Σ bid(YES_i) > $1` short-the-basket edge. Detection-safe to *model*;
   execution needs the adapter and a wallet (a later, separate phase).
3. **Cross-venue parity (the unified-system payoff).** Once `kalshi-scout` and `pmscan` share
   the venue-agnostic `Market`/`OrderBook` shapes, the same mutually-exclusive event priced on
   both venues becomes a cross-venue arbitrage surface — Polymarket NegRisk basket vs. the
   matching Kalshi event bracket. That is the structural, non-latency edge this codebase is
   being shaped toward.

## Project layout

```
polymarket-scout/
  scan.py              # CLI entry (detection + logging only): binary + --negrisk paths
  test_scanner.py      # synthetic self-tests (4 binary + 6 NegRisk)
  requirements.txt     # (stdlib only)
  pmscan/
    __init__.py        # package exports
    models.py          # Market / OrderBook / Opportunity / NegRiskEvent / NegRiskOpportunity
    client.py          # Gamma + CLOB read-only clients, parse_market
    scanner.py         # scan_market (Phase 1) + group_negrisk / scan_negrisk (Phase 1b)
```

## Boundaries (by design)

- Read-only public endpoints. No keys, no wallet, no signing, no orders — anywhere, this phase.
- Top-of-book only; not a sizing/execution model.
- NegRisk **buy** side only in detection; the convert/sell side is flagged, not implemented.
- Edges must be proven to **persist** (via `--interval` logging) before anything resembling
  execution is even designed.
