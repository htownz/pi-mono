# kalshi-scout

Temperature-market intelligence scanner for Kalshi.

The thesis: temperature markets are path-dependent, station-specific, thinly
watched, and often mispriced because traders treat them like generic weather
forecasts instead of settlement contracts. The scout doesn't try to forecast
better than the NWS. It looks for markets where the **settlement question has
already been answered** by the official station and the orderbook hasn't
caught up.

## What this version (V0.5 + V0.6 + V0.7) does

- **Crawls** every open Kalshi temperature event under known series prefixes
  (`KXHIGH*`, `KXLOW*`, `KXTEMP*`) without authentication.
- **Parses** each market ticker + `yes_sub_title` into a structured contract:
  city, metric (HIGH/LOW), market date, and bracket — six operators matching
  Kalshi's GLOBALTEMPERATURE rulebook (`GT`/`LT`/`GTE`/`LTE`/`EQ`/`BETWEEN`).
- **Resolves the settlement source** (V0.4): parses each market's
  `rules_primary` text to extract the official NWS station; falls back to the
  hand registry only as a tagged lower-trust signal; refuses to grade
  (`F`) when neither produces a verified source (invariant I4).
- **Pulls** NWS station observations for the market's local-day window, the
  station's hourly forecast, and the latest CLI product. Discards the CLI if
  its report date doesn't match the market date.
- **Classifies** each contract's state via a deterministic state machine:
  `LOCKED_YES` / `DEAD_NO` / `BRACKET_HIT_VULNERABLE` / `NOT_REACHED` /
  `FORECAST_DEPENDENT`.
- **Cross-bracket coherence** (V0.4 / invariant I7): when any sibling in an
  event is `LOCKED_YES`, all others are automatically demoted to `DEAD_NO`.
  Flags overpriced (sum > 105c) and underpriced (sum < 95c) books across
  the event.
- **Grades** each contract A+ → F based on settlement-conclusiveness, edge
  vs. tradable price (Yes ask or derived No ask), and spread / liquidity.
- **Outputs** a ranked opportunity board (rich table or JSON).
- **Snapshot store** (V0.7): every scan/evaluate run can persist to a SQLite
  database (`--store path.db`). Every contract evaluation becomes one row
  capturing engine inputs (station identity, running max/min, CLI values,
  market price) AND outputs (state, fair-prob, grade). The store is what
  activates invariants D1 (replayability) and D2 (backtestability).
- **Settlement backfill** (V0.7): `backfill-settlements --date YYYY-MM-DD`
  pulls the day's CLI report per stored station and joins to snapshots,
  writing one `SettlementRow` per market (resolved_yes computed from
  `bracket.contains(cli_value)`).
- **Backtester** (V0.7): `backtest --grade A --since YYYY-MM-DD` joins
  snapshots ↔ settlements, computes side (Yes if LOCKED_YES or fair≥0.5,
  else No), assumes fill at the recorded ask, reports hit rate + total P&L.
- **Replay verifier** (V0.7): `replay <snapshot_id>` re-runs the engine
  against a snapshot's stored inputs and asserts state + grade match. CI-
  friendly (non-zero exit on drift).
- **Regime classifier** (V0.5): one of `CLEAR_AND_DRY`, `RAIN_COOLED`,
  `MARINE_LAYER`, `COLD_FRONT_NEAR`, `CALM_HUMID_RADIATIONAL`, `UNKNOWN`
  per station per evaluation. Notes-only — does not auto-shift fair_prob
  or grade until backtest evidence supports calibrated weights (invariant
  I9). The reasoning string is appended to every evaluation in the event.
- **Orderbook depth** (V0.6): `evaluate --depth N` fetches each contract's
  orderbook and computes the average fill price for N contracts on the
  natural trade side, with a partial-fill flag. Confirms that A+/A edges
  are actually fillable before you act on them.

What it deliberately does *not* do yet: trade execution, alert delivery,
regime-shifted grade ladder (V0.8).

See `AGENTS.md` for the engineering invariants every change must respect.

## Install

```bash
cd kalshi-scout
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.10+. Outbound HTTPS to `api.elections.kalshi.com` and
`api.weather.gov` (whose terms require a real User-Agent — set in
`src/kalshi_scout/nws.py`).

## Commands

### `scan` — universe scanner

```bash
kalshi-scout scan                  # everything open, grade C+ by default
kalshi-scout scan --city HOUSTON   # one city
kalshi-scout scan --min-grade A    # only settlement-conclusive edges
kalshi-scout scan --json           # machine-readable
```

### `evaluate` — single event or market

```bash
# All contracts in one event
kalshi-scout evaluate KXLOWHOUSTON-26MAY28

# One specific market
kalshi-scout evaluate KXLOWHOUSTON-26MAY28-B70-71

# JSON output
kalshi-scout evaluate KXLOWHOUSTON-26MAY28 --json
```

### `watch` — poll an event on a loop

```bash
# Tonight's KHOU low, every 5 minutes, only B+ or better
kalshi-scout watch KXLOWHOUSTON-26MAY28 --interval 300 --min-grade B
```

`watch` only prints when a contract's state *changes* — so the table you see
is the news, not the noise.

### `snapshots` / `backfill-settlements` / `backtest` / `replay` — V0.7

```bash
# Persist every scan to ./scout.db
kalshi-scout scan --store scout.db --min-grade C

# Inspect what scan recorded
kalshi-scout snapshots --store scout.db --market-date 2026-05-27 --min-grade A

# After CLI is published for a settled day, derive realized outcomes
kalshi-scout backfill-settlements --store scout.db --date 2026-05-27

# Backtest every A+/A alert that has a known settlement
kalshi-scout backtest --store scout.db --min-grade A --since 2026-05-01

# Verify a single alert is replayable (CI-friendly: exits non-zero on drift)
kalshi-scout replay --store scout.db 42
```

### `cities` — what the scout settles against

```bash
kalshi-scout cities
```

Each city slug maps to exactly one NWS ICAO + CLI product. **Always verify
the Kalshi contract's stated settlement source matches before trusting a
grade.** Settlement-source mismatch is the worst-case error.

## The model

### Six contract shapes (matching Kalshi's GLOBALTEMPERATURE rulebook)

| `BracketKind` | Operator | Rulebook word | Typical title             |
|---------------|----------|---------------|---------------------------|
| `GT`          | `>`      | "above"       | "above 80°"               |
| `LT`          | `<`      | "below"       | "below 75°"               |
| `GTE`         | `≥`      | "at least"    | "85° or above"            |
| `LTE`         | `≤`      | "at most"     | "78° or below"            |
| `EQ`          | `=` (1dp)| "exactly"     | "exactly 80°"             |
| `BETWEEN`     | `[lo,hi]`| "between"     | "79° to 80°"              |

The parser refuses to guess direction for a `T<n>` ticker without title
disambiguation (returns `None`) — silent skip beats wrong settlement.
The naming distinguishes Kalshi's *colloquial* English ("X or below" =
inclusive `≤` = `LTE`) from the *rulebook strict operator* ("below X" = `<` =
`LT`). See `AGENTS.md` invariant I6.

### State machine (the engine)

| Metric | Bracket   | Observation                          | State                       |
|--------|-----------|--------------------------------------|-----------------------------|
| HIGH   | GTE(t)    | running_max ≥ t                      | `LOCKED_YES`                |
| HIGH   | GTE(t)    | running_max < t                      | `NOT_REACHED`               |
| HIGH   | LTE(t)    | running_max > t                      | `DEAD_NO`                   |
| HIGH   | LTE(t)    | running_max ≤ t                      | `FORECAST_DEPENDENT`        |
| HIGH   | btwn(l,h) | running_max > h                      | `DEAD_NO`                   |
| HIGH   | btwn(l,h) | l ≤ running_max ≤ h                  | `BRACKET_HIT_VULNERABLE`    |
| HIGH   | btwn(l,h) | running_max < l                      | `NOT_REACHED`               |
| LOW    | LTE(t)    | running_min ≤ t                      | `LOCKED_YES` (low can't undo)|
| LOW    | LTE(t)    | running_min > t                      | `FORECAST_DEPENDENT`        |
| LOW    | GTE(t)    | running_min < t                      | `DEAD_NO`                   |
| LOW    | btwn(l,h) | running_min < l                      | `DEAD_NO`                   |
| LOW    | btwn(l,h) | l ≤ running_min ≤ h                  | `BRACKET_HIT_VULNERABLE`    |
| LOW    | btwn(l,h) | running_min > h                      | `NOT_REACHED`               |

### Grade ladder

| Grade | Trigger                                                     |
|-------|-------------------------------------------------------------|
| A+    | Settlement state already decisive; tradable price stale ≥8c, spread tight. |
| A     | Settlement state already decisive; price stale 3–8c.        |
| B+    | Bracket hit, forecast escape improbable, price stale.       |
| B     | Forecast-dependent edge ≥12c, forecast in agreement.        |
| C     | Forecast-dependent edge 5–12c.                              |
| D     | Edge <5c or spread/liquidity makes it unfillable.           |
| F     | Settlement source ambiguous or data missing — never trade.  |

## Tonight's KHOU low (the worked example)

```bash
kalshi-scout watch KXLOWHOUSTON-26MAY28 --interval 300 --min-grade B
```

What it does inside the engine:

1. Builds the market-day window: `2026-05-28 00:00 → 23:59 America/Chicago`.
2. Pulls KHOU observations *only inside that window* (post-midnight only).
   Before midnight, `running_min_f` is `None`, so every contract is
   `FORECAST_DEPENDENT` and the hourly forecast drives the fair probability.
3. Fetches the latest CLIHOU. If its `report_date` ≠ 2026-05-28, the values
   are discarded — preventing the "today's CLI used for tomorrow's market"
   trap.
4. Classifies every bracket every 5 minutes. The first time the running min
   drops to the `below(t)` strike, that contract flips to `LOCKED_YES` and
   the table prints — that's the alert.

The spec's path-dependence rule baked in: **once a low is observed, it
cannot be undone by later warming.** The `LOCKED_YES` transition is the
trade.

## Roadmap

Current: **V0.7 + V0.5 + V0.6** — universe scanner, settlement-source
resolver, cross-bracket coherence, snapshot store, backtester, replay
verifier, regime classifier, orderbook depth. Next:

- **V0.8** alert delivery + backtest-tuned grade thresholds: webhook / push
  when a contract transitions into A+/A; replace the magic-number grade
  cutoffs in `ranker.py` with values calibrated against stored history.
  Activates deferred invariant D3. Also threads `regime` into a calibrated
  fair-probability adjustment.

## Testing

```bash
pytest                              # 93 tests, all offline
```

The state machine, ranker, parser, resolver, coherence pass, snapshot
store, settlement derivation, backtester, and replay verifier all have
zero network dependencies — every test runs against synthetic fixtures or
temporary SQLite databases. The Kalshi and NWS clients are covered by the
type system and exercised at runtime.
