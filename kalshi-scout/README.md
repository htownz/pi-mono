# kalshi-scout

Temperature-market intelligence scanner for Kalshi.

The thesis: temperature markets are path-dependent, station-specific, thinly
watched, and often mispriced because traders treat them like generic weather
forecasts instead of settlement contracts. The scout doesn't try to forecast
better than the NWS. It looks for markets where the **settlement question has
already been answered** by the official station and the orderbook hasn't
caught up.

## What this version (V0.5 → V1.0) does

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
- **Alert delivery** (V0.8): `scan --notify stdout`, `--notify jsonl:path`,
  `--notify webhook:URL`. Alerts fire on **grade-improvement transitions**
  detected against the snapshot store — a market's first appearance at
  the alert grade fires once; subsequent identical grades do not. Per
  invariant I8, alerts require `--store` so they're replayable.
- **Calibration report** (V0.8): `kalshi-scout calibrate --store db.sqlite`
  prints realized hit rate, average / total P&L, sample N, and median
  edge per grade tier from stored history.
- **Auto-tuned ranker config** (V0.9): `kalshi-scout calibrate --apply
  config.json` derives per-(state, grade) edge cutoffs from stored history.
  Tiers with N < MIN_N_PER_TIER fall back to defaults (invariant I10).
  Pass `--config config.json` to `scan` / `evaluate` / `watch` to use it.
- **Regime-shifted fair_probability** (V0.9): the same `--apply` step also
  derives per-(regime, metric, bracket-kind) shift coefficients from
  realized history. Applied only to non-deterministic states
  (BRACKET_HIT_VULNERABLE / FORECAST_DEPENDENT / NOT_REACHED); LOCKED_YES
  and DEAD_NO are settlement-conclusive and never shifted. Shifts are
  clamped to ±20%.
- **Daemon mode** (V1.0): `kalshi-scout serve` runs the scanner on a loop
  with SIGTERM-safe shutdown, structured logs to stderr, and rotating
  JSONL alerts. `--once` for cron.
- **FastAPI dashboard** (V1.0): `kalshi-scout dashboard` serves a
  read-only HTML view at :8080 — alerts feed, current state per market,
  calibration table, full risk report. Auto-refreshes every 30s.
- **Position tracking + pre-flight risk** (V1.0): `positions add/list/close`
  records manually-tracked positions. `risk` aggregates open exposure by
  city / market date / regime / event and explicitly flags **event
  collisions** (holding Yes on multiple brackets of the same event
  guarantees partial loss).
- **Containerized deploy** (V1.0): `Dockerfile`, `docker-compose.yml` (scout
  + dashboard sharing a volume), and `fly.toml` for one-command fly.io
  deploys.

What it deliberately does *not* do yet: trade execution (Kalshi
authenticated API); ML regime classifier.

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

### Alerts on grade-improvement transitions — V0.8

```bash
# Alert to stdout when a contract first hits grade A or better
kalshi-scout scan --store scout.db --notify stdout --notify-min-grade A

# Append every alert to a JSONL feed (great for dashboards)
kalshi-scout scan --store scout.db --notify jsonl:./alerts.jsonl

# Webhook to Slack/Discord/ntfy.sh
kalshi-scout scan --store scout.db --notify webhook:https://hooks.slack.com/services/...

# Multiple sinks at once
kalshi-scout scan --store scout.db \
    --notify stdout \
    --notify jsonl:./alerts.jsonl \
    --notify webhook:https://example.com/scout

# Realized stats per grade — observability for the magic-number cutoffs
kalshi-scout calibrate --store scout.db
kalshi-scout calibrate --store scout.db --since 2026-05-01 --json
```

### Auto-tuned config — V0.9

```bash
# Derive a config from stored history (sample-size gated per invariant I10)
kalshi-scout calibrate --store scout.db --apply config.json

# Use it for future scans (custom cutoffs + regime shifts)
kalshi-scout scan --store scout.db --config config.json --notify stdout

# Verify the tuning was applied to a snapshot
kalshi-scout snapshots --store scout.db --min-grade A
```

### Daemon + dashboard + risk — V1.0

```bash
# Run the scanner as a daemon (every 5 minutes, alerts to stdout + JSONL)
kalshi-scout serve --store scout.db --interval 300 \
    --notify stdout --notify jsonl:./alerts.jsonl --notify-min-grade A

# Single scan via cron
kalshi-scout serve --store scout.db --once --notify stdout

# Read-only HTML dashboard at http://127.0.0.1:8080
kalshi-scout dashboard --store scout.db

# Manually track an open position
kalshi-scout positions add --store scout.db --side yes \
    --size 100 --price 71 KXHIGHHOUSTON-26MAY27-B79-80
kalshi-scout positions list --store scout.db
kalshi-scout positions close --store scout.db 1

# Pre-flight risk: bucketed exposure + event-collision flags
kalshi-scout risk --store scout.db
```

### Deploy

```bash
# Local: docker compose brings up scout daemon + dashboard sharing /data
docker compose up -d

# fly.io: one-command after `fly volumes create scout_data`
fly deploy
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

Current: **V1.0 (+ V0.3 → V0.9)** — universe scanner, settlement-source
resolver, cross-bracket coherence, snapshot store, backtester, replay
verifier, regime classifier, orderbook depth, alert delivery, calibration
report, auto-tuned ranker + regime-shifted fair_probability, daemon mode,
FastAPI dashboard, position tracking, pre-flight risk aggregation,
containerized deploy. All ten hard invariants (I1-I10) active.

## Testing

```bash
pytest                              # 164 tests, all offline
```

The state machine, ranker, parser, resolver, coherence pass, snapshot
store, settlement derivation, backtester, and replay verifier all have
zero network dependencies — every test runs against synthetic fixtures or
temporary SQLite databases. The Kalshi and NWS clients are covered by the
type system and exercised at runtime.
