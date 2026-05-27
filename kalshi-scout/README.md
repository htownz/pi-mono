# kalshi-scout

Temperature-market intelligence scanner for Kalshi.

The thesis: temperature markets are path-dependent, station-specific, thinly
watched, and often mispriced because traders treat them like generic weather
forecasts instead of settlement contracts. The scout doesn't try to forecast
better than the NWS. It looks for markets where the **settlement question has
already been answered** by the official station and the orderbook hasn't
caught up.

## What this version (v0.3) does

- **Crawls** every open Kalshi temperature event under known series prefixes
  (`KXHIGH*`, `KXLOW*`, `KXTEMP*`) without authentication.
- **Parses** each market ticker + `yes_sub_title` into a structured contract:
  city, metric (HIGH/LOW), market date, and bracket (above/below/between).
- **Pulls** NWS station observations for the market's local-day window, the
  station's hourly forecast, and the latest CLI product. Discards the CLI if
  its report date doesn't match the market date (one of the v0.2 traps).
- **Classifies** each contract's state via a deterministic state machine:
  `LOCKED_YES` / `DEAD_NO` / `BRACKET_HIT_VULNERABLE` / `NOT_REACHED` /
  `FORECAST_DEPENDENT`.
- **Grades** each contract A+ → F based on settlement-conclusiveness, edge
  vs. tradable price (Yes ask or derived No ask), and spread / liquidity.
- **Outputs** a ranked opportunity board (rich table or JSON).

What it deliberately does *not* do yet: trade execution, persistence /
backtesting, alert delivery, fancy forecast models. See "Roadmap" below.

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

### `cities` — what the scout settles against

```bash
kalshi-scout cities
```

Each city slug maps to exactly one NWS ICAO + CLI product. **Always verify
the Kalshi contract's stated settlement source matches before trusting a
grade.** Settlement-source mismatch is the worst-case error.

## The model

### Three contract shapes

```
above(t)        settlement >= t        suffix: T<t>   title: "<t>° or above"
below(t)        settlement <= t        suffix: T<t>   title: "<t>° or below"
between(lo,hi)  lo <= settlement <= hi suffix: B<lo>-<hi>
```

The parser refuses to guess direction for a `T<n>` ticker without title
disambiguation (returns `None`) — silent skip beats wrong settlement.

### State machine (the engine)

| Metric | Bracket   | Observation                          | State                       |
|--------|-----------|--------------------------------------|-----------------------------|
| HIGH   | above(t)  | running_max ≥ t                      | `LOCKED_YES`                |
| HIGH   | above(t)  | running_max < t                      | `NOT_REACHED`               |
| HIGH   | below(t)  | running_max > t                      | `DEAD_NO`                   |
| HIGH   | below(t)  | running_max ≤ t                      | `FORECAST_DEPENDENT`        |
| HIGH   | btwn(l,h) | running_max > h                      | `DEAD_NO`                   |
| HIGH   | btwn(l,h) | l ≤ running_max ≤ h                  | `BRACKET_HIT_VULNERABLE`    |
| HIGH   | btwn(l,h) | running_max < l                      | `NOT_REACHED`               |
| LOW    | below(t)  | running_min ≤ t                      | `LOCKED_YES` (low can't undo)|
| LOW    | below(t)  | running_min > t                      | `FORECAST_DEPENDENT`        |
| LOW    | above(t)  | running_min < t                      | `DEAD_NO`                   |
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

This is V0.3 (universe scanner skeleton). Next slices, in order:

- **V0.4** station/source resolver: confirm each market's actual settlement
  station from Kalshi's rules rather than the hand-curated map.
- **V0.5** forecast engine: cloud/precip/front regime detection, better
  remaining-extreme bounds.
- **V0.6** orderbook depth: use full `/orderbook` to estimate fill quality,
  not just top-of-book ask.
- **V0.7** backtester: SQLite-backed snapshot store + settlement matcher.
- **V0.8** alert delivery: webhook / push when a contract transitions into
  A+/A or a high-grade B+.

## Testing

```bash
pytest                              # 27 tests, all offline
```

The state machine and ranker have no network dependencies — every test runs
against synthetic `StationState` objects. The Kalshi and NWS clients are
covered by the type system and exercised at runtime; integration tests will
land with V0.7.
