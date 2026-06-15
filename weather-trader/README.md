# weather-trader

A **forecast-driven** trading bot for Kalshi temperature markets.

Where `kalshi-scout` deliberately *doesn't* forecast — it reads the official
station truth and looks for settlement mispricings — `weather-trader` flips
the thesis: it builds its **own probabilistic forecast** of each station's
daily high/low, prices every Kalshi bracket off that forecast, and surfaces
(and optionally trades) the contracts where the market disagrees with the
model.

## The core idea: a scenario ensemble

The heart of the bot (`forecast.py`) is a **scenario ensemble** of the day's
extreme temperature at a station. Each scenario is one plausible realized
daily high (or low), and the bot's fair probability for any bracket is simply
**the weighted fraction of scenarios that land inside it**.

A scenario is built by combining three things:

1. **Observed-so-far truth (path dependence).** Whatever the station has
   already recorded today sets a hard floor on the daily high (or ceiling on
   the daily low). A high can't un-happen. So every scenario starts from the
   running max/min observed inside the local market day.
2. **The remaining-day forecast.** For the hours left in the day we draw from
   two sources and blend them:
   - the **Open-Meteo ensemble** (GFS ens, ~31 members) — one scenario per
     member, giving a calibrated spread, and
   - the **NWS deterministic hourly forecast** — added as additional weighted
     scenarios so the official forecast anchors the distribution.
3. **A per-station bias correction.** An additive `bias_f` term shifts the
   whole distribution to correct for systematic model error at that station
   (e.g. Open-Meteo running cool at a coastal site).

```
scenario_extremum = combine(observed_so_far, member_remaining_extremum) + bias
fair_prob(bracket) = weighted_fraction(scenarios where bracket.contains(scenario))
```

This folds the path-dependence insight that makes temperature markets
tractable directly into the forecaster: when the day's window is fully
observed the distribution collapses to a single point (`locked`) and the fair
probability is 0 or 1 — the same settlement-conclusive signal `kalshi-scout`
grades A+, but now it's just the degenerate case of the forecast.

### Hybrid: blend now, learned correction on top

The forecaster ships in **blend mode** (`bias_f = 0`) — it needs no training
data and runs the moment the APIs are reachable. The **learned half** is the
`calibrate` command (`calibration.py`): every forecast is logged (`store.py`),
`backfill` joins those logs to realized observations to produce
`(forecast, actual)` residual rows, and `calibrate` turns that history into a
per-(station, metric) **bias model**:

```
bias_f = clamp( mean(actual − predicted) )      # applied only when n ≥ min_samples
```

Adding `bias_f` to the forecast is, to first order, the additive correction
that minimizes the corrected prediction's squared error. The same residuals
also yield a per-station spread (`sigma_f`) used to widen/narrow the synthetic
fallback. Pass the resulting config to `scan`/`forecast` with `--calibration`
and it flows through the *same* scenario interface — no rewiring of the
pricing or grading layers. The unit of correction is (station, metric);
lead-time / regime splits are a future refinement.

## Pipeline

```
Kalshi /markets ──► parse ──► contract (city, metric, date, bracket)
                                   │
NWS obs + hourly ──┐               ▼
Open-Meteo ens   ──┼──► forecast distribution (scenario ensemble + bias)
observed so far  ──┘               │
                                   ▼
                         fair probability per bracket
                                   │
              market price ───────►├──► edge ──► grade (A+…F)
                                   │
                                   ├──► alert (stdout / jsonl)
                                   └──► execute (paper; live = guarded TODO)
```

## Install

```bash
cd weather-trader
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.10+.

### Network egress

Running live needs outbound HTTPS to three hosts:

| Host                          | Used for                                  |
|-------------------------------|-------------------------------------------|
| `api.elections.kalshi.com`    | open temperature markets + prices (read)  |
| `api.weather.gov`             | station observations + NWS hourly forecast |
| `ensemble-api.open-meteo.com` | ensemble member temperatures              |

In a sandboxed/remote environment these must be on the egress allowlist.
Check reachability any time with:

```bash
weather-trader doctor
```

`doctor` exits non-zero if any required host is blocked, and tells you exactly
which one — so you know whether an empty scan is "no good trades" or "no
network."

## Commands

```bash
# Which cities/stations the bot knows
weather-trader cities

# Pre-flight: are the three required hosts reachable?
weather-trader doctor

# Build + print the forecast distribution and per-bracket fair prob for an event
weather-trader forecast KXHIGHNYC-26JUN16
weather-trader forecast KXHIGHNYC-26JUN16 --json

# Scan every open temperature market, grade each contract, rank the board
weather-trader scan                      # grade C+ by default
weather-trader scan --city HOUSTON
weather-trader scan --min-grade B
weather-trader scan --json

# Alert on strong edges and log every forecast for the model-later loop
weather-trader scan --log forecasts.jsonl --notify stdout --notify-min-grade B

# After a day settles, join logged forecasts to realized highs/lows
weather-trader backfill --log forecasts.jsonl --date 2026-06-15 --out residuals.jsonl

# Learn per-station bias corrections from accumulated residuals
weather-trader calibrate --residuals residuals.jsonl --out calibration.json

# Apply the learned bias to future forecasts/scans
weather-trader scan --calibration calibration.json --min-grade B
weather-trader forecast KXHIGHNYC-26JUN16 --calibration calibration.json
```

### The calibration loop

```
scan --log ──► forecasts.jsonl
                    │  (after each day settles)
backfill --out ──► residuals.jsonl
                    │
calibrate --out ──► calibration.json ──► scan/forecast --calibration
```

## Grade ladder

Forecast-driven, so grades reflect **edge × forecast confidence × fillability**:

| Grade | Trigger                                                                 |
|-------|-------------------------------------------------------------------------|
| A+    | `locked` (day fully observed → 0/1 outcome) and price stale, tight spread |
| A     | `locked` with a smaller but real price gap                              |
| B+    | Big edge (≥ ~15c) with a narrow scenario band (high agreement)          |
| B     | Edge ≥ ~12c, reasonable confidence                                      |
| C     | Edge 5–12c                                                              |
| D     | Edge < 5c, or spread/liquidity makes it unfillable                      |
| F     | No usable forecast or no price — never trade                            |

## Execution safety

`execution.py` ships a **`PaperExecutor`** that records intended orders to a
log and never touches the network — the default. A `LiveKalshiExecutor`
interface is scaffolded but **deliberately not wired to place real orders**:
authenticated Kalshi trading (RSA-PSS request signing, real money) is the next
milestone and must be enabled explicitly with credentials. The bot will not
spend money by accident.

## Testing

```bash
pytest
```

All logic — parser, scenario ensemble, fair-probability, grader, residual
backfill, alerts — is tested **offline** against synthetic data. The Kalshi /
NWS / Open-Meteo clients are exercised against mocked transports, so the suite
needs no network.

## Lifting this into its own repo

This package is self-contained (no imports from `kalshi-scout` or the
surrounding monorepo). To make it a standalone GitHub repo:

```bash
# from the monorepo root
git subtree split --prefix=weather-trader -b weather-trader-standalone
# then push that branch to a fresh empty repo, or just copy the directory:
cp -r weather-trader /path/to/new/checkout && cd /path/to/new/checkout && git init
```
