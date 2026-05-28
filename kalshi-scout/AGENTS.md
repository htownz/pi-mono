# kalshi-scout — Engineering Rulebook

The scout is a **Kalshi temperature-market scanner with a settlement-first brain**.
It is not a single-market script, not a weather app, not a forecast model.
Every change to this package must keep that identity intact.

Read this file before opening a PR against `kalshi-scout/`. If a change requires
breaking an invariant below, the change must say so explicitly and propose a
replacement invariant in the same PR.

---

## Hard invariants (V0.3+)

These are enforced now. Violations are bugs.

### I1. No city-specific code paths

City logic lives **only** in the station resolver / registry. Engine code
(`parser.py`, `state.py`, `ranker.py`, `coherence.py`) must never branch on a
city slug. If you're tempted to write `if city == "HOUSTON"`, you're writing a
script, not a scanner.

### I2. No hardcoded tickers in production code

Tickers belong in tests, docs, and CLI examples — never in the engine. If a
ticker appears outside `tests/` or `README.md`, that's a leak.

### I3. Universe-first CLI

`scan` is the front door. `evaluate` and `watch` are debugging / active-window
tools. Any new functionality should land in `scan` first; specialized commands
are exceptions, not the default.

### I4. Refuse to grade if settlement source is not verified

If the resolver cannot produce a `Settlement` for a market (no station, no CLI
product, ambiguous area), the contract is graded **F** with reason
`unverified-settlement-source`. We never trade on a guessed station. The hand
registry in `stations.py` is a **fallback for resolver gaps**, not a primary
truth source — and registry-only matches are tagged as such in the
evaluation's notes.

### I5. Parser refuses to guess

If `parse_market()` cannot unambiguously determine metric + date + bracket
from ticker + `yes_sub_title`, it returns `None` and the scanner silently
skips the market. Silent skip beats wrong settlement.

### I6. Bracket semantics match Kalshi's GLOBALTEMPERATURE rulebook exactly

The five operators in the rulebook map to `BracketKind` values:

| Rulebook word | Operator | `BracketKind` |
|---------------|----------|---------------|
| above         | `>`      | `GT`          |
| below         | `<`      | `LT`          |
| at least      | `≥`      | `GTE`         |
| at most       | `≤`      | `LTE`         |
| exactly       | `=` (1dp)| `EQ`          |
| between       | `[lo,hi]`| `BETWEEN`     |

Note: Kalshi's `yes_sub_title` of "X° or below" colloquially means "at most X"
(≤), which maps to `LTE`, **not** `LT`. The naming exists precisely to prevent
confusion between the colloquial English and the rulebook operators.

### I7. Cross-bracket coherence

When any contract in an event transitions to `LOCKED_YES`, all siblings in the
same event are automatically demoted to `DEAD_NO` in the same evaluation pass.
Contracts in one event are mutually exclusive by Kalshi's rules; the engine
must reflect that without waiting for the orderbook to catch up.

### I8. No alert the engine can't replay from stored state (active since V0.7)

Every alert / grade decision must be a pure function of stored snapshots.
Live-only logic (e.g. consulting an observation the snapshot store did not
record) is a bug. The `replay` command + `tests/test_store.py` enforce this:
a snapshot whose stored inputs would produce a different grade under the
current engine fails the replay check. CI-wireable.

### I9. Every signal must be backtestable (active since V0.7)

A new edge category, state, or grade threshold must ship with a backtest
column showing it would have been profitable (or at least non-degrading) on
historical settled markets. Author intuition is not evidence. Use
`kalshi-scout backtest` against a stored history; add the result to the PR.

---

## Deferred invariants (activate when the named milestone ships)

These describe the long-term identity. They become enforced on the listed
milestone.

### D3. All grade thresholds derived from backtest (activates with V0.8)

The magic numbers in `ranker.py` (`>= 0.08` for A+, `>= 0.03` for A, etc.) get
replaced with values tuned from realized outcomes. The current numbers are
starting points, not the final word.

---

## Soft guidance (style, not invariant)

- Prefer adding station coverage via the resolver, not by expanding
  `stations.py` by hand. Each hand-added station is technical debt against I4.
- Prefer fixing the parser to handle a new title shape over special-casing the
  market in the CLI.
- Keep `cli.py` thin. Business logic belongs in `state.py` / `resolver.py` /
  `coherence.py`. The CLI orchestrates; it shouldn't decide.
- New tests must be offline by default. Live API tests are reserved for
  integration suites that don't run in `pytest` by default.

---

## When in doubt

When in doubt, ask: *"Does this make us a better scanner across the universe,
or does it make us better at one market?"* If it's the second, push back.
