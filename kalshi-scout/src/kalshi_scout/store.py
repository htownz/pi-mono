"""SQLite-backed snapshot store + settlement matcher + backtester.

This module is what activates the deferred invariants in AGENTS.md:

  D1. No alert the engine can't replay from stored state.
  D2. Every signal must be backtestable.

Two tables:

  snapshots   — one row per (scan, contract). Captures the engine's inputs
                (running max/min, CLI values, station identity, market price)
                AND outputs (state, fair prob, grade, edge). Enough to
                deterministically re-derive the grade.

  settlements — one row per settled market. Populated by `backfill-settlements`
                which reads the official CLI report for a past date and joins
                its max/min against the contract bracket to determine the
                realized outcome (Yes/No).

The backtester is a simple join: for each snapshot at grade ≥ X within a
date range, look up its market_ticker in settlements; if present, compute
the P&L assuming we'd taken the alert at the snapshot's tradable price.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from kalshi_scout.models import (
    Bracket,
    BracketKind,
    ContractEvaluation,
    ContractState,
    KalshiMarket,
    Metric,
    ParsedContract,
)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id TEXT NOT NULL,
    scanned_at_utc TEXT NOT NULL,

    market_ticker TEXT NOT NULL,
    event_ticker TEXT NOT NULL,
    city_slug TEXT NOT NULL,
    metric TEXT NOT NULL,
    market_date TEXT NOT NULL,
    bracket_kind TEXT NOT NULL,
    bracket_lo REAL,
    bracket_hi REAL,

    station_icao TEXT,
    cli_product TEXT,
    source_provenance TEXT NOT NULL,

    regime TEXT,

    running_max_f REAL,
    running_min_f REAL,
    projected_extremum_f REAL,
    cli_report_date TEXT,
    cli_max_f REAL,
    cli_min_f REAL,

    state TEXT NOT NULL,
    reason TEXT NOT NULL,
    fair_prob_low REAL NOT NULL,
    fair_prob_high REAL NOT NULL,

    yes_bid INTEGER, yes_ask INTEGER,
    no_bid INTEGER, no_ask INTEGER,
    last_price INTEGER,
    volume INTEGER NOT NULL DEFAULT 0,
    open_interest INTEGER NOT NULL DEFAULT 0,

    edge_yes REAL, edge_no REAL,
    grade TEXT NOT NULL,
    notes_json TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_snapshots_market_date ON snapshots(market_ticker, market_date);
CREATE INDEX IF NOT EXISTS idx_snapshots_market_scanned ON snapshots(market_ticker, scanned_at_utc);
CREATE INDEX IF NOT EXISTS idx_snapshots_grade_scanned ON snapshots(grade, scanned_at_utc);
CREATE INDEX IF NOT EXISTS idx_snapshots_event_scan ON snapshots(event_ticker, scanned_at_utc);

CREATE TABLE IF NOT EXISTS settlements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_ticker TEXT NOT NULL UNIQUE,
    event_ticker TEXT NOT NULL,
    market_date TEXT NOT NULL,
    city_slug TEXT NOT NULL,
    metric TEXT NOT NULL,

    station_icao TEXT NOT NULL,
    cli_product TEXT NOT NULL,
    cli_report_date TEXT NOT NULL,
    cli_value_f REAL NOT NULL,

    resolved_yes INTEGER NOT NULL,
    settled_at_utc TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_settlements_event ON settlements(event_ticker);
CREATE INDEX IF NOT EXISTS idx_settlements_market_date ON settlements(market_date);

CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_ticker TEXT NOT NULL,
    event_ticker TEXT NOT NULL,
    side TEXT NOT NULL,                  -- 'yes' or 'no'
    size_contracts INTEGER NOT NULL,
    avg_price_cents INTEGER NOT NULL,
    opened_at_utc TEXT NOT NULL,
    closed_at_utc TEXT,                  -- NULL while open
    closed_at_price_cents INTEGER,       -- NULL until closed; payout=100 for win, 0 for loss
    notes TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_positions_open ON positions(closed_at_utc, market_ticker);
CREATE INDEX IF NOT EXISTS idx_positions_event ON positions(event_ticker);
"""


# -- DTOs --------------------------------------------------------------------

@dataclass(frozen=True)
class SnapshotRow:
    """In-memory mirror of a row in the `snapshots` table.

    Mirrors the schema 1:1 so the row can be passed to replay() without
    further translation.
    """
    id: int
    scan_id: str
    scanned_at_utc: datetime
    market_ticker: str
    event_ticker: str
    city_slug: str
    metric: str
    market_date: date
    bracket_kind: str
    bracket_lo: Optional[float]
    bracket_hi: Optional[float]
    station_icao: Optional[str]
    cli_product: Optional[str]
    source_provenance: str
    regime: Optional[str]
    running_max_f: Optional[float]
    running_min_f: Optional[float]
    projected_extremum_f: Optional[float]
    cli_report_date: Optional[date]
    cli_max_f: Optional[float]
    cli_min_f: Optional[float]
    state: str
    reason: str
    fair_prob_low: float
    fair_prob_high: float
    yes_bid: Optional[int]
    yes_ask: Optional[int]
    no_bid: Optional[int]
    no_ask: Optional[int]
    last_price: Optional[int]
    volume: int
    open_interest: int
    edge_yes: Optional[float]
    edge_no: Optional[float]
    grade: str
    notes: list[str]


@dataclass(frozen=True)
class SettlementRow:
    id: int
    market_ticker: str
    event_ticker: str
    market_date: date
    city_slug: str
    metric: str
    station_icao: str
    cli_product: str
    cli_report_date: date
    cli_value_f: float
    resolved_yes: bool
    settled_at_utc: datetime


@dataclass(frozen=True)
class PositionRow:
    """An open or closed position. Manually tracked — we don't (yet) read
    trades from Kalshi's authenticated API."""
    id: int
    market_ticker: str
    event_ticker: str
    side: str                 # 'yes' or 'no'
    size_contracts: int
    avg_price_cents: int
    opened_at_utc: datetime
    closed_at_utc: Optional[datetime]
    closed_at_price_cents: Optional[int]  # NULL until closed with an exit price
    notes: str

    @property
    def is_open(self) -> bool:
        return self.closed_at_utc is None

    @property
    def cost_basis_cents(self) -> int:
        """Total cash deployed = size × avg fill price, in cents."""
        return self.size_contracts * self.avg_price_cents

    @property
    def realized_pnl_cents(self) -> Optional[int]:
        """(exit - entry) × size, in cents. None if no exit price recorded.

        Convention: `closed_at_price_cents` is the per-contract realized
        value of the side we hold. For a settled YES that won, that's 100;
        for a settled YES that lost, 0. For a mid-trade close, it's the
        opposite side's bid (proceeds of selling our position).
        """
        if self.closed_at_price_cents is None:
            return None
        return (self.closed_at_price_cents - self.avg_price_cents) * self.size_contracts

    @property
    def max_loss_cents(self) -> int:
        """If this side loses, we forfeit price_paid_cents per contract."""
        return self.cost_basis_cents


@dataclass(frozen=True)
class BacktestRow:
    """Per-snapshot backtest outcome — what would have happened if we'd
    taken this snapshot's alert.
    """
    snapshot_id: int
    market_ticker: str
    market_date: date
    grade: str
    state: str
    side: str           # 'yes' or 'no'
    price_paid_cents: int
    resolved_yes: bool
    payout_cents: int   # 100 if our side won, 0 if it lost
    pnl_cents: int      # payout_cents - price_paid_cents

    @property
    def won(self) -> bool:
        return self.payout_cents > 0


# -- Date/time helpers --------------------------------------------------------

def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def _iso_date(d: date) -> str:
    return d.isoformat()


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    return date.fromisoformat(s)


# -- Store -------------------------------------------------------------------

class SnapshotStore:
    """Append-only SQLite store of scan snapshots + settlements.

    Single-process design — no concurrent-writer locking beyond SQLite's
    default journal mode. If concurrent scans become a requirement, switch
    to WAL mode and add retry-on-locked.
    """

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(
            str(self.db_path),
            detect_types=sqlite3.PARSE_DECLTYPES,
            isolation_level=None,  # autocommit; we manage transactions explicitly
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SnapshotStore":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        # V0.9 migration: add `regime` column to pre-existing snapshots table
        # if it isn't there yet. SQLite's CREATE TABLE IF NOT EXISTS only
        # creates fresh tables; it does not add columns to existing ones.
        cur = self._conn.execute("PRAGMA table_info(snapshots)")
        cols = {row["name"] for row in cur.fetchall()}
        if "regime" not in cols:
            self._conn.execute("ALTER TABLE snapshots ADD COLUMN regime TEXT")
        # V1.2 migration: per-snapshot forecast projection so the calibration
        # tuner can compute (projected - realized) residuals from settled rows.
        if "projected_extremum_f" not in cols:
            self._conn.execute("ALTER TABLE snapshots ADD COLUMN projected_extremum_f REAL")
        # Add closed-at exit price on positions for realized-P&L tracking.
        cur = self._conn.execute("PRAGMA table_info(positions)")
        pos_cols = {row["name"] for row in cur.fetchall()}
        if "closed_at_price_cents" not in pos_cols:
            self._conn.execute(
                "ALTER TABLE positions ADD COLUMN closed_at_price_cents INTEGER"
            )
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('version', ?)",
            (str(SCHEMA_VERSION),),
        )
        cur.close()

    @contextmanager
    def _txn(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN")
        try:
            yield self._conn
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    # -- Snapshot writes -----------------------------------------------------

    def record_scan(
        self,
        evaluations: list[ContractEvaluation],
        scan_id: Optional[str] = None,
        scanned_at: Optional[datetime] = None,
        station_state_map: Optional[dict[str, dict]] = None,
    ) -> str:
        """Persist a batch of evaluations as one logical scan.

        `station_state_map` is optional per-ticker StationState data
        (running_max_f, running_min_f, cli_*). When omitted, those columns
        are stored as NULL — replay of those snapshots will only verify the
        grade/state given recorded fair_prob, not re-derive state from
        observations.
        """
        scan_id = scan_id or str(uuid.uuid4())
        scanned_at = scanned_at or datetime.now(timezone.utc)
        station_state_map = station_state_map or {}
        rows: list[tuple] = []
        for e in evaluations:
            ss = station_state_map.get(e.market.ticker, {})
            rows.append((
                scan_id, _iso_utc(scanned_at),
                e.market.ticker, e.market.event_ticker, e.contract.city_slug,
                e.contract.metric.value, _iso_date(e.contract.market_date),
                e.contract.bracket.kind.value, e.contract.bracket.lo, e.contract.bracket.hi,
                ss.get("station_icao"), ss.get("cli_product"),
                ss.get("source_provenance", "registry"),
                ss.get("regime"),
                ss.get("running_max_f"), ss.get("running_min_f"),
                ss.get("projected_extremum_f"),
                _iso_date(ss["cli_report_date"]) if ss.get("cli_report_date") else None,
                ss.get("cli_max_f"), ss.get("cli_min_f"),
                e.state.value, e.reason, e.fair_prob_low, e.fair_prob_high,
                e.market.yes_bid, e.market.yes_ask, e.market.no_bid, e.market.no_ask,
                e.market.last_price, e.market.volume, e.market.open_interest,
                e.edge_yes, e.edge_no, e.grade, json.dumps(list(e.notes)),
            ))
        with self._txn() as conn:
            conn.executemany(
                """
                INSERT INTO snapshots (
                    scan_id, scanned_at_utc,
                    market_ticker, event_ticker, city_slug, metric, market_date,
                    bracket_kind, bracket_lo, bracket_hi,
                    station_icao, cli_product, source_provenance,
                    regime,
                    running_max_f, running_min_f, projected_extremum_f,
                    cli_report_date, cli_max_f, cli_min_f,
                    state, reason, fair_prob_low, fair_prob_high,
                    yes_bid, yes_ask, no_bid, no_ask, last_price,
                    volume, open_interest,
                    edge_yes, edge_no, grade, notes_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
        return scan_id

    # -- Snapshot reads ------------------------------------------------------

    def get_snapshot(self, snapshot_id: int) -> Optional[SnapshotRow]:
        cur = self._conn.execute("SELECT * FROM snapshots WHERE id = ?", (snapshot_id,))
        row = cur.fetchone()
        return _row_to_snapshot(row) if row else None

    def query_snapshots(
        self,
        market_ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
        market_date: Optional[date] = None,
        min_grade: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> list[SnapshotRow]:
        clauses: list[str] = []
        params: list = []
        if market_ticker:
            clauses.append("market_ticker = ?")
            params.append(market_ticker)
        if event_ticker:
            clauses.append("event_ticker = ?")
            params.append(event_ticker)
        if market_date:
            clauses.append("market_date = ?")
            params.append(_iso_date(market_date))
        if min_grade:
            grade_order = ["A+", "A", "B+", "B", "C", "D", "F"]
            if min_grade in grade_order:
                allowed = grade_order[: grade_order.index(min_grade) + 1]
                placeholders = ",".join("?" * len(allowed))
                clauses.append(f"grade IN ({placeholders})")
                params.extend(allowed)
        if since:
            clauses.append("scanned_at_utc >= ?")
            params.append(_iso_utc(since))
        if until:
            clauses.append("scanned_at_utc <= ?")
            params.append(_iso_utc(until))
        sql = "SELECT * FROM snapshots"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY scanned_at_utc DESC, id DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        cur = self._conn.execute(sql, params)
        return [_row_to_snapshot(r) for r in cur.fetchall()]

    def count_snapshots(
        self,
        market_ticker: Optional[str] = None,
        before: Optional[datetime] = None,
        keep_grades: Optional[tuple[str, ...]] = None,
    ) -> int:
        """Cheap row counter used by the prune CLI for a dry-run preview.

        `keep_grades` mirrors `prune_snapshots`: rows whose grade is in the
        set are excluded from the count, so a dry-run accurately reflects
        what the destructive call would delete.
        """
        clauses: list[str] = []
        params: list = []
        if market_ticker:
            clauses.append("market_ticker = ?")
            params.append(market_ticker)
        if before:
            clauses.append("scanned_at_utc < ?")
            params.append(_iso_utc(before))
        if keep_grades:
            placeholders = ",".join("?" * len(keep_grades))
            clauses.append(f"grade NOT IN ({placeholders})")
            params.extend(keep_grades)
        sql = "SELECT COUNT(*) AS n FROM snapshots"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return int(self._conn.execute(sql, params).fetchone()["n"])

    def prune_snapshots(
        self,
        before: datetime,
        keep_grades: Optional[tuple[str, ...]] = None,
    ) -> int:
        """Delete snapshots older than `before`, optionally preserving rows whose
        grade is in `keep_grades`. Returns the number of rows deleted.

        Use cases:
          - Daily housekeeping: prune older than 30d, keep A+/A history forever
          - Compact for backup: prune older than 7d, no grade exclusion
        """
        clauses = ["scanned_at_utc < ?"]
        params: list = [_iso_utc(before)]
        if keep_grades:
            placeholders = ",".join("?" * len(keep_grades))
            clauses.append(f"grade NOT IN ({placeholders})")
            params.extend(keep_grades)
        with self._txn() as conn:
            cur = conn.execute(
                "DELETE FROM snapshots WHERE " + " AND ".join(clauses), params,
            )
            return cur.rowcount

    # -- Settlement writes ---------------------------------------------------

    def record_settlement(self, settlement: SettlementRow) -> None:
        with self._txn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO settlements (
                    market_ticker, event_ticker, market_date, city_slug, metric,
                    station_icao, cli_product, cli_report_date, cli_value_f,
                    resolved_yes, settled_at_utc
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    settlement.market_ticker, settlement.event_ticker,
                    _iso_date(settlement.market_date), settlement.city_slug, settlement.metric,
                    settlement.station_icao, settlement.cli_product,
                    _iso_date(settlement.cli_report_date), settlement.cli_value_f,
                    1 if settlement.resolved_yes else 0,
                    _iso_utc(settlement.settled_at_utc),
                ),
            )

    def get_settlement(self, market_ticker: str) -> Optional[SettlementRow]:
        cur = self._conn.execute(
            "SELECT * FROM settlements WHERE market_ticker = ?", (market_ticker,)
        )
        row = cur.fetchone()
        return _row_to_settlement(row) if row else None

    # -- Position writes/reads -----------------------------------------------

    def add_position(
        self,
        market_ticker: str,
        event_ticker: str,
        side: str,
        size_contracts: int,
        avg_price_cents: int,
        opened_at: Optional[datetime] = None,
        notes: str = "",
    ) -> int:
        if side not in ("yes", "no"):
            raise ValueError(f"side must be 'yes' or 'no', got {side!r}")
        if size_contracts <= 0 or avg_price_cents <= 0 or avg_price_cents >= 100:
            raise ValueError("size_contracts > 0 and 0 < avg_price_cents < 100 required")
        opened_at = opened_at or datetime.now(timezone.utc)
        with self._txn() as conn:
            cur = conn.execute(
                """
                INSERT INTO positions (
                    market_ticker, event_ticker, side, size_contracts,
                    avg_price_cents, opened_at_utc, closed_at_utc, notes
                ) VALUES (?,?,?,?,?,?,NULL,?)
                """,
                (market_ticker, event_ticker, side, size_contracts,
                 avg_price_cents, _iso_utc(opened_at), notes),
            )
            return cur.lastrowid

    def close_position(
        self,
        position_id: int,
        closed_at: Optional[datetime] = None,
        at_price_cents: Optional[int] = None,
    ) -> bool:
        """Mark a position closed. `at_price_cents` is the per-contract exit
        value (100 if our side won, 0 if it lost, mid-trade close price
        otherwise) and enables realized-P&L on the listing."""
        closed_at = closed_at or datetime.now(timezone.utc)
        with self._txn() as conn:
            cur = conn.execute(
                """UPDATE positions
                   SET closed_at_utc = ?, closed_at_price_cents = ?
                   WHERE id = ? AND closed_at_utc IS NULL""",
                (_iso_utc(closed_at), at_price_cents, position_id),
            )
            return cur.rowcount > 0

    def query_positions(self, open_only: bool = True) -> list[PositionRow]:
        sql = "SELECT * FROM positions"
        if open_only:
            sql += " WHERE closed_at_utc IS NULL"
        sql += " ORDER BY opened_at_utc DESC"
        cur = self._conn.execute(sql)
        return [_row_to_position(r) for r in cur.fetchall()]

    def query_settlements(
        self,
        event_ticker: Optional[str] = None,
        market_date: Optional[date] = None,
    ) -> list[SettlementRow]:
        clauses, params = [], []
        if event_ticker:
            clauses.append("event_ticker = ?")
            params.append(event_ticker)
        if market_date:
            clauses.append("market_date = ?")
            params.append(_iso_date(market_date))
        sql = "SELECT * FROM settlements"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        cur = self._conn.execute(sql, params)
        return [_row_to_settlement(r) for r in cur.fetchall()]


# -- Settlement derivation from CLI value ------------------------------------

def settlement_from_cli(
    market_ticker: str,
    event_ticker: str,
    market_date: date,
    city_slug: str,
    metric: Metric,
    bracket: Bracket,
    station_icao: str,
    cli_product: str,
    cli_report_date: date,
    cli_value_f: float,
    settled_at: Optional[datetime] = None,
) -> SettlementRow:
    """Build a SettlementRow by applying the bracket's contains() check to
    the official CLI max/min value.

    This is the *only* place we decide a market's realized outcome — every
    backtest reads from here. Single source of truth.
    """
    settled_at = settled_at or datetime.now(timezone.utc)
    resolved_yes = bracket.contains(cli_value_f)
    return SettlementRow(
        id=0,
        market_ticker=market_ticker,
        event_ticker=event_ticker,
        market_date=market_date,
        city_slug=city_slug,
        metric=metric.value,
        station_icao=station_icao,
        cli_product=cli_product,
        cli_report_date=cli_report_date,
        cli_value_f=cli_value_f,
        resolved_yes=resolved_yes,
        settled_at_utc=settled_at,
    )


# -- Replay ------------------------------------------------------------------

@dataclass(frozen=True)
class ReplayResult:
    """Outcome of replaying a snapshot through the live engine."""
    snapshot_id: int
    stored_state: str
    replayed_state: str
    stored_grade: str
    replayed_grade: str
    matches: bool
    drift_reason: Optional[str] = None


def replay(store: SnapshotStore, snapshot_id: int) -> ReplayResult:
    """Re-derive state + grade from the stored snapshot.

    This is invariant D1 in code form: a snapshot that can't be replayed
    deterministically is a snapshot the engine shouldn't have alerted on.

    Replay procedure:
      1. Reconstruct ParsedContract + KalshiMarket from stored columns.
      2. Re-run state.classify() against the stored running_max_f /
         running_min_f. If those are NULL (legacy/light snapshot), skip the
         state check and only verify grade.
      3. Re-run ranker.grade() with the stored fair_prob and price; assert
         the resulting grade matches.
    """
    from kalshi_scout.ranker import grade as grade_fn  # local import to avoid cycles
    from kalshi_scout.state import _high_state, _low_state  # type: ignore[attr-defined]

    snap = store.get_snapshot(snapshot_id)
    if snap is None:
        return ReplayResult(
            snapshot_id=snapshot_id,
            stored_state="?", replayed_state="?",
            stored_grade="?", replayed_grade="?",
            matches=False, drift_reason="snapshot not found",
        )

    bracket = Bracket(
        kind=BracketKind(snap.bracket_kind),
        lo=snap.bracket_lo,
        hi=snap.bracket_hi,
    )
    contract = ParsedContract(
        market_ticker=snap.market_ticker,
        event_ticker=snap.event_ticker,
        city_slug=snap.city_slug,
        metric=Metric(snap.metric),
        market_date=snap.market_date,
        bracket=bracket,
    )
    market = KalshiMarket(
        ticker=snap.market_ticker,
        event_ticker=snap.event_ticker,
        title="",
        yes_sub_title="",
        status="closed",
        close_time=None,
        yes_bid=snap.yes_bid, yes_ask=snap.yes_ask,
        no_bid=snap.no_bid, no_ask=snap.no_ask,
        last_price=snap.last_price,
        volume=snap.volume, open_interest=snap.open_interest,
    )

    # Re-run classify if we stored the inputs.
    replayed_state_str = snap.state
    drift_reason: Optional[str] = None
    if snap.metric == Metric.HIGH.value and snap.running_max_f is not None:
        replayed_state, _ = _high_state(bracket, snap.running_max_f)
        replayed_state_str = replayed_state.value
    elif snap.metric == Metric.LOW.value and snap.running_min_f is not None:
        replayed_state, _ = _low_state(bracket, snap.running_min_f)
        replayed_state_str = replayed_state.value

    # Re-grade with stored fair-prob.
    replayed_eval = grade_fn(
        contract=contract,
        market=market,
        state=ContractState(replayed_state_str),
        reason="(replay)",
        fair_lo=snap.fair_prob_low,
        fair_hi=snap.fair_prob_high,
    )

    state_matches = replayed_state_str == snap.state
    grade_matches = replayed_eval.grade == snap.grade
    if not state_matches:
        drift_reason = f"state {snap.state} -> {replayed_state_str}"
    elif not grade_matches:
        drift_reason = f"grade {snap.grade} -> {replayed_eval.grade}"

    return ReplayResult(
        snapshot_id=snapshot_id,
        stored_state=snap.state, replayed_state=replayed_state_str,
        stored_grade=snap.grade, replayed_grade=replayed_eval.grade,
        matches=state_matches and grade_matches,
        drift_reason=drift_reason,
    )


# -- Backtest ----------------------------------------------------------------

def backtest(
    store: SnapshotStore,
    min_grade: str = "A",
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> list[BacktestRow]:
    """For each snapshot at grade >= min_grade with a known settlement,
    compute what would have happened if we'd taken the alert.

    Trade-side selection rule:
      - LOCKED_YES (or fair_prob midpoint > 0.5): buy Yes at yes_ask
      - DEAD_NO (or fair_prob midpoint < 0.5): buy No at no_ask
      - Skip if the chosen side has no price (unfillable).

    P&L per contract = 100c - price_paid if our side won, else -price_paid.
    """
    snapshots = store.query_snapshots(min_grade=min_grade, since=since, until=until)
    out: list[BacktestRow] = []
    for snap in snapshots:
        settlement = store.get_settlement(snap.market_ticker)
        if settlement is None:
            continue

        fair_mid = (snap.fair_prob_low + snap.fair_prob_high) / 2.0
        side = "yes" if (snap.state == ContractState.LOCKED_YES.value or fair_mid >= 0.5) else "no"
        if side == "yes":
            price = snap.yes_ask if snap.yes_ask is not None else (
                (100 - snap.no_bid) if snap.no_bid is not None else None
            )
        else:
            price = snap.no_ask if snap.no_ask is not None else (
                (100 - snap.yes_bid) if snap.yes_bid is not None else None
            )
        if price is None or price <= 0 or price > 100:
            continue

        side_won = (side == "yes" and settlement.resolved_yes) or (
            side == "no" and not settlement.resolved_yes
        )
        payout = 100 if side_won else 0
        out.append(BacktestRow(
            snapshot_id=snap.id,
            market_ticker=snap.market_ticker,
            market_date=snap.market_date,
            grade=snap.grade,
            state=snap.state,
            side=side,
            price_paid_cents=price,
            resolved_yes=settlement.resolved_yes,
            payout_cents=payout,
            pnl_cents=payout - price,
        ))
    return out


# -- Internal row -> dataclass adapters --------------------------------------

def _row_to_snapshot(row: sqlite3.Row) -> SnapshotRow:
    return SnapshotRow(
        id=row["id"],
        scan_id=row["scan_id"],
        scanned_at_utc=_parse_utc(row["scanned_at_utc"]),
        market_ticker=row["market_ticker"],
        event_ticker=row["event_ticker"],
        city_slug=row["city_slug"],
        metric=row["metric"],
        market_date=date.fromisoformat(row["market_date"]),
        bracket_kind=row["bracket_kind"],
        bracket_lo=row["bracket_lo"],
        bracket_hi=row["bracket_hi"],
        station_icao=row["station_icao"],
        cli_product=row["cli_product"],
        source_provenance=row["source_provenance"],
        regime=row["regime"] if "regime" in row.keys() else None,
        running_max_f=row["running_max_f"],
        running_min_f=row["running_min_f"],
        projected_extremum_f=(
            row["projected_extremum_f"] if "projected_extremum_f" in row.keys() else None
        ),
        cli_report_date=_parse_date(row["cli_report_date"]),
        cli_max_f=row["cli_max_f"],
        cli_min_f=row["cli_min_f"],
        state=row["state"],
        reason=row["reason"],
        fair_prob_low=row["fair_prob_low"],
        fair_prob_high=row["fair_prob_high"],
        yes_bid=row["yes_bid"], yes_ask=row["yes_ask"],
        no_bid=row["no_bid"], no_ask=row["no_ask"],
        last_price=row["last_price"],
        volume=row["volume"], open_interest=row["open_interest"],
        edge_yes=row["edge_yes"], edge_no=row["edge_no"],
        grade=row["grade"],
        notes=json.loads(row["notes_json"]),
    )


def _row_to_position(row: sqlite3.Row) -> PositionRow:
    return PositionRow(
        id=row["id"],
        market_ticker=row["market_ticker"],
        event_ticker=row["event_ticker"],
        side=row["side"],
        size_contracts=row["size_contracts"],
        avg_price_cents=row["avg_price_cents"],
        opened_at_utc=_parse_utc(row["opened_at_utc"]),
        closed_at_utc=_parse_utc(row["closed_at_utc"]) if row["closed_at_utc"] else None,
        closed_at_price_cents=(
            row["closed_at_price_cents"] if "closed_at_price_cents" in row.keys() else None
        ),
        notes=row["notes"] or "",
    )


def _row_to_settlement(row: sqlite3.Row) -> SettlementRow:
    return SettlementRow(
        id=row["id"],
        market_ticker=row["market_ticker"],
        event_ticker=row["event_ticker"],
        market_date=date.fromisoformat(row["market_date"]),
        city_slug=row["city_slug"],
        metric=row["metric"],
        station_icao=row["station_icao"],
        cli_product=row["cli_product"],
        cli_report_date=date.fromisoformat(row["cli_report_date"]),
        cli_value_f=row["cli_value_f"],
        resolved_yes=bool(row["resolved_yes"]),
        settled_at_utc=_parse_utc(row["settled_at_utc"]),
    )
