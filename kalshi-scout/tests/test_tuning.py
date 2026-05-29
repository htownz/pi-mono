"""Tests for V0.9 tuning derivation.

Builds a synthetic snapshot+settlement history and verifies the tuner:
  - keeps defaults for buckets below the sample-size threshold
  - derives a median-of-winners cutoff for buckets above it
  - computes per-regime fair-prob bias correctly
  - never auto-applies a regime shift below MIN_N_PER_REGIME
"""

from datetime import date
from pathlib import Path

import pytest

from kalshi_scout.config import (
    DEFAULT_LOCKED_YES,
    MIN_N_PER_REGIME,
    MIN_N_PER_TIER,
    regime_key,
)
from kalshi_scout.models import (
    Bracket,
    BracketKind,
    ContractEvaluation,
    ContractState,
    KalshiMarket,
    Metric,
    ParsedContract,
)
from kalshi_scout.store import SnapshotStore, settlement_from_cli
from kalshi_scout.tuning import derive_config, derive_regime_shifts, derive_tier_thresholds


def _eval(ticker: str, grade: str, state: ContractState,
          edge_yes: float = 0.10, yes_ask: int = 70,
          fair_lo: float = 0.95, fair_hi: float = 0.99,
          bracket: Bracket | None = None,
          metric: Metric = Metric.HIGH) -> ContractEvaluation:
    bracket = bracket or Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0)
    contract = ParsedContract(
        market_ticker=ticker, event_ticker="E",
        city_slug="HOUSTON", metric=metric,
        market_date=date(2026, 5, 27), bracket=bracket,
    )
    market = KalshiMarket(
        ticker=ticker, event_ticker="E",
        title="", yes_sub_title="", status="open", close_time=None,
        yes_bid=yes_ask - 1, yes_ask=yes_ask,
        no_bid=100 - yes_ask - 1, no_ask=100 - yes_ask,
        last_price=yes_ask, volume=10, open_interest=100,
    )
    return ContractEvaluation(
        contract=contract, market=market, state=state, reason="",
        fair_prob_low=fair_lo, fair_prob_high=fair_hi,
        yes_ask_cents=yes_ask, no_ask_cents=100 - yes_ask,
        edge_yes=edge_yes, edge_no=None,
        grade=grade, notes=[],
    )


def _record_win(store: SnapshotStore, ticker: str, grade: str,
                edge_yes: float = 0.10, yes_ask: int = 70,
                regime: str = "clear_and_dry",
                state: ContractState = ContractState.LOCKED_YES,
                metric: Metric = Metric.HIGH, fair_mid: float = 0.97) -> None:
    """Persist a snapshot + Yes-resolved settlement so it counts as a win."""
    e = _eval(ticker, grade, state, edge_yes=edge_yes, yes_ask=yes_ask,
              fair_lo=fair_mid - 0.01, fair_hi=fair_mid + 0.01, metric=metric)
    store.record_scan([e], station_state_map={
        ticker: {"regime": regime, "station_icao": "KHOU", "cli_product": "CLIHOU",
                 "source_provenance": "resolver"}
    })
    bracket = Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0)
    store.record_settlement(settlement_from_cli(
        market_ticker=ticker, event_ticker="E",
        market_date=date(2026, 5, 27),
        city_slug="HOUSTON", metric=metric,
        bracket=bracket, station_icao="KHOU", cli_product="CLIHOU",
        cli_report_date=date(2026, 5, 27), cli_value_f=80.0,
    ))


def _record_loss(store: SnapshotStore, ticker: str, grade: str,
                 edge_yes: float = 0.10, yes_ask: int = 70,
                 regime: str = "clear_and_dry",
                 state: ContractState = ContractState.LOCKED_YES,
                 metric: Metric = Metric.HIGH, fair_mid: float = 0.97) -> None:
    e = _eval(ticker, grade, state, edge_yes=edge_yes, yes_ask=yes_ask,
              fair_lo=fair_mid - 0.01, fair_hi=fair_mid + 0.01, metric=metric)
    store.record_scan([e], station_state_map={
        ticker: {"regime": regime, "station_icao": "KHOU", "cli_product": "CLIHOU",
                 "source_provenance": "resolver"}
    })
    bracket = Bracket(BracketKind.BETWEEN, lo=79.0, hi=80.0)
    # cli=82 falls outside 79-80 -> resolves No
    store.record_settlement(settlement_from_cli(
        market_ticker=ticker, event_ticker="E",
        market_date=date(2026, 5, 27),
        city_slug="HOUSTON", metric=metric,
        bracket=bracket, station_icao="KHOU", cli_product="CLIHOU",
        cli_report_date=date(2026, 5, 27), cli_value_f=82.0,
    ))


@pytest.fixture
def store(tmp_path: Path) -> SnapshotStore:
    s = SnapshotStore(tmp_path / "tune.db")
    yield s
    s.close()


# -- Tier threshold derivation -----------------------------------------------

def test_empty_store_keeps_all_defaults(store: SnapshotStore):
    thresholds, report = derive_tier_thresholds(store)
    assert thresholds[ContractState.LOCKED_YES.value] == DEFAULT_LOCKED_YES
    # Every tier row marks applied=False with "no settled samples".
    for tier in report:
        assert tier.applied is False
        assert "no settled samples" in tier.note


def test_low_n_bucket_keeps_default(store: SnapshotStore):
    """5 wins is below MIN_N_PER_TIER (30) — defaults kept."""
    for i in range(5):
        _record_win(store, f"K{i}", "A+", edge_yes=0.20, yes_ask=70)
    thresholds, report = derive_tier_thresholds(store)
    # Default cutoff is preserved.
    assert thresholds[ContractState.LOCKED_YES.value].high_cutoff == DEFAULT_LOCKED_YES.high_cutoff
    # Audit row reflects below-threshold status.
    a_plus_row = next(
        r for r in report
        if r.state == ContractState.LOCKED_YES.value and r.grade == "A+"
    )
    assert a_plus_row.applied is False
    assert "below threshold" in a_plus_row.note


def test_above_threshold_bucket_derives_median_edge(store: SnapshotStore):
    """MIN_N_PER_TIER winners -> derived cutoff = median of winning edges."""
    n = MIN_N_PER_TIER
    edges = [0.05 + 0.005 * i for i in range(n)]  # 0.05 to 0.05 + 0.005*(n-1)
    for i, edge in enumerate(edges):
        _record_win(store, f"K{i}", "A+", edge_yes=edge, yes_ask=70)
    thresholds, report = derive_tier_thresholds(store)
    a_plus_row = next(
        r for r in report
        if r.state == ContractState.LOCKED_YES.value and r.grade == "A+"
    )
    assert a_plus_row.applied is True
    assert a_plus_row.n_settled == n
    # Median of a sorted list of length n.
    import statistics as _s
    expected = _s.median([abs(e) for e in edges])
    assert abs(a_plus_row.suggested_cutoff - expected) < 1e-9
    assert thresholds[ContractState.LOCKED_YES.value].high_cutoff == pytest.approx(expected)


def test_all_losers_keeps_default(store: SnapshotStore):
    """Enough samples but no winners — keep default (can't derive a cutoff)."""
    n = MIN_N_PER_TIER
    for i in range(n):
        _record_loss(store, f"K{i}", "A+", edge_yes=0.20, yes_ask=70)
    thresholds, report = derive_tier_thresholds(store)
    assert thresholds[ContractState.LOCKED_YES.value].high_cutoff == DEFAULT_LOCKED_YES.high_cutoff
    a_plus_row = next(
        r for r in report
        if r.state == ContractState.LOCKED_YES.value and r.grade == "A+"
    )
    assert a_plus_row.applied is False
    assert "zero winners" in a_plus_row.note


# -- Regime shift derivation -------------------------------------------------

def test_no_shifts_in_empty_store(store: SnapshotStore):
    shifts, report = derive_regime_shifts(store)
    assert shifts == {}
    assert report == []


def test_low_n_regime_keeps_zero_shift(store: SnapshotStore):
    """Under MIN_N_PER_REGIME samples -> shift not applied."""
    for i in range(5):
        _record_win(store, f"K{i}", "C", regime="rain_cooled")
    shifts, report = derive_regime_shifts(store)
    # rain_cooled / HIGH / BETWEEN should appear as not-applied
    key = regime_key("rain_cooled", "high", "between")
    assert key in shifts
    assert shifts[key].applied is False
    assert shifts[key].delta == 0.0
    row = next(r for r in report if r.regime == "rain_cooled")
    assert row.applied is False
    assert "below threshold" in row.note


def test_high_n_regime_applies_avg_bias(store: SnapshotStore):
    """MIN_N_PER_REGIME samples where realized always > fair_mid -> positive shift."""
    n = MIN_N_PER_REGIME
    # fair_mid = 0.50, all resolve Yes (realized=1.0) -> bias = +0.50, clamped to +0.20
    for i in range(n):
        _record_win(store, f"K{i}", "C", regime="rain_cooled",
                    fair_mid=0.50)
    shifts, report = derive_regime_shifts(store)
    key = regime_key("rain_cooled", "high", "between")
    assert shifts[key].applied is True
    # Bias is clamped to +0.20 by RegimeShift.of()
    assert shifts[key].delta == pytest.approx(0.20)
    row = next(r for r in report if r.regime == "rain_cooled")
    assert row.applied is True
    assert row.n_settled == n


def test_separates_regimes(store: SnapshotStore):
    """Two regimes with different bias signatures stay separated."""
    n = MIN_N_PER_REGIME
    for i in range(n):
        _record_win(store, f"R{i}", "C", regime="rain_cooled", fair_mid=0.50)
    for i in range(n):
        _record_loss(store, f"M{i}", "C", regime="marine_layer", fair_mid=0.50)
    shifts, _report = derive_regime_shifts(store)
    rain = shifts[regime_key("rain_cooled", "high", "between")]
    marine = shifts[regime_key("marine_layer", "high", "between")]
    # rain_cooled: all wins, fair_mid 0.50 -> bias +0.50 -> clamped to +0.20
    assert rain.delta == pytest.approx(0.20)
    # marine_layer: all losses, fair_mid 0.50 -> bias -0.50 -> clamped to -0.20
    assert marine.delta == pytest.approx(-0.20)


# -- End-to-end derive_config -----------------------------------------------

def test_derive_config_returns_loadable_config(store: SnapshotStore, tmp_path: Path):
    """derive_config -> save_json -> load_json must round-trip."""
    for i in range(MIN_N_PER_TIER):
        _record_win(store, f"K{i}", "A+", edge_yes=0.15, yes_ask=70,
                    regime="clear_and_dry")
    cfg, report = derive_config(store)
    path = tmp_path / "tuned.json"
    cfg.save_json(path)
    from kalshi_scout.config import RankerConfig
    loaded = RankerConfig.load_json(path)
    assert loaded.based_on_snapshots > 0
    # The A+ LOCKED_YES tier was tuned.
    assert loaded.locked_yes.high_cutoff == cfg.locked_yes.high_cutoff
