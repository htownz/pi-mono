import json
from datetime import date, datetime, timezone

from weather_trader.alerts import AlertDispatcher, JsonlSink, alert_from_eval
from weather_trader.execution import LiveKalshiExecutor, PaperExecutor, order_from_eval
from weather_trader.forecast import ForecastDistribution, Scenario
from weather_trader.grade import evaluate
from weather_trader.models import Bracket, BracketKind, Contract, KalshiMarket, Metric, Station
from weather_trader.nws import Observation
from weather_trader.store import ForecastLog, backfill_residuals

STATION = Station("KTST", "Test", "TEST", "UTC", 0.0, 0.0)
BRACKET = Bracket(BracketKind.GTE, lo=80, hi=None)


def _contract():
    return Contract("KXHIGHTEST-26JUN16-X", "KXHIGHTEST-26JUN16", "TEST",
                    Metric.HIGH, date(2026, 6, 16), BRACKET)


def _market(yes_ask=90, yes_bid=88, no_ask=12):
    return KalshiMarket(
        ticker="KXHIGHTEST-26JUN16-X", event_ticker="KXHIGHTEST-26JUN16", title="",
        yes_sub_title="80° or above", status="open", close_time=None,
        yes_bid=yes_bid, yes_ask=yes_ask, no_bid=None, no_ask=no_ask,
        last_price=None, volume=10, open_interest=0,
    )


def _locked_eval():
    dist = ForecastDistribution(Metric.HIGH, date(2026, 6, 16), STATION,
                                [Scenario(85.0, "observed", 1.0)], 85.0, True, 0.0, 0, [])
    return evaluate(_contract(), _market(), dist), dist


# -- alerts -----------------------------------------------------------------------

def test_dispatcher_fires_only_at_or_above_min_grade(tmp_path):
    e, _ = _locked_eval()       # grades A+
    path = tmp_path / "alerts.jsonl"
    disp = AlertDispatcher([JsonlSink(str(path))], min_grade="B")
    fired = disp.dispatch([e])
    assert len(fired) == 1
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["grade"] == "A+" and rows[0]["side"] == "yes"


def test_dispatcher_skips_below_threshold():
    e, _ = _locked_eval()
    e.grade = "D"
    assert AlertDispatcher([], min_grade="B").dispatch([e]) == []


def test_alert_from_eval_fields():
    e, _ = _locked_eval()
    a = alert_from_eval(e, now_utc=datetime(2026, 6, 16, tzinfo=timezone.utc))
    assert a.ticker == "KXHIGHTEST-26JUN16-X"
    assert a.fair_mid == 1.0 and a.locked is True


# -- store / backfill -------------------------------------------------------------

class _FakeNws:
    """Returns canned observations regardless of args."""
    def __init__(self, temps):
        self._obs = [Observation(datetime(2026, 6, 16, 15, tzinfo=timezone.utc), t) for t in temps]

    def observations(self, icao, start=None, end=None, limit=500):
        return self._obs


def test_forecast_log_roundtrip_and_backfill(tmp_path):
    log = tmp_path / "f.jsonl"
    e, dist = _locked_eval()    # predicted q50 == observed == 85.0
    flog = ForecastLog(str(log))
    flog.append_evaluation(e, dist, now_utc=datetime(2026, 6, 16, 18, tzinfo=timezone.utc))
    rows = flog.read()
    assert len(rows) == 1 and rows[0]["station"] == "KTST" and rows[0]["predicted_q50_f"] == 85.0

    # Realized high was 83 -> residual = 83 - 85 = -2.
    residuals = backfill_residuals(str(log), date(2026, 6, 16), _FakeNws([83.0, 70.0]))
    assert len(residuals) == 1
    assert residuals[0]["actual_f"] == 83.0
    assert residuals[0]["residual_f"] == -2.0


def test_backfill_empty_when_no_actuals(tmp_path):
    log = tmp_path / "f.jsonl"
    e, dist = _locked_eval()
    ForecastLog(str(log)).append_evaluation(e, dist)
    assert backfill_residuals(str(log), date(2026, 6, 16), _FakeNws([])) == []


# -- execution --------------------------------------------------------------------

def test_order_from_eval_builds_best_side_order():
    e, _ = _locked_eval()       # yes edge ~0.10
    order = order_from_eval(e, count=5)
    assert order is not None
    assert order.side == "yes" and order.count == 5 and order.limit_price_cents == 90


def test_order_from_eval_none_below_min_edge():
    e, _ = _locked_eval()
    assert order_from_eval(e, min_edge=0.5) is None


def test_paper_executor_records_and_logs(tmp_path):
    e, _ = _locked_eval()
    order = order_from_eval(e)
    log = tmp_path / "orders.jsonl"
    ex = PaperExecutor(log_path=str(log))
    fill = ex.submit(order)
    assert fill.status == "paper"
    assert len(ex.orders) == 1
    assert log.exists() and "KXHIGHTEST" in log.read_text()


def test_live_executor_refuses():
    e, _ = _locked_eval()
    order = order_from_eval(e)
    try:
        LiveKalshiExecutor().submit(order)
        assert False, "live executor must not run yet"
    except NotImplementedError:
        pass
