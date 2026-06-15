import httpx

from weather_trader.kalshi import KalshiClient, market_from_dict
from weather_trader.openmeteo import parse_ensemble_response


def test_market_from_dict_parses_dollar_prices():
    m = market_from_dict({
        "ticker": "KXHIGHNYC-26JUN16-B79-80",
        "event_ticker": "KXHIGHNYC-26JUN16",
        "yes_sub_title": "79° to 80°",
        "yes_ask_dollars": "0.0900",
        "no_ask_dollars": "0.9200",
        "volume_fp": "12.0",
        "status": "open",
    })
    assert m.yes_ask == 9
    assert m.no_ask == 92
    assert m.volume == 12


def test_kalshi_iter_markets_follows_cursor():
    def handler(request: httpx.Request) -> httpx.Response:
        cursor = request.url.params.get("cursor")
        if not cursor:
            return httpx.Response(200, json={
                "markets": [{"ticker": "A-1", "event_ticker": "A"}], "cursor": "next",
            })
        return httpx.Response(200, json={
            "markets": [{"ticker": "A-2", "event_ticker": "A"}], "cursor": "",
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    kc = KalshiClient(client=client, pace_seconds=0.0)
    tickers = [m.ticker for m in kc.iter_markets()]
    assert tickers == ["A-1", "A-2"]


def test_parse_ensemble_response_members():
    data = {
        "timezone": "UTC",
        "hourly": {
            "time": ["2026-06-16T12:00", "2026-06-16T13:00"],
            "temperature_2m_member01": [70.0, 72.0],
            "temperature_2m_member02": [74.0, 76.0],
        },
    }
    points = parse_ensemble_response(data, tz="UTC")
    assert len(points) == 2
    assert points[0].members_f == (70.0, 74.0)
    assert points[0].mean_f == 72.0
    assert points[0].std_f > 0


def test_parse_ensemble_response_falls_back_to_deterministic():
    data = {"hourly": {"time": ["2026-06-16T12:00"], "temperature_2m": [80.0]}}
    points = parse_ensemble_response(data, tz="UTC")
    assert len(points) == 1 and points[0].members_f == (80.0,)


def test_parse_ensemble_response_empty_without_hourly():
    assert parse_ensemble_response({}, tz="UTC") == []
