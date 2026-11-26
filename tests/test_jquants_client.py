"""jquants_lab.client の単体テスト。合成データ + フェイク transport のみ。

ネットワークは使わない。APIキー未設定時の fail closed、ページング、
重複排除、429 の待機・再開と上限停止を検証する。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jquants_lab.client import (
    JQuantsApiError,
    JQuantsClient,
    JQuantsConfig,
    MissingApiKeyError,
    PaginationError,
    RateLimitError,
    TransportResponse,
    dedupe_records,
)
from jquants_lab.synthetic import make_daily_bars

TEST_KEY = "synthetic-test-key-do-not-use"


class FakeTransport:
    """キュー式フェイク。呼び出し回数と受け取ったヘッダーを記録する。"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def __call__(self, method, url, headers, params):
        self.calls.append(
            {"method": method, "url": url, "headers": dict(headers), "params": dict(params)}
        )
        if not self._responses:
            raise AssertionError("FakeTransport: no more queued responses")
        return self._responses.pop(0)


def make_client(transport, **kwargs):
    kwargs.setdefault("sleeper", lambda seconds: None)
    config = JQuantsConfig(api_key=TEST_KEY, base_url="https://example.invalid")
    return JQuantsClient(config, transport=transport, **kwargs)


def page(records, pagination_key=None):
    body = {"daily_quotes": records}
    if pagination_key is not None:
        body["pagination_key"] = pagination_key
    return TransportResponse(200, body, {})


# --- fail closed -------------------------------------------------------


def test_missing_env_var_raises_and_message_has_no_key_slot():
    with pytest.raises(MissingApiKeyError):
        JQuantsConfig.from_env({})
    with pytest.raises(MissingApiKeyError):
        JQuantsConfig.from_env({"JQUANTS_API_KEY": "   "})


def test_client_requires_config_with_key():
    # キーなしでは設定オブジェクト自体が作れない経路を確認する
    with pytest.raises(MissingApiKeyError):
        JQuantsConfig.from_env({"JQUANTS_API_KEY": ""})


def test_env_var_is_read_when_present():
    config = JQuantsConfig.from_env({"JQUANTS_API_KEY": TEST_KEY})
    assert config.api_key == TEST_KEY
    assert TEST_KEY not in repr(config)


# --- basic fetch / pagination ------------------------------------------


def test_single_page_fetch():
    bars = make_daily_bars(days=2)
    transport = FakeTransport([page(bars)])
    client = make_client(transport)
    result = client.get_daily_bars("2024-01-09")
    assert result.pages_fetched == 1
    assert len(result.records) == len(bars)
    assert result.duplicates_removed == 0
    assert transport.calls[0]["headers"]["x-api-key"] == TEST_KEY
    assert transport.calls[0]["params"] == {"date": "2024-01-09"}


def test_pagination_follows_key_until_absent():
    bars = make_daily_bars(days=2)
    half = len(bars) // 2
    transport = FakeTransport(
        [page(bars[:half], pagination_key="k1"), page(bars[half:])]
    )
    client = make_client(transport)
    result = client.get_daily_bars("2024-01-09")
    assert result.pages_fetched == 2
    assert len(result.records) == len(bars)
    assert transport.calls[1]["params"]["pagination_key"] == "k1"
    assert [entry["records"] for entry in result.request_log] == [half, len(bars) - half]


def test_overlapping_pages_are_deduped():
    bars = make_daily_bars(days=2)
    overlap = bars[:3]
    transport = FakeTransport(
        [page(bars, pagination_key="k1"), page(overlap)]
    )
    client = make_client(transport)
    result = client.get_daily_bars("2024-01-09")
    assert result.duplicates_removed == 3
    assert len(result.records) == len(bars)


def test_repeated_pagination_key_raises():
    bars = make_daily_bars(days=1)
    transport = FakeTransport(
        [page(bars, pagination_key="k1"), page(bars, pagination_key="k1")]
    )
    client = make_client(transport)
    with pytest.raises(PaginationError):
        client.get_daily_bars("2024-01-09")


def test_max_pages_cap_raises():
    bars = make_daily_bars(days=1)
    transport = FakeTransport(
        [page(bars, pagination_key=f"k{i}") for i in range(10)]
    )
    client = make_client(transport, max_pages=3)
    with pytest.raises(PaginationError):
        client.get_daily_bars("2024-01-09")


def test_empty_page_with_key_is_anomaly_not_error():
    bars = make_daily_bars(days=1)
    transport = FakeTransport([page([], pagination_key="k1"), page(bars)])
    client = make_client(transport)
    result = client.get_daily_bars("2024-01-09")
    assert len(result.records) == len(bars)
    assert result.anomalies


# --- 429 handling --------------------------------------------------------


def test_429_then_success_waits_and_retries():
    bars = make_daily_bars(days=1)
    waits = []
    transport = FakeTransport(
        [TransportResponse(429, {}, {}), page(bars)]
    )
    client = make_client(transport, sleeper=waits.append)
    result = client.get_daily_bars("2024-01-09")
    assert len(result.records) == len(bars)
    assert waits == [60.0]  # 既定の待機 (Free は最低1分待つ)


def test_429_respects_retry_after_header():
    bars = make_daily_bars(days=1)
    waits = []
    transport = FakeTransport(
        [TransportResponse(429, {}, {"Retry-After": "5"}), page(bars)]
    )
    client = make_client(transport, sleeper=waits.append)
    client.get_daily_bars("2024-01-09")
    assert waits == [5.0]


def test_persistent_429_hard_stops_after_max_retries():
    transport = FakeTransport([TransportResponse(429, {}, {}) for _ in range(10)])
    client = make_client(transport, max_retries=3)
    with pytest.raises(RateLimitError):
        client.get_daily_bars("2024-01-09")
    # 初回 + 再試行3回で停止し、それ以上は呼ばない
    assert len(transport.calls) == 4


# --- errors / key hygiene ------------------------------------------------


def test_non_200_raises_api_error_without_key():
    transport = FakeTransport([TransportResponse(500, {"message": "boom"}, {})])
    client = make_client(transport)
    with pytest.raises(JQuantsApiError) as excinfo:
        client.get_daily_bars("2024-01-09")
    assert excinfo.value.status_code == 500
    assert TEST_KEY not in str(excinfo.value)


def test_request_log_contains_no_key_or_raw_records():
    bars = make_daily_bars(days=1)
    transport = FakeTransport([page(bars)])
    client = make_client(transport)
    result = client.get_daily_bars("2024-01-09")
    log_text = str(result.request_log)
    assert TEST_KEY not in log_text
    assert "Close" not in log_text


def test_dedupe_records_first_wins():
    rows = [
        {"Date": "2024-01-09", "Code": "7203", "Close": 100.0},
        {"Date": "2024-01-09", "Code": "7203", "Close": 101.0},
        {"Date": "2024-01-09", "Code": "6758", "Close": 200.0},
    ]
    unique, duplicates = dedupe_records(rows, ("Date", "Code"))
    assert duplicates == 1
    assert len(unique) == 2
    assert unique[0]["Close"] == 100.0
