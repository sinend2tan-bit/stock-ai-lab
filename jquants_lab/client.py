"""J-Quants V2 API 読み取り専用クライアント（実験準備段階）。

設計方針:
- APIキーは環境変数 JQUANTS_API_KEY からのみ読み込む。未設定・空なら
  MissingApiKeyError で即座に停止し、ネットワーク呼び出しは一切行わない
  (fail closed)。
- APIキーは例外メッセージ・ログ・戻り値に含めない。
- 本モジュールは本番アプリ (app.py 等) から import されない独立モジュール。
  実APIへの接続は、ユーザーが自分のキーで安全に設定した後に別途行う。
  このモジュールのテストはすべて合成データ + フェイク transport で行う。

対象エンドポイント (J-Quants V2、Notion のパイロット設計 2026-11-26 に基づく):
- GET /v2/equities/bars/daily   日足 (date 指定で全銘柄)
- GET /v2/equities/master       銘柄一覧
- GET /v2/fins/summary          財務サマリー

ページネーションは pagination_key をたどる。レスポンスのデータ配列キーは
エンドポイントごとに異なる可能性があるため、明示指定がなければ本文中の
唯一のリスト型フィールドを自動検出する (初回実接続時に要確認)。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

API_KEY_ENV_VAR = "JQUANTS_API_KEY"
BASE_URL_ENV_VAR = "JQUANTS_BASE_URL"
DEFAULT_BASE_URL = "https://api.jquants.com"
DEFAULT_TIMEOUT_SECONDS = 30.0
# Free プランは 5 リクエスト/分。429 のときは最低 1 分待つのが公式案内。
DEFAULT_RATE_LIMIT_WAIT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_PAGES = 10000


class JQuantsError(Exception):
    """J-Quants クライアント基底エラー。"""


class MissingApiKeyError(JQuantsError):
    """APIキーが環境変数に設定されていない (fail closed)。"""


class JQuantsApiError(JQuantsError):
    """API が 200/429 以外を返した。メッセージに認証情報は含めない。"""

    def __init__(self, status_code: int, path: str):
        self.status_code = status_code
        self.path = path
        super().__init__(f"J-Quants API error: status={status_code} path={path}")


class RateLimitError(JQuantsError):
    """429 が再試行上限を超えた。無限リトライはしない (hard stop)。"""


class PaginationError(JQuantsError):
    """pagination_key の繰り返しやページ上限超過など、ページング異常。"""


@dataclass(frozen=True)
class TransportResponse:
    status_code: int
    body: Mapping[str, Any]
    headers: Mapping[str, str] = field(default_factory=dict)


# transport(method, url, headers, params) -> TransportResponse
Transport = Callable[
    [str, str, Mapping[str, str], Mapping[str, Any]], TransportResponse
]


def urllib_transport(
    method: str,
    url: str,
    headers: Mapping[str, str],
    params: Mapping[str, Any],
) -> TransportResponse:
    """実ネットワーク用 transport (テストでは使わない)。"""
    query = urllib.parse.urlencode(
        {k: v for k, v in params.items() if v is not None}
    )
    full_url = url + ("?" + query if query else "")
    request = urllib.request.Request(full_url, method=method, headers=dict(headers))
    try:
        with urllib.request.urlopen(
            request, timeout=DEFAULT_TIMEOUT_SECONDS
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return TransportResponse(
                response.status, payload, dict(response.headers.items())
            )
    except urllib.error.HTTPError as error:
        try:
            payload = json.loads(error.read().decode("utf-8"))
        except Exception:
            payload = {}
        error_headers = dict(error.headers.items()) if error.headers else {}
        return TransportResponse(error.code, payload, error_headers)


@dataclass(frozen=True)
class JQuantsConfig:
    api_key: str = field(repr=False)
    base_url: str = DEFAULT_BASE_URL

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "JQuantsConfig":
        source = os.environ if env is None else env
        key = (source.get(API_KEY_ENV_VAR) or "").strip()
        if not key:
            raise MissingApiKeyError(
                f"環境変数 {API_KEY_ENV_VAR} が未設定です。"
                "自分の J-Quants APIキーを環境変数に設定してください"
                "（チャットやソースコードにキーを貼らないでください）。"
            )
        base_url = (source.get(BASE_URL_ENV_VAR) or DEFAULT_BASE_URL).rstrip("/")
        return cls(api_key=key, base_url=base_url)


@dataclass
class FetchResult:
    records: list[dict[str, Any]]
    pages_fetched: int
    duplicates_removed: int
    # ログにはパス・応答コード・件数だけを残す。認証ヘッダーや生データは残さない。
    request_log: list[dict[str, Any]] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)


def dedupe_records(
    records: Sequence[Mapping[str, Any]],
    key_fields: Sequence[str],
) -> tuple[list[dict[str, Any]], int]:
    """key_fields で重複を除く。先勝ち。キーを作れない行は行全体で判定する。"""
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    duplicates = 0
    for record in records:
        if all(record.get(name) is not None for name in key_fields):
            key = "|".join(str(record.get(name)) for name in key_fields)
        else:
            key = json.dumps(dict(record), sort_keys=True, default=str)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(dict(record))
    return unique, duplicates


def _extract_records(
    body: Mapping[str, Any], data_key: Optional[str]
) -> list[Mapping[str, Any]]:
    if data_key is not None:
        value = body.get(data_key)
        if not isinstance(value, list):
            raise JQuantsApiError(200, f"response field '{data_key}' is not a list")
        return value
    list_fields = [
        name
        for name, value in body.items()
        if isinstance(value, list) and name != "pagination_key"
    ]
    if len(list_fields) != 1:
        raise JQuantsApiError(
            200, f"cannot identify data field in response (candidates={list_fields})"
        )
    return body[list_fields[0]]


class JQuantsClient:
    """読み取り専用クライアント。書き込み系メソッドは持たない。"""

    def __init__(
        self,
        config: JQuantsConfig,
        transport: Optional[Transport] = None,
        sleeper: Optional[Callable[[float], None]] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        rate_limit_wait_seconds: float = DEFAULT_RATE_LIMIT_WAIT_SECONDS,
        max_pages: int = DEFAULT_MAX_PAGES,
    ):
        self._config = config
        self._transport = transport if transport is not None else urllib_transport
        self._sleeper = sleeper if sleeper is not None else time.sleep
        self._max_retries = max_retries
        self._rate_limit_wait_seconds = rate_limit_wait_seconds
        self._max_pages = max_pages

    def _headers(self) -> dict[str, str]:
        # J-Quants V2 の APIキー認証ヘッダー。
        return {"x-api-key": self._config.api_key}

    @staticmethod
    def _retry_after_seconds(headers: Mapping[str, str]) -> Optional[float]:
        for name, value in headers.items():
            if name.lower() == "retry-after":
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
        return None

    def _get(self, path: str, params: Mapping[str, Any]) -> TransportResponse:
        url = self._config.base_url + path
        retries = 0
        while True:
            response = self._transport("GET", url, self._headers(), params)
            if response.status_code == 200:
                return response
            if response.status_code == 429:
                if retries >= self._max_retries:
                    raise RateLimitError(
                        f"rate limited (429) after {retries} retries: path={path}"
                    )
                wait = self._retry_after_seconds(response.headers)
                if wait is None:
                    wait = self._rate_limit_wait_seconds
                self._sleeper(wait)
                retries += 1
                continue
            raise JQuantsApiError(response.status_code, path)

    def fetch_all_pages(
        self,
        path: str,
        params: Optional[Mapping[str, Any]] = None,
        data_key: Optional[str] = None,
        key_fields: Sequence[str] = ("Date", "Code"),
    ) -> FetchResult:
        """pagination_key をたどって全ページ取得し、重複を除いて返す。"""
        current_params: dict[str, Any] = dict(params or {})
        raw_records: list[Mapping[str, Any]] = []
        request_log: list[dict[str, Any]] = []
        anomalies: list[str] = []
        seen_page_keys: set[str] = set()
        pages = 0
        while True:
            if pages >= self._max_pages:
                raise PaginationError(
                    f"page limit exceeded ({self._max_pages}): path={path}"
                )
            response = self._get(path, current_params)
            page_records = _extract_records(response.body, data_key)
            pages += 1
            request_log.append(
                {
                    "path": path,
                    "status_code": response.status_code,
                    "records": len(page_records),
                }
            )
            raw_records.extend(page_records)
            next_key = response.body.get("pagination_key")
            if not next_key:
                break
            if len(page_records) == 0:
                anomalies.append("empty page returned with a pagination_key")
            if next_key in seen_page_keys:
                raise PaginationError(
                    f"pagination_key repeated (possible loop): path={path}"
                )
            seen_page_keys.add(str(next_key))
            current_params = {**current_params, "pagination_key": next_key}
        unique, duplicates = dedupe_records(raw_records, key_fields)
        return FetchResult(
            records=unique,
            pages_fetched=pages,
            duplicates_removed=duplicates,
            request_log=request_log,
            anomalies=anomalies,
        )

    def get_daily_bars(self, date: str) -> FetchResult:
        """単一日の全銘柄日足。調整済み/未調整は行内のフィールドで区別される。"""
        return self.fetch_all_pages(
            "/v2/equities/bars/daily",
            params={"date": date},
            key_fields=("Date", "Code"),
        )

    def get_listed_info(self, date: Optional[str] = None) -> FetchResult:
        """銘柄一覧。過去日付を指定すれば当時の上場銘柄が得られる想定。"""
        params: dict[str, Any] = {}
        if date is not None:
            params["date"] = date
        return self.fetch_all_pages(
            "/v2/equities/master",
            params=params,
            key_fields=("Date", "Code"),
        )

    def get_financial_summary(
        self,
        code: Optional[str] = None,
        date: Optional[str] = None,
    ) -> FetchResult:
        """財務サマリー。開示日 (DisclosedDate) 基準で扱うこと。"""
        params: dict[str, Any] = {}
        if code is not None:
            params["code"] = code
        if date is not None:
            params["date"] = date
        return self.fetch_all_pages(
            "/v2/fins/summary",
            params=params,
            key_fields=("DisclosureNumber",),
        )
