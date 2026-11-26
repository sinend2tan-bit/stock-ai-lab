"""jquants_lab: J-Quants 読み取り専用接続の実験準備パッケージ。

本番アプリ (app.py 等) からは import しない独立パッケージ。
実データ取得は行わない。詳細は同梱 README.md を参照。
"""

from .client import (
    API_KEY_ENV_VAR,
    FetchResult,
    JQuantsApiError,
    JQuantsClient,
    JQuantsConfig,
    MissingApiKeyError,
    PaginationError,
    RateLimitError,
)

__all__ = [
    "API_KEY_ENV_VAR",
    "FetchResult",
    "JQuantsApiError",
    "JQuantsClient",
    "JQuantsConfig",
    "MissingApiKeyError",
    "PaginationError",
    "RateLimitError",
]
