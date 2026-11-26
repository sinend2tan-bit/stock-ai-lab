"""合成データ生成器 (テスト・検証専用)。

実データではない。J-Quants の日足・銘柄一覧・財務サマリーの
フィールド形状だけを模した、決定的 (seed 固定) なダミーデータを作る。
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Any

DEFAULT_CODES = ("7203", "6758", "8306", "9984", "5401")


def business_days(start: date, count: int) -> list[date]:
    days: list[date] = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def make_daily_bars(
    codes: tuple[str, ...] = DEFAULT_CODES,
    start: date = date(2024, 1, 9),
    days: int = 20,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """銘柄×営業日の日足ダミー。調整済み系列は生系列と同値 (分割なし前提)。"""
    rng = random.Random(seed)
    trading_days = business_days(start, days)
    records: list[dict[str, Any]] = []
    for code in codes:
        price = 1000.0 + (int(code) % 500)
        for day in trading_days:
            drift = rng.uniform(-0.02, 0.02)
            open_price = price
            close_price = max(10.0, price * (1 + drift))
            high = max(open_price, close_price) * 1.005
            low = min(open_price, close_price) * 0.995
            records.append(
                {
                    "Date": day.isoformat(),
                    "Code": code,
                    "Open": round(open_price, 1),
                    "High": round(high, 1),
                    "Low": round(low, 1),
                    "Close": round(close_price, 1),
                    "Volume": rng.randint(10000, 500000),
                    "AdjustmentFactor": 1.0,
                    "AdjustmentOpen": round(open_price, 1),
                    "AdjustmentHigh": round(high, 1),
                    "AdjustmentLow": round(low, 1),
                    "AdjustmentClose": round(close_price, 1),
                }
            )
            price = close_price
    return records


def make_master(
    codes: tuple[str, ...] = DEFAULT_CODES,
    on: date = date(2024, 1, 9),
) -> list[dict[str, Any]]:
    return [
        {
            "Date": on.isoformat(),
            "Code": code,
            "CompanyName": f"Synthetic {code}",
            "Market": "Prime",
        }
        for code in codes
    ]


def make_financials(
    codes: tuple[str, ...] = DEFAULT_CODES,
    disclosed: date = date(2024, 2, 9),
) -> list[dict[str, Any]]:
    """財務サマリーダミー。数値は四半期累計を模した合成値。"""
    return [
        {
            "Code": code,
            "DisclosedDate": disclosed.isoformat(),
            "DisclosureNumber": f"SYN-{code}-001",
            "TypeOfDocument": "3Q Financial Results",
            # 累計値の形を模した合成数値。単四半期値ではない。
            "NetSales": 1000000000 * (index + 1),
            "OperatingProfit": 50000000 * (index + 1),
        }
        for index, code in enumerate(codes)
    ]
