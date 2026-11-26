"""J-Quants 形式レコードのデータ品質検証ヘルパー（準備段階）。

合成データに対しても実データに対しても使える純粋関数群。
ネットワークアクセスはしない。価格の生系列 (Open/High/Low/Close) と
調整済み系列 (AdjustmentOpen/...) は区別して報告する。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping, Sequence

RAW_PRICE_FIELDS = ("Open", "High", "Low", "Close")
ADJUSTED_PRICE_FIELDS = (
    "AdjustmentOpen",
    "AdjustmentHigh",
    "AdjustmentLow",
    "AdjustmentClose",
)
BAR_REQUIRED_FIELDS = ("Date", "Code")


@dataclass
class BarsQualityReport:
    total_records: int = 0
    duplicate_count: int = 0
    missing_field_counts: dict[str, int] = field(default_factory=dict)
    null_value_counts: dict[str, int] = field(default_factory=dict)
    invalid_date_count: int = 0
    nonpositive_close_count: int = 0
    inconsistent_ohlc_count: int = 0
    out_of_order_count: int = 0
    # {"code": ..., "from": ..., "to": ..., "gap_days": ...}
    date_gaps: list[dict[str, Any]] = field(default_factory=list)
    has_raw_series: bool = False
    has_adjusted_series: bool = False

    @property
    def is_clean(self) -> bool:
        return (
            self.duplicate_count == 0
            and self.invalid_date_count == 0
            and self.nonpositive_close_count == 0
            and self.inconsistent_ohlc_count == 0
            and not self.missing_field_counts
            and not self.date_gaps
        )


def _parse_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def validate_bars(
    records: Sequence[Mapping[str, Any]],
    max_gap_days: int = 5,
) -> BarsQualityReport:
    """日足レコードの検証。

    max_gap_days: 同一銘柄でこの日数を超える日付間隔を gap として報告する
    (週末+祝日程度は通常範囲として既定 5 日)。
    """
    report = BarsQualityReport(total_records=len(records))
    key_counter: Counter[str] = Counter()
    dates_by_code: dict[str, list[date]] = {}

    for record in records:
        for name in BAR_REQUIRED_FIELDS:
            if record.get(name) is None:
                report.missing_field_counts[name] = (
                    report.missing_field_counts.get(name, 0) + 1
                )
        for name in RAW_PRICE_FIELDS + ADJUSTED_PRICE_FIELDS + ("Volume",):
            if name in record and record.get(name) is None:
                report.null_value_counts[name] = (
                    report.null_value_counts.get(name, 0) + 1
                )
        if any(record.get(name) is not None for name in RAW_PRICE_FIELDS):
            report.has_raw_series = True
        if any(record.get(name) is not None for name in ADJUSTED_PRICE_FIELDS):
            report.has_adjusted_series = True

        code = record.get("Code")
        day = _parse_date(record.get("Date"))
        if record.get("Date") is not None and day is None:
            report.invalid_date_count += 1
        if code is not None and day is not None:
            key_counter[f"{record.get('Date')}|{code}"] += 1
            dates_by_code.setdefault(str(code), []).append(day)

        close = record.get("Close")
        if isinstance(close, (int, float)) and close <= 0:
            report.nonpositive_close_count += 1
        open_, high, low = record.get("Open"), record.get("High"), record.get("Low")
        if all(isinstance(v, (int, float)) for v in (open_, high, low, close)):
            if high < low or not (low <= close <= high) or not (low <= open_ <= high):
                report.inconsistent_ohlc_count += 1

    report.duplicate_count = sum(
        count - 1 for count in key_counter.values() if count > 1
    )

    for code, days in dates_by_code.items():
        for previous, current in zip(days, days[1:]):
            if current < previous:
                report.out_of_order_count += 1
        ordered = sorted(days)
        for previous, current in zip(ordered, ordered[1:]):
            gap = (current - previous).days
            if gap > max_gap_days:
                report.date_gaps.append(
                    {
                        "code": code,
                        "from": previous.isoformat(),
                        "to": current.isoformat(),
                        "gap_days": gap,
                    }
                )
    return report


@dataclass
class FinancialsQualityReport:
    total_records: int = 0
    duplicate_count: int = 0
    missing_disclosure_date_count: int = 0
    missing_code_count: int = 0

    @property
    def is_point_in_time_safe(self) -> bool:
        """開示日が無い行は「いつ知り得たか」を再現できないため安全でない。"""
        return self.missing_disclosure_date_count == 0 and self.missing_code_count == 0


def validate_financials(
    records: Sequence[Mapping[str, Any]],
) -> FinancialsQualityReport:
    """財務レコードの検証。開示日基準 (point-in-time) の前提をチェックする。

    注意: J-Quants の財務数値には四半期累計値が含まれる。ここでは累計値と
    単四半期値を混同しないよう、値そのものの解釈は行わず、開示日と銘柄の
    欠落・重複だけを数える。
    """
    report = FinancialsQualityReport(total_records=len(records))
    counter: Counter[str] = Counter()
    for record in records:
        if record.get("DisclosedDate") is None:
            report.missing_disclosure_date_count += 1
        if record.get("Code") is None:
            report.missing_code_count += 1
        number = record.get("DisclosureNumber")
        if number is not None:
            counter[str(number)] += 1
    report.duplicate_count = sum(
        count - 1 for count in counter.values() if count > 1
    )
    return report
