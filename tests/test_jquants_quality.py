"""jquants_lab.quality の単体テスト。合成データのみ。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jquants_lab.quality import validate_bars, validate_financials
from jquants_lab.synthetic import make_daily_bars, make_financials


def test_clean_synthetic_bars_pass():
    report = validate_bars(make_daily_bars(days=10))
    assert report.total_records == 50
    assert report.is_clean
    assert report.has_raw_series
    assert report.has_adjusted_series


def test_duplicates_are_counted():
    bars = make_daily_bars(days=2)
    report = validate_bars(bars + bars[:2])
    assert report.duplicate_count == 2
    assert not report.is_clean


def test_missing_and_null_fields_are_counted():
    bars = make_daily_bars(days=1)
    bars[0] = dict(bars[0])
    del bars[0]["Code"]
    bars[1] = dict(bars[1])
    bars[1]["Close"] = None
    report = validate_bars(bars)
    assert report.missing_field_counts.get("Code") == 1
    assert report.null_value_counts.get("Close") == 1


def test_invalid_date_and_nonpositive_close():
    bars = make_daily_bars(days=1)
    bars[0] = dict(bars[0], Date="2024-13-40")
    bars[1] = dict(bars[1], Close=0)
    report = validate_bars(bars)
    assert report.invalid_date_count == 1
    assert report.nonpositive_close_count == 1


def test_inconsistent_ohlc_detected():
    bars = make_daily_bars(days=1)
    bars[0] = dict(bars[0], High=50.0, Low=100.0)
    report = validate_bars(bars)
    assert report.inconsistent_ohlc_count >= 1


def test_date_gap_detected_per_code():
    bars = make_daily_bars(codes=("7203",), days=3)
    gapped = bars[:1] + bars[2:]  # 中日を抜く
    gapped.append(dict(bars[2], Date="2024-02-01"))
    report = validate_bars(gapped, max_gap_days=5)
    assert any(gap["code"] == "7203" for gap in report.date_gaps)


def test_out_of_order_detected():
    bars = make_daily_bars(codes=("7203",), days=3)
    shuffled = [bars[1], bars[0], bars[2]]
    report = validate_bars(shuffled)
    assert report.out_of_order_count >= 1


def test_financials_point_in_time_checks():
    fins = make_financials()
    report = validate_financials(fins)
    assert report.is_point_in_time_safe
    broken = [dict(fins[0])]
    del broken[0]["DisclosedDate"]
    broken_report = validate_financials(broken)
    assert broken_report.missing_disclosure_date_count == 1
    assert not broken_report.is_point_in_time_safe


def test_financials_duplicates_by_disclosure_number():
    fins = make_financials()
    report = validate_financials(fins + fins[:1])
    assert report.duplicate_count == 1
