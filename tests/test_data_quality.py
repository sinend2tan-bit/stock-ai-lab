"""data_quality.py の単体テスト。合成データのみ・ネットワークは使わない。

対象は診断専用モジュール data_quality.diagnose_history と、
既存の一括/個別取得経路の整形一致（既存関数は変更しない）。
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis
import data_quality
import scan
from data_quality import diagnose_history

AS_OF = "2024-12-31"
# 2024-01-04(木) 05(金) 08(月) 09(火) 10(水)：金→月に週末をまたぐ通常系列
NORMAL_DATES = ["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"]


def make_frame(dates, *, open_=None, high=None, low=None, close=None, volume=None):
    n = len(dates)

    def col(values, default):
        return list(values) if values is not None else [default] * n

    return pd.DataFrame(
        {
            "Open": col(open_, 100.0),
            "High": col(high, 101.0),
            "Low": col(low, 99.0),
            "Close": col(close, 100.0),
            "Volume": col(volume, 1000.0),
        },
        index=pd.DatetimeIndex(pd.to_datetime(dates)),
    )


# ---------- 1. 日付系列 ----------


def test_dates_normal_no_issues():
    r = diagnose_history(make_frame(NORMAL_DATES), as_of=AS_OF)
    assert r.nat_count == 0
    assert r.duplicate_dates == []
    assert r.monotonic_increasing is True
    assert r.out_of_order == []
    assert r.future_dates == []
    # 金→月の週末またぎは「要確認」にもしない
    assert r.review_gaps == []
    assert r.anomaly_count == 0


def test_dates_duplicate_detected():
    dates = ["2024-01-04", "2024-01-05", "2024-01-05", "2024-01-08"]
    r = diagnose_history(make_frame(dates), as_of=AS_OF)
    assert len(r.duplicate_dates) == 1
    dup = r.duplicate_dates[0]
    assert dup["date"] == "2024-01-05"
    assert dup["count"] == 2
    assert dup["rows"] == [1, 2]
    assert r.duplicate_row_count == 2


def test_dates_descending_detected():
    dates = list(reversed(NORMAL_DATES))
    r = diagnose_history(make_frame(dates), as_of=AS_OF)
    assert r.monotonic_increasing is False
    assert [e["row"] for e in r.out_of_order] == [1, 2, 3, 4]


def test_dates_mixed_order_detected():
    dates = ["2024-01-04", "2024-01-09", "2024-01-08", "2024-01-10"]
    r = diagnose_history(make_frame(dates), as_of=AS_OF)
    assert r.monotonic_increasing is False
    assert [e["row"] for e in r.out_of_order] == [2]
    assert r.out_of_order[0]["previous_date"] == "2024-01-09"


def test_dates_nat_detected():
    dates = ["2024-01-04", None, "2024-01-08"]
    r = diagnose_history(make_frame(dates), as_of=AS_OF)
    assert r.nat_count == 1
    assert r.nat_rows == [1]


def test_dates_future_detected():
    r = diagnose_history(make_frame(NORMAL_DATES), as_of="2024-01-09")
    assert r.future_dates == [{"row": 4, "date": "2024-01-10"}]


def test_dates_future_reference_must_be_explicit():
    # as_of も market も指定しない既定動作は、実行環境の日付で曖昧に判定せずエラー
    with pytest.raises(ValueError):
        diagnose_history(make_frame(NORMAL_DATES))


def test_dates_future_unknown_market_rejected():
    with pytest.raises(ValueError):
        diagnose_history(make_frame(NORMAL_DATES), market="UK")


# 2024-01-10 01:00 UTC = 東京 2024-01-10 10:00 / ニューヨーク 2024-01-09 20:00
BOUNDARY_NOW_UTC = datetime(2024, 1, 10, 1, 0, tzinfo=timezone.utc)


def test_dates_future_market_jp_boundary():
    # UTCホストでも、市場（東京）の当日バーは未来日付と誤判定しない
    r = diagnose_history(
        make_frame(NORMAL_DATES), market="JP", now_utc=BOUNDARY_NOW_UTC
    )
    assert r.future_dates == []


def test_dates_future_market_us_boundary():
    # 同じ時刻でもニューヨークは前日なので、2024-01-10 のバーは未来日付になる
    r = diagnose_history(
        make_frame(NORMAL_DATES), market="US", now_utc=BOUNDARY_NOW_UTC
    )
    assert r.future_dates == [{"row": 4, "date": "2024-01-10"}]


def test_dates_future_market_us_same_day_bar_not_future():
    # ニューヨーク取引時間内（2024-01-10 16:00 UTC = 現地 11:00）は当日バーを未来にしない
    now = datetime(2024, 1, 10, 16, 0, tzinfo=timezone.utc)
    r = diagnose_history(make_frame(NORMAL_DATES), market="US", now_utc=now)
    assert r.future_dates == []


def test_dates_future_as_of_takes_precedence_over_market():
    # as_of を明示した場合は、market / now_utc より as_of を優先する
    r = diagnose_history(
        make_frame(NORMAL_DATES),
        as_of="2024-01-09",
        market="JP",
        now_utc=BOUNDARY_NOW_UTC,
    )
    assert r.future_dates == [{"row": 4, "date": "2024-01-10"}]


def test_gap_skipping_weekday_is_review_only():
    # 金→火で月曜を飛ばす間隔は、休場日の可能性があるため異常確定せず「要確認」
    dates = ["2024-01-05", "2024-01-09"]
    r = diagnose_history(make_frame(dates), as_of=AS_OF)
    assert len(r.review_gaps) == 1
    assert r.review_gaps[0]["from_date"] == "2024-01-05"
    assert r.review_gaps[0]["to_date"] == "2024-01-09"
    assert r.review_gaps[0]["skipped_weekdays"] == 1
    assert r.review_count == 1
    assert r.anomaly_count == 0


def test_dates_string_index_accepted():
    df = make_frame(NORMAL_DATES)
    df.index = pd.Index([d for d in NORMAL_DATES])
    r = diagnose_history(df, as_of=AS_OF)
    assert r.index_is_datetime is True
    assert r.anomaly_count == 0


# ---------- 2. OHLC整合性 ----------


def test_ohlc_normal_passes():
    r = diagnose_history(make_frame(NORMAL_DATES), as_of=AS_OF)
    for kind in (
        "non_finite",
        "non_positive",
        "high_below_open_close",
        "low_above_open_close",
        "high_below_low",
    ):
        assert r.price_issues[kind] == []
    assert r.anomaly_count == 0


def test_ohlc_high_below_open_close():
    df = make_frame(NORMAL_DATES, high=[101.0, 99.5, 101.0, 101.0, 101.0])
    r = diagnose_history(df, as_of=AS_OF)
    rows = [e["row"] for e in r.price_issues["high_below_open_close"]]
    assert rows == [1]
    assert r.price_issues["high_below_open_close"][0]["column"] == "High"
    # 元データは補正されない
    assert df.iloc[1]["High"] == 99.5


def test_ohlc_low_above_open_close():
    df = make_frame(NORMAL_DATES, low=[99.0, 99.0, 100.5, 99.0, 99.0])
    r = diagnose_history(df, as_of=AS_OF)
    rows = [e["row"] for e in r.price_issues["low_above_open_close"]]
    assert rows == [2]
    assert r.price_issues["low_above_open_close"][0]["column"] == "Low"


def test_ohlc_high_compared_when_open_missing():
    # Open が欠損でも、残っている Close と High の整合性は検査する
    df = make_frame(
        NORMAL_DATES,
        open_=[100.0, np.nan, 100.0, 100.0, 100.0],
        high=[101.0, 99.0, 101.0, 101.0, 101.0],
    )
    r = diagnose_history(df, as_of=AS_OF)
    rows = [e["row"] for e in r.price_issues["high_below_open_close"]]
    assert rows == [1]
    # Open の NaN 自体は従来どおり non_finite に記録される
    assert (1, "Open") in {
        (e["row"], e["column"]) for e in r.price_issues["non_finite"]
    }


def test_ohlc_high_compared_when_close_missing():
    # Close が欠損でも、残っている Open と High の整合性は検査する
    df = make_frame(
        NORMAL_DATES,
        close=[100.0, np.nan, 100.0, 100.0, 100.0],
        high=[101.0, 99.0, 101.0, 101.0, 101.0],
    )
    r = diagnose_history(df, as_of=AS_OF)
    rows = [e["row"] for e in r.price_issues["high_below_open_close"]]
    assert rows == [1]
    assert (1, "Close") in {
        (e["row"], e["column"]) for e in r.price_issues["non_finite"]
    }


def test_ohlc_low_compared_when_open_missing():
    # Open が欠損でも、残っている Close と Low の整合性は検査する
    df = make_frame(
        NORMAL_DATES,
        open_=[100.0, 100.0, np.nan, 100.0, 100.0],
        low=[99.0, 99.0, 100.5, 99.0, 99.0],
    )
    r = diagnose_history(df, as_of=AS_OF)
    rows = [e["row"] for e in r.price_issues["low_above_open_close"]]
    assert rows == [2]
    assert (2, "Open") in {
        (e["row"], e["column"]) for e in r.price_issues["non_finite"]
    }


def test_ohlc_low_compared_when_close_missing():
    # Close が欠損でも、残っている Open と Low の整合性は検査する
    df = make_frame(
        NORMAL_DATES,
        close=[100.0, 100.0, np.nan, 100.0, 100.0],
        low=[99.0, 99.0, 100.5, 99.0, 99.0],
    )
    r = diagnose_history(df, as_of=AS_OF)
    rows = [e["row"] for e in r.price_issues["low_above_open_close"]]
    assert rows == [2]
    assert (2, "Close") in {
        (e["row"], e["column"]) for e in r.price_issues["non_finite"]
    }


def test_ohlc_one_side_missing_consistent_no_false_positive():
    # 片方が欠損でも、残っている値と整合していれば検出しない
    df = make_frame(
        NORMAL_DATES,
        open_=[100.0, np.nan, 100.0, 100.0, 100.0],
        close=[100.0, 100.0, np.nan, 100.0, 100.0],
    )
    r = diagnose_history(df, as_of=AS_OF)
    assert r.price_issues["high_below_open_close"] == []
    assert r.price_issues["low_above_open_close"] == []


def test_ohlc_high_below_low():
    df = make_frame(NORMAL_DATES, high=[101.0, 101.0, 101.0, 98.0, 101.0])
    r = diagnose_history(df, as_of=AS_OF)
    rows = [e["row"] for e in r.price_issues["high_below_low"]]
    assert rows == [3]


def test_ohlc_non_positive():
    df = make_frame(
        NORMAL_DATES,
        open_=[100.0, -5.0, 100.0, 100.0, 100.0],
        close=[100.0, 100.0, 100.0, 0.0, 100.0],
    )
    r = diagnose_history(df, as_of=AS_OF)
    entries = {(e["row"], e["column"]) for e in r.price_issues["non_positive"]}
    assert entries == {(1, "Open"), (3, "Close")}


def test_ohlc_nan_detected():
    df = make_frame(NORMAL_DATES, close=[100.0, 100.0, np.nan, 100.0, 100.0])
    r = diagnose_history(df, as_of=AS_OF)
    entries = r.price_issues["non_finite"]
    assert len(entries) == 1
    assert entries[0]["row"] == 2
    assert entries[0]["column"] == "Close"
    assert entries[0]["detail"] == "NaN"


def test_ohlc_inf_detected():
    df = make_frame(
        NORMAL_DATES,
        high=[101.0, np.inf, 101.0, 101.0, 101.0],
        low=[99.0, 99.0, 99.0, -np.inf, 99.0],
    )
    r = diagnose_history(df, as_of=AS_OF)
    by_col = {(e["column"], e["detail"]) for e in r.price_issues["non_finite"]}
    assert by_col == {("High", "inf"), ("Low", "-inf")}


# ---------- 3. 出来高 ----------


def test_volume_categories_distinguished():
    df = make_frame(
        NORMAL_DATES, volume=[1000.0, 0.0, -10.0, np.nan, np.inf]
    )
    r = diagnose_history(df, as_of=AS_OF)
    assert [e["row"] for e in r.volume_issues["zero"]] == [1]
    assert [e["row"] for e in r.volume_issues["negative"]] == [2]
    assert [e["row"] for e in r.volume_issues["non_finite"]] == [3, 4]
    # ゼロは警告候補のみ、負値・非有限値は異常候補
    assert r.warning_count == 1
    assert r.anomaly_count == 3


def test_volume_normal_only():
    r = diagnose_history(make_frame(NORMAL_DATES), as_of=AS_OF)
    assert r.volume_issues == {"zero": [], "negative": [], "non_finite": []}


# ---------- 4. 欠損の可視化 ----------


def test_missing_visualization():
    dates = ["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11"]
    df = make_frame(
        dates,
        open_=[100.0, 100.0, 100.0, 100.0, 100.0, np.nan],
        close=[100.0, np.nan, np.nan, 100.0, 100.0, 100.0],
    )
    r = diagnose_history(df, as_of=AS_OF)
    assert r.row_count == 6  # 元の観測本数が残る
    assert r.missing_per_column == {
        "Open": 1,
        "High": 0,
        "Low": 0,
        "Close": 2,
        "Volume": 0,
    }
    assert r.rows_with_missing == 3
    assert r.rows_dropped_by_dropna == 3
    assert r.row_count_after_dropna == 3
    assert r.last_row_missing is True
    assert r.consecutive_missing_runs == [
        {"start_row": 1, "length": 2},
        {"start_row": 5, "length": 1},
    ]


def test_missing_none():
    r = diagnose_history(make_frame(NORMAL_DATES), as_of=AS_OF)
    assert r.rows_with_missing == 0
    assert r.row_count_after_dropna == r.row_count
    assert r.last_row_missing is False
    assert r.consecutive_missing_runs == []


def test_missing_column_absent_reported():
    df = make_frame(NORMAL_DATES).drop(columns=["Volume"])
    r = diagnose_history(df, as_of=AS_OF)
    assert r.missing_columns == ["Volume"]
    assert r.volume_issues == {"zero": [], "negative": [], "non_finite": []}
    assert r.row_count == 5


def test_empty_and_none_input():
    r = diagnose_history(pd.DataFrame(), as_of=AS_OF)
    assert r.row_count == 0
    assert set(r.missing_columns) == {"Open", "High", "Low", "Close", "Volume"}
    assert r.anomaly_count == 0
    r_none = diagnose_history(None, as_of=AS_OF)
    assert r_none.row_count == 0


# ---------- 5. 一括/個別経路の整形一致 ----------


def _ohlcv_arrays(nan_row=None):
    base = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    close = base + 0.5
    if nan_row is not None:
        close = close.copy()
        close[nan_row] = np.nan
    return {
        "Open": base,
        "High": base + 1.0,
        "Low": base - 1.0,
        "Close": close,
        "Volume": np.full(6, 1000.0),
    }


def _plain_frame(nan_row=None):
    idx = pd.bdate_range("2024-01-04", periods=6)
    return pd.DataFrame(_ohlcv_arrays(nan_row), index=idx)


def _multi_frame(nan_row=None):
    idx = pd.bdate_range("2024-01-04", periods=6)
    arrays = _ohlcv_arrays(nan_row)
    cols = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["7203.T"]]
    )
    data = np.column_stack([arrays[c] for c in ["Open", "High", "Low", "Close", "Volume"]])
    return pd.DataFrame(data, index=idx, columns=cols)


def test_shape_plain_and_multiindex_agree():
    plain = scan._shape(_plain_frame())
    multi = scan._shape(_multi_frame())
    assert not plain.empty
    pd.testing.assert_frame_equal(plain, multi)
    assert list(plain.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_fetch_history_mock_agrees_with_shape(monkeypatch):
    for frame in (_plain_frame(), _multi_frame()):
        monkeypatch.setattr(analysis.yf, "download", lambda *a, **k: frame.copy())
        fetched, err = analysis.fetch_history("7203.T")
        assert err is None
        pd.testing.assert_frame_equal(fetched, scan._shape(frame))


def test_paths_drop_same_rows_and_diagnostics_match(monkeypatch):
    plain_raw = _plain_frame(nan_row=2)
    multi_raw = _multi_frame(nan_row=2)

    shaped_plain = scan._shape(plain_raw)
    shaped_multi = scan._shape(multi_raw)
    pd.testing.assert_frame_equal(shaped_plain, shaped_multi)
    # 欠損行（元の3行目）は両経路とも除外される
    assert len(shaped_plain) == 5
    assert pd.Timestamp("2024-01-08") not in shaped_plain.index

    monkeypatch.setattr(analysis.yf, "download", lambda *a, **k: plain_raw.copy())
    fetched, err = analysis.fetch_history("7203.T")
    assert err is None
    pd.testing.assert_frame_equal(fetched, shaped_plain)

    # 整形前の診断も、通常列・MultiIndex列で一致する
    d_plain = diagnose_history(plain_raw, as_of=AS_OF).to_dict()
    d_multi = diagnose_history(multi_raw, as_of=AS_OF).to_dict()
    assert d_plain == d_multi
    assert d_plain["rows_dropped_by_dropna"] == 1
    # 整形後の診断も一致する
    assert diagnose_history(shaped_plain, as_of=AS_OF).to_dict() == diagnose_history(
        shaped_multi, as_of=AS_OF
    ).to_dict()


# ---------- 6. 非破壊性・非接続 ----------


def test_non_destructive_plain():
    df = make_frame(NORMAL_DATES, close=[100.0, np.nan, 100.0, 100.0, 100.0])
    snapshot = df.copy(deep=True)
    diagnose_history(df, as_of=AS_OF)
    pd.testing.assert_frame_equal(df, snapshot)


def test_non_destructive_multiindex():
    df = _multi_frame(nan_row=1)
    snapshot = df.copy(deep=True)
    diagnose_history(df, as_of=AS_OF)
    pd.testing.assert_frame_equal(df, snapshot)


def test_module_is_independent_of_scoring_paths():
    source = Path(data_quality.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "import analysis",
        "import scan",
        "import scoring_v2",
        "import forward",
        "import app",
        "yfinance",
        "yf.download",
    ):
        assert forbidden not in source
    # モジュール属性としても既存モジュールへの参照を持たない
    assert not hasattr(data_quality, "analysis")
    assert not hasattr(data_quality, "scan")


def test_report_to_dict_summary_counts():
    df = make_frame(
        NORMAL_DATES, volume=[1000.0, 0.0, 1000.0, 1000.0, 1000.0]
    )
    d = diagnose_history(df, as_of=AS_OF).to_dict()
    assert d["row_count"] == 5
    assert d["warning_count"] == 1
    assert d["anomaly_count"] == 0
    assert d["review_count"] == 0
    assert d["index_is_datetime"] is True
