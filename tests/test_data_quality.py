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
    # 必須列の欠落は異常件数にも反映される（欠落1列 = 異常1件）
    assert r.anomaly_count == 1
    assert r.has_anomalies is True
    assert r.to_dict()["anomaly_count"] == 1


def test_empty_and_none_input():
    r = diagnose_history(pd.DataFrame(), as_of=AS_OF)
    assert r.row_count == 0
    assert set(r.missing_columns) == {"Open", "High", "Low", "Close", "Volume"}
    # 空DataFrameは必須5列すべてが欠落するため、異常件数は5
    assert r.anomaly_count == 5
    assert r.has_anomalies is True
    assert r.to_dict()["anomaly_count"] == 5
    r_none = diagnose_history(None, as_of=AS_OF)
    assert r_none.row_count == 0
    # None も必須5列すべてが欠落したものとして異常件数に反映される
    assert set(r_none.missing_columns) == {"Open", "High", "Low", "Close", "Volume"}
    assert r_none.anomaly_count == 5
    assert r_none.has_anomalies is True
    assert r_none.to_dict()["anomaly_count"] == 5


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


# ---------- 7. 境界条件：重複列名（入力契約） ----------

from datetime import date as _date
from decimal import Decimal


def test_boundary_duplicate_columns_plain_rejected():
    # 必須列の重複は列の特定が曖昧になるため、契約外として明示的にエラー
    df = pd.concat([make_frame(NORMAL_DATES)[["Open"]], make_frame(NORMAL_DATES)], axis=1)
    assert list(df.columns).count("Open") == 2
    with pytest.raises(ValueError, match="重複"):
        diagnose_history(df, as_of=AS_OF)


def test_boundary_duplicate_columns_multiindex_two_tickers_rejected():
    # 複数銘柄の MultiIndex を先頭レベルだけ平坦化すると列名が重複する
    idx = pd.bdate_range("2024-01-04", periods=5)
    cols = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["7203.T", "6758.T"]]
    )
    data = np.tile(np.array([100.0, 101.0, 99.0, 100.0, 1000.0]), (5, 2))
    df = pd.DataFrame(data, index=idx, columns=cols)
    with pytest.raises(ValueError, match="重複"):
        diagnose_history(df, as_of=AS_OF)


def test_boundary_duplicate_nonrequired_column_rejected():
    # 必須列でない列の重複も、列名による特定が曖昧になるため契約外とする
    df = make_frame(NORMAL_DATES)
    df["Adj Close"] = 100.0
    df = pd.concat([df, df[["Adj Close"]]], axis=1)
    with pytest.raises(ValueError, match="重複"):
        diagnose_history(df, as_of=AS_OF)


# ---------- 8. 境界条件：日付インデックス ----------


def test_boundary_index_mixed_timezone_strings():
    # 形式の混在した文字列は例外にせず、解釈できた日付だけを使う。
    # 先頭要素と形式が異なる要素は NaT として記録する（現行仕様の固定）
    df = make_frame(NORMAL_DATES)
    df.index = pd.Index(
        [
            "2024-01-04",
            "2024-01-05T00:00:00+09:00",
            "2024-01-08",
            "2024-01-09T00:00:00-05:00",
            "2024-01-10",
        ]
    )
    r = diagnose_history(df, as_of=AS_OF)
    assert r.index_is_datetime is True
    assert r.nat_rows == [1, 3]
    assert r.nat_count == 2



def test_dates_mixed_string_formats_accepted():
    df = make_frame(NORMAL_DATES)
    df.index = pd.Index(["2024-01-04", "2024/01/05", "2024-01-08T00:00:00", "2024/01/09", "2024-01-10"])
    r = diagnose_history(df, as_of=AS_OF)
    assert r.index_is_datetime is True
    assert r.nat_count == 0
    assert r.anomaly_count == 0


def test_boundary_index_unparseable_strings_all_nat():
    df = make_frame(NORMAL_DATES)
    df.index = pd.Index(["foo", "bar", "baz", "qux", "zzz"])
    r = diagnose_history(df, as_of=AS_OF)
    assert r.index_is_datetime is False
    assert r.nat_count == 5
    assert r.nat_rows == [0, 1, 2, 3, 4]


def test_boundary_index_numeric_not_datetime():
    # 数値インデックスは日付として解釈せず、全行 NaT として扱う（例外にしない）
    df = make_frame(NORMAL_DATES)
    df.index = pd.RangeIndex(5)
    r = diagnose_history(df, as_of=AS_OF)
    assert r.index_is_datetime is False
    assert r.nat_count == 5


def test_boundary_index_int_yyyymmdd_not_parsed_as_date():
    # YYYYMMDD 形式の整数も日付とは解釈しない（黙って日付化しない）
    df = make_frame(NORMAL_DATES)
    df.index = pd.Index([20240104, 20240105, 20240108, 20240109, 20240110])
    r = diagnose_history(df, as_of=AS_OF)
    assert r.index_is_datetime is False
    assert r.nat_count == 5


def test_boundary_index_empty_datetime_with_columns():
    # 列はあるが0行（空の DatetimeIndex）は例外なく空の診断になる
    df = make_frame([])
    assert isinstance(df.index, pd.DatetimeIndex)
    r = diagnose_history(df, as_of=AS_OF)
    assert r.row_count == 0
    assert r.missing_columns == []
    assert r.anomaly_count == 0
    assert r.last_row_missing is False


def test_boundary_index_dst_tz_aware():
    # DST 境界（春の切り替え・秋の切り替え）を含む tz 付きインデックスでも
    # 例外なく診断でき、当日バー等を誤検出しない
    for start, as_of in (("2024-03-08", "2024-03-31"), ("2024-11-01", "2024-11-30")):
        idx = pd.date_range(start, periods=5, freq="D", tz="America/New_York")
        df = make_frame(NORMAL_DATES)
        df.index = idx
        r = diagnose_history(df, as_of=as_of)
        assert r.index_is_datetime is True
        assert r.nat_count == 0
        assert r.anomaly_count == 0
        assert r.review_gaps == []


def test_boundary_index_date_objects():
    # datetime.date オブジェクトのインデックスは日付として解釈する
    df = make_frame(NORMAL_DATES)
    df.index = pd.Index(
        [_date(2024, 1, 4), _date(2024, 1, 5), _date(2024, 1, 8),
         _date(2024, 1, 9), _date(2024, 1, 10)]
    )
    r = diagnose_history(df, as_of=AS_OF)
    assert r.index_is_datetime is True
    assert r.nat_count == 0
    assert r.anomaly_count == 0


# ---------- 9. 境界条件：値の型 ----------


def test_boundary_numeric_strings_diagnosed_as_numbers():
    # 数値文字列は数値として診断する（非数値として誤検出しない）
    df = make_frame(
        NORMAL_DATES,
        close=["100.5", "101", "99.9", "100", "102"],
        volume=["1000"] * 5,
    )
    r = diagnose_history(df, as_of=AS_OF)
    assert r.price_issues["non_finite"] == []
    # Close=102 > High=101 の行は、文字列でも整合性違反として検出する
    assert [e["row"] for e in r.price_issues["high_below_open_close"]] == [4]


def test_boundary_non_numeric_strings_nan_in_prices_not_in_missing():
    # 数値化できない文字列は価格診断では NaN として記録する。
    # 一方、欠損の可視化は既存 dropna() と同じ基準（元の値の isna）なので
    # 文字列は欠損に数えない（rows_dropped_by_dropna も増えない）
    df = make_frame(NORMAL_DATES, close=["abc", "100", "1,000", "", "100"])
    r = diagnose_history(df, as_of=AS_OF)
    assert [e["row"] for e in r.price_issues["non_finite"]] == [0, 2, 3]
    assert all(
        e["detail"] == "NaN" for e in r.price_issues["non_finite"]
    )
    assert r.missing_per_column["Close"] == 0
    assert r.rows_dropped_by_dropna == 0


def test_boundary_decimal_values():
    df = make_frame(
        NORMAL_DATES,
        close=[Decimal("100.5"), Decimal("NaN"), Decimal("101"),
               Decimal("100"), Decimal("100")],
    )
    r = diagnose_history(df, as_of=AS_OF)
    assert [(e["row"], e["detail"]) for e in r.price_issues["non_finite"]] == [
        (1, "NaN")
    ]
    # Decimal("NaN") は元の値自体が欠損なので、欠損にも数える
    assert r.missing_per_column["Close"] == 1


def test_boundary_nullable_dtype_with_pd_na():
    df = make_frame(NORMAL_DATES)
    df["Close"] = pd.array([100.0, pd.NA, 100.0, 100.0, 100.0], dtype="Float64")
    df["Volume"] = pd.array([1000, pd.NA, 1000, 1000, 1000], dtype="Int64")
    r = diagnose_history(df, as_of=AS_OF)
    assert [(e["row"], e["detail"]) for e in r.price_issues["non_finite"]] == [
        (1, "NaN")
    ]
    assert [(e["row"], e["detail"]) for e in r.volume_issues["non_finite"]] == [
        (1, "NaN")
    ]
    assert r.missing_per_column["Close"] == 1
    assert r.missing_per_column["Volume"] == 1


def test_boundary_inf_strings():
    df = make_frame(
        NORMAL_DATES,
        high=["inf", "101", "101", "101", "101"],
        low=["99", "99", "99", "99", "-inf"],
    )
    r = diagnose_history(df, as_of=AS_OF)
    by_col = {(e["column"], e["detail"]) for e in r.price_issues["non_finite"]}
    assert by_col == {("High", "inf"), ("Low", "-inf")}


def test_boundary_object_column_pd_na_and_none():
    df = make_frame(NORMAL_DATES, close=[100.0, pd.NA, 100.0, 100.0, None])
    r = diagnose_history(df, as_of=AS_OF)
    assert [e["row"] for e in r.price_issues["non_finite"]] == [1, 4]
    assert r.missing_per_column["Close"] == 2


# ---------- 10. 境界条件：列構成 ----------


def test_boundary_missing_high_low_columns():
    # High/Low が無くても例外にせず、不足列として記録し、比較は行わない
    df = make_frame(NORMAL_DATES).drop(columns=["High", "Low"])
    r = diagnose_history(df, as_of=AS_OF)
    assert r.missing_columns == ["High", "Low"]
    # 必須列の欠落2列は異常件数にも2件として反映される
    assert r.anomaly_count == 2
    assert r.has_anomalies is True
    assert r.price_issues["high_below_open_close"] == []
    assert r.price_issues["low_above_open_close"] == []
    assert r.price_issues["high_below_low"] == []


def test_boundary_missing_open_close_columns():
    df = make_frame(NORMAL_DATES).drop(columns=["Open", "Close"])
    r = diagnose_history(df, as_of=AS_OF)
    assert r.missing_columns == ["Open", "Close"]
    assert r.price_issues["high_below_open_close"] == []
    assert r.price_issues["low_above_open_close"] == []
    # 参照する価格が無い場合でも High/Low 自体の検査は行う（ここでは問題なし）。
    # 異常件数は必須列の欠落2列分のみで、価格検査由来の異常は加わらない
    assert r.anomaly_count == 2
    assert r.has_anomalies is True
    assert r.to_dict()["anomaly_count"] == 2


def test_boundary_reordered_columns_same_diagnosis():
    base = diagnose_history(make_frame(NORMAL_DATES), as_of=AS_OF)
    reordered_df = make_frame(NORMAL_DATES)[["Volume", "Close", "Low", "High", "Open"]]
    r = diagnose_history(reordered_df, as_of=AS_OF)
    # 列順は入力どおり記録し、診断結果は変わらない
    assert r.columns == ["Volume", "Close", "Low", "High", "Open"]
    assert r.missing_columns == base.missing_columns
    assert r.missing_per_column == base.missing_per_column
    assert r.price_issues == base.price_issues
    assert r.volume_issues == base.volume_issues
    assert r.anomaly_count == base.anomaly_count


def test_boundary_extra_column_ignored_for_diagnosis():
    df = make_frame(NORMAL_DATES)
    df["Adj Close"] = [1.0, np.nan, 1.0, 1.0, 1.0]  # 欠損があっても診断対象外
    r = diagnose_history(df, as_of=AS_OF)
    assert "Adj Close" in r.columns
    assert "Adj Close" not in r.missing_per_column
    assert r.rows_with_missing == 0
    assert r.anomaly_count == 0


def test_boundary_zero_rows_with_columns():
    df = make_frame(NORMAL_DATES).iloc[0:0]
    r = diagnose_history(df, as_of=AS_OF)
    assert r.row_count == 0
    assert r.missing_columns == []
    # 必須列が全て揃った0行DataFrameは欠落がないため異常件数は0のまま
    assert r.anomaly_count == 0
    assert r.has_anomalies is False
    assert r.to_dict()["anomaly_count"] == 0
    assert r.consecutive_missing_runs == []


# ---------- 11. 境界条件：出力の堅牢性 ----------


def test_boundary_to_dict_mutation_does_not_affect_report():
    # to_dict() の戻り値（入れ子を含む）を変更しても、レポート内部の状態や
    # 次の to_dict() の結果に副作用がないこと
    dates = ["2024-01-04", "2024-01-05", "2024-01-05", "2024-01-09", "2024-01-10"]
    df = make_frame(dates, volume=[1000.0, 0.0, 1000.0, 1000.0, 1000.0])
    r = diagnose_history(df, as_of=AS_OF)
    before = r.to_dict()
    assert before["duplicate_row_count"] == 2
    assert before["warning_count"] == 1

    d = r.to_dict()
    d["price_issues"]["non_finite"].append(
        {"row": 99, "date": "x", "column": "Open", "detail": "NaN"}
    )
    d["volume_issues"]["zero"].clear()
    d["duplicate_dates"][0]["count"] = 999
    d["duplicate_dates"][0]["rows"].append(99)
    d["out_of_order"].append({"row": 0, "date": "x", "previous_date": "y"})
    d["future_dates"].append({"row": 0, "date": "2999-01-01"})
    d["review_gaps"].append({"from_date": "x", "to_date": "y", "skipped_weekdays": 9})
    d["consecutive_missing_runs"].append({"start_row": 0, "length": 9})
    d["columns"].append("HACK")
    d["missing_columns"].append("HACK")
    d["nat_rows"].append(99)
    d["missing_per_column"]["Open"] = 999

    after = r.to_dict()
    assert after == before
    assert r.price_issues["non_finite"] == []
    assert r.warning_count == 1
    assert r.anomaly_count == before["anomaly_count"]
    assert r.columns == ["Open", "High", "Low", "Close", "Volume"]


# ---------- 12. 境界条件：集計定義 ----------


def test_boundary_duplicate_row_count_counts_all_involved_rows():
    # duplicate_row_count は「重複に関係する全行数」（出現回数の合計）。
    # 「余分な重複行数」（出現回数−1 の合計）ではない
    dates = ["2024-01-04", "2024-01-05", "2024-01-05", "2024-01-05", "2024-01-08"]
    r = diagnose_history(make_frame(dates), as_of=AS_OF)
    assert r.duplicate_dates[0]["count"] == 3
    assert r.duplicate_row_count == 3  # 余分な行数なら 2 になる


def test_boundary_duplicate_row_count_multiple_groups():
    dates = ["2024-01-04", "2024-01-04", "2024-01-08", "2024-01-08", "2024-01-08"]
    r = diagnose_history(make_frame(dates), as_of=AS_OF)
    assert [d["count"] for d in r.duplicate_dates] == [2, 3]
    assert r.duplicate_row_count == 5  # 2 + 3（余分な行数なら 3）


# ---------- 13. 集計定義：必須列欠落の異常件数への反映 ----------


def test_missing_columns_added_to_anomaly_count():
    # anomaly_count には必須列の欠落数（len(missing_columns)）が加算され、
    # 他の異常とは合算される。has_anomalies と to_dict() も整合する
    dates = ["2024-01-04", "2024-01-05", "2024-01-05", "2024-01-09", "2024-01-10"]
    df = make_frame(dates).drop(columns=["Volume"])
    r = diagnose_history(df, as_of=AS_OF)
    assert r.missing_columns == ["Volume"]
    assert r.duplicate_row_count == 2
    # 欠落1列 + 重複日付に関係する2行 = 3
    assert r.anomaly_count == 3
    assert r.has_anomalies is True
    assert r.to_dict()["anomaly_count"] == 3


def test_no_missing_columns_keeps_anomaly_count_zero():
    # 必須列が全て揃った通常入力では、欠落由来の加算はなく異常件数は0
    r = diagnose_history(make_frame(NORMAL_DATES), as_of=AS_OF)
    assert r.missing_columns == []
    assert r.anomaly_count == 0
    assert r.has_anomalies is False
    assert r.to_dict()["anomaly_count"] == 0


# ---------- 14. 必須列のカスタム指定（required_columns 引数の契約） ----------


def test_required_columns_custom_subset_only_specified_are_required():
    # 標準OHLCV以外の列構成でも、指定した列だけが必須列として扱われる。
    # Close と Volume だけの入力を required_columns=("Close", "Volume") で
    # 診断すると、Open/High/Low は欠落に数えない
    df = make_frame(NORMAL_DATES)[["Close", "Volume"]]
    r = diagnose_history(df, as_of=AS_OF, required_columns=("Close", "Volume"))
    assert r.missing_columns == []
    assert r.anomaly_count == 0
    assert r.has_anomalies is False
    # 同じ入力を既定（OHLCV 5列）で診断すると、欠けている3列が必須列欠落になる
    r_default = diagnose_history(df, as_of=AS_OF)
    assert r_default.missing_columns == ["Open", "High", "Low"]
    assert r_default.anomaly_count == 3


def test_required_columns_custom_non_ohlcv_name_missing():
    # 標準OHLCVに無い列名を指定し、入力に無ければ欠落として記録され、
    # missing_columns / anomaly_count / has_anomalies / to_dict() が整合する
    r = diagnose_history(
        make_frame(NORMAL_DATES),
        as_of=AS_OF,
        required_columns=("Open", "Close", "Adj Close"),
    )
    assert r.missing_columns == ["Adj Close"]
    assert r.anomaly_count == 1
    assert r.has_anomalies is True
    d = r.to_dict()
    assert d["missing_columns"] == ["Adj Close"]
    assert d["anomaly_count"] == 1
    # 欠落した必須列は入力に存在しないため、欠損集計の対象にはならない
    assert set(r.missing_per_column) == {"Open", "Close"}


def test_required_columns_missing_order_follows_argument():
    # missing_columns は required_columns で指定した順序で記録される
    df = make_frame(NORMAL_DATES).drop(columns=["Open", "Volume"])
    r = diagnose_history(df, as_of=AS_OF, required_columns=("Volume", "Open"))
    assert r.missing_columns == ["Volume", "Open"]
    assert r.anomaly_count == 2
    assert r.has_anomalies is True
    assert r.to_dict()["missing_columns"] == ["Volume", "Open"]


def test_required_columns_missing_aggregation_scoped_to_present_required():
    # 欠損行数／dropna相当の集計は「指定した必須列のうち入力に存在する列」
    # だけが対象になる。指定外の列の欠損は集計に入らない
    dates = [
        "2024-01-04", "2024-01-05", "2024-01-08",
        "2024-01-09", "2024-01-10", "2024-01-11",
    ]
    df = make_frame(
        dates,
        close=[100.0, np.nan, 100.0, 100.0, 100.0, 100.0],
        volume=[1000.0, 1000.0, np.nan, 1000.0, 1000.0, 1000.0],
    )
    # Close だけを必須にすると、Volume の欠損行は集計に入らない
    r_close = diagnose_history(df, as_of=AS_OF, required_columns=("Close",))
    assert r_close.missing_per_column == {"Close": 1}
    assert r_close.rows_with_missing == 1
    assert r_close.rows_dropped_by_dropna == 1
    assert r_close.row_count_after_dropna == 5
    assert r_close.consecutive_missing_runs == [{"start_row": 1, "length": 1}]
    # Volume だけを必須にすると、Close の欠損行は集計に入らない
    r_volume = diagnose_history(df, as_of=AS_OF, required_columns=("Volume",))
    assert r_volume.missing_per_column == {"Volume": 1}
    assert r_volume.rows_with_missing == 1
    assert r_volume.rows_dropped_by_dropna == 1
    assert r_volume.row_count_after_dropna == 5
    assert r_volume.consecutive_missing_runs == [{"start_row": 2, "length": 1}]
    # 両方を必須にすると、欠損行は合算される
    r_both = diagnose_history(
        df, as_of=AS_OF, required_columns=("Close", "Volume")
    )
    assert r_both.missing_per_column == {"Close": 1, "Volume": 1}
    assert r_both.rows_with_missing == 2
    assert r_both.rows_dropped_by_dropna == 2
    assert r_both.row_count_after_dropna == 4
    assert r_both.consecutive_missing_runs == [{"start_row": 1, "length": 2}]


def test_required_columns_none_and_empty_with_custom():
    # None・空DataFrameでは、指定した必須列がすべて欠落として記録され、
    # 異常件数にも反映される
    r_none = diagnose_history(
        None, as_of=AS_OF, required_columns=("Close", "Volume")
    )
    assert r_none.row_count == 0
    assert r_none.missing_columns == ["Close", "Volume"]
    assert r_none.anomaly_count == 2
    assert r_none.has_anomalies is True
    assert r_none.to_dict()["anomaly_count"] == 2
    r_empty = diagnose_history(
        pd.DataFrame(), as_of=AS_OF, required_columns=("Close",)
    )
    assert r_empty.missing_columns == ["Close"]
    assert r_empty.anomaly_count == 1
    assert r_empty.has_anomalies is True


def test_required_columns_default_matches_explicit_ohlcv():
    # 既定値（引数省略）は REQUIRED_COLUMNS（OHLCV 5列）と同一であり、
    # 明示指定しても既存挙動は変わらない。リストで渡しても同じ扱いになる
    assert data_quality.REQUIRED_COLUMNS == (
        "Open", "High", "Low", "Close", "Volume"
    )
    df = make_frame(
        NORMAL_DATES, close=[100.0, np.nan, 100.0, 100.0, 100.0]
    )
    omitted = diagnose_history(df, as_of=AS_OF).to_dict()
    explicit = diagnose_history(
        df, as_of=AS_OF, required_columns=data_quality.REQUIRED_COLUMNS
    ).to_dict()
    assert omitted == explicit
    as_list = diagnose_history(
        df,
        as_of=AS_OF,
        required_columns=["Open", "High", "Low", "Close", "Volume"],
    ).to_dict()
    assert as_list == explicit
