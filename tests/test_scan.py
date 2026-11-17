"""スキャン機能（scan.py）の単体テスト。ネットワークは使わない。"""

import ast
from pathlib import Path

import pandas as pd
import pytest

import scan


def make_frame(closes, end=None):
    if end is None:
        end = pd.Timestamp.now(tz="Asia/Tokyo").date()
    idx = pd.bdate_range(end=end, periods=len(closes))
    return pd.DataFrame(
        {
            "Open": closes, "High": closes, "Low": closes,
            "Close": closes, "Volume": [1000] * len(closes),
        },
        index=idx,
    )


def rising(n=300):
    return make_frame([100.0 + 0.2 * k for k in range(n)])


def falling(n=300):
    return make_frame([200.0 - 0.2 * k for k in range(n)])


def flat(n=300):
    return make_frame([100.0] * n)


UNIVERSE3 = [
    ("AAA.T", "上昇社", "テスト"),
    ("BBB.T", "横ばい社", "テスト"),
    ("CCC.T", "下落社", "テスト"),
]
FRAMES = {"AAA.T": rising(), "BBB.T": flat(), "CCC.T": falling()}


def batch_ok(symbols):
    return {s: (FRAMES[s], None) for s in symbols}


def single_ok(symbol):
    return FRAMES[symbol], None


# --- 銘柄リストの整合性 ---

def test_universe_integrity():
    codes = [u[0] for u in scan.UNIVERSE_JP]
    assert len(codes) == 60
    assert len(set(codes)) == 60  # 重複なし
    assert all(c.endswith(".T") for c in codes)  # yfinanceの日本株形式
    assert all(u[1] and u[2] for u in scan.UNIVERSE_JP)  # 会社名・業種が空でない
    assert "9613.T" not in codes  # 上場廃止のため含めない
    assert "4307.T" in codes  # 差し替え先


# --- ランキングの並び順 ---

def test_ranking_order_by_mid_score():
    ranked, excluded, meta = scan.run_scan(
        UNIVERSE3, batch_ok, single_ok, run_at="2026-10-29 00:00:00"
    )
    assert excluded == []
    assert [r["コード"] for r in ranked] == ["AAA.T", "BBB.T", "CCC.T"]
    assert [r["順位"] for r in ranked] == [1, 2, 3]
    assert ranked[0]["中期スコア"] > ranked[1]["中期スコア"] > ranked[2]["中期スコア"]
    assert meta["ランキング件数"] == 3 and meta["参考外件数"] == 0
    # 選定理由が付いていること
    assert ranked[0]["選定理由"] != ""


def test_ranking_is_deterministic_for_ties():
    universe = [("ZZZ.T", "同点社Z", "テスト"), ("AAA.T", "同点社A", "テスト")]
    frames = {"ZZZ.T": rising(), "AAA.T": rising()}
    ranked, _, _ = scan.run_scan(
        universe,
        lambda syms: {s: (frames[s], None) for s in syms},
        lambda s: (frames[s], None),
        run_at="x",
    )
    # 同点の場合はコード順で決定的に並ぶ
    assert [r["コード"] for r in ranked] == ["AAA.T", "ZZZ.T"]


# --- 参考外への分類 ---

def test_fetch_failure_goes_to_excluded_with_reason():
    def batch(symbols):
        out = {}
        for s in symbols:
            if s == "BBB.T":
                out[s] = (pd.DataFrame(), "一括取得でデータが取れませんでした")
            else:
                out[s] = (FRAMES[s], None)
        return out

    def single(symbol):
        if symbol == "BBB.T":
            return pd.DataFrame(), "データが取得できませんでした"
        return FRAMES[symbol], None

    ranked, excluded, meta = scan.run_scan(UNIVERSE3, batch, single, run_at="x")
    assert [r["コード"] for r in ranked] == ["AAA.T", "CCC.T"]
    assert len(excluded) == 1
    assert excluded[0]["コード"] == "BBB.T"
    assert "取得失敗" in excluded[0]["警告・理由"]
    assert meta["参考外件数"] == 1


def test_batch_failure_falls_back_to_single_fetch():
    def batch(symbols):
        return {s: (pd.DataFrame(), "一括取得でデータが取れませんでした") for s in symbols}

    ranked, excluded, _ = scan.run_scan(UNIVERSE3, batch, single_ok, run_at="x")
    # 一括が全滅しても、個別取得で取り直せればランキングに入る
    assert len(ranked) == 3 and excluded == []


def test_stale_data_goes_to_excluded():
    old = make_frame(
        [100.0 + 0.2 * k for k in range(300)],
        end=pd.Timestamp.now(tz="Asia/Tokyo").date() - pd.Timedelta(days=15),
    )
    universe = [("OLD.T", "古い社", "テスト")]
    ranked, excluded, _ = scan.run_scan(
        universe,
        lambda syms: {"OLD.T": (old, None)},
        lambda s: (old, None),
        run_at="x",
    )
    assert ranked == []
    assert len(excluded) == 1
    assert "古い可能性" in excluded[0]["警告・理由"]


def test_scan_holds_invalid_ohlc_before_ranking(monkeypatch):
    frame = rising()
    frame.iloc[-1, frame.columns.get_loc("High")] = (
        frame.iloc[-1]["Close"] * 0.5
    )
    universe = [("BADPRICE.T", "価格異常社", "テスト")]
    ranked, excluded, _ = scan.run_scan(
        universe,
        lambda syms: {"BADPRICE.T": (frame, None)},
        lambda symbol: (frame, None),
        run_at="quality-warning-test",
    )
    assert ranked == []
    assert len(excluded) == 1
    assert "OHLC価格異常" in excluded[0]["警告・理由"]


def test_scan_diagnostic_failure_fails_closed(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("diagnostic unavailable")

    monkeypatch.setattr(scan.data_quality, "diagnose_history", fail)
    frame = rising()
    universe = [("DIAGFAIL.T", "診断失敗社", "テスト")]
    ranked, excluded, _ = scan.run_scan(
        universe,
        lambda syms: {"DIAGFAIL.T": (frame, None)},
        lambda symbol: (frame, None),
        run_at="quality-diagnostic-failure-test",
    )
    assert ranked == []
    assert len(excluded) == 1
    assert "データ品質診断不能" in excluded[0]["警告・理由"]


def test_scan_shape_retains_raw_for_quality_gate_and_drops_volume_only_row():
    frame = rising()
    frame.iloc[-1, frame.columns.get_loc("Volume")] = float("nan")
    shaped = scan._shape(frame, symbol="7203.T")
    assert len(shaped) == len(frame) - 1
    assert "_quality_report" in shaped.attrs
    assert shaped.attrs["_quality_report"].rows_with_missing == 1
    warnings, blocks = scan._data_quality_assessment(shaped, "7203.T")
    assert blocks == []
    assert any("欠損行の除外 1行" in item for item in warnings)


def test_scan_shape_blocks_price_missing_row_but_not_volume_only_row():
    frame = rising()
    frame.iloc[-1, frame.columns.get_loc("Close")] = float("nan")
    shaped = scan._shape(frame, symbol="7203.T")
    warnings, blocks = scan._data_quality_assessment(shaped, "7203.T")
    assert any("最新行の価格データ欠損：Close" in item for item in blocks)


def test_app_batch_and_single_fetch_callbacks_capture_sanitized_frames():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "clean_frame_for_forward"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "scan"
    ]
    assert len(calls) == 2


def test_scan_quality_gate_synthetic_60_ticker_universe_exclusion_reasons():
    universe = []
    frames = {}

    def add_group(prefix, count, defect=None):
        for i in range(count):
            symbol = f"{prefix}{i:02d}.T"
            frame = rising()
            if defect == "volume_missing":
                frame.iloc[10, frame.columns.get_loc("Volume")] = float("nan")
            elif defect == "latest_price_missing":
                frame.iloc[-1, frame.columns.get_loc("Close")] = float("nan")
            elif defect == "ohlc_contradiction":
                frame.iloc[10, frame.columns.get_loc("High")] = (
                    frame.iloc[10]["Close"] * 0.5
                )
            elif defect == "malformed_price":
                frame["Close"] = frame["Close"].astype(object)
                frame.iloc[10, frame.columns.get_loc("Close")] = "not-a-price"
            frames[symbol] = frame
            universe.append((symbol, symbol, "合成テスト"))

    add_group("N", 50)
    add_group("V", 4, "volume_missing")
    add_group("L", 2, "latest_price_missing")
    add_group("O", 2, "ohlc_contradiction")
    add_group("M", 2, "malformed_price")
    assert len(universe) == 60

    def batch_fetcher(symbols):
        return {
            symbol: (scan._shape(frames[symbol], symbol=symbol), None)
            for symbol in symbols
        }

    ranked, excluded, _ = scan.run_scan(
        universe,
        batch_fetcher,
        lambda symbol: (scan._shape(frames[symbol], symbol=symbol), None),
        run_at="synthetic-60-quality-gate",
    )

    assert len(ranked) == 54
    assert len(excluded) == 6
    assert sum("最新行の価格データ欠損" in row["警告・理由"] for row in excluded) == 2
    assert sum("OHLC価格異常" in row["警告・理由"] for row in excluded) == 4
    ranked_symbols = {row["コード"] for row in ranked}
    assert all(f"V{i:02d}.T" in ranked_symbols for i in range(4))
    assert all(
        "欠損行の除外 1行" in next(row for row in ranked if row["コード"] == f"V{i:02d}.T")["警告・理由"]
        for i in range(4)
    )


def test_clean_frame_for_forward_strips_raw_diagnostic_attrs():
    frame = rising()
    frame.attrs["_quality_raw"] = frame.copy()
    frame.attrs["_quality_report"] = object()
    frame.attrs["_quality_diagnostic_error"] = "RuntimeError"
    frame.attrs["rows_dropped_by_dropna"] = 1
    frame.attrs["raw_row_count"] = len(frame)

    captured = scan.clean_frame_for_forward(frame)

    assert captured.equals(frame)
    assert captured.attrs == {}
    # Sanitization is non-mutating; scan can still consume the original metadata.
    assert "_quality_raw" in frame.attrs


def test_scan_quality_gap_warning_ignores_official_jp_holiday():
    dates = pd.to_datetime(["2026-11-02", "2026-11-04"])
    frame = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [101.0, 102.0],
            "Low": [99.0, 100.0],
            "Close": [100.5, 101.5],
            "Volume": [1000, 1100],
        },
        index=dates,
    )
    warnings = scan._data_quality_warnings(frame, "7203.T")
    assert not any("日付間隔の要確認" in warning for warning in warnings)
    assert not any("判定不能" in warning for warning in warnings)


def test_scan_quality_gap_without_calendar_is_reported_as_unverified():
    dates = pd.to_datetime(["2024-01-05", "2024-01-09"])
    frame = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [101.0, 102.0],
            "Low": [99.0, 100.0],
            "Close": [100.5, 101.5],
            "Volume": [1000, 1100],
        },
        index=dates,
    )
    warnings = scan._data_quality_warnings(frame, "7203.T")
    assert any("判定不能" in warning for warning in warnings)
    assert not any("日付間隔の要確認" in warning for warning in warnings)


def test_scan_quality_diagnostic_failure_is_visible_and_fails_closed(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("diagnostic unavailable")

    monkeypatch.setattr(scan.data_quality, "diagnose_history", fail)
    frame = rising()
    universe = [("DIAGFAIL.T", "診断失敗社", "テスト")]
    ranked, excluded, _ = scan.run_scan(
        universe,
        lambda syms: {"DIAGFAIL.T": (frame, None)},
        lambda symbol: (frame, None),
        run_at="quality-diagnostic-failure-test",
    )
    assert ranked == []
    assert len(excluded) == 1
    assert "データ品質診断不能" in excluded[0]["警告・理由"]


def test_future_dated_data_is_excluded_from_ranking():
    future = (
        pd.Timestamp.now(tz="Asia/Tokyo").normalize().tz_localize(None)
        + pd.Timedelta(days=5)
    )
    frame = rising()
    frame.index = frame.index[:-1].append(pd.DatetimeIndex([future]))
    universe = [("FUTURE.T", "未来日付社", "テスト")]
    ranked, excluded, _ = scan.run_scan(
        universe,
        lambda syms: {"FUTURE.T": (frame, None)},
        lambda symbol: (frame, None),
        run_at="future-index-test",
    )
    assert ranked == []
    assert len(excluded) == 1
    assert "未来日付" in excluded[0]["警告・理由"]


def test_insufficient_bars_goes_to_excluded():
    short = make_frame([100.0 + 0.2 * k for k in range(50)])  # 中期の最低80本に不足
    universe = [("NEW.T", "短い社", "テスト")]
    ranked, excluded, _ = scan.run_scan(
        universe,
        lambda syms: {"NEW.T": (short, None)},
        lambda s: (short, None),
        run_at="x",
    )
    assert ranked == []
    assert "中期評価ができません" in excluded[0]["警告・理由"]


# --- CSV ---

def test_scan_csv_contains_both_sections():
    def batch(symbols):
        out = {s: (FRAMES[s], None) for s in symbols}
        return out

    ranked, excluded, _ = scan.run_scan(UNIVERSE3, batch, single_ok, run_at="x")
    text = scan.scan_to_csv(ranked, excluded).decode("utf-8-sig")
    assert "ランキング" in text
    assert "AAA.T" in text and "上昇社" in text
    first_line = text.splitlines()[0]
    for col in ["コード", "会社名", "中期スコア", "選定理由", "データ最終日", "取得日時"]:
        assert col in first_line


# --- 欠損行の除外数（表示専用） ---


def test_price_first_batch_multi_ticker_falls_back_to_individual_fetch(monkeypatch):
    target = rising()
    other = falling()
    price_first = pd.concat(
        {"7203.T": target, "6758.T": other}, axis=1
    ).swaplevel(0, 1, axis=1)
    monkeypatch.setattr(scan.yf, "download", lambda *args, **kwargs: price_first)

    universe = [
        ("7203.T", "対象社", "テスト"),
        ("6758.T", "別銘柄社", "テスト"),
    ]
    individual_calls = []

    def single_fetch(symbol):
        individual_calls.append(symbol)
        return (target if symbol == "7203.T" else other), None

    ranked, excluded, _ = scan.run_scan(
        universe,
        lambda symbols: scan.fetch_batch(symbols, period="1y"),
        single_fetch,
        run_at="price-first-batch-fallback-test",
    )

    assert excluded == []
    assert {row["コード"] for row in ranked} == {"7203.T", "6758.T"}
    assert set(individual_calls) == {"7203.T", "6758.T"}
    assert len(individual_calls) == 2

def test_shape_multiindex_ticker_first_selects_requested_symbol():
    target = rising()
    other = falling()
    source = pd.concat({"7203.T": target, "6758.T": other}, axis=1)

    shaped = scan._shape(source, symbol="7203.T")

    pd.testing.assert_frame_equal(shaped, target)


def test_shape_multiindex_price_first_selects_requested_symbol():
    target = rising()
    other = falling()
    source = pd.concat({"7203.T": target, "6758.T": other}, axis=1).swaplevel(0, 1, axis=1)

    shaped = scan._shape(source, symbol="7203.T")

    pd.testing.assert_frame_equal(shaped, target)


def test_shape_multiindex_unknown_symbol_is_rejected():
    target = rising()
    other = falling()
    source = pd.concat({"7203.T": target, "6758.T": other}, axis=1)

    shaped = scan._shape(source, symbol="9999.T")

    assert shaped.empty


def test_scan_shape_preserves_missing_row_drop_count():
    frame = rising()
    frame.iloc[10, frame.columns.get_loc("Close")] = float("nan")
    snapshot = frame.copy(deep=True)

    shaped = scan._shape(frame)

    assert len(shaped) == len(frame) - 1
    assert shaped.attrs["rows_dropped_by_dropna"] == 1
    assert shaped.attrs["raw_row_count"] == len(frame)
    pd.testing.assert_frame_equal(frame, snapshot)


def test_scan_displays_historical_missing_row_warning_without_blocking_unverified_gap():
    frame = rising()
    frame.iloc[10, frame.columns.get_loc("Close")] = float("nan")
    shaped = scan._shape(frame, symbol="7203.T")
    universe = [("7203.T", "欠損確認社", "テスト")]

    ranked, excluded, _ = scan.run_scan(
        universe,
        lambda symbols: {"7203.T": (shaped, None)},
        lambda symbol: (shaped, None),
        run_at="missing-row-warning-test",
    )
    assert len(ranked) == 1 and excluded == []
    assert "欠損行の除外 1行" in ranked[0]["警告・理由"]
    assert "品質ゲートで分析保留" not in ranked[0]["警告・理由"]


def test_scan_displays_missing_row_drop_count_on_single_fetch_fallback():
    frame = rising()
    frame.iloc[10, frame.columns.get_loc("Close")] = float("nan")
    _, clean, error = scan.analysis._split_raw_clean(frame)
    assert error is None
    universe = [("7203.T", "個別取得確認社", "テスト")]

    ranked, excluded, _ = scan.run_scan(
        universe,
        lambda symbols: {"7203.T": (pd.DataFrame(), "batch unavailable")},
        lambda symbol: (clean, None),
        run_at="missing-row-fallback-test",
    )

    assert len(ranked) == 1 and excluded == []
    assert "欠損行の除外 1行" in ranked[0]["警告・理由"]



def test_scan_reports_all_rows_dropped_when_fallback_has_no_data():
    frame = rising()
    frame["Close"] = float("nan")
    shaped = scan._shape(frame)
    assert shaped.empty
    assert shaped.attrs["rows_dropped_by_dropna"] == len(frame)
    universe = [("ALLMISSING.T", "全行欠損社", "テスト")]

    ranked, excluded, _ = scan.run_scan(
        universe,
        lambda symbols: {"ALLMISSING.T": (shaped, "batch has no clean rows")},
        lambda symbol: (pd.DataFrame(), "取得したデータが空でした（欠損のみ）。"),
        run_at="all-missing-row-test",
    )

    assert ranked == []
    assert len(excluded) == 1
    assert "欠損行除外：300行" in excluded[0]["警告・理由"]



def test_successful_single_fallback_does_not_inherit_batch_dropna_count():
    batch_empty = pd.DataFrame()
    batch_empty.attrs["rows_dropped_by_dropna"] = 7
    fallback = rising()  # No missing rows and no batch-derived metadata.
    universe = [("FALLBACKOK.T", "個別取得成功社", "テスト")]

    ranked, excluded, _ = scan.run_scan(
        universe,
        lambda symbols: {
            "FALLBACKOK.T": (batch_empty, "batch unavailable")
        },
        lambda symbol: (fallback, None),
        run_at="successful-fallback-no-leak-test",
    )

    assert len(ranked) == 1 and excluded == []
    assert "欠損行の除外" not in ranked[0]["警告・理由"]
