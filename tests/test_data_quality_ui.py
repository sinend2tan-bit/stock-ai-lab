"""個別分析タブの品質診断表示（AppTest）。

取得は analysis の関数をモックして実通信をしない。診断表示が出ることと、
診断があってもスコア表示が従来どおり出ることを確認する。
画面の目視確認ではなく、AppTest 上の要素テキストによる確認である。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis
import data_quality
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


def _frame(n=120):
    closes = [100.0 + i for i in range(n)]
    idx = pd.bdate_range(end=pd.Timestamp.now(tz="Asia/Tokyo").date(), periods=n)
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": [1000.0] * n,
        },
        index=idx,
    )


def _all_text(at):
    texts = []
    for attr in ("markdown", "caption", "write", "warning", "error", "info", "success"):
        for element in getattr(at, attr, []):
            texts.append(str(element.value))
    return texts


def test_watchlist_overview_uses_raw_bundle_and_shared_quality_gate():
    source = Path(APP_PATH).read_text(encoding="utf-8")
    assert "raw, clean, err = get_history_bundle(sym, period)" in source
    assert "quality_report, quality_blocks, quality_error = assess_data_quality(raw, sym)" in source


def test_detail_tab_volume_warning_does_not_block_scores(monkeypatch):
    raw = _frame()
    raw.iloc[10, raw.columns.get_loc("Volume")] = 0.0  # スコア計算対象外の警告候補
    clean = raw.copy()
    monkeypatch.setattr(
        analysis, "fetch_history_bundle", lambda s, p="2y": (raw, clean, None)
    )
    monkeypatch.setattr(analysis, "fetch_history", lambda s, p="2y": (clean, None))
    st.cache_data.clear()
    at = AppTest.from_file(APP_PATH).run(timeout=180)
    assert not at.exception
    texts = _all_text(at)
    assert any("データ品質診断（取得直後データ）" in t for t in texts)
    assert any("警告候補" in t for t in texts)
    assert any("売買シグナルではありません" in t for t in texts)
    # Volume-only warning does not block because current scoring does not consume Volume.
    metric_labels = [m.label for m in at.metric]
    assert "条件スコア" in metric_labels


def test_detail_tab_missing_columns_shows_diagnosis_and_no_analysis(monkeypatch):
    raw = _frame().drop(columns=["Volume"])
    err = "取得したデータに必要な列（始値・高値・安値・終値・出来高）がそろっていません。"
    monkeypatch.setattr(
        analysis, "fetch_history_bundle", lambda s, p="2y": (raw, pd.DataFrame(), err)
    )
    monkeypatch.setattr(analysis, "fetch_history", lambda s, p="2y": (pd.DataFrame(), err))
    st.cache_data.clear()
    at = AppTest.from_file(APP_PATH).run(timeout=180)
    assert not at.exception
    texts = _all_text(at)
    assert any("必要な列" in t for t in texts)
    assert any("分析は行いません" in t for t in texts)
    # 分析不可でも、取得直後データの診断（必須列欠落）は表示される
    assert any("データ品質診断（取得直後データ）" in t for t in texts)
    assert any("必須列の欠落" in t or "異常候補" in t for t in texts)
    metric_labels = [m.label for m in at.metric]
    assert "条件スコア" not in metric_labels


def test_detail_tab_diagnosis_failure_holds_scores(monkeypatch):
    # 診断不能は正常扱いせず、個別スコア判定を保留する。
    raw = _frame()
    clean = raw.copy()

    def _raise(*args, **kwargs):
        raise ValueError("診断テスト用の失敗")

    monkeypatch.setattr(data_quality, "diagnose_history", _raise)
    monkeypatch.setattr(
        analysis, "fetch_history_bundle", lambda s, p="2y": (raw, clean, None)
    )
    monkeypatch.setattr(analysis, "fetch_history", lambda s, p="2y": (clean, None))
    st.cache_data.clear()
    at = AppTest.from_file(APP_PATH).run(timeout=180)
    assert not at.exception
    texts = _all_text(at)
    assert any("データ品質診断（取得直後データ）" in t for t in texts)
    assert any("診断不能" in t for t in texts)
    metric_labels = [m.label for m in at.metric]
    assert "条件スコア" not in metric_labels
    assert any("診断不能のため、個別スコア判定を保留します" in t for t in texts)


def test_detail_tab_invalid_latest_price_holds_scores(monkeypatch):
    raw = _frame()
    raw.iloc[-1, raw.columns.get_loc("Close")] = np.nan
    clean = raw.dropna().copy()
    monkeypatch.setattr(
        analysis, "fetch_history_bundle", lambda s, p="2y": (raw, clean, None)
    )
    monkeypatch.setattr(analysis, "fetch_history", lambda s, p="2y": (clean, None))
    st.cache_data.clear()
    at = AppTest.from_file(APP_PATH).run(timeout=180)
    assert not at.exception
    texts = _all_text(at)
    assert any("最新行の価格データ欠損：Close" in t for t in texts)
    metric_labels = [m.label for m in at.metric]
    assert "条件スコア" not in metric_labels


def test_detail_tab_invalid_ohlc_holds_scores(monkeypatch):
    raw = _frame()
    raw.iloc[20, raw.columns.get_loc("High")] = 1.0
    clean = raw.copy()
    monkeypatch.setattr(
        analysis, "fetch_history_bundle", lambda s, p="2y": (raw, clean, None)
    )
    monkeypatch.setattr(analysis, "fetch_history", lambda s, p="2y": (clean, None))
    st.cache_data.clear()
    at = AppTest.from_file(APP_PATH).run(timeout=180)
    assert not at.exception
    texts = _all_text(at)
    assert any("OHLC価格異常" in t for t in texts)
    metric_labels = [m.label for m in at.metric]
    assert "条件スコア" not in metric_labels


def test_detail_tab_malformed_price_holds_scores(monkeypatch):
    raw = _frame()
    raw["Close"] = raw["Close"].astype(object)
    raw.iloc[20, raw.columns.get_loc("Close")] = "not-a-price"
    clean = raw.copy()
    monkeypatch.setattr(
        analysis, "fetch_history_bundle", lambda s, p="2y": (raw, clean, None)
    )
    monkeypatch.setattr(analysis, "fetch_history", lambda s, p="2y": (clean, None))
    st.cache_data.clear()
    at = AppTest.from_file(APP_PATH).run(timeout=180)
    assert not at.exception
    texts = _all_text(at)
    assert any("OHLC価格異常" in t for t in texts)
    metric_labels = [m.label for m in at.metric]
    assert "条件スコア" not in metric_labels


def test_detail_tab_diagnoses_raw_not_clean(monkeypatch):
    # 診断対象は clean ではなく raw：raw の欠損行が「dropna除外候補」として
    # 件数に現れること、分析は clean で行われスコアが出ることを確認する。
    # Use a 2025 historical gap, where the built-in holiday calendar is
    # unverified; the isolated missing row is warned, not promoted to a confirmed gap.
    raw = _frame(n=300)
    raw.iloc[15, raw.columns.get_loc("Close")] = np.nan
    clean = raw.dropna().copy()
    assert len(raw) == 300 and len(clean) == 299
    monkeypatch.setattr(
        analysis, "fetch_history_bundle", lambda s, p="2y": (raw, clean, None)
    )
    monkeypatch.setattr(analysis, "fetch_history", lambda s, p="2y": (clean, None))
    st.cache_data.clear()
    at = AppTest.from_file(APP_PATH).run(timeout=180)
    assert not at.exception
    texts = _all_text(at)
    assert any("欠損行（dropna除外候補）：1行" in t for t in texts)
    metric_labels = [m.label for m in at.metric]
    assert "条件スコア" in metric_labels


def test_overview_and_detail_share_bundle_cache(monkeypatch):
    # Watchlist overview and individual detail share the same cached raw/clean bundle;
    # the selected ticker must not be downloaded a second time in the same run.
    raw = _frame()
    clean = raw.copy()
    calls = []

    def _bundle(symbol, period="2y"):
        calls.append((symbol, period))
        return raw, clean, None

    monkeypatch.setattr(analysis, "fetch_history_bundle", _bundle)
    monkeypatch.setattr(analysis, "fetch_history", lambda s, p="2y": (clean, None))
    st.cache_data.clear()
    at = AppTest.from_file(APP_PATH).run(timeout=180)
    assert not at.exception
    assert len(calls) == 4
    assert calls.count(("7203.T", "2y")) == 1
    assert {symbol for symbol, _ in calls} == {"7203.T", "6758.T", "8306.T", "9984.T"}
