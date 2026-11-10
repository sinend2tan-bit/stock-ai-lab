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


def test_detail_tab_shows_diagnosis_and_scores(monkeypatch):
    raw = _frame()
    raw.iloc[10, raw.columns.get_loc("Volume")] = 0.0  # 警告候補
    raw.iloc[20, raw.columns.get_loc("High")] = 1.0  # 異常候補
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
    assert any("異常候補" in t and "件あります" in t for t in texts)
    assert any("売買シグナルではありません" in t for t in texts)
    # 診断があってもスコア表示は従来どおり出る（診断はスコアを変えない）
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


def test_detail_tab_diagnosis_failure_shows_unavailable(monkeypatch):
    # 診断自体が失敗した場合は正常扱いにせず「診断不能」と表示し、
    # 分析・スコア表示は従来どおり行う（診断不能表示の境界）。
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
    assert "条件スコア" in metric_labels


def test_detail_tab_diagnoses_raw_not_clean(monkeypatch):
    # 診断対象は clean ではなく raw：raw の欠損行が「dropna除外候補」として
    # 件数に現れること、分析は clean で行われスコアが出ることを確認する。
    raw = _frame()
    raw.iloc[15, raw.columns.get_loc("Close")] = np.nan
    clean = raw.dropna().copy()
    assert len(raw) == 120 and len(clean) == 119
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


def test_detail_tab_fetches_bundle_once(monkeypatch):
    # 1銘柄・1期間あたりの外部データ取得は bundle の1回だけ
    # （個別分析タブが取得を重複させないことの境界）。
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
    assert len(calls) == 1
