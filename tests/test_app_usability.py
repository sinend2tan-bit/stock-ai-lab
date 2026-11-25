"""使いやすさ改善（2026-11-25）の AppTest。

対象：タブ順、銘柄プリセット、スキャン結果CSVの復元、スキャン表の
既定列の絞り込みと全列切り替え、スキャン行選択から個別分析への連動、
仮想売買メモCSVの復元、品質診断の正常時の1行表示。

取得は analysis の関数をモックして実通信をしない（復元・連動の確認が目的で、
取得の成否は問わない）。
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis
import analysis_quality
import scan
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")
SCAN_ID = "2026-10-19 15:30:00"

SCAN_MAIN_COLUMNS = ["順位", "コード", "会社名", "中期スコア", "中期ラベル", "終値", "データ最終日"]


def scan_row(**overrides):
    row = {
        "区分": "ランキング", "順位": 1, "コード": "1000.T", "会社名": "テスト株",
        "業種": "テスト", "中期スコア": 4, "中期ラベル": "強い",
        "短期スコア": 3, "短期ラベル": "強い", "長期スコア": 1, "長期ラベル": "やや強い",
        "終値": 110.0, "選定理由": "中期条件", "警告・理由": "",
        "データ最終日": "2026-10-19", "本数": 300, "取得日時": SCAN_ID,
    }
    row.update(overrides)
    return row


def _mock_empty_fetch(monkeypatch):
    def empty_response(_symbol, _period="2y"):
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            analysis.NO_DATA_ERROR_PREFIX + "テスト用の空レスポンス",
        )

    monkeypatch.setattr(analysis, "fetch_history_bundle", empty_response)
    monkeypatch.setattr(
        analysis, "resolve_company_name", lambda _symbol, *_: "テスト銘柄"
    )
    st.cache_data.clear()


def _seed_scan(at):
    at.session_state["scan_ranked"] = [scan_row()]
    at.session_state["scan_excluded"] = []
    at.session_state["scan_meta"] = {
        "実行日時": SCAN_ID, "対象銘柄数": 1, "ランキング件数": 1,
        "参考外件数": 0, "取得期間": "2y",
    }
    at.session_state["scan_frames"] = {}


def _scan_dataframe(at):
    for element in at.dataframe:
        value = element.value
        if hasattr(value, "columns") and "順位" in value.columns:
            return value
    raise AssertionError("scan dataframe was not rendered")


def test_scan_tab_is_first():
    at = AppTest.from_file(APP_PATH).run(timeout=180)
    assert not at.exception
    assert [t.label for t in at.tabs][0] == "スキャン"


def test_watchlist_preset_button_sets_symbols(monkeypatch):
    _mock_empty_fetch(monkeypatch)
    at = AppTest.from_file(APP_PATH).run(timeout=180)
    assert not at.exception
    button = [b for b in at.sidebar.button if b.label == "監視銘柄"]
    assert len(button) == 1
    button[0].click().run(timeout=180)
    assert not at.exception
    assert at.session_state["symbols_text_日本株"] == "5401.T, 8766.T, 4689.T"


def test_scan_csv_restore_renders_previous_results(monkeypatch):
    _mock_empty_fetch(monkeypatch)
    csv_bytes = scan.scan_to_csv([scan_row()], [])
    at = AppTest.from_file(APP_PATH).run(timeout=180)
    uploader = [u for u in at.file_uploader if u.key == "scan_csv_upload"]
    assert len(uploader) == 1
    uploader[0].set_value([("stock_ai_scan.csv", csv_bytes, "text/csv")])
    at.run(timeout=180)
    assert not at.exception
    assert at.session_state["scan_meta"]["取得期間"] == "保存CSVから復元"
    assert at.session_state["scan_ranked"][0]["コード"] == "1000.T"
    frame = _scan_dataframe(at)
    assert list(frame.columns) == SCAN_MAIN_COLUMNS
    assert frame.iloc[0]["コード"] == "1000.T"


def test_scan_csv_restore_rejects_wrong_format(monkeypatch):
    _mock_empty_fetch(monkeypatch)
    at = AppTest.from_file(APP_PATH).run(timeout=180)
    uploader = [u for u in at.file_uploader if u.key == "scan_csv_upload"][0]
    uploader.set_value([("bad.csv", b"a,b,c\n1,2,3\n", "text/csv")])
    at.run(timeout=180)
    assert not at.exception
    assert at.session_state["scan_meta"] is None
    errors = [str(e.value) for e in at.error]
    assert any("スキャン結果の形式ではありません" in e for e in errors)


def test_scan_table_show_all_columns_toggle(monkeypatch):
    _mock_empty_fetch(monkeypatch)
    at = AppTest.from_file(APP_PATH)
    _seed_scan(at)
    at.run(timeout=180)
    assert not at.exception
    assert list(_scan_dataframe(at).columns) == SCAN_MAIN_COLUMNS
    toggle = [c for c in at.checkbox if c.key == "scan_show_all"]
    assert len(toggle) == 1
    toggle[0].set_value(True).run(timeout=180)
    assert not at.exception
    assert list(_scan_dataframe(at).columns) == scan.SCAN_COLUMNS


def test_scan_row_selection_links_to_detail_symbol(monkeypatch):
    _mock_empty_fetch(monkeypatch)
    at = AppTest.from_file(APP_PATH)
    _seed_scan(at)
    at.session_state["scan_ranked_df"] = {"selection": {"rows": [0], "columns": []}}
    at.run(timeout=180)
    assert not at.exception
    assert at.session_state["selected_symbol_日本株"] == "1000.T"
    assert "1000.T" in at.session_state["symbols_text_日本株"]


def test_journal_csv_restore_adds_rows_once(monkeypatch):
    _mock_empty_fetch(monkeypatch)
    columns = [
        "日時", "銘柄", "スタイル", "方向", "想定エントリー", "想定損切り",
        "想定利確", "終了価格", "メモ", "結果",
    ]
    journal_df = pd.DataFrame(
        [["2026-11-20 10:00:00", "5401.T", "中期", "買い", 670.0, 660.0,
          700.0, 0.0, "テストメモ", ""]],
        columns=columns,
    )
    csv_bytes = journal_df.to_csv(index=False).encode("utf-8-sig")
    at = AppTest.from_file(APP_PATH).run(timeout=180)
    uploader = [u for u in at.file_uploader if u.key == "journal_csv_upload"]
    assert len(uploader) == 1
    uploader[0].set_value([("stock_paper_journal.csv", csv_bytes, "text/csv")])
    at.run(timeout=180)
    assert not at.exception
    journal = at.session_state["journal"]
    assert len(journal) == 1
    assert journal.iloc[0]["銘柄"] == "5401.T"
    assert journal.iloc[0]["想定エントリー"] == 670.0
    # 同じファイルが残ったまま再描画しても二重に追加しない
    at.run(timeout=180)
    assert len(at.session_state["journal"]) == 1


def test_quality_clean_case_shows_single_success_line(monkeypatch):
    n = 120
    closes = [100.0 + i for i in range(n)]
    idx = pd.bdate_range(end=pd.Timestamp.now(tz="Asia/Tokyo").date(), periods=n)
    raw = pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": [1000.0] * n,
        },
        index=idx,
    )
    clean_report = {
        "row_count": n, "rows_dropped_by_dropna": 0,
        "anomaly_count": 0, "warning_count": 0, "review_count": 0,
        "unverified_gap_count": 0, "missing_columns": [], "nat_count": 0,
        "duplicate_dates": [], "out_of_order": [], "future_dates": [],
        "price_issues": {}, "volume_issues": {}, "missing_per_column": {},
        "consecutive_missing_runs": [], "last_row_missing": False,
        "review_gaps": [], "unverified_gaps": [],
    }
    report_obj = SimpleNamespace(to_dict=lambda: clean_report)
    monkeypatch.setattr(
        analysis_quality, "assess", lambda raw_, symbol: (report_obj, [], None)
    )
    monkeypatch.setattr(
        analysis, "fetch_history_bundle", lambda s, p="2y": (raw, raw.copy(), None)
    )
    monkeypatch.setattr(analysis, "fetch_history", lambda s, p="2y": (raw, None))
    st.cache_data.clear()
    at = AppTest.from_file(APP_PATH).run(timeout=180)
    assert not at.exception
    successes = [str(s.value) for s in at.success]
    assert any("異常候補・警告候補・要確認はありませんでした" in s for s in successes)
    # 正常時は品質診断由来の警告・情報では注意を引かない
    # （画面上部の常時警告バナーは対象外）
    warning_texts = [str(w.value) for w in at.warning]
    assert not any("異常候補が" in t or "品質上の問題" in t for t in warning_texts)
    info_texts = [str(i.value) for i in at.info]
    assert not any("異常候補はありませんが" in t for t in info_texts)
