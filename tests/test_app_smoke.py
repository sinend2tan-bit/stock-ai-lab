"""app.py の AppTest スモーク。画面が例外なく描画されることを確認する。

ネットワークを使う取得は、失敗しても画面が壊れない設計のため、
取得の成否自体はここでは問わない。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import forward
import scoring_v2
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")
SCAN_ID = "2026-10-19 15:30:00"
META = {"実行日時": SCAN_ID}


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


def test_app_loads_without_exception():
    at = AppTest.from_file(APP_PATH).run(timeout=180)
    assert not at.exception
    subheaders = [s.value for s in at.subheader]
    assert "フォワード登録と事後評価" in subheaders


def test_app_renders_v1v2_comparison_with_seeded_records():
    v1 = forward.build_records([scan_row()], [], META, {})
    v2 = forward.build_records(
        [scan_row(中期スコア=-3)], [], META, {},
        scoring_version=2, direction_fn=scoring_v2.direction_from_score_v2,
    )
    results = [{
        "スキャンID": SCAN_ID, "コード": "1000.T", "評価営業日数": 20,
        "スコア仕様バージョン": 1, "状態": "評価済み", "リターンpct": 5.0,
        "TOPIX超過リターンpct": 4.0, "日経超過リターンpct": 4.5,
        "最大上昇率pct": 7.0, "最大下落率pct": -2.0,
    }]
    at = AppTest.from_file(APP_PATH)
    at.session_state["fwd_records"] = v1 + v2
    at.session_state["fwd_results"] = results
    at.run(timeout=180)
    assert not at.exception
    expander_labels = [e.label for e in at.expander]
    assert "V1・V2比較（新仕様の並行評価）" in expander_labels


def test_app_dual_registration_from_seeded_scan():
    import pandas as pd

    dates = list(pd.bdate_range(start="2025-10-01", periods=300, tz="Asia/Tokyo"))
    closes = [100.0 + 0.1 * i for i in range(300)]
    frame = pd.DataFrame(
        {"Open": closes, "High": [c + 1.0 for c in closes],
         "Low": [c - 1.0 for c in closes], "Close": closes, "Volume": [1000] * 300},
        index=pd.DatetimeIndex(dates),
    )
    base_date = str(dates[-1].date())
    at = AppTest.from_file(APP_PATH)
    at.session_state["scan_ranked"] = [scan_row(データ最終日=base_date)]
    at.session_state["scan_excluded"] = []
    at.session_state["scan_meta"] = {
        "実行日時": SCAN_ID, "対象銘柄数": 1, "ランキング件数": 1,
        "参考外件数": 0, "取得期間": "2y",
    }
    at.session_state["scan_frames"] = {"1000.T": frame}
    at.run(timeout=180)
    assert not at.exception
    register = [
        b for b in at.button if "フォワード登録する" in (b.label or "")
    ]
    assert len(register) == 1
    register[0].click().run(timeout=180)
    assert not at.exception
    records = at.session_state["fwd_records"]
    versions = sorted({int(r["スコア仕様バージョン"]) for r in records})
    assert versions == [1, 2]
    assert "V1 1 件・V2 1 件" in at.session_state["fwd_register_note"]
