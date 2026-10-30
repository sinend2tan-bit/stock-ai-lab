"""forward.py の V2 並行比較まわりのテスト。ネットワークは使わない。

確認すること:
- 旧形式CSV（追加列なし）を既定値で補って読み込めること。
- V1 と V2 の記録が版で区別され、重複・競合が版ごとに処理されること。
- 記録形式の版（記録仕様版数）が変わっても既存データが競合にならないこと。
- 評価結果に元の記録の版が引き継がれること。
- V2 用スキャン行の構築（順位付け・参考外・失敗の切り分け）。
- 再取得データの基準日検査（切り取り・日付・価格差）。
- 16セルの全組み合わせ集計と完全性チェック、見送りの分離、片側登録、
  共通母集団、採用ゲート（コホート信頼区間・銘柄依存）。
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import forward
import scoring_v2

TZ = "Asia/Tokyo"
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


def rec(scan_id, code, version, mid="中立", short="中立", long="中立", warning=""):
    return {
        "スキャンID": scan_id, "コード": code, "スコア仕様バージョン": version,
        "短期予測方向": short, "中期予測方向": mid, "長期予測方向": long,
        "警告・理由": warning,
    }


def res(scan_id, code, n, version, ret, exc=None, max_down=-3.0):
    return {
        "スキャンID": scan_id, "コード": code, "評価営業日数": n,
        "スコア仕様バージョン": version, "状態": "評価済み",
        "リターンpct": ret, "TOPIX超過リターンpct": exc, "最大下落率pct": max_down,
    }


def pair(scan_id, code, d1, d2, horizon="中期"):
    kw = {"中期": "mid", "短期": "short", "長期": "long"}[horizon]
    return [rec(scan_id, code, 1, **{kw: d1}), rec(scan_id, code, 2, **{kw: d2})]


# ---------------- CSV 互換 ----------------

def test_legacy_records_csv_loads_with_defaults():
    records = forward.build_records([scan_row()], [], META, {})
    old_columns = [c for c in forward.RECORD_COLUMNS if c != "過熱フラグ"]
    legacy = pd.DataFrame(records, columns=old_columns).to_csv(index=False).encode("utf-8-sig")
    loaded = forward.records_from_csv(legacy)
    assert len(loaded) == 1
    assert loaded[0]["過熱フラグ"] == ""
    assert loaded[0]["スコア仕様バージョン"] == 1


def test_legacy_results_csv_loads_with_version_1():
    row = res(SCAN_ID, "1000.T", 20, 1, 5.0, exc=4.0)
    row["記録仕様版数"] = 1
    old_columns = [c for c in forward.RESULT_COLUMNS if c != "スコア仕様バージョン"]
    legacy = pd.DataFrame([row], columns=old_columns).to_csv(index=False).encode("utf-8-sig")
    loaded = forward.results_from_csv(legacy)
    assert len(loaded) == 1
    assert loaded[0]["スコア仕様バージョン"] == 1


def test_v2_record_csv_roundtrip():
    records = forward.build_records(
        [scan_row(中期スコア=-3)], [], META, {},
        scoring_version=2, direction_fn=scoring_v2.direction_from_score_v2,
        overheat_flags={"1000.T": True},
    )
    assert records[0]["中期予測方向"] == "下落注意"
    assert records[0]["過熱フラグ"] == 1
    loaded = forward.records_from_csv(forward.records_to_csv(records))
    assert loaded[0]["スコア仕様バージョン"] == 2
    assert str(loaded[0]["過熱フラグ"]) == "1"
    assert loaded[0]["中期予測方向"] == "下落注意"


def test_merge_keeps_versions_separate():
    v1 = forward.build_records([scan_row()], [], META, {})
    v2 = forward.build_records(
        [scan_row(中期スコア=-3)], [], META, {},
        scoring_version=2, direction_fn=scoring_v2.direction_from_score_v2,
    )
    merged, conflicts = forward.merge_records([], v1 + v2)
    assert len(merged) == 2 and conflicts == []
    merged2, conflicts2 = forward.merge_records(merged, v1 + v2)
    assert len(merged2) == 2 and conflicts2 == []
    changed_v2 = forward.build_records(
        [scan_row(中期スコア=-3, 終値=111.0)], [], META, {},
        scoring_version=2, direction_fn=scoring_v2.direction_from_score_v2,
    )
    merged3, conflicts3 = forward.merge_records(merged2, changed_v2)
    assert len(merged3) == 2 and len(conflicts3) == 1
    assert conflicts3[0]["スコア仕様バージョン"] == 2


def test_merge_format_version_bump_does_not_conflict():
    records = forward.build_records([scan_row()], [], META, {})
    legacy = [dict(r, **{"記録仕様版数": 1, "過熱フラグ": ""}) for r in records]
    merged, conflicts = forward.merge_records(legacy, records)
    assert len(merged) == 1 and conflicts == []


# ---------------- 評価への版の引き継ぎ ----------------

def biz_frame(start="2026-10-05", n=40, base=100.0, step=0.5):
    dates = list(pd.bdate_range(start=start, periods=n, tz=TZ))
    closes = [base + step * i for i in range(n)]
    return pd.DataFrame(
        {"Open": closes, "High": [c + 1.0 for c in closes],
         "Low": [c - 1.0 for c in closes], "Close": closes, "Volume": [1000] * n},
        index=pd.DatetimeIndex(dates),
    )


def test_evaluate_stamps_record_version():
    frame = biz_frame()
    base_date = str(frame.index[0].date())
    v1 = forward.build_records([scan_row(データ最終日=base_date)], [], META, {})
    v2 = forward.build_records(
        [scan_row(データ最終日=base_date)], [], META, {},
        scoring_version=2, direction_fn=scoring_v2.direction_from_score_v2,
    )
    results, pending, failed = forward.evaluate_records(v1 + v2, lambda s: (frame, None))
    assert failed == []
    versions = sorted({r["スコア仕様バージョン"] for r in results})
    assert versions == [1, 2]
    assert len(results) == 8  # 4チェックポイント × 2版
    merged, conflicts = forward.merge_results([], results)
    assert len(merged) == 8 and conflicts == []


def test_summarize_version_filter():
    frame = biz_frame()
    base_date = str(frame.index[0].date())
    v1 = forward.build_records([scan_row(データ最終日=base_date)], [], META, {})
    v2 = forward.build_records(
        [scan_row(データ最終日=base_date, 中期スコア=-3)], [], META, {},
        scoring_version=2, direction_fn=scoring_v2.direction_from_score_v2,
    )
    results, _, _ = forward.evaluate_records(v1 + v2, lambda s: (frame, None))
    only_v1 = forward.summarize([r for r in results if r["スコア仕様バージョン"] == 1], v1)
    filtered = forward.summarize(results, v1 + v2, version=1)
    assert filtered == only_v1
    assert forward.summarize(results, v1 + v2, version=2)["時点別"]


# ---------------- 基準日検査と V2 行の構築 ----------------

def test_prepare_v2_frame_truncates_future_bars():
    frame = biz_frame(n=30)
    dates = [str(d.date()) for d in frame.index]
    truncated, reason = forward.prepare_v2_frame(frame, dates[10], float(frame["Close"].iloc[10]))
    assert reason == ""
    assert str(truncated.index[-1].date()) == dates[10]
    assert len(truncated) == 11


def test_prepare_v2_frame_rejects_missing_date_and_price_gap():
    frame = biz_frame(n=30)
    dates = [str(d.date()) for d in frame.index]
    out, reason = forward.prepare_v2_frame(frame, "2026-12-31", 100.0)
    assert out is None and reason != ""
    close = float(frame["Close"].iloc[10])
    out, reason = forward.prepare_v2_frame(frame, dates[10], close * 1.03)
    assert out is None and reason != ""
    out, reason = forward.prepare_v2_frame(frame, dates[10], close * 1.01)
    assert out is not None and reason == ""


def assessment(mid_score, available=True, short_score=0, long_score=0, overheat=False):
    def one(horizon, score, avail):
        return {"horizon": horizon, "available": avail, "score": score if avail else None,
                "label": "x", "reasons": ["理由"], "warnings": []}
    return {"results": {
        "短期": one("短期", short_score, True),
        "中期": one("中期", mid_score, available),
        "長期": one("長期", long_score, True),
    }, "overheat": overheat}


def test_build_v2_scan_rows_ranking_exclusion_failure():
    ranked = [scan_row(コード="A.T"), scan_row(コード="B.T", 順位=2),
              scan_row(コード="C.T", 順位=3), scan_row(コード="D.T", 順位=4)]
    excluded_src = scan_row(区分="参考外", 順位=None, コード="E.T", 終値=None)
    excluded_src["警告・理由"] = "取得失敗：データなし"
    assessments = {
        "A.T": assessment(1),
        "B.T": assessment(5, overheat=True),
        "C.T": assessment(0, available=False),
    }
    ranked_v2, excluded_v2, flags, skipped = forward.build_v2_scan_rows(
        ranked, [excluded_src], assessments, failed={"D.T": "基準日検査に失敗しました"},
    )
    assert [r["コード"] for r in ranked_v2] == ["B.T", "A.T"]  # V2中期スコア順
    assert ranked_v2[0]["順位"] == 1
    assert flags == {"A.T": False, "B.T": True, "C.T": False}
    excluded_codes = {r["コード"]: r["警告・理由"] for r in excluded_v2}
    assert "データ不足" in excluded_codes["C.T"]
    # 計算できなかった銘柄は V2 の行を作らず、skipped（片側登録の対象）になる
    skipped_map = {s["コード"]: s["理由"] for s in skipped}
    assert skipped_map["D.T"] == "基準日検査に失敗しました"
    assert skipped_map["E.T"].startswith("取得失敗")
    assert "D.T" not in excluded_codes and "E.T" not in excluded_codes


def test_build_v2_scan_rows_never_substitutes_v1_scores():
    # V1 では参考外（鮮度）だがスコアが計算されていた銘柄。V2 の行には
    # V1 のスコアではなく、V2 自身が計算したスコアが入らなければならない
    stale_src = scan_row(区分="参考外", 順位=None, コード="S.T",
                         中期スコア=4, 短期スコア=3, 長期スコア=2)
    stale_src["警告・理由"] = "データが古い可能性があります：最終日が古い"
    ranked_v2, excluded_v2, flags, skipped = forward.build_v2_scan_rows(
        [], [stale_src], {"S.T": assessment(-3)},
    )
    assert ranked_v2 == [] and skipped == []
    row = excluded_v2[0]
    assert row["区分"] == "参考外"
    assert row["警告・理由"].startswith("データが古い可能性があります")  # 鮮度の理由は引き継ぐ
    assert row["中期スコア"] == -3 and row["短期スコア"] == 0 and row["長期スコア"] == 0
    # 方向も V2 の規則で V2 のスコアから決まる（V1 の +4 由来ではない）
    records = forward.build_records(
        [], [row], META, {}, scoring_version=2,
        direction_fn=scoring_v2.direction_from_score_v2,
    )
    assert records[0]["中期予測方向"] == "下落注意"


def test_build_v2_scan_rows_missing_assessment_is_skipped():
    ranked_v2, excluded_v2, flags, skipped = forward.build_v2_scan_rows(
        [scan_row(コード="X.T")], [], {},
    )
    assert ranked_v2 == [] and excluded_v2 == []
    assert skipped == [{"コード": "X.T", "理由": "V2計算用データが用意できませんでした"}]


# ---------------- 比較：行列・完全性・見送り・片側登録 ----------------

def test_compare_all_16_cells_and_completeness():
    records, results = [], []
    combos = [(d1, d2) for d1 in forward.V1_DIRECTIONS for d2 in forward.V2_DIRECTIONS]
    for i, (d1, d2) in enumerate(combos):
        code = f"{1000 + i}.T"
        records += pair(SCAN_ID, code, d1, d2)
        results.append(res(SCAN_ID, code, 20, 1, 5.0, exc=4.0))
    out = forward.compare_versions(results, records)
    matrix = out["行列"][(20, "中期")]
    assert len(matrix) == 16
    assert all(cell["件数"] == 1 for cell in matrix.values())
    comp = out["完全性"][(20, "中期")]
    assert comp["対象件数"] == 16 and comp["セル合計件数"] == 16
    assert comp["未分類件数"] == 0 and comp["一致"] is True
    assert comp["差分合計pct"] == comp["セル差分合計pct"]


def test_compare_unclassified_is_detected_not_dropped():
    records = [rec(SCAN_ID, "1000.T", 1, mid="上昇"), rec(SCAN_ID, "1000.T", 2, mid="不明")]
    results = [res(SCAN_ID, "1000.T", 20, 1, 5.0, exc=4.0)]
    out = forward.compare_versions(results, records)
    comp = out["完全性"][(20, "中期")]
    assert comp["未分類件数"] == 1
    assert (20, "中期") not in out["行列"] or out["行列"][(20, "中期")] == {}


def test_compare_skip_separation():
    records, results = [], []
    cases = [
        ("A.T", "下落", "下落注意", 6.0),   # 回避
        ("B.T", "下落", "下落注意", -4.0),  # 機会損失
        ("C.T", "下落", "中立", 3.0),       # スコア変化による見送り（別枠）
        ("D.T", "上昇", "下落注意", 5.0),   # 立場の反転。見送りに入れない
    ]
    for code, d1, d2, ret in cases:
        records += pair(SCAN_ID, code, d1, d2)
        results.append(res(SCAN_ID, code, 20, 1, ret, exc=ret))
    out = forward.compare_versions(results, records)
    skip = out["見送り"][(20, "中期")]
    caution = skip["下落注意"]
    assert caution["件数"] == 2 and caution["回避件数"] == 1 and caution["回避合計pct"] == 6.0
    assert caution["機会損失件数"] == 1 and caution["機会損失合計pct"] == 4.0
    assert caution["純影響pct"] == 2.0
    assert skip["中立"]["件数"] == 1 and skip["中立"]["回避合計pct"] == 3.0


def test_compare_one_sided_registration():
    records = [rec(SCAN_ID, "A.T", 1), rec(SCAN_ID, "B.T", 2)]
    results = [res(SCAN_ID, "A.T", 20, 1, 5.0), res(SCAN_ID, "B.T", 20, 2, 5.0)]
    out = forward.compare_versions(results, records)
    assert out["片側登録件数"] == {"V1のみ": 1, "V2のみ": 1}
    missing = {entry["コード"]: entry["不足側"] for entry in out["片側登録"]}
    assert missing == {"A.T": "V2", "B.T": "V1"}
    assert out["行列"] == {}


def test_compare_one_sided_reason_inference():
    fetch_failed = rec(SCAN_ID, "A.T", 1)
    fetch_failed["データ取得状況"] = "取得失敗"
    calc_failed = rec(SCAN_ID, "C.T", 1)
    calc_failed["データ取得状況"] = "取得成功"
    out = forward.compare_versions([], [fetch_failed, calc_failed])
    reasons = {e["コード"]: e["理由"] for e in out["片側登録"]}
    assert "取得に失敗" in reasons["A.T"]
    assert "基準日検査" in reasons["C.T"]


def test_compare_common_population():
    records, results = [], []
    cases = [("A.T", "上昇", 5.0), ("B.T", "下落", 5.0), ("C.T", "下落", -5.0)]
    for code, d1, ret in cases:
        records += pair(SCAN_ID, code, d1, "上昇")
        results.append(res(SCAN_ID, code, 20, 1, ret, exc=ret - 1.0))
    out = forward.compare_versions(results, records)
    common = out["共通母集団"][(20, "中期")]
    assert common["件数"] == 3
    assert common["V1的中率pct"] == 66.7 and common["V2的中率pct"] == 66.7
    assert common["常時上昇pct"] == 66.7
    assert common["ペア差分平均TOPIX超過pct"] is not None


def test_compare_pending_counts():
    records = pair(SCAN_ID, "A.T", "上昇", "上昇")
    results = [res(SCAN_ID, "A.T", 1, 1, 1.0)]
    out = forward.compare_versions(results, records)
    assert out["評価待ち"] == {5: 1, 10: 1, 20: 1}


# ---------------- 採用ゲート ----------------

def test_gate_insufficient_data_and_regime_unknown():
    records = pair(SCAN_ID, "A.T", "下落", "下落注意")
    results = [res(SCAN_ID, "A.T", 20, 1, 5.0, exc=4.0)]
    out = forward.compare_versions(results, records)
    gate = out["ゲート"]
    assert gate["充足"] is False
    regime_item = [i for i in gate["項目"] if "相場" in i["項目"]][0]
    assert regime_item["判定"] is None  # 指数データなしでは未判定


def test_gate_cohort_ci_and_symbol_dependence():
    records, results = [], []
    # 20コホート：主エンドポイントで各コホート差分 +2.0（TOPIX超過ベース）
    for k in range(20):
        scan_id = f"2026-{1 + k // 4:02d}-{1 + (k % 4) * 7:02d} 15:30:00"
        records += pair(scan_id, "A.T", "下落", "下落注意")
        results.append(res(scan_id, "A.T", 20, 1, 3.0, exc=2.0))
    out = forward.compare_versions(results, records, regime_by_scan={})
    gate = out["ゲート"]
    cohort_item = [i for i in gate["項目"] if "コホート数" in i["項目"]][0]
    assert cohort_item["判定"] is True
    assert gate["コホート平均差分pct"] == 2.0
    assert gate["信頼区間95"] == (2.0, 2.0)
    # 1銘柄だけのデータなので依存警告が出る
    assert gate["銘柄依存"]["依存警告"] is True
    assert gate["銘柄依存"]["最多銘柄"] == "A.T"


def test_gate_counts_fallback_cohorts():
    records = pair(SCAN_ID, "A.T", "下落", "下落注意")
    records[1]["警告・理由"] = forward.FALLBACK_NOTE
    results = [res(SCAN_ID, "A.T", 20, 1, 3.0, exc=2.0)]
    out = forward.compare_versions(results, records)
    fallback = out["ゲート"]["再取得フォールバック"]
    assert fallback["コホート数"] == 1 and fallback["スキャンID"] == [SCAN_ID]


# ---------------- 未登録理由の保存と下落注意の参考グループ ----------------

def test_skipped_store_roundtrip_and_merge():
    entries = [{"スキャンID": SCAN_ID, "コード": "A.T", "理由": "基準日検査に失敗しました"}]
    loaded = forward.skipped_from_csv(forward.skipped_to_csv(entries))
    assert loaded == entries
    merged = forward.merge_skipped(loaded, entries + [
        {"スキャンID": SCAN_ID, "コード": "B.T", "理由": "取得失敗：データなし"},
    ])
    assert len(merged) == 2
    assert forward.skipped_from_csv(b"not,a,csv") == []


def test_caution_reference_group():
    records = [
        rec(SCAN_ID, "A.T", 2, mid="下落注意"),
        rec(SCAN_ID, "B.T", 2, mid="上昇"),
        rec(SCAN_ID, "C.T", 1, mid="下落"),
    ]
    results = [
        res(SCAN_ID, "A.T", 20, 2, 6.0, max_down=-2.0),
        res(SCAN_ID, "B.T", 20, 2, 3.0),
        res(SCAN_ID, "C.T", 20, 1, -5.0),
    ]
    out = forward.caution_reference(results, records)
    group = out[(20, "中期")]
    assert group["件数"] == 1
    assert group["平均リターンpct"] == 6.0 and group["平均最大下落率pct"] == -2.0
