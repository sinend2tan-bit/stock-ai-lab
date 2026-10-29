"""検証ロジック（verify.py）の単体テスト。ネットワークは使わない。"""

import pandas as pd
import pytest

import analysis
import verify


def make_frame(closes):
    idx = pd.bdate_range(start="2021-01-04", periods=len(closes))
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": [1000] * len(closes),
        },
        index=idx,
    )


def rising_frame(n=400):
    return make_frame([100.0 + 0.1 * k for k in range(n)])


# --- 未来情報が混入しないこと ---

def test_future_prices_do_not_change_past_evaluation():
    frame = rising_frame(400)
    records1, _ = verify.verify_symbol("7203.T", frame, run_at="2026-10-29 00:00:00")
    changed = frame.copy()
    cutoff = frame.index[299]
    changed.loc[changed.index > cutoff, "Close"] = (
        changed.loc[changed.index > cutoff, "Close"] * 1.5
    )
    changed.loc[changed.index > cutoff, "Open"] = changed.loc[changed.index > cutoff, "Close"]
    changed.loc[changed.index > cutoff, "High"] = changed.loc[changed.index > cutoff, "Close"]
    changed.loc[changed.index > cutoff, "Low"] = changed.loc[changed.index > cutoff, "Close"]
    records2, _ = verify.verify_symbol("7203.T", changed, run_at="2026-10-29 00:00:00")

    def key(r):
        return (r["評価日"], r["時間軸"])

    by_key2 = {key(r): r for r in records2}
    compared = 0
    for r1 in records1:
        # 測定日まで含めて未来変更の影響を受けない記録だけを比べる
        if r1["測定日"] and r1["測定日"] <= str(cutoff.date()):
            r2 = by_key2[key(r1)]
            for field in ["スコア", "予測方向", "評価ラベル", "騰落率pct", "実際の方向", "判定"]:
                assert r1[field] == r2[field], f"{field} が未来価格の変更で変わった"
            compared += 1
    assert compared > 50  # 十分な件数を実際に比較したこと


def test_slice_reproduction_matches_direct_assessment():
    frame = rising_frame(400)
    records, _ = verify.verify_symbol("7203.T", frame, run_at="x")
    target = next(
        r for r in records if r["時間軸"] == "短期" and r["評価日"] == str(frame.index[99].date())
    )
    sliced = frame.iloc[:100]
    direct = analysis.assess_all(analysis.add_indicators(sliced))["短期"]
    assert target["スコア"] == direct["score"]
    assert target["予測方向"] == verify.direction_from_result(direct)


# --- 測定窓と騰落率 ---

def test_forward_return_exact_value():
    closes = [100.0] * 120
    closes[39] = 102.0  # 評価行インデックス34の短期測定日（+5）
    frame = make_frame(closes)
    records, _ = verify.verify_symbol("7203.T", frame, run_at="x")
    target = next(
        r for r in records
        if r["時間軸"] == "短期" and r["評価日"] == str(frame.index[34].date())
    )
    assert target["騰落率pct"] == pytest.approx(2.0)
    assert target["測定日"] == str(frame.index[39].date())


def test_tail_points_are_excluded_and_counted():
    frame = rising_frame(120)
    _, exclusions = verify.verify_symbol("7203.T", frame, run_at="x")
    tail = [e for e in exclusions if "測定窓" in e["理由"]]
    assert tail and all(e["件数"] > 0 for e in tail)


# --- 方向の対応 ---

def test_direction_from_score_thresholds():
    assert verify.direction_from_result({"available": True, "score": 2}) == "上昇"
    assert verify.direction_from_result({"available": True, "score": 5}) == "上昇"
    assert verify.direction_from_result({"available": True, "score": -2}) == "下落"
    assert verify.direction_from_result({"available": True, "score": 1}) == "中立"
    assert verify.direction_from_result({"available": True, "score": 0}) == "中立"
    assert verify.direction_from_result({"available": False, "score": None}) == "判定保留"


def test_flat_band_boundaries():
    assert verify.actual_direction(0.5, "短期") == "小動き"
    assert verify.actual_direction(1.0, "短期") == "小動き"
    assert verify.actual_direction(1.5, "短期") == "上昇"
    assert verify.actual_direction(-2.0, "中期") == "小動き"
    assert verify.actual_direction(-2.5, "中期") == "下落"
    assert verify.actual_direction(3.5, "長期") == "上昇"


# --- 集計 ---

def _rec(**kw):
    base = {
        "実行日時": "x", "仕様版数": 1, "銘柄": "7203.T", "会社名": "", "市場": "JP",
        "評価日": "2025-01-06", "時間軸": "短期", "スコア": 3, "予測方向": "上昇",
        "評価ラベル": "", "評価日終値": 100.0, "測定日": "2025-01-13",
        "測定日終値": 103.0, "騰落率pct": 3.0, "実際の方向": "上昇",
        "判定": "的中", "対象外理由": "", "方向付き騰落率pct": 3.0,
        "データ最終日": "2026-10-29", "取得期間": "5y",
    }
    base.update(kw)
    return base


def test_aggregate_denominator_and_pnl_exclusions():
    records = [
        _rec(判定="的中", 方向付き騰落率pct=3.0),
        _rec(判定="不的中", 予測方向="下落", 方向付き騰落率pct=-2.0,
             実際の方向="上昇", 騰落率pct=2.0),
        _rec(判定="中立予測", 予測方向="中立", 方向付き騰落率pct=None),
        _rec(判定="実際小動き", 実際の方向="小動き", 方向付き騰落率pct=0.5,
             騰落率pct=0.5),
    ]
    exclusions = [
        {"銘柄": "7203.T", "時間軸": "短期", "理由": "履歴不足のため評価開始前の地点", "件数": 7},
        {"銘柄": "7203.T", "時間軸": "全体", "理由": "データ取得に失敗", "件数": 1},
    ]
    summary = verify.aggregate(records, exclusions)
    short = next(r for r in summary["時間軸別"] if r["時間軸"] == "短期")
    assert short["的中"] == 1 and short["不的中"] == 1
    assert short["的中率pct"] == 50.0  # 分母は的中+不的中の2件のみ
    assert short["中立予測"] == 1 and short["実際小動き"] == 1
    # 中立割合は予測記録の総数（4件）に対する割合
    assert short["中立割合pct"] == 25.0
    # 対象外は予測記録ではなく対象外集計から数える（全体の取得失敗は時間軸別に入らない）
    assert short["対象外"] == 7
    assert len(summary["取得失敗"]) == 1
    # 方向付き騰落率の平均は中立（None）を除いた 3件（3.0, -2.0, 0.5）の平均
    assert short["方向付き騰落率_平均pct"] == pytest.approx(0.5)
    assert short["精度判断"].startswith("件数不足")
    # 実際上昇の内訳：上昇予測1（的中）・下落予測1（不的中）・中立予測1（見逃し側）
    bd = next(
        r for r in summary["実際方向別内訳"]
        if r["時間軸"] == "短期" and r["実際の方向"] == "上昇"
    )
    assert bd["上昇予測"] == 1 and bd["下落予測"] == 1 and bd["中立予測"] == 1


# --- 取得失敗の扱い ---

def test_fetch_failure_becomes_exclusion_and_others_continue():
    good = rising_frame(120)

    def fetcher(symbol):
        if symbol == "BAD":
            return pd.DataFrame(), "取得制限のため失敗"
        return good, None

    records, exclusions, info = verify.verify_all(
        ["BAD", "7203.T"], fetcher, run_at="x"
    )
    # 取得失敗は予測記録には入れず、対象外集計に区別して記録する
    assert [r for r in records if r["銘柄"] == "BAD"] == []
    bad_excl = next(e for e in exclusions if e["銘柄"] == "BAD")
    assert bad_excl["時間軸"] == "全体" and "取得に失敗" in bad_excl["理由"]
    assert any(r["銘柄"] == "7203.T" for r in records)
    bad_info = next(i for i in info if i["銘柄"] == "BAD")
    assert bad_info["取得"] == "失敗" and bad_info["本数"] == 0


# --- 履歴不足（長期の開始前） ---

def test_long_horizon_warmup_exclusion():
    # 長期は250本の履歴＋60営業日の測定窓が必要なため、最低310本以上で評価可能になる
    frame = rising_frame(400)
    records, exclusions = verify.verify_symbol("7203.T", frame, run_at="x")
    long_records = [r for r in records if r["時間軸"] == "長期"]
    assert long_records  # 400本あれば長期も評価できる地点がある
    first_long_date = min(r["評価日"] for r in long_records)
    # 長期の最初の評価日は250本目以降のはず
    assert first_long_date >= str(frame.index[249].date())
    warm = next(
        e for e in exclusions if e["時間軸"] == "長期" and "履歴不足" in e["理由"]
    )
    assert warm["件数"] > 0


# --- CSVの往復と重複除外 ---

def test_csv_roundtrip_and_dedupe():
    frame = rising_frame(200)
    records, _ = verify.verify_symbol("7203.T", frame, run_at="2026-10-29 00:00:00")
    assert records
    loaded = verify.records_from_csv(verify.records_to_csv(records))
    assert len(loaded) == len(records)
    for orig, back in zip(records, loaded):
        for field in ["銘柄", "評価日", "時間軸", "予測方向", "判定", "仕様版数"]:
            assert orig[field] == back[field]
        assert orig["騰落率pct"] == pytest.approx(back["騰落率pct"])
    merged = verify.dedupe_records(records + loaded)
    assert len(merged) == len(records)


def test_records_from_invalid_csv_is_empty():
    assert verify.records_from_csv(b"not,a,valid,verification,csv") == []


# --- 重複キーの統一と競合検出 ---

def test_merge_records_identical_is_duplicate_not_conflict():
    original = _rec()
    copy_rec = dict(original)
    copy_rec["実行日時"] = "2026-11-01 00:00:00"  # 実行日時だけ違う同一記録
    merged, conflicts = verify.merge_records([original], [copy_rec])
    assert len(merged) == 1
    assert conflicts == []


def test_merge_records_conflict_is_detected_and_not_adopted():
    existing = _rec(判定="的中", スコア=3)
    incoming = _rec(判定="不的中", スコア=-3, 予測方向="下落")
    merged, conflicts = verify.merge_records([existing], [incoming])
    assert len(merged) == 1
    assert merged[0]["判定"] == "的中"  # 既存を残し、新規は取り込まない
    assert len(conflicts) == 1
    c = conflicts[0]
    assert c["銘柄"] == "7203.T" and c["時間軸"] == "短期"
    assert c["既存の判定"] == "的中" and c["新規の判定"] == "不的中"


def test_merge_records_new_key_is_added():
    merged, conflicts = verify.merge_records([_rec()], [_rec(評価日="2025-01-07")])
    assert len(merged) == 2 and conflicts == []


# --- 対象外集計のCSV保存・復元と二重計上防止 ---

def test_exclusions_csv_roundtrip_and_no_double_count():
    exclusions = [
        {"銘柄": "7203.T", "時間軸": "長期", "理由": "履歴不足のため評価開始前の地点", "件数": 49},
        {"銘柄": "BAD", "時間軸": "全体", "理由": "データ取得に失敗したため、この銘柄は検証対象外", "件数": 1},
    ]
    loaded = verify.exclusions_from_csv(verify.exclusions_to_csv(exclusions))
    assert loaded == exclusions
    # 同じ集計を読み直しても、件数は足し合わされず二重計上にならない
    merged = verify.merge_exclusions(exclusions, loaded)
    assert len(merged) == 2
    assert next(e for e in merged if e["銘柄"] == "7203.T")["件数"] == 49


def test_exclusions_from_invalid_csv_is_empty():
    assert verify.exclusions_from_csv(b"a,b,c") == []


def test_fetch_failure_exclusion_not_counted_in_horizons():
    # 取得失敗（時間軸「全体」）は、時間軸別・銘柄別の対象外件数に重複計上しない
    records = [_rec()]
    exclusions = [
        {"銘柄": "BAD", "時間軸": "全体", "理由": "データ取得に失敗したため、この銘柄は検証対象外", "件数": 1},
        {"銘柄": "7203.T", "時間軸": "短期", "理由": "履歴不足のため評価開始前の地点", "件数": 5},
        {"銘柄": "7203.T", "時間軸": "中期", "理由": "履歴不足のため評価開始前の地点", "件数": 15},
    ]
    summary = verify.aggregate(records, exclusions)
    by_h = {r["時間軸"]: r for r in summary["時間軸別"]}
    assert by_h["短期"]["対象外"] == 5
    assert by_h["中期"]["対象外"] == 15
    # 長期には記録も対象外もないため行自体がなく、時間軸別の対象外合計は
    # 履歴不足分（5+15=20）だけになる。取得失敗の1件はどこにも加算されていない
    assert sum(r["対象外"] for r in summary["時間軸別"]) == 20
    # 取得失敗は専用の欄でのみ数える
    assert len(summary["取得失敗"]) == 1
    assert summary["取得失敗"][0]["銘柄"] == "BAD"
    # 銘柄別でも、取得失敗銘柄の行は作られない
    assert all(r["銘柄"] != "BAD" for r in summary["銘柄別"])


def test_repeated_csv_import_does_not_grow_records_or_exclusions():
    frame = rising_frame(200)
    records, exclusions = verify.verify_symbol("7203.T", frame, run_at="2026-10-29 00:00:00")
    assert records and exclusions
    records_csv = verify.records_to_csv(records)
    exclusions_csv = verify.exclusions_to_csv(exclusions)

    # アプリの読み込み動作を再現：同じCSVを3回読み込んでも増えない
    session_records = []
    session_exclusions = []
    for _ in range(3):
        loaded = verify.records_from_csv(records_csv)
        session_records, conflicts = verify.merge_records(session_records, loaded)
        assert conflicts == []
        loaded_ex = verify.exclusions_from_csv(exclusions_csv)
        session_exclusions = verify.merge_exclusions(session_exclusions, loaded_ex)
    assert len(session_records) == len(records)
    assert session_exclusions == exclusions
