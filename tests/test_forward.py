"""フォワード登録と事後評価（forward.py）のテスト。ネットワークは使わない。

確認すること:
- 登録は全銘柄（参考外を含む）を対象に、スキャン時点の情報だけで作られる。
- 同一スキャンの再登録は重複、別スキャンは別記録、内容違いは競合になる。
- 評価は基準日より後の足だけで、営業日（足の本数）単位で数える。
- リターンは評価時に取得した系列内の基準価格で計算する（分割に強い）。
- 最大上昇率・最大下落率は期間内の高値・安値から計算する。
- 指数比較は基準日・評価日に対応する同じ期間で計算し、対応データが
  なければ無理に計算せず理由を残す。
- 集計の的中率は中立・判定保留を分母に入れない。
"""

import pandas as pd
import pytest

import analysis
import forward

TZ = "Asia/Tokyo"


def biz_dates(start="2026-10-05", n=40):
    return list(pd.bdate_range(start=start, periods=n, tz=TZ))


def make_frame(dates, closes, highs=None, lows=None):
    closes = [float(c) for c in closes]
    highs = highs if highs is not None else [c + 1.0 for c in closes]
    lows = lows if lows is not None else [c - 1.0 for c in closes]
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [float(h) for h in highs],
            "Low": [float(l) for l in lows],
            "Close": closes,
            "Volume": [1000] * len(closes),
        },
        index=pd.DatetimeIndex(dates),
    )


def scan_row(**overrides):
    row = {
        "区分": "ランキング", "順位": 1, "コード": "1000.T", "会社名": "テスト株",
        "業種": "テスト", "中期スコア": 4, "中期ラベル": "強い",
        "短期スコア": 3, "短期ラベル": "強い", "長期スコア": 1, "長期ラベル": "やや強い",
        "終値": 110.0, "選定理由": "中期条件", "警告・理由": "",
        "データ最終日": "2026-10-19", "本数": 300, "取得日時": "2026-10-19 15:30:00",
    }
    row.update(overrides)
    return row


META = {"実行日時": "2026-10-19 15:30:00", "対象銘柄数": 3, "ランキング件数": 2, "参考外件数": 1}


def test_build_records_covers_all_and_stamps():
    ranked = [scan_row(), scan_row(コード="2000.T", 順位=2, 中期スコア=-3, 短期スコア=0, 長期スコア=None)]
    excluded_row = scan_row(区分="参考外", 順位=None, コード="3000.T", 終値=None)
    excluded_row["警告・理由"] = "取得失敗：データなし"
    excluded = [excluded_row]
    benchmarks = {"^N225": (60000.0, "2026-10-19"), "1306.T": (400.0, "2026-10-19")}
    records = forward.build_records(ranked, excluded, META, benchmarks)

    assert len(records) == 3
    first = records[0]
    assert first["スキャンID"] == "2026-10-19 15:30:00"
    assert first["株価基準日"] == "2026-10-19"
    assert first["中期予測方向"] == "上昇"  # スコア4
    assert first["短期予測方向"] == "上昇"  # スコア3
    assert first["長期予測方向"] == "中立"  # スコア1
    assert first["スコア仕様バージョン"] == analysis.SCORING_VERSION == 1
    assert first["日経平均_基準値"] == 60000.0
    assert first["TOPIX連動ETF_基準値"] == 400.0
    assert first["データ取得状況"] == "取得成功"
    second = records[1]
    assert second["中期予測方向"] == "下落"  # スコア-3
    assert second["短期予測方向"] == "中立"  # スコア0
    assert second["長期予測方向"] == "判定保留"  # スコアなし
    third = records[2]
    assert third["区分"] == "参考外"
    assert third["データ取得状況"] == "取得失敗"


def test_build_records_missing_benchmark_is_blank():
    records = forward.build_records([scan_row()], [], META, {})
    assert records[0]["日経平均_基準値"] is None
    assert records[0]["日経平均_基準日"] == ""


def test_merge_records_dedup_new_scan_and_conflict():
    records = forward.build_records([scan_row()], [], META, {})
    merged, conflicts = forward.merge_records([], records)
    assert len(merged) == 1 and conflicts == []

    # 同一スキャンの再登録は重複
    merged2, conflicts2 = forward.merge_records(merged, records)
    assert len(merged2) == 1 and conflicts2 == []

    # 別スキャン（実行日時が違う）は別記録
    other_meta = dict(META, 実行日時="2026-10-20 15:30:00")
    other = forward.build_records([scan_row()], [], other_meta, {})
    merged3, conflicts3 = forward.merge_records(merged2, other)
    assert len(merged3) == 2 and conflicts3 == []

    # 同じキーで内容が違えば競合（既存を残す）
    changed = forward.build_records([scan_row(終値=999.0)], [], META, {})
    merged4, conflicts4 = forward.merge_records(merged3, changed)
    assert len(merged4) == 2
    assert len(conflicts4) == 1
    kept = [r for r in merged4 if r["スキャンID"] == META["実行日時"]][0]
    assert kept["記録時終値"] == 110.0


def test_records_csv_round_trip():
    records = forward.build_records(
        [scan_row()], [], META, {"^N225": (60000.0, "2026-10-19")}
    )
    loaded = forward.records_from_csv(forward.records_to_csv(records))
    assert len(loaded) == 1
    rec = loaded[0]
    assert rec["コード"] == "1000.T"
    assert rec["中期スコア"] == 4
    assert rec["記録時終値"] == pytest.approx(110.0)
    assert rec["日経平均_基準値"] == pytest.approx(60000.0)
    # 往復した記録を再統合しても重複として処理される
    merged, conflicts = forward.merge_records(records, loaded)
    assert len(merged) == 1 and conflicts == []


def make_fetcher(frames):
    def fetcher(symbol):
        if symbol not in frames:
            return pd.DataFrame(), "データなし"
        return frames[symbol], None

    return fetcher


def base_setup():
    """基準日を10本目（2026-10-19）に置いた40本の系列と登録レコード。"""
    dates = biz_dates(n=40)
    closes = [100.0 + i for i in range(40)]  # 基準日（10本目）の終値は110
    frame = make_frame(dates, closes)
    benchmarks = {
        "^N225": make_frame(dates, [60000.0 * (1.001 ** i) for i in range(40)]),
        "1306.T": make_frame(dates, [400.0 * (1.001 ** i) for i in range(40)]),
    }
    frames = {"1000.T": frame, **benchmarks}
    records = forward.build_records([scan_row()], [], META, {})
    return dates, frames, records


def test_evaluate_basic_and_benchmark_same_period():
    dates, frames, records = base_setup()
    results, pending, failed = forward.evaluate_records(
        records, make_fetcher(frames), evaluated_at="2026-11-20 10:00:00"
    )
    assert failed == [] and pending == []
    by_n = {r["評価営業日数"]: r for r in results}
    assert set(by_n) == {1, 5, 10, 20}
    r5 = by_n[5]
    assert r5["評価日"] == dates[15].date().isoformat()
    assert r5["評価用基準価格"] == pytest.approx(110.0)
    assert r5["評価価格"] == pytest.approx(115.0)
    assert r5["リターンpct"] == pytest.approx(round((115 / 110 - 1) * 100, 3))
    # 最大上昇率は期間内（翌足〜評価日）の高値、最大下落率は安値から
    assert r5["最大上昇率pct"] == pytest.approx(round((116 / 110 - 1) * 100, 3))
    assert r5["最大下落率pct"] == pytest.approx(round((110 / 110 - 1) * 100, 3))
    bench_ret = round((1.001 ** 5 - 1) * 100, 3)
    assert r5["日経平均_リターンpct"] == pytest.approx(bench_ret, abs=0.01)
    assert r5["日経超過リターンpct"] == pytest.approx(
        round(r5["リターンpct"] - r5["日経平均_リターンpct"], 3)
    )
    assert r5["TOPIX超過リターンpct"] is not None


def test_evaluate_pending_when_checkpoint_not_reached():
    dates = biz_dates(n=40)
    frame = make_frame(dates, [100.0 + i for i in range(40)])
    frames = {"1000.T": frame}
    rec = scan_row(データ最終日=dates[35].date().isoformat(), 終値=135.0)
    records = forward.build_records([rec], [], META, {})
    results, pending, failed = forward.evaluate_records(
        records, make_fetcher(frames), evaluated_at="2026-12-01 10:00:00"
    )
    assert [r["評価営業日数"] for r in results] == [1]
    assert sorted(p["評価営業日数"] for p in pending) == [5, 10, 20]


def test_evaluate_uses_series_base_price_split_safe():
    dates, frames, records = base_setup()
    # 記録時終値は分割前の200。系列（調整済み）の基準日終値は110。
    records[0]["記録時終値"] = 200.0
    results, _, _ = forward.evaluate_records(
        records, make_fetcher(frames), evaluated_at="2026-11-20 10:00:00"
    )
    r5 = [r for r in results if r["評価営業日数"] == 5][0]
    assert r5["評価用基準価格"] == pytest.approx(110.0)
    assert r5["リターンpct"] == pytest.approx(round((115 / 110 - 1) * 100, 3))


def test_evaluate_base_date_missing_is_unevaluable():
    dates, frames, records = base_setup()
    records[0]["株価基準日"] = "2026-10-20"  # 系列にない日（祝日の想定）
    # 2026-10-20 は営業日のため系列にあるので、存在しない日に差し替える
    records[0]["株価基準日"] = "2026-10-18"  # 日曜
    results, pending, failed = forward.evaluate_records(
        records, make_fetcher(frames), evaluated_at="2026-11-20 10:00:00"
    )
    assert len(results) == 1
    assert results[0]["状態"] == "評価不能"
    assert "基準日" in results[0]["理由"]


def test_evaluate_benchmark_missing_leaves_reason():
    dates, frames, records = base_setup()
    frames["^N225"] = make_frame(dates[:20], [60000.0] * 20)  # 評価日側が欠ける
    results, _, _ = forward.evaluate_records(
        records, make_fetcher(frames), evaluated_at="2026-11-20 10:00:00"
    )
    r20 = [r for r in results if r["評価営業日数"] == 20][0]
    assert r20["状態"] == "評価済み"
    assert r20["リターンpct"] is not None
    assert r20["日経超過リターンpct"] is None
    assert "日経平均" in r20["理由"]
    assert r20["TOPIX超過リターンpct"] is not None


def test_merge_results_dedup_and_conflict():
    dates, frames, records = base_setup()
    results, _, _ = forward.evaluate_records(
        records, make_fetcher(frames), evaluated_at="2026-11-20 10:00:00"
    )
    merged, conflicts = forward.merge_results([], results)
    assert len(merged) == 4 and conflicts == []
    merged2, conflicts2 = forward.merge_results(merged, results)
    assert len(merged2) == 4 and conflicts2 == []
    changed = [dict(r) for r in results]
    changed[0]["リターンpct"] = 999.0
    merged3, conflicts3 = forward.merge_results(merged2, changed)
    assert len(merged3) == 4 and len(conflicts3) == 1


def test_results_csv_round_trip():
    dates, frames, records = base_setup()
    results, _, _ = forward.evaluate_records(
        records, make_fetcher(frames), evaluated_at="2026-11-20 10:00:00"
    )
    loaded = forward.results_from_csv(forward.results_to_csv(results))
    assert len(loaded) == 4
    merged, conflicts = forward.merge_results(results, loaded)
    assert len(merged) == 4 and conflicts == []


def test_summarize_hit_rate_and_bands():
    records = forward.build_records(
        [
            scan_row(順位=1, 中期スコア=4),
            scan_row(コード="2000.T", 順位=6, 中期スコア=-4, 短期スコア=-4, 長期スコア=-4),
            scan_row(コード="3000.T", 順位=7, 中期スコア=0, 短期スコア=0, 長期スコア=0),
        ],
        [], META, {},
    )

    def result(code, ret):
        return {
            "記録仕様版数": 1, "スキャンID": META["実行日時"], "コード": code,
            "会社名": "", "評価営業日数": 5, "株価基準日": "2026-10-19",
            "評価日": "2026-10-26", "評価用基準価格": 100.0, "評価価格": 100.0,
            "リターンpct": ret, "最大上昇率pct": None, "最大下落率pct": None,
            "日経平均_基準値": None, "日経平均_評価値": None,
            "日経平均_リターンpct": None, "日経超過リターンpct": None,
            "TOPIX連動ETF_基準値": None, "TOPIX連動ETF_評価値": None,
            "TOPIX連動ETF_リターンpct": None, "TOPIX超過リターンpct": 1.0,
            "状態": "評価済み", "理由": "", "実行日時": "2026-11-20 10:00:00",
        }

    results = [result("1000.T", 5.0), result("2000.T", 3.0), result("3000.T", -2.0)]
    summary = forward.summarize(results, records)

    point = summary["時点別"][5]
    assert point["件数"] == 3
    assert point["平均リターンpct"] == pytest.approx(2.0)
    assert point["上昇割合pct"] == pytest.approx(66.7)

    bands = summary["順位帯別"][5]
    assert bands["上位5"]["件数"] == 1
    assert bands["上位5"]["平均リターンpct"] == pytest.approx(5.0)
    assert bands["全体"]["件数"] == 3

    # 中期：上昇予測（+5%で的中）と下落予測（+3%で不的中）→ 的中率50%
    mid = summary["時間軸別"][(5, "中期")]
    assert mid["方向件数"] == 2
    assert mid["的中率pct"] == pytest.approx(50.0)
    # 中立の銘柄は分母に入らない（短期は方向2件のまま）
    short = summary["時間軸別"][(5, "短期")]
    assert short["方向件数"] == 2
