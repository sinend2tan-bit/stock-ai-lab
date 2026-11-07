"""検証A（短期フォワード仮想売買評価、verification_a.py）の単体テスト。

ネットワークは使わない（フェイク fetcher で完結させる）。
確認すること:
- 手数料は買い・売りそれぞれの約定代金から個別に計算する（固定198円にしない）。
- エントリーは翌営業日の始値、決済は day0 起算の day5/day10 終値。
- 2% 照合ガード、基準日の足なし、取得失敗は「評価不能」として理由付きで残る。
- 決済日が未到来のものは「保留」として理由付きで残る。
- 重複 dedupe、対象外（V2・中立など）の分離。
- 未来情報の混入防止（決済日より後の行を変えても結果が変わらない）。
- 単元不足は理論評価と混ぜずに分離する。
- DD-1（同時保有を考慮しない）の計算。
- 最低証拠条件が未達なら「判定保留」で不足数を出す。
- snapshot の往復（保存→再評価で一致）とハッシュ改変検出。
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

import forward
import verification_a as va

START = "2026-10-05"  # 月曜日


def biz_dates(n=40, start=START):
    return [d.date().isoformat() for d in pd.bdate_range(start=start, periods=n)]


def make_frame(dates, opens, closes, volumes=None):
    n = len(dates)
    return pd.DataFrame(
        {
            "Open": [float(x) for x in opens],
            "High": [float(max(o, c) + 1) for o, c in zip(opens, closes)],
            "Low": [float(min(o, c) - 1) for o, c in zip(opens, closes)],
            "Close": [float(x) for x in closes],
            "Volume": volumes or [1000] * n,
        },
        index=pd.to_datetime(dates),
    )


def rising_frame(n=40, start_close=1000.0, step=10.0):
    dates = biz_dates(n)
    closes = [start_close + step * i for i in range(n)]
    opens = [c - 5.0 for c in closes]
    return make_frame(dates, opens, closes), dates, opens, closes


def make_record(symbol="7203.T", scan="scan-1", base_date=START, close=1000.0,
                direction="上昇", version=1):
    rec = {col: "" for col in forward.RECORD_COLUMNS}
    rec.update({
        "記録仕様版数": 2,
        "スキャンID": scan,
        "株価基準日": base_date,
        "コード": symbol,
        "会社名": "テスト会社",
        "区分": "ランキング",
        "短期予測方向": direction,
        "記録時終値": close,
        "スコア仕様バージョン": version,
        "日経平均_基準値": 38000.0,
        "TOPIX連動ETF_基準値": 1500.0,
    })
    return rec


def fake_fetcher(mapping):
    def fetcher(symbol):
        df = mapping.get(symbol)
        if df is None:
            return pd.DataFrame(), "データが取得できませんでした"
        return df, None
    return fetcher


def topix_frame(n=40):
    dates = biz_dates(n)
    closes = [1500.0 + 2.0 * i for i in range(n)]
    return make_frame(dates, closes, closes)


def standard_setup(n=40, **rec_kwargs):
    """7203.T の上昇系列と TOPIX代理を用意した標準の評価結果を返す。"""
    frame, dates, opens, closes = rising_frame(n=n)
    fetcher = fake_fetcher({"7203.T": frame, va.TOPIX_PROXY_SYMBOL: topix_frame(n)})
    rec = make_record(base_date=dates[0], close=closes[0], **rec_kwargs)
    result = va.evaluate([rec], fetcher, run_id="test-run")
    return result, dates, opens, closes


# ---------------------------------------------------------------
# 手数料
# ---------------------------------------------------------------

@pytest.mark.parametrize("amount,expected", [
    (50_000, 55), (50_001, 99), (100_000, 99), (100_001, 115),
    (200_000, 115), (200_001, 275), (500_000, 275), (500_001, 535),
    (1_000_000, 535), (1_000_001, 640), (1_500_000, 640), (1_500_001, 1_013),
    (30_000_000, 1_013), (30_000_001, 1_070),
])
def test_fee_boundaries(amount, expected):
    assert va.fee_for_amount(amount) == expected


def test_fee_not_fixed_198():
    """売り約定代金が10万円を超えると売り側の区分が変わり、合計は198円にならない。"""
    frame, dates, opens, closes = rising_frame()
    # 買い約定代金は想定元本ちょうど 100,000（手数料99）。
    # day5 終値を 1015 にして、売り約定代金を 100,000 超にする。
    closes2 = list(closes)
    closes2[6] = 1015.0
    frame = make_frame(dates, opens, closes2)
    fetcher = fake_fetcher({"7203.T": frame, va.TOPIX_PROXY_SYMBOL: topix_frame()})
    rec = make_record(base_date=dates[0], close=closes[0])
    result = va.evaluate([rec], fetcher, run_id="test-run")
    t5 = [t for t in result["trades"] if t["期間"] == 5][0]
    assert t5["状態"] == va.STATUS_EVALUATED
    assert t5["買い約定代金"] == pytest.approx(100_000.0)
    assert t5["売り約定代金"] > 100_000
    assert t5["手数料_ケース1"] == 99 + 115


# ---------------------------------------------------------------
# 営業日カウントと価格の取り方
# ---------------------------------------------------------------

def test_business_day_counting():
    result, dates, opens, closes = standard_setup()
    t5 = [t for t in result["trades"] if t["期間"] == 5][0]
    t10 = [t for t in result["trades"] if t["期間"] == 10][0]
    # エントリーは基準日の翌営業日、その始値
    assert t5["エントリー日"] == dates[1]
    assert t5["エントリー価格_翌営業日始値"] == opens[1]
    # エントリー日を day0 として day5 / day10 の終値
    assert t5["決済日"] == dates[6]
    assert t5["決済価格_終値"] == closes[6]
    assert t10["決済日"] == dates[11]
    assert t10["決済価格_終値"] == closes[11]
    # ベースライン（同一銘柄買い持ち）は基準日終値→決済日終値のグロス
    assert t5["ベースライン_同一銘柄買い持ちpct"] == pytest.approx(
        (closes[6] / closes[0] - 1) * 100)
    assert t5["ベースライン_TOPIX代理pct"] is not None


# ---------------------------------------------------------------
# 評価不能・保留の切り分け
# ---------------------------------------------------------------

def test_price_guard_mismatch():
    frame, dates, opens, closes = rising_frame()
    fetcher = fake_fetcher({"7203.T": frame, va.TOPIX_PROXY_SYMBOL: topix_frame()})
    rec = make_record(base_date=dates[0], close=closes[0] * 1.03)  # 3% ずれている
    result = va.evaluate([rec], fetcher, run_id="test-run")
    for t in result["trades"]:
        assert t["状態"] == va.STATUS_UNEVALUABLE
        assert va.REASON_PRICE_MISMATCH in t["理由"]


def test_price_guard_within_tolerance():
    frame, dates, opens, closes = rising_frame()
    fetcher = fake_fetcher({"7203.T": frame, va.TOPIX_PROXY_SYMBOL: topix_frame()})
    rec = make_record(base_date=dates[0], close=closes[0] * 1.01)  # 1% は許容内
    result = va.evaluate([rec], fetcher, run_id="test-run")
    t5 = [t for t in result["trades"] if t["期間"] == 5][0]
    assert t5["状態"] == va.STATUS_EVALUATED


def test_no_base_bar():
    frame, dates, opens, closes = rising_frame()
    fetcher = fake_fetcher({"7203.T": frame, va.TOPIX_PROXY_SYMBOL: topix_frame()})
    rec = make_record(base_date="2026-10-03", close=1000.0)  # 土曜日、系列にない
    result = va.evaluate([rec], fetcher, run_id="test-run")
    for t in result["trades"]:
        assert t["状態"] == va.STATUS_UNEVALUABLE
        assert t["理由"] == va.REASON_NO_BASE_BAR


def test_fetch_failure():
    fetcher = fake_fetcher({va.TOPIX_PROXY_SYMBOL: topix_frame()})  # 対象銘柄なし
    rec = make_record()
    result = va.evaluate([rec], fetcher, run_id="test-run")
    for t in result["trades"]:
        assert t["状態"] == va.STATUS_UNEVALUABLE
        assert va.REASON_FETCH_FAILED in t["理由"]


def test_pending_when_exit_not_reached():
    # 系列が8行しかない: day5 は評価できるが day10 は未到来
    result, dates, opens, closes = standard_setup(n=8)
    t5 = [t for t in result["trades"] if t["期間"] == 5][0]
    t10 = [t for t in result["trades"] if t["期間"] == 10][0]
    assert t5["状態"] == va.STATUS_EVALUATED
    assert t10["状態"] == va.STATUS_PENDING
    assert va.REASON_EXIT_NOT_YET in t10["理由"]


def test_pending_when_entry_not_reached():
    result, dates, opens, closes = standard_setup(n=1)
    for t in result["trades"]:
        assert t["状態"] == va.STATUS_PENDING
        assert t["理由"] == va.REASON_ENTRY_NOT_YET


# ---------------------------------------------------------------
# 対象の分離と重複
# ---------------------------------------------------------------

def test_scope_and_dedupe():
    frame, dates, opens, closes = rising_frame()
    fetcher = fake_fetcher({"7203.T": frame, va.TOPIX_PROXY_SYMBOL: topix_frame()})
    records = [
        make_record(base_date=dates[0], close=closes[0]),
        make_record(base_date=dates[0], close=closes[0]),  # 重複
        make_record(base_date=dates[0], close=closes[0], direction="中立"),
        make_record(base_date=dates[0], close=closes[0], version=2),
    ]
    result = va.evaluate(records, fetcher, run_id="test-run")
    c = result["counts"]
    assert c["入力記録数"] == 4
    assert c["対象外件数"] == 2
    assert c["重複除去件数"] == 1
    assert c["対象記録数"] == 1
    assert len(result["trades"]) == 2  # day5 と day10


# ---------------------------------------------------------------
# 未来情報の混入防止
# ---------------------------------------------------------------

def test_no_lookahead():
    frame, dates, opens, closes = rising_frame()
    fetcher = fake_fetcher({"7203.T": frame, va.TOPIX_PROXY_SYMBOL: topix_frame()})
    rec = make_record(base_date=dates[0], close=closes[0])
    before = va.evaluate([rec], fetcher, run_id="test-run")["trades"]
    # 決済日（day10 = dates[11]）より後の行を大きく変えても結果は変わらない
    closes2 = list(closes)
    opens2 = list(opens)
    for i in range(12, len(closes2)):
        closes2[i] = closes2[i] * 3
        opens2[i] = opens2[i] * 3
    frame2 = make_frame(dates, opens2, closes2)
    fetcher2 = fake_fetcher({"7203.T": frame2, va.TOPIX_PROXY_SYMBOL: topix_frame()})
    after = va.evaluate([rec], fetcher2, run_id="test-run")["trades"]
    assert va.compare_trades(before, after) == []


# ---------------------------------------------------------------
# コストの大小関係
# ---------------------------------------------------------------

def test_cost_case_ordering():
    result, dates, opens, closes = standard_setup()
    t5 = [t for t in result["trades"] if t["期間"] == 5][0]
    assert t5["グロス損益"] > t5["ネット損益_ケース2手数料0"]  # スリッページ分だけ小さい
    assert t5["ネット損益_ケース2手数料0"] > t5["ネット損益_主結果"]  # 手数料分だけ小さい
    assert t5["ネット損益_主結果"] > t5["ネット損益_ストレス"]  # ストレスはさらに小さい
    assert t5["手数料_ケース2"] == 0
    assert t5["手数料_ケース1"] > 0


# ---------------------------------------------------------------
# 実行可能性評価（単元）
# ---------------------------------------------------------------

def test_unit_shortage_separated():
    # エントリー始値 1500 → 単元価格 150,000 > 100,000 で単元不足
    frame, dates, opens, closes = rising_frame(start_close=1505.0)
    fetcher = fake_fetcher({"7203.T": frame, va.TOPIX_PROXY_SYMBOL: topix_frame()})
    rec = make_record(base_date=dates[0], close=closes[0])
    result = va.evaluate([rec], fetcher, run_id="test-run")
    t5 = [t for t in result["trades"] if t["期間"] == 5][0]
    assert t5["状態"] == va.STATUS_EVALUATED  # 理論評価は成立する
    assert t5["単元区分"] == "単元不足"
    assert t5["単元ネット損益_参考"] is None
    s = va.summarize_horizon(result["trades"], 5)
    assert s["実行可能性"]["単元不足件数"] == 1
    assert s["実行可能性"]["購入可能件数"] == 0
    assert s["実行可能性"]["購入可能割合"] == 0.0


def test_unit_feasible():
    # エントリー始値 505 → 単元価格 50,500。100,000 // 50,500 = 1単元 = 100株。
    frame, dates, opens, closes = rising_frame(start_close=500.0)
    fetcher = fake_fetcher({"7203.T": frame, va.TOPIX_PROXY_SYMBOL: topix_frame()})
    rec = make_record(base_date=dates[0], close=closes[0])
    result = va.evaluate([rec], fetcher, run_id="test-run")
    t = [x for x in result["trades"] if x["期間"] == 5][0]
    assert t["単元区分"] == "購入可能"
    assert t["単元価格"] == opens[1] * 100 == 50_500.0
    assert t["単元株数"] == 100
    assert t["単元ネット損益_参考"] is not None
    s = va.summarize_horizon(result["trades"], 5)
    assert s["実行可能性"]["購入可能割合"] == 1.0


# ---------------------------------------------------------------
# DD-1
# ---------------------------------------------------------------

def test_dd1_known_series():
    def trade(day, code, net):
        return {"決済日": day, "コード": code, "ネット損益_主結果": net,
                "状態": va.STATUS_EVALUATED}
    trades = [
        trade("2026-10-06", "7203.T", 100.0),
        trade("2026-10-07", "7203.T", -300.0),
        trade("2026-10-08", "6758.T", -100.0),
        trade("2026-10-08", "7203.T", 50.0),
    ]
    # 累積: +100 → -200 → （同日はコード順）6758.T -100 → -300 → 7203.T +50 → -250
    # ピーク 100 からの最大落差は 100 - (-300) = 400
    assert va.compute_dd1(trades) == 400.0
    assert va.compute_dd1([]) == 0.0


# ---------------------------------------------------------------
# 最低証拠条件
# ---------------------------------------------------------------

def test_evidence_shortfall():
    result, dates, opens, closes = standard_setup()
    s = va.summarize_horizon(result["trades"], 5)
    ev = s["最低証拠条件"]
    assert ev["達成"] is False
    assert ev["現在"]["評価済み取引"] == 1
    assert ev["不足"]["評価済み取引"] == 99
    assert ev["不足"]["コホート"] == 7
    assert ev["不足"]["観測期間日数"] == 60  # 基準日が1日だけなので観測期間は0日
    assert s["判定"] == "判定保留（最低証拠条件未達）"
    assert s["判定条件"] is None


# ---------------------------------------------------------------
# snapshot 往復と改変検出
# ---------------------------------------------------------------

def test_snapshot_roundtrip(tmp_path):
    frame, dates, opens, closes = rising_frame()
    fetcher = fake_fetcher({"7203.T": frame, va.TOPIX_PROXY_SYMBOL: topix_frame()})
    rec = make_record(base_date=dates[0], close=closes[0])
    fetched_at = datetime(2026, 11, 7, 2, 0, 0, tzinfo=timezone.utc)
    result = va.evaluate([rec], fetcher, run_id="test-run", fetched_at=fetched_at)
    snap_dir = str(tmp_path / "snapshot")
    manifest = va.write_snapshot(result, snap_dir)
    assert manifest["manifest_SHA256"]
    assert manifest["ファイル"]["7203.T"]["行数"] == 12  # 基準日〜day10決済日
    assert manifest["ファイル"][va.TOPIX_PROXY_SYMBOL]["行数"] == 12
    assert va.verify_snapshot(snap_dir) == []
    reproduced = va.reevaluate_from_snapshot([rec], snap_dir, run_id="test-run")
    assert va.compare_trades(result["trades"], reproduced["trades"]) == []


def test_snapshot_tamper_detection(tmp_path):
    frame, dates, opens, closes = rising_frame()
    fetcher = fake_fetcher({"7203.T": frame, va.TOPIX_PROXY_SYMBOL: topix_frame()})
    rec = make_record(base_date=dates[0], close=closes[0])
    result = va.evaluate([rec], fetcher, run_id="test-run")
    snap_dir = str(tmp_path / "snapshot")
    va.write_snapshot(result, snap_dir)
    path = tmp_path / "snapshot" / "7203.T.csv"
    data = bytearray(path.read_bytes())
    data[-3] = (data[-3] + 1) % 256  # 1バイト変える
    path.write_bytes(bytes(data))
    problems = va.verify_snapshot(snap_dir)
    assert problems and "7203.T" in problems[0]
    with pytest.raises(ValueError):
        va.reevaluate_from_snapshot([rec], snap_dir, run_id="test-run")


# ---------------------------------------------------------------
# 実行一式（run_evaluation）の出力
# ---------------------------------------------------------------

def test_run_evaluation_outputs(tmp_path):
    import hashlib
    import json
    import os

    frame, dates, opens, closes = rising_frame()
    fetcher = fake_fetcher({"7203.T": frame, va.TOPIX_PROXY_SYMBOL: topix_frame()})
    rec = make_record(base_date=dates[0], close=closes[0])
    out_dir = str(tmp_path / "out")
    run_summary = va.run_evaluation(
        [rec], fetcher, out_dir, run_id="fixture-test-001",
        note="fixture（合成データ）による動作確認",
        fetched_at=datetime(2026, 11, 7, 2, 0, 0, tzinfo=timezone.utc),
    )
    for name in ("verification_a_results.csv", "verification_a_summary.json",
                 "verification_a_summary.md", "run_summary.json",
                 "fixture-test-001_backup.zip"):
        assert os.path.exists(os.path.join(out_dir, name)), name
    assert os.path.exists(os.path.join(out_dir, "snapshot", "manifest.json"))
    # ZIP のハッシュが実行サマリと一致する
    with open(os.path.join(out_dir, "fixture-test-001_backup.zip"), "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == run_summary["ZIP_SHA256"]
    # manifest のハッシュが結果CSV（二重記録）と集計に残っている
    results_text = open(os.path.join(out_dir, "verification_a_results.csv"),
                        encoding="utf-8-sig").read()
    assert run_summary["manifest_SHA256"] in results_text
    summary = json.loads(open(os.path.join(out_dir, "verification_a_summary.json"),
                              encoding="utf-8").read())
    assert summary["manifest_SHA256"] == run_summary["manifest_SHA256"]
    md = open(os.path.join(out_dir, "verification_a_summary.md"), encoding="utf-8").read()
    assert va.DISCLAIMER in md
    assert va.DD1_LABEL in md
    assert "fixture" in md


# ---------------------------------------------------------------
# 登録CSV経由の読み込み（forward.records_from_csv との接続）
# ---------------------------------------------------------------

def test_records_csv_roundtrip_input():
    frame, dates, opens, closes = rising_frame()
    rec = make_record(base_date=dates[0], close=closes[0])
    loaded = forward.records_from_csv(forward.records_to_csv([rec]))
    assert len(loaded) == 1
    fetcher = fake_fetcher({"7203.T": frame, va.TOPIX_PROXY_SYMBOL: topix_frame()})
    result = va.evaluate(loaded, fetcher, run_id="test-run")
    t5 = [t for t in result["trades"] if t["期間"] == 5][0]
    assert t5["状態"] == va.STATUS_EVALUATED
