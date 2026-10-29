"""スキャン機能（scan.py）の単体テスト。ネットワークは使わない。"""

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
