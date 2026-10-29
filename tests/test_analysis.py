"""analysis.py の単体テスト。合成データで決定的に検証する。"""
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis


def make_frame(closes, start="2024-01-02"):
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range(start=start, periods=len(closes))
    return pd.DataFrame(
        {
            "Open": closes, "High": closes * 1.01, "Low": closes * 0.99,
            "Close": closes, "Volume": np.full(len(closes), 1_000_000.0),
        },
        index=idx,
    )


def rising(n, start_price=100.0, step=0.5):
    return make_frame([start_price + step * i for i in range(n)])


def test_long_unavailable_under_250_bars():
    df = analysis.add_indicators(rising(120))
    result = analysis.assess_long(df)
    assert result["available"] is False
    assert result["label"] == "データ不足のため長期判定不可"
    assert result["score"] is None


def test_mid_and_short_available_at_120_bars():
    df = analysis.add_indicators(rising(120))
    assert analysis.assess_short(df)["available"] is True
    assert analysis.assess_mid(df)["available"] is True


def test_all_unavailable_under_30_bars():
    df = analysis.add_indicators(rising(25))
    results = analysis.assess_all(df)
    assert all(r["available"] is False for r in results.values())


def test_ema200_is_nan_before_200_bars():
    df = analysis.add_indicators(rising(300))
    assert pd.isna(df["EMA200"].iloc[150])
    assert not pd.isna(df["EMA200"].iloc[250])


def test_horizons_differ_on_recent_dip_in_uptrend():
    # 長期上昇の末尾で急落：短期は下向き、長期は上向きが残るはず
    closes = [100.0 + 0.6 * i for i in range(290)] + [274.0 - 3.0 * i for i in range(10)]
    df = analysis.add_indicators(make_frame(closes))
    short = analysis.assess_short(df)
    long = analysis.assess_long(df)
    assert short["available"] and long["available"]
    assert short["score"] < 0
    assert long["score"] > 0
    assert short["label"] != long["label"]


def test_rising_series_scores_positive():
    df = analysis.add_indicators(rising(300))
    results = analysis.assess_all(df)
    assert all(r["available"] for r in results.values())
    assert results["中期"]["score"] >= 2
    assert results["長期"]["score"] >= 2
    # 一方通行の上昇ではRSIが100になり、短期は過熱として注意が付き、
    # スコアが割り引かれる（RSIのゼロケース修正後の正しい動き）
    assert results["短期"]["score"] >= 1
    assert any("買われ過ぎ" in w for w in results["短期"]["warnings"])


def test_freshness_levels():
    today = date(2026, 10, 29)
    df_today = make_frame([100.0] * 40, start="2026-09-03")
    df_today.index = pd.bdate_range(end="2026-10-29", periods=40)
    fresh = analysis.freshness_check(df_today, "7203.T", today=today)
    assert fresh["status"] == "fresh"
    assert fresh["rows"] == 40

    df_old = make_frame([100.0] * 40)
    df_old.index = pd.bdate_range(end="2026-10-19", periods=40)
    stale = analysis.freshness_check(df_old, "7203.T", today=today)
    assert stale["status"] == "stale"
    assert "暫定" in stale["note"]

    empty = analysis.freshness_check(pd.DataFrame(), "7203.T", today=today)
    assert empty["status"] == "no_data"


def test_fetch_empty_result_returns_error(monkeypatch):
    # 通信をモック化し、空データが返るケースを決定的に検証する（実通信なし）
    monkeypatch.setattr(analysis, "FETCH_RETRY_WAIT_SECONDS", 0)
    monkeypatch.setattr(analysis.yf, "download", lambda *a, **k: pd.DataFrame())
    df, err = analysis.fetch_history("INVALIDXYZ123", period="1y")
    assert df.empty
    assert err is not None


def test_fetch_none_result_returns_error(monkeypatch):
    monkeypatch.setattr(analysis, "FETCH_RETRY_WAIT_SECONDS", 0)
    monkeypatch.setattr(analysis.yf, "download", lambda *a, **k: None)
    df, err = analysis.fetch_history("AAPL", period="1y")
    assert df.empty
    assert err is not None


def test_fetch_exception_is_caught(monkeypatch):
    def boom(*args, **kwargs):
        raise ConnectionError("network down")

    monkeypatch.setattr(analysis, "FETCH_RETRY_WAIT_SECONDS", 0)
    monkeypatch.setattr(analysis.yf, "download", boom)
    df, err = analysis.fetch_history("AAPL", period="1y")
    assert df.empty
    assert err is not None and "エラー" in err


def test_fetch_retries_once_and_succeeds(monkeypatch):
    # 1回目は例外、2回目は成功するケースで、再試行によりデータが取れること
    monkeypatch.setattr(analysis, "FETCH_RETRY_WAIT_SECONDS", 0)
    good = make_frame([100.0] * 40)
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("temporary")
        return good

    monkeypatch.setattr(analysis.yf, "download", flaky)
    df, err = analysis.fetch_history("AAPL", period="1y")
    assert err is None
    assert len(df) == 40
    assert calls["n"] == 2


def test_fetch_retries_empty_result_once(monkeypatch):
    # 1回目が空（取得制限など）でも、2回目で成功すればデータが取れること
    monkeypatch.setattr(analysis, "FETCH_RETRY_WAIT_SECONDS", 0)
    good = make_frame([100.0] * 40)
    calls = {"n": 0}

    def empty_then_good(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return pd.DataFrame()
        return good

    monkeypatch.setattr(analysis.yf, "download", empty_then_good)
    df, err = analysis.fetch_history("7203.T", period="1y")
    assert err is None
    assert len(df) == 40
    assert calls["n"] == 2


def test_rsi_rising_only_is_100():
    df = analysis.add_indicators(rising(300))
    assert df["RSI14"].iloc[-1] == 100.0


def test_rsi_falling_only_is_0():
    df = analysis.add_indicators(rising(300, start_price=400.0, step=-0.5))
    assert df["RSI14"].iloc[-1] == 0.0


def test_rsi_flat_is_nan_and_not_scored():
    df = analysis.add_indicators(make_frame([100.0] * 300))
    assert pd.isna(df["RSI14"].iloc[-1])
    short = analysis.assess_short(df)
    # 値動きがない場合、RSIは欠損として扱い、スコアに反映しない
    assert not any("RSI" in r for r in short["reasons"])
    assert short["score"] == 0


def test_rsi_normal_data_within_range():
    rng = np.random.default_rng(42)
    closes = 100.0 + np.cumsum(rng.normal(0, 1.5, 300))
    df = analysis.add_indicators(make_frame(closes))
    rsi = df["RSI14"].iloc[-1]
    assert 0.0 < rsi < 100.0


def test_market_today_uses_market_timezone():
    # 2026-10-29 16:30 UTC は、日本では10月30日未明、ニューヨークでは10月29日昼
    now_utc = datetime(2026, 10, 29, 16, 30, tzinfo=timezone.utc)
    assert analysis.market_today("7203.T", now_utc=now_utc) == date(2026, 10, 30)
    assert analysis.market_today("AAPL", now_utc=now_utc) == date(2026, 10, 29)


def test_freshness_weekend_gap_is_check_not_stale():
    # 米国市場：金曜のデータに対して月曜（市場日付）では「間隔あり」まで
    df_us = make_frame([100.0] * 40)
    df_us.index = pd.bdate_range(end="2026-10-23", periods=40)  # 金曜
    result = analysis.freshness_check(df_us, "AAPL", today=date(2026, 10, 26))  # 月曜
    assert result["status"] == "check"
    assert result["age_days"] == 3


def test_freshness_us_recent_close_is_fresh():
    # 米国市場：水曜の終値データに対して木曜（市場日付）では新しい
    df_us = make_frame([100.0] * 40)
    df_us.index = pd.bdate_range(end="2026-10-28", periods=40)
    result = analysis.freshness_check(df_us, "AAPL", today=date(2026, 10, 29))
    assert result["status"] == "fresh"


def test_freshness_jp_monday_after_friday_is_check():
    # 日本市場：金曜のデータに対して月曜では「間隔あり」まで
    df_jp = make_frame([100.0] * 40)
    df_jp.index = pd.bdate_range(end="2026-10-23", periods=40)
    result = analysis.freshness_check(df_jp, "7203.T", today=date(2026, 10, 26))
    assert result["status"] == "check"


def test_company_name_known_jp():
    assert analysis.resolve_company_name("7203.T") == "トヨタ自動車"


def test_company_name_known_us():
    assert analysis.resolve_company_name("AAPL") == "Apple"


def test_company_name_map_wins_over_fetched():
    assert (
        analysis.resolve_company_name("7203.T", "Toyota Motor Corporation")
        == "トヨタ自動車"
    )


def test_company_name_unknown_uses_fetched():
    assert analysis.resolve_company_name("9999.T", "Example Corp") == "Example Corp"


def test_company_name_unknown_without_name_is_empty():
    assert analysis.resolve_company_name("9999.T") == ""


def test_display_label_with_name():
    assert analysis.display_label("7203.T", "トヨタ自動車") == "7203.T トヨタ自動車"


def test_display_label_without_name():
    assert analysis.display_label("9999.T") == "9999.T"


def test_select_label_with_name():
    assert analysis.select_label("7203.T", "トヨタ自動車") == "トヨタ自動車（7203.T）"


def test_select_label_without_name():
    assert analysis.select_label("9999.T") == "9999.T"
