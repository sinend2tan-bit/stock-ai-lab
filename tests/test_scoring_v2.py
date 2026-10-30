"""scoring_v2.py の単体テスト。合成データで決定的に検証する。

z の境界など条件を分離して検証するため、add_indicators の後に最終行の
指標値（EMA・RSI・Volatility20）を上書きして評価関数へ渡すケースを含む。
評価関数は最終行の指標と終値系列だけを読むため、この方法で各条件を
独立に検証できる。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis
import scoring_v2


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


def enriched(closes):
    return analysis.add_indicators(make_frame(closes))


def set_last(df, **values):
    for col, value in values.items():
        df.iloc[-1, df.columns.get_loc(col)] = value
    return df


def flat(n, price=100.0):
    return [price] * n


def test_spec_constants_frozen():
    assert scoring_v2.SCORING_VERSION_V2 == 2
    assert scoring_v2.Z_THRESHOLD == 0.3


def test_short_z_boundary():
    # r5 = +0.7%、vol = 1.0 → z = 0.7 / (1.0 × √5) ≒ 0.313 > 0.3 で +1
    closes = flat(40)
    closes[-1] = 100.7
    df = set_last(enriched(closes), EMA5=100.0, EMA20=100.0, RSI14=40.0, Volatility20_pct=1.0)
    assert scoring_v2.assess_short_v2(df)["score"] == 1
    # r5 = +0.6% → z ≒ 0.268 < 0.3 で 0 点
    closes[-1] = 100.6
    df = set_last(enriched(closes), EMA5=100.0, EMA20=100.0, RSI14=40.0, Volatility20_pct=1.0)
    assert scoring_v2.assess_short_v2(df)["score"] == 0


def test_short_high_vol_suppresses_raw_band():
    # V1 なら +1 になる r5 = +2.0% でも、vol = 3.0 では z ≒ 0.298 で 0 点
    closes = flat(40)
    closes[-1] = 102.0
    df = set_last(enriched(closes), EMA5=100.0, EMA20=100.0, RSI14=40.0, Volatility20_pct=3.0)
    assert analysis.assess_short(df)["score"] == 1
    assert scoring_v2.assess_short_v2(df)["score"] == 0


def test_short_vol_nan_and_zero_are_invalid():
    closes = flat(40)
    closes[-1] = 102.0
    for vol in (np.nan, 0.0):
        df = set_last(enriched(closes), EMA5=100.0, EMA20=100.0, RSI14=40.0, Volatility20_pct=vol)
        result = scoring_v2.assess_short_v2(df)
        assert result["score"] == 0
        assert any("無効" in reason for reason in result["reasons"])


def test_short_negative_z_gives_minus_one():
    closes = flat(40)
    closes[-1] = 98.0
    df = set_last(enriched(closes), EMA5=100.0, EMA20=100.0, RSI14=40.0, Volatility20_pct=0.5)
    assert scoring_v2.assess_short_v2(df)["score"] == -1


def test_mid_z_conditions():
    # r20 = r60 = +1.0%、vol = 0.5 → z20 ≒ 0.447 で +1、z60 ≒ 0.258 で 0
    closes = flat(100)
    closes[-1] = 101.0
    df = set_last(enriched(closes), EMA20=101.0, EMA50=101.0, Volatility20_pct=0.5)
    assert scoring_v2.assess_mid_v2(df)["score"] == 1
    # V1 は固定帯（+2% / +3%）に届かず 0 点
    assert analysis.assess_mid(df)["score"] == 0


def test_long_z_conditions():
    # r60 = r120 = +2.9%、vol = 0.4 → z60 ≒ 0.936、z120 ≒ 0.662 で合計 +2
    closes = flat(260)
    closes[-1] = 102.9
    df = set_last(enriched(closes), EMA50=102.9, EMA200=102.9, Volatility20_pct=0.4)
    assert scoring_v2.assess_long_v2(df)["score"] == 2
    # V1 は固定帯（+3.0% / +5.0%）に届かず 0 点
    assert analysis.assess_long(df)["score"] == 0


def test_long_unavailable_when_ema200_nan():
    df = set_last(enriched(flat(260)), EMA200=np.nan)
    result = scoring_v2.assess_long_v2(df)
    assert result["available"] is False
    assert result["score"] is None


def test_unavailable_under_min_bars():
    df = enriched(flat(10))
    assert scoring_v2.assess_short_v2(df)["available"] is False
    assert scoring_v2.assess_mid_v2(df)["available"] is False
    assert scoring_v2.assess_long_v2(df)["available"] is False


def test_direction_mapping():
    assert scoring_v2.direction_from_score_v2(2) == "上昇"
    assert scoring_v2.direction_from_score_v2(5) == "上昇"
    assert scoring_v2.direction_from_score_v2(-2) == "下落注意"
    assert scoring_v2.direction_from_score_v2(-5) == "下落注意"
    assert scoring_v2.direction_from_score_v2(1) == "中立"
    assert scoring_v2.direction_from_score_v2(0) == "中立"
    assert scoring_v2.direction_from_score_v2(None) == "判定保留"


def test_overheat_by_deviation():
    closes = flat(40)
    closes[-1] = 106.0
    df = set_last(enriched(closes), EMA20=100.0)
    assert scoring_v2.overheat_flag(df) is True


def test_overheat_by_bollinger_only():
    closes = flat(40)
    closes[-1] = 103.0
    # 乖離は 103 / 102.5 - 1 ≒ +0.49% で基準未満。バンド上限突破のみで判定する
    df = set_last(enriched(closes), EMA20=102.5)
    assert scoring_v2.overheat_flag(df) is True


def test_overheat_false_when_calm():
    df = set_last(enriched(flat(40)), EMA20=100.0)
    assert scoring_v2.overheat_flag(df) is False


def test_overheat_false_when_short_history():
    assert scoring_v2.overheat_flag(enriched(flat(10))) is False


def test_parity_with_v1_when_returns_are_zero():
    # 上昇後に横ばいが続く系列では、測定区間のリターンが全て 0 になり、
    # リターン条件は両版とも 0 点。EMA・RSI 条件は共通なのでスコアが一致する
    closes = [100.0 + 0.5 * i for i in range(170)] + [185.0] * 130
    df = enriched(closes)
    v1 = analysis.assess_all(df)
    v2 = scoring_v2.assess_all_v2(df)
    for horizon in ("短期", "中期", "長期"):
        assert v1[horizon]["available"] is True
        assert v2[horizon]["score"] == v1[horizon]["score"]


def test_parity_on_flat_frame():
    df = enriched(flat(300))
    v1 = analysis.assess_all(df)
    v2 = scoring_v2.assess_all_v2(df)
    for horizon in ("短期", "中期", "長期"):
        assert v2[horizon]["score"] == v1[horizon]["score"] == 0
