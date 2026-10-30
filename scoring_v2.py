"""実験版スコア V2（SCORING_VERSION_V2 = 2）。凍結仕様。

V1（analysis.py、SCORING_VERSION = 1）の条件を土台に、次の3点だけを変更する。
V1 のコード・条件は一切変更しない。V2 の条件を変える場合はこのファイルを
書き換えず、scoring_v3.py を新設して版数を上げる（不変ルール）。

1. リターン条件のボラティリティ正規化
   期間リターンの固定％閾値の代わりに、z = 期間リターン（%）÷
   （Volatility20_pct（%/日）× √対象営業日数）を使い、z > 0.3 で +1、
   z < −0.3 で −1 とする。適用先は短期の5日、中期の20日・60日、
   長期の60日・120日。EMA 条件と短期の RSI 条件は V1 のまま。
   Volatility20_pct は analysis.add_indicators の既存の定義をそのまま使う
   （日次リターンの直近20本の標本標準偏差、%/日、年率換算なし）。
   閾値 0.3 は 2023年までの設計期間で校正した値であり、フォワードデータで
   再調整しない。
   Volatility20 が NaN または 0 の場合、その条件は 0 点とし理由に残す。

2. 下落の分離
   スコアは V1 と同じ範囲で計算するが、予測方向の対応を変える：
   スコア −2 以下は「下落」ではなく「下落注意」とする（direction_from_score_v2）。
   下落注意は方向予測ではなく、的中率の集計対象から外す参考情報として扱う。

3. 過熱フラグ（表示のみ）
   overheat_flag：基準日の終値が EMA20 から +5% 超乖離している、または
   ボリンジャーバンド（20日・2σ、標本標準偏差）上限を終値が上回っている
   場合に True。スコア・方向・順位には影響しない。
"""

import numpy as np
import pandas as pd

import analysis

SCORING_VERSION_V2 = 2
Z_THRESHOLD = 0.3
OVERHEAT_DEVIATION_PCT = 5.0
BB_WINDOW = 20
BB_SIGMA = 2.0


def _z_return_points(df, bars):
    """正規化した期間リターン条件の点数と理由を返す。

    戻り値: (points, reason)。リターン自体が取れない場合は (0, None)
    （V1 が理由を残さない扱いに合わせる）。
    """
    r = analysis._return_pct(df, bars)
    if r is None:
        return 0, None
    vol = df.iloc[-1]["Volatility20_pct"]
    if pd.isna(vol) or vol <= 0:
        return 0, (
            f"直近{bars}営業日は {r:+.1f}%"
            "（ボラティリティが算出できないため正規化条件は無効・0点）"
        )
    z = r / (vol * np.sqrt(bars))
    if z > Z_THRESHOLD:
        return 1, f"直近{bars}営業日で {r:+.1f}% 上昇（変動比 z={z:+.2f}）"
    if z < -Z_THRESHOLD:
        return -1, f"直近{bars}営業日で {r:+.1f}% 下落（変動比 z={z:+.2f}）"
    return 0, f"直近{bars}営業日は {r:+.1f}%（変動比 z={z:+.2f}、基準内）"


def assess_short_v2(df):
    """短期 V2：V1 と同じ EMA5/20 と RSI、5日リターンだけ正規化。"""
    horizon = "短期"
    if len(df) < analysis.MIN_BARS_SHORT:
        return analysis._unavailable(horizon, analysis.MIN_BARS_SHORT, len(df))
    last = df.iloc[-1]
    score = 0
    reasons = []
    warnings = []

    if last["EMA5"] > last["EMA20"]:
        score += 1
        reasons.append("EMA5がEMA20より上（短期の勢いは上向き）")
    elif last["EMA5"] < last["EMA20"]:
        score -= 1
        reasons.append("EMA5がEMA20より下（短期の勢いは下向き）")
    else:
        reasons.append("EMA5とEMA20がほぼ同じ水準")

    points, reason = _z_return_points(df, 5)
    score += points
    if reason is not None:
        reasons.append(reason)

    rsi = last["RSI14"]
    if pd.notna(rsi):
        if 50 <= rsi <= 70:
            score += 1
            reasons.append(f"RSI14は {rsi:.0f}（上昇モメンタムの範囲）")
        elif rsi > 75:
            score -= 1
            reasons.append(f"RSI14は {rsi:.0f}（高水準）")
            warnings.append("RSIが高く、買われ過ぎによる反落の可能性があります。")
        elif rsi > 70:
            reasons.append(f"RSI14は {rsi:.0f}（やや高め）")
            warnings.append("RSIが70を超えています。過熱に注意してください。")
        elif rsi < 30:
            reasons.append(f"RSI14は {rsi:.0f}（低水準）")
            warnings.append("RSIが低く売られ過ぎの水準です。ただし反発を保証するものではありません。")
        else:
            reasons.append(f"RSI14は {rsi:.0f}（中立〜弱め）")

    return {
        "horizon": horizon, "available": True, "score": score,
        "label": analysis._label_from_score(score), "reasons": reasons, "warnings": warnings,
    }


def assess_mid_v2(df):
    """中期 V2：V1 と同じ EMA 条件、20日・60日リターンだけ正規化。"""
    horizon = "中期"
    if len(df) < analysis.MIN_BARS_MID:
        return analysis._unavailable(horizon, analysis.MIN_BARS_MID, len(df))
    last = df.iloc[-1]
    score = 0
    reasons = []
    warnings = []

    trend_up = last["EMA20"] > last["EMA50"]
    if trend_up:
        score += 1
        reasons.append("EMA20がEMA50より上（中期トレンドは上向き）")
    elif last["EMA20"] < last["EMA50"]:
        score -= 1
        reasons.append("EMA20がEMA50より下（中期トレンドは下向き）")
    else:
        reasons.append("EMA20とEMA50がほぼ同じ水準")

    if last["Close"] > last["EMA20"]:
        score += 1
        reasons.append("終値がEMA20より上")
    elif last["Close"] < last["EMA20"]:
        score -= 1
        reasons.append("終値がEMA20より下")
    else:
        reasons.append("終値がEMA20とほぼ同じ水準")

    for bars in (20, 60):
        points, reason = _z_return_points(df, bars)
        score += points
        if reason is not None:
            reasons.append(reason)

    # 押し目・戻りの参考情報（スコアには入れない。V1 と同じ）
    r5 = analysis._return_pct(df, 5)
    if r5 is not None:
        if trend_up and r5 < -1.0:
            reasons.append("参考：中期トレンドは上向きですが直近5日は下げており、押し目の可能性があります（断定ではありません）")
        elif not trend_up and last["EMA20"] < last["EMA50"] and r5 > 1.0:
            reasons.append("参考：中期トレンドは下向きですが直近5日は戻しており、戻りの可能性があります（断定ではありません）")

    return {
        "horizon": horizon, "available": True, "score": score,
        "label": analysis._label_from_score(score), "reasons": reasons, "warnings": warnings,
    }


def assess_long_v2(df):
    """長期 V2：V1 と同じ EMA 条件、60日・120日リターンだけ正規化。"""
    horizon = "長期"
    if len(df) < analysis.MIN_BARS_LONG:
        result = analysis._unavailable(horizon, analysis.MIN_BARS_LONG, len(df))
        result["label"] = "データ不足のため長期判定不可"
        return result
    last = df.iloc[-1]
    if pd.isna(last["EMA200"]):
        return {
            "horizon": horizon, "available": False, "score": None,
            "label": "データ不足のため長期判定不可",
            "reasons": ["EMA200の計算に必要な履歴（200本以上）がそろっていません。"],
            "warnings": [],
        }
    score = 0
    reasons = []
    warnings = []

    if last["EMA50"] > last["EMA200"]:
        score += 1
        reasons.append("EMA50がEMA200より上（長期トレンドは上向き）")
    elif last["EMA50"] < last["EMA200"]:
        score -= 1
        reasons.append("EMA50がEMA200より下（長期トレンドは下向き）")
    else:
        reasons.append("EMA50とEMA200がほぼ同じ水準")

    if last["Close"] > last["EMA200"]:
        score += 1
        reasons.append("終値がEMA200より上")
    elif last["Close"] < last["EMA200"]:
        score -= 1
        reasons.append("終値がEMA200より下")

    for bars in (60, 120):
        points, reason = _z_return_points(df, bars)
        score += points
        if reason is not None:
            reasons.append(reason)

    label = analysis._label_from_score(score)
    if score >= 2:
        label = "長期トレンドは上向き（条件一致が多い）"
    elif score <= -2:
        label = "長期トレンドは下向き（条件一致が多い）"
    return {
        "horizon": horizon, "available": True, "score": score,
        "label": label, "reasons": reasons, "warnings": warnings,
    }


def assess_all_v2(df):
    """短期・中期・長期の V2 評価をまとめて返す（V1 の assess_all と同形式）。"""
    return {
        "短期": assess_short_v2(df),
        "中期": assess_mid_v2(df),
        "長期": assess_long_v2(df),
    }


def direction_from_score_v2(score):
    """V2 の予測方向。スコア −2 以下は「下落注意」（方向予測ではない）。"""
    if score is None:
        return "判定保留"
    if score >= 2:
        return "上昇"
    if score <= -2:
        return "下落注意"
    return "中立"


def overheat_flag(df):
    """過熱フラグ（表示のみ）。df は add_indicators 適用後のフレーム。"""
    if df is None or len(df) < BB_WINDOW:
        return False
    last = df.iloc[-1]
    close = float(last["Close"])
    ema20 = last["EMA20"]
    if pd.notna(ema20) and ema20 > 0:
        if (close / float(ema20) - 1.0) * 100.0 > OVERHEAT_DEVIATION_PCT:
            return True
    closes20 = df["Close"].iloc[-BB_WINDOW:]
    if len(closes20) == BB_WINDOW:
        upper = closes20.mean() + BB_SIGMA * closes20.std(ddof=1)
        if pd.notna(upper) and close > float(upper):
            return True
    return False
