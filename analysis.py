"""Stock AI Lab の分析ロジック（UIから分離した純粋な計算部分）。

- 売買判定ではなく、テクニカル条件の一致度をルールベースで評価する。
- スコアは条件の一致度であり、利益や将来価格の予測確率ではない。
- データ取得は無料の yfinance のみ。取得できない場合は失敗を明示する。
"""

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

# 市場ごとのタイムゾーン。新旧判定の「今日」は市場の現地日付で数える。
MARKET_TIMEZONES = {
    "JP": ZoneInfo("Asia/Tokyo"),
    "US": ZoneInfo("America/New_York"),
}

# 区分ごとの最低データ本数（初期の運用基準。精度を保証する数字ではない）
MIN_BARS_SHORT = 30
MIN_BARS_MID = 80
MIN_BARS_LONG = 250

NEEDED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# 初期銘柄の会社名（表示専用）。これ以外の銘柄は画面側でYahooから取得を試み、
# 取得できなければコードのみ表示する。分析ロジックには一切使わない。
KNOWN_NAMES = {
    "7203.T": "トヨタ自動車",
    "6758.T": "ソニーグループ",
    "8306.T": "三菱UFJフィナンシャル・グループ",
    "9984.T": "ソフトバンクグループ",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "SPY": "SPDR S&P 500 ETF",
}


def resolve_company_name(symbol, fetched_name=""):
    """表示用の会社名を決める。内蔵の対応表を優先し、次に取得した名前を使う。"""
    if symbol in KNOWN_NAMES:
        return KNOWN_NAMES[symbol]
    return fetched_name or ""


def display_label(symbol, name=""):
    """「7203.T トヨタ自動車」のような表示ラベルを作る。名前が無ければコードのみ。"""
    return f"{symbol} {name}" if name else symbol


def select_label(symbol, name=""):
    """選択欄用に「トヨタ自動車（7203.T）」のような表示を作る。名前が無ければコードのみ。"""
    return f"{name}（{symbol}）" if name else symbol


def fetch_history(symbol, period="2y"):
    """株価の日足を取得する。

    戻り値: (DataFrame, error_message)
    成功時は (データ, None)、失敗時は (空のDataFrame, 理由の説明)。
    例外は外に出さず、画面全体が止まらないようにする。
    """
    try:
        df = yf.download(
            symbol, period=period, auto_adjust=True, progress=False, threads=False
        )
    except Exception as exc:  # 通信エラー・取得制限など
        return pd.DataFrame(), f"データ取得でエラーが発生しました（{type(exc).__name__}: {str(exc)[:120]}）"
    if df is None or df.empty:
        return pd.DataFrame(), "データが取得できませんでした。銘柄コードの誤り、または提供側でデータがない可能性があります。"
    try:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not all(c in df.columns for c in NEEDED_COLUMNS):
            return pd.DataFrame(), "取得したデータに必要な列（始値・高値・安値・終値・出来高）がそろっていません。"
        df = df[NEEDED_COLUMNS].dropna().copy()
    except Exception as exc:
        return pd.DataFrame(), f"取得データの整形でエラーが発生しました（{type(exc).__name__}: {str(exc)[:120]}）"
    if df.empty:
        return pd.DataFrame(), "取得したデータが空でした（欠損のみ）。"
    return df, None


def add_indicators(df):
    """テクニカル指標を付ける。EMA200は200本そろうまで NaN（有効でない）とする。"""
    x = df.copy()
    x["EMA5"] = x["Close"].ewm(span=5, adjust=False).mean()
    x["EMA20"] = x["Close"].ewm(span=20, adjust=False).mean()
    x["EMA50"] = x["Close"].ewm(span=50, adjust=False).mean()
    # min_periods=200: 200本未満では EMA200 を出さない（育っていない値を判定に使わない）
    x["EMA200"] = x["Close"].ewm(span=200, adjust=False, min_periods=200).mean()
    delta = x["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    # RSIのゼロケースを明示する：
    # - 平均損失が0で平均上昇が正（上昇のみ）→ RSIは100
    # - 平均上昇が0で平均損失が正（下落のみ）→ RSIは0
    # - 値動きがまったくない（両方0）→ 勢いを判断できないため欠損（NaN）とし、
    #   スコアには反映しない（assess_short は NaN のときRSIを評価しない）
    rs = gain / loss.where(loss > 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(~((loss == 0) & (gain > 0)), 100.0)
    rsi = rsi.where(~((gain == 0) & (loss > 0)), 0.0)
    rsi = rsi.where(~((gain == 0) & (loss == 0)), np.nan)
    x["RSI14"] = rsi
    ema12 = x["Close"].ewm(span=12, adjust=False).mean()
    ema26 = x["Close"].ewm(span=26, adjust=False).mean()
    x["MACD"] = ema12 - ema26
    x["MACD_Signal"] = x["MACD"].ewm(span=9, adjust=False).mean()
    x["Return_1d_pct"] = x["Close"].pct_change() * 100
    x["Volatility20_pct"] = x["Return_1d_pct"].rolling(20).std()
    x["Volume_MA20"] = x["Volume"].rolling(20).mean()
    return x


def _return_pct(df, bars):
    """直近 bars 営業日の騰落率（%）。本数が足りなければ None。"""
    if len(df) <= bars:
        return None
    base = df["Close"].iloc[-bars - 1]
    last = df["Close"].iloc[-1]
    if base == 0 or pd.isna(base) or pd.isna(last):
        return None
    return (last / base - 1.0) * 100.0


def _label_from_score(score):
    if score >= 2:
        return "上向き条件が多い"
    if score <= -2:
        return "下向き条件が多い"
    return "混在・中立"


def _unavailable(horizon, needed, actual):
    return {
        "horizon": horizon,
        "available": False,
        "score": None,
        "label": "判定保留（データ不足）",
        "reasons": [f"{horizon}の判定には最低{needed}本のデータが必要です（現在{actual}本）。"],
        "warnings": [],
    }


def assess_short(df):
    """短期：EMA5とEMA20の位置関係、直近5営業日の値動き、RSI14の勢い。"""
    horizon = "短期"
    if len(df) < MIN_BARS_SHORT:
        return _unavailable(horizon, MIN_BARS_SHORT, len(df))
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

    r5 = _return_pct(df, 5)
    if r5 is not None:
        if r5 > 1.0:
            score += 1
            reasons.append(f"直近5営業日で {r5:+.1f}% 上昇")
        elif r5 < -1.0:
            score -= 1
            reasons.append(f"直近5営業日で {r5:+.1f}% 下落")
        else:
            reasons.append(f"直近5営業日は {r5:+.1f}% と小動き")

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
        "label": _label_from_score(score), "reasons": reasons, "warnings": warnings,
    }


def assess_mid(df):
    """中期：EMA20とEMA50の位置関係、株価とEMA20、直近20・60営業日の値動き。"""
    horizon = "中期"
    if len(df) < MIN_BARS_MID:
        return _unavailable(horizon, MIN_BARS_MID, len(df))
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

    r20 = _return_pct(df, 20)
    if r20 is not None:
        if r20 > 2.0:
            score += 1
            reasons.append(f"直近20営業日で {r20:+.1f}% 上昇")
        elif r20 < -2.0:
            score -= 1
            reasons.append(f"直近20営業日で {r20:+.1f}% 下落")
        else:
            reasons.append(f"直近20営業日は {r20:+.1f}%")

    r60 = _return_pct(df, 60)
    if r60 is not None:
        if r60 > 3.0:
            score += 1
            reasons.append(f"直近60営業日で {r60:+.1f}% 上昇")
        elif r60 < -3.0:
            score -= 1
            reasons.append(f"直近60営業日で {r60:+.1f}% 下落")
        else:
            reasons.append(f"直近60営業日は {r60:+.1f}%")

    # 押し目・戻りの参考情報（スコアには入れない）
    r5 = _return_pct(df, 5)
    if r5 is not None:
        if trend_up and r5 < -1.0:
            reasons.append("参考：中期トレンドは上向きですが直近5日は下げており、押し目の可能性があります（断定ではありません）")
        elif not trend_up and last["EMA20"] < last["EMA50"] and r5 > 1.0:
            reasons.append("参考：中期トレンドは下向きですが直近5日は戻しており、戻りの可能性があります（断定ではありません）")

    return {
        "horizon": horizon, "available": True, "score": score,
        "label": _label_from_score(score), "reasons": reasons, "warnings": warnings,
    }


def assess_long(df):
    """長期：EMA50とEMA200の位置関係、株価とEMA200、直近60・120営業日の値動き。

    日足250本以上があり、EMA200が有効な値（NaNでない）場合のみ判定する。
    """
    horizon = "長期"
    if len(df) < MIN_BARS_LONG:
        result = _unavailable(horizon, MIN_BARS_LONG, len(df))
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

    r60 = _return_pct(df, 60)
    if r60 is not None:
        if r60 > 3.0:
            score += 1
            reasons.append(f"直近60営業日で {r60:+.1f}% 上昇")
        elif r60 < -3.0:
            score -= 1
            reasons.append(f"直近60営業日で {r60:+.1f}% 下落")
        else:
            reasons.append(f"直近60営業日は {r60:+.1f}%")

    r120 = _return_pct(df, 120)
    if r120 is not None:
        if r120 > 5.0:
            score += 1
            reasons.append(f"直近120営業日で {r120:+.1f}% 上昇")
        elif r120 < -5.0:
            score -= 1
            reasons.append(f"直近120営業日で {r120:+.1f}% 下落")
        else:
            reasons.append(f"直近120営業日は {r120:+.1f}%")

    label = _label_from_score(score)
    if score >= 2:
        label = "長期トレンドは上向き（条件一致が多い）"
    elif score <= -2:
        label = "長期トレンドは下向き（条件一致が多い）"
    return {
        "horizon": horizon, "available": True, "score": score,
        "label": label, "reasons": reasons, "warnings": warnings,
    }


def assess_all(df):
    """短期・中期・長期の評価をまとめて返す。"""
    return {
        "短期": assess_short(df),
        "中期": assess_mid(df),
        "長期": assess_long(df),
    }


def market_of(symbol):
    """銘柄コードから市場を推定する（.T は日本、それ以外は米国として扱う）。"""
    return "JP" if str(symbol).upper().endswith(".T") else "US"


def market_today(symbol, now_utc=None):
    """市場の現地タイムゾーンでの「今日」の日付を返す。

    実行環境の日付（サーバーのタイムゾーン）ではなく、市場の日付で数える。
    日本株は Asia/Tokyo、米国株は America/New_York を使う。
    now_utc を渡すと、その時刻（UTC）から市場日付を求める（テスト用）。
    """
    tz = MARKET_TIMEZONES[market_of(symbol)]
    if now_utc is None:
        return datetime.now(tz).date()
    return now_utc.astimezone(tz).date()


def freshness_check(df, symbol, today=None):
    """データの新しさを暦日で暫定判定する。

    「今日」は市場の現地日付（market_today）で数える。日本時間の日付だけで
    米国市場のデータを判定することはしない。
    公式の営業日カレンダーとは照合していない暫定判定であることを、
    戻り値の note に必ず含める。
    today を渡した場合は、その日付を市場の現地日付として扱う（テスト用）。
    戻り値: dict(status, last_date, rows, age_days, message, note)
      status: "fresh"（新しい）/ "check"（やや間隔あり）/ "stale"（古い可能性）/ "no_data"
    """
    note = "この新旧判定は暦日による暫定チェックです。公式の営業日カレンダー（祝日・臨時休場）とは照合していません。"
    if df is None or df.empty:
        return {
            "status": "no_data", "last_date": None, "rows": 0,
            "age_days": None, "message": "データがありません。", "note": note,
        }
    last_date = df.index[-1].date()
    if today is None:
        today = market_today(symbol)
    age = (today - last_date).days
    rows = len(df)
    if age <= 1:
        status = "fresh"
        message = f"データ最終日は {last_date} です（本日から{age}日前）。"
    elif age <= 3:
        status = "check"
        message = f"データ最終日は {last_date} です（本日から{age}日前）。週末・祝日を挟んだ可能性があります。"
    else:
        status = "stale"
        message = (
            f"データ最終日は {last_date} で、本日から{age}日経っています。"
            "市場の休場だけでは説明しにくい間隔があり、データが古い可能性があります。"
            "このデータによる分析は慎重に扱ってください。"
        )
    return {
        "status": status, "last_date": last_date, "rows": rows,
        "age_days": age, "message": message, "note": note,
    }
