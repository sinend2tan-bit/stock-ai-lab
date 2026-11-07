"""検証A：短期フォワード仮想売買評価（Stock AI Lab 検証計画 修正版4、2.1/2.5/2.6/2.6.1/2.8/2.10）。

既存の forward.py の評価（チェックポイントのリターン）には一切手を入れず、
その上に重ねる独立した集計モジュール。既存ファイルは変更しない。

対象と定義（本モジュールで固定する定義）:
- 対象はスコア仕様バージョン=1（V1）の登録記録のうち、短期予測方向が
  「上昇」のものだけ。中期・長期・V2 は対象外。
- エントリーは株価基準日（シグナル日）の翌営業日の始値。決済はエントリー日を
  day0 として day5 の終値（主評価）、day10 の終値（補助評価）。
- 価格系列は評価時に fetcher で再取得する（auto_adjust=True の調整済み日足）。
  再取得系列の基準日終値が記録時終値と 2% を超えて異なる記録は
  「評価不能（価格照合不一致）」とする。
- 評価は、取得した価格を snapshot 書式（小数6桁）に正規化してから行う。
  これにより、保存した snapshot だけからの再評価が元の評価と一致する。
- ベースライン定義（本実装の定義として明記）:
  - 同一銘柄買い持ち = 同一の再取得系列で「株価基準日の終値で買い、
    決済日の終値で売る」グロスリターン（コストなし）。
  - TOPIX代理 = 1306.T の同一系列・同一日付（基準日終値→決済日終値）の
    グロスリターン。TOPIX指数そのものではなく連動ETFによる代替である。
- DD-1（同時保有を考慮しない）のみを算出する。DD-2 は実装しない。
- 本評価は調整済み価格に基づく理論評価であり、実際の約定損益とは異なる。
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import zipfile
from datetime import datetime, timedelta, timezone

import pandas as pd

import forward

# ---------------------------------------------------------------
# 定数（計画書 修正版4 の確定事項）
# ---------------------------------------------------------------

PLAN_VERSION = "修正版4"
SCORING_VERSION = 1
TARGET_DIRECTION = "上昇"
NOTIONAL_YEN = 100_000
HORIZONS = (5, 10)
PRIMARY_HORIZON = 5
UNIT_SHARES = 100
PRICE_TOLERANCE = 0.02  # 基準日終値の照合ガード（2%）
SLIPPAGE_STANDARD = 0.0005  # 片道 0.05%
SLIPPAGE_STRESS = 0.0015  # 片道 0.15%
TOPIX_PROXY_SYMBOL = forward.TOPIX_PROXY_SYMBOL  # "1306.T"

DISCLAIMER = "調整済み価格に基づく理論評価であり、実際の約定損益とは異なる"
DD1_LABEL = "DD-1（同時保有を考慮しない）"
BASELINE_DEFINITION = (
    "同一銘柄買い持ち = 同一の再取得系列で株価基準日の終値で買い、決済日の終値で売る"
    "グロスリターン（コストなし）。TOPIX代理 = 1306.T の同一系列・同一日付"
    "（基準日終値→決済日終値）のグロスリターン（TOPIX指数そのものではなく連動ETFによる代替）"
)

# 最低証拠条件（2.1）
MIN_EVALUATED_TRADES = 100
MIN_COHORTS = 8
MIN_SPAN_DAYS = 60

# SBI証券スタンダードプラン（現物、1注文の約定代金→税込手数料）
# 料金表の確認日: 2026年11月7日、参照先: SBI証券公式サイト「国内株式委託手数料」
FEE_BRACKETS = (
    (50_000, 55),
    (100_000, 99),
    (200_000, 115),
    (500_000, 275),
    (1_000_000, 535),
    (1_500_000, 640),
    (30_000_000, 1_013),
)
FEE_OVER_TOP = 1_070  # 3,000万円超

STATUS_EVALUATED = "評価済み"
STATUS_UNEVALUABLE = "評価不能"
STATUS_PENDING = "保留"

REASON_FETCH_FAILED = "系列取得失敗"
REASON_NO_BASE_BAR = "基準日の足なし"
REASON_NO_RECORDED_CLOSE = "記録時終値なし（価格照合不能）"
REASON_PRICE_MISMATCH = "価格照合不一致"
REASON_ENTRY_NOT_YET = "エントリー日（翌営業日）がまだ到来していない"
REASON_EXIT_NOT_YET = "決済日がまだ到来していない"

RESULT_COLUMNS = [
    "snapshot_id", "評価実行ID", "計画バージョン", "スコア仕様バージョン",
    "スキャンID", "コード", "会社名", "期間", "期間区分",
    "株価基準日", "エントリー日", "エントリー価格_翌営業日始値",
    "決済日", "決済価格_終値", "保有営業日数",
    "状態", "理由", "結果区分",
    "想定元本", "株数_端数", "買い約定代金", "売り約定代金", "グロス損益",
    "手数料_ケース1", "スリッページ_標準", "ネット損益_主結果", "ネット損益pct_主結果",
    "手数料_ケース2", "ネット損益_ケース2手数料0",
    "スリッページ_ストレス", "ネット損益_ストレス",
    "ベースライン_同一銘柄買い持ちpct", "超過リターン_対同一銘柄pct",
    "ベースライン_TOPIX代理pct", "超過リターン_対TOPIX代理pct",
    "単元価格", "単元区分", "単元株数", "単元ネット損益_参考",
    "manifest_SHA256",
]


# ---------------------------------------------------------------
# 手数料（2.6 修正版4：買い・売りそれぞれの約定代金から個別に計算）
# ---------------------------------------------------------------

def fee_for_amount(amount):
    """1注文の約定代金に対する税込手数料（SBIスタンダードプラン、現物）。"""
    if amount is None:
        return None
    for limit, fee in FEE_BRACKETS:
        if amount <= limit:
            return fee
    return FEE_OVER_TOP


# ---------------------------------------------------------------
# 価格系列の正規化
# ---------------------------------------------------------------

def _round6(x):
    return round(float(x), 6)


def series_rows(df):
    """fetcher の DataFrame を [{date, Open, High, Low, Close, Volume}] にする。

    価格は snapshot 書式（小数6桁）に正規化する。評価はこの正規化済みの
    値だけで行うため、snapshot からの再評価が元の評価と一致する。
    """
    rows = []
    if df is None or len(df) == 0:
        return rows
    cols = {c.lower(): c for c in df.columns}
    for ts, row in df.iterrows():
        try:
            date_str = pd.Timestamp(ts).date().isoformat()
        except Exception:
            date_str = str(ts)[:10]
        try:
            vol = row[cols["volume"]]
            vol = 0 if pd.isna(vol) else int(float(vol))
        except Exception:
            vol = 0
        rows.append({
            "date": date_str,
            "Open": _round6(row[cols["open"]]),
            "High": _round6(row[cols["high"]]),
            "Low": _round6(row[cols["low"]]),
            "Close": _round6(row[cols["close"]]),
            "Volume": vol,
        })
    rows.sort(key=lambda r: r["date"])
    return rows


def _fetch_series(fetcher, symbol):
    """(rows, error) を返す。例外はエラー文字列に変換する。"""
    try:
        df, err = fetcher(symbol)
    except Exception as exc:  # fetcher 自体の例外も取得失敗として扱う
        return [], f"{REASON_FETCH_FAILED}：{type(exc).__name__}: {str(exc)[:120]}"
    if err is not None:
        return [], f"{REASON_FETCH_FAILED}：{err}"
    rows = series_rows(df)
    if not rows:
        return [], f"{REASON_FETCH_FAILED}：データなし"
    return rows, None


# ---------------------------------------------------------------
# 対象記録の抽出と重複除去
# ---------------------------------------------------------------

def _version_of(rec):
    try:
        return int(float(rec.get("スコア仕様バージョン")))
    except (TypeError, ValueError):
        return 1


def select_targets(records):
    """(対象記録, 対象外件数, 重複除去件数) を返す。

    対象: スコア仕様バージョン=1 かつ 短期予測方向=「上昇」。
    重複は (スキャンID, コード, バージョン) で判定し、先勝ちで1件に絞る
    （期間は評価側で day5/day10 に展開するため、記録単位のキーで等価）。
    """
    targets = []
    seen = set()
    n_out_of_scope = 0
    n_duplicates = 0
    for rec in records:
        if _version_of(rec) != SCORING_VERSION or rec.get("短期予測方向") != TARGET_DIRECTION:
            n_out_of_scope += 1
            continue
        key = (rec.get("スキャンID"), rec.get("コード"), SCORING_VERSION)
        if key in seen:
            n_duplicates += 1
            continue
        seen.add(key)
        targets.append(rec)
    return targets, n_out_of_scope, n_duplicates


# ---------------------------------------------------------------
# 1取引の計算
# ---------------------------------------------------------------

def _cost_breakdown(buy_amount, sell_amount):
    """(グロス, 手数料ケース1, スリッページ標準, スリッページ・ストレス)。"""
    gross = sell_amount - buy_amount
    fee1 = fee_for_amount(buy_amount) + fee_for_amount(sell_amount)
    slip_std = buy_amount * SLIPPAGE_STANDARD + sell_amount * SLIPPAGE_STANDARD
    slip_stress = buy_amount * SLIPPAGE_STRESS + sell_amount * SLIPPAGE_STRESS
    return gross, fee1, slip_std, slip_stress


def _evaluate_trade(rec, horizon, rows, topix_rows, base):
    """1記録×1期間の評価結果 dict を返す。

    base は呼び出し側で判定済みの基礎情報:
      {"status": ..., "reason": ..., "base_idx": int|None,
       "base_close": float|None, "entry_idx": int|None}
    """
    symbol = rec.get("コード", "")
    trade = {
        "評価実行ID": base.get("run_id", ""),
        "計画バージョン": PLAN_VERSION,
        "スコア仕様バージョン": SCORING_VERSION,
        "スキャンID": rec.get("スキャンID", ""),
        "コード": symbol,
        "会社名": rec.get("会社名", ""),
        "期間": horizon,
        "期間区分": "主評価" if horizon == PRIMARY_HORIZON else "補助評価",
        "株価基準日": str(rec.get("株価基準日", "")),
        "エントリー日": "",
        "エントリー価格_翌営業日始値": None,
        "決済日": "",
        "決済価格_終値": None,
        "保有営業日数": horizon,
        "状態": "",
        "理由": "",
        "結果区分": "",
        "想定元本": NOTIONAL_YEN,
    }
    # 基礎段階での評価不能・保留（系列取得失敗／基準日の足なし／照合不一致／入口未到来）
    if base["status"] != STATUS_EVALUATED:
        trade["状態"] = base["status"]
        trade["理由"] = base["reason"]
        return trade
    entry_idx = base["entry_idx"]
    exit_idx = entry_idx + horizon
    if exit_idx >= len(rows):
        trade["状態"] = STATUS_PENDING
        trade["理由"] = f"{REASON_EXIT_NOT_YET}（day{horizon}）"
        return trade
    entry_row = rows[entry_idx]
    exit_row = rows[exit_idx]
    entry_open = entry_row["Open"]
    exit_close = exit_row["Close"]
    base_close = base["base_close"]
    if entry_open is None or entry_open <= 0 or exit_close is None or base_close <= 0:
        trade["状態"] = STATUS_UNEVALUABLE
        trade["理由"] = "価格が不正（始値・終値が0以下または欠損）"
        return trade

    shares = NOTIONAL_YEN / entry_open
    buy_amount = entry_open * shares
    sell_amount = exit_close * shares
    gross, fee1, slip_std, slip_stress = _cost_breakdown(buy_amount, sell_amount)
    net_main = gross - fee1 - slip_std
    net_case2 = gross - slip_std  # 手数料0円ケース（標準スリッページ）
    net_stress = gross - fee1 - slip_stress
    net_pct = net_main / NOTIONAL_YEN * 100.0

    # ベースライン: 同一銘柄買い持ち（基準日終値→決済日終値、グロス）
    baseline_self_pct = (exit_close / base_close - 1.0) * 100.0
    # ベースライン: TOPIX代理 1306.T（同一日付、基準日終値→決済日終値、グロス）
    topix_pct = None
    if topix_rows:
        tmap = {r["date"]: r["Close"] for r in topix_rows}
        t_base = tmap.get(trade["株価基準日"])
        t_exit = tmap.get(exit_row["date"])
        if t_base and t_exit:
            topix_pct = (t_exit / t_base - 1.0) * 100.0

    # 実行可能性評価（2.6.1）: 単元ベースで買い直した参考損益
    unit_price = entry_open * UNIT_SHARES
    if unit_price <= NOTIONAL_YEN:
        units = int(NOTIONAL_YEN // unit_price)
        unit_shares = units * UNIT_SHARES
        u_buy = entry_open * unit_shares
        u_sell = exit_close * unit_shares
        u_gross, u_fee, u_slip, _ = _cost_breakdown(u_buy, u_sell)
        unit_net = u_gross - u_fee - u_slip
        unit_status = "購入可能"
    else:
        unit_shares = None
        unit_net = None
        unit_status = "単元不足"

    trade.update({
        "エントリー日": entry_row["date"],
        "エントリー価格_翌営業日始値": entry_open,
        "決済日": exit_row["date"],
        "決済価格_終値": exit_close,
        "状態": STATUS_EVALUATED,
        "理由": "",
        "結果区分": "勝ち" if net_main > 0 else ("負け" if net_main < 0 else "引き分け"),
        "株数_端数": shares,
        "買い約定代金": buy_amount,
        "売り約定代金": sell_amount,
        "グロス損益": gross,
        "手数料_ケース1": fee1,
        "スリッページ_標準": slip_std,
        "ネット損益_主結果": net_main,
        "ネット損益pct_主結果": net_pct,
        "手数料_ケース2": 0,
        "ネット損益_ケース2手数料0": net_case2,
        "スリッページ_ストレス": slip_stress,
        "ネット損益_ストレス": net_stress,
        "ベースライン_同一銘柄買い持ちpct": baseline_self_pct,
        "超過リターン_対同一銘柄pct": net_pct - baseline_self_pct,
        "ベースライン_TOPIX代理pct": topix_pct,
        "超過リターン_対TOPIX代理pct": (net_pct - topix_pct) if topix_pct is not None else None,
        "単元価格": unit_price,
        "単元区分": unit_status,
        "単元株数": unit_shares,
        "単元ネット損益_参考": unit_net,
    })
    return trade


# ---------------------------------------------------------------
# 評価本体
# ---------------------------------------------------------------

def evaluate(records, fetcher, run_id="", fetched_at=None):
    """登録記録リストを検証Aで評価する。

    fetcher(symbol) -> (DataFrame, error)。forward.evaluate_records と同じ契約。
    戻り値: dict（trades, counts, series, topix_error など）。
    評価は決済日までの行だけを使う（決済日より後の行は結果に影響しない）。
    """
    if fetched_at is None:
        fetched_at = datetime.now(timezone.utc)
    targets, n_out, n_dup = select_targets(records)

    cache = {}

    def get_series(symbol):
        if symbol not in cache:
            cache[symbol] = _fetch_series(fetcher, symbol)
        return cache[symbol]

    topix_rows, topix_error = get_series(TOPIX_PROXY_SYMBOL)

    trades = []
    series_used = {}
    for rec in targets:
        symbol = rec.get("コード", "")
        rows, err = get_series(symbol)
        series_used[symbol] = {"rows": rows, "error": err}
        base_date = str(rec.get("株価基準日", ""))
        base = {"run_id": run_id}
        if err is not None:
            base.update({"status": STATUS_UNEVALUABLE, "reason": err})
        else:
            idx = {r["date"]: i for i, r in enumerate(rows)}
            if base_date not in idx:
                base.update({"status": STATUS_UNEVALUABLE, "reason": REASON_NO_BASE_BAR})
            else:
                base_idx = idx[base_date]
                base_close = rows[base_idx]["Close"]
                recorded = rec.get("記録時終値")
                if recorded is None:
                    base.update({"status": STATUS_UNEVALUABLE,
                                 "reason": REASON_NO_RECORDED_CLOSE})
                elif base_close <= 0 or abs(base_close / float(recorded) - 1.0) > PRICE_TOLERANCE:
                    base.update({"status": STATUS_UNEVALUABLE,
                                 "reason": f"{REASON_PRICE_MISMATCH}"
                                           f"（再取得 {base_close} / 記録 {recorded}）"})
                elif base_idx + 1 >= len(rows):
                    base.update({"status": STATUS_PENDING, "reason": REASON_ENTRY_NOT_YET})
                else:
                    base.update({"status": STATUS_EVALUATED, "reason": "",
                                 "base_idx": base_idx, "base_close": base_close,
                                 "entry_idx": base_idx + 1})
        for horizon in HORIZONS:
            trades.append(_evaluate_trade(rec, horizon, rows, topix_rows, base))

    return {
        "run_id": run_id,
        "fetched_at": fetched_at,
        "trades": trades,
        "counts": {
            "入力記録数": len(records),
            "対象外件数": n_out,
            "重複除去件数": n_dup,
            "対象記録数": len(targets),
        },
        "series": series_used,
        "topix": {"rows": topix_rows, "error": topix_error},
        "targets": targets,
    }


# ---------------------------------------------------------------
# 集計（2.5）
# ---------------------------------------------------------------

def compute_dd1(evaluated):
    """DD-1（同時保有を考慮しない）。

    評価済み取引を決済日順（同日はコード順）に並べた累積ネット損益曲線の
    ピーク→トラフ最大落差（円、正の値）。取引が無ければ 0。
    """
    ordered = sorted(evaluated, key=lambda t: (t["決済日"], t["コード"]))
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in ordered:
        cum += t["ネット損益_主結果"]
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    return max_dd


def _mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def summarize_horizon(trades, horizon):
    """1期間分の集計 dict を返す。"""
    subset = [t for t in trades if t["期間"] == horizon]
    evaluated = [t for t in subset if t["状態"] == STATUS_EVALUATED]
    unevaluable = [t for t in subset if t["状態"] == STATUS_UNEVALUABLE]
    pending = [t for t in subset if t["状態"] == STATUS_PENDING]
    n = len(evaluated)

    nets = [t["ネット損益_主結果"] for t in evaluated]
    grosses = [t["グロス損益"] for t in evaluated]
    wins = [v for v in nets if v > 0]
    losses = [v for v in nets if v < 0]
    avg_win = _mean(wins)
    avg_loss = _mean(losses)
    payoff = None
    if avg_win is not None and avg_loss not in (None, 0):
        payoff = avg_win / abs(avg_loss)
    expectancy = _mean(nets)
    expectancy_pct = _mean([t["ネット損益pct_主結果"] for t in evaluated])

    base_dates = [t["株価基準日"] for t in evaluated if t["株価基準日"]]
    span_days = 0
    if base_dates:
        d0 = datetime.strptime(min(base_dates), "%Y-%m-%d").date()
        d1 = datetime.strptime(max(base_dates), "%Y-%m-%d").date()
        span_days = (d1 - d0).days
    cohorts = sorted({t["スキャンID"] for t in evaluated})

    # コホート別
    cohort_rows = []
    for cid in sorted({t["スキャンID"] for t in subset}):
        c_ev = [t for t in evaluated if t["スキャンID"] == cid]
        cohort_rows.append({
            "スキャンID": cid,
            "対象件数": len([t for t in subset if t["スキャンID"] == cid]),
            "評価済み件数": len(c_ev),
            "評価不能件数": len([t for t in subset
                               if t["スキャンID"] == cid and t["状態"] == STATUS_UNEVALUABLE]),
            "保留件数": len([t for t in subset
                             if t["スキャンID"] == cid and t["状態"] == STATUS_PENDING]),
            "ネット損益合計": sum(t["ネット損益_主結果"] for t in c_ev) if c_ev else None,
            "勝率": (sum(1 for t in c_ev if t["ネット損益_主結果"] > 0) / len(c_ev)
                     if c_ev else None),
        })

    feasible = [t for t in evaluated if t.get("単元区分") == "購入可能"]
    shortfall = {
        "評価済み取引": max(0, MIN_EVALUATED_TRADES - n),
        "コホート": max(0, MIN_COHORTS - len(cohorts)),
        "観測期間日数": max(0, MIN_SPAN_DAYS - span_days),
    }
    evidence_met = all(v == 0 for v in shortfall.values())
    judgement = None
    conditions = None
    if evidence_met:
        cond_expectancy = expectancy is not None and expectancy > 0
        excess_self = _mean([t["超過リターン_対同一銘柄pct"] for t in evaluated])
        cond_baseline = excess_self is not None and excess_self > 0
        cond_payoff = payoff is not None and payoff >= 1.0
        conditions = {
            "コスト控除後の期待値がプラス": cond_expectancy,
            "同一銘柄買い持ちベースラインを上回る": cond_baseline,
            "ペイオフレシオが1.0以上": cond_payoff,
        }
        judgement = ("短期シグナルに優位性の兆しがある"
                     if all(conditions.values()) else "優位性を確認できず")
    else:
        judgement = "判定保留（最低証拠条件未達）"

    return {
        "期間": horizon,
        "期間区分": "主評価" if horizon == PRIMARY_HORIZON else "補助評価",
        "対象件数": len(subset),
        "評価済み件数": n,
        "評価不能件数": len(unevaluable),
        "保留件数": len(pending),
        "評価不能理由": _reason_counts(unevaluable),
        "保留理由": _reason_counts(pending),
        "コホート数": len(cohorts),
        "観測期間日数": span_days,
        "勝率_コスト控除後": (len(wins) / n) if n else None,
        "勝率_コスト前": (sum(1 for v in grosses if v > 0) / n) if n else None,
        "平均利益": avg_win,
        "平均損失": avg_loss,
        "ペイオフレシオ": payoff,
        "期待値_円": expectancy,
        "期待値_pct": expectancy_pct,
        "累積損益": sum(nets) if nets else 0.0,
        DD1_LABEL: compute_dd1(evaluated),
        "コスト橋渡し": {
            "グロス合計": sum(grosses) if grosses else 0.0,
            "手数料合計_ケース1": sum(t["手数料_ケース1"] for t in evaluated),
            "スリッページ合計_標準": sum(t["スリッページ_標準"] for t in evaluated),
            "ネット合計_主結果": sum(nets) if nets else 0.0,
            "ネット合計_ケース2手数料0": sum(t["ネット損益_ケース2手数料0"] for t in evaluated),
            "スリッページ合計_ストレス": sum(t["スリッページ_ストレス"] for t in evaluated),
            "ネット合計_ストレス": sum(t["ネット損益_ストレス"] for t in evaluated),
        },
        "ベースライン比較": {
            "定義": BASELINE_DEFINITION,
            "平均超過リターン_対同一銘柄pct": _mean(
                [t["超過リターン_対同一銘柄pct"] for t in evaluated]),
            "平均超過リターン_対TOPIX代理pct": _mean(
                [t["超過リターン_対TOPIX代理pct"] for t in evaluated]),
            "TOPIX代理比較可能件数": sum(
                1 for t in evaluated if t["超過リターン_対TOPIX代理pct"] is not None),
        },
        "実行可能性": {
            "購入可能件数": len(feasible),
            "単元不足件数": len([t for t in evaluated if t.get("単元区分") == "単元不足"]),
            "購入可能割合": (len(feasible) / n) if n else None,
        },
        "コホート別": cohort_rows,
        "最低証拠条件": {
            "必要": {"評価済み取引": MIN_EVALUATED_TRADES, "コホート": MIN_COHORTS,
                     "観測期間日数": MIN_SPAN_DAYS},
            "現在": {"評価済み取引": n, "コホート": len(cohorts), "観測期間日数": span_days},
            "不足": shortfall,
            "達成": evidence_met,
        },
        "判定": judgement,
        "判定条件": conditions,
    }


def _reason_counts(items):
    counts = {}
    for t in items:
        key = t["理由"].split("（")[0] if t["理由"] else ""
        counts[key] = counts.get(key, 0) + 1
    return counts


def summarize(result):
    """評価結果全体の集計（期間別＋全体情報）。"""
    return {
        "評価実行ID": result["run_id"],
        "計画バージョン": PLAN_VERSION,
        "スコア仕様バージョン": SCORING_VERSION,
        "注記_価格": DISCLAIMER,
        "ベースライン定義": BASELINE_DEFINITION,
        "件数": result["counts"],
        "期間別": {str(h): summarize_horizon(result["trades"], h) for h in HORIZONS},
    }


# ---------------------------------------------------------------
# snapshot 保存（2.10）
# ---------------------------------------------------------------

SNAPSHOT_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def snapshot_csv_bytes(rows):
    """固定書式（UTF-8 / LF / 価格は小数6桁 / 出来高は整数）のCSVバイト列。"""
    lines = [",".join(SNAPSHOT_COLUMNS)]
    for r in rows:
        lines.append(
            f"{r['date']},{r['Open']:.6f},{r['High']:.6f},{r['Low']:.6f},"
            f"{r['Close']:.6f},{int(r['Volume'])}"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_filename(symbol):
    return symbol.replace("/", "_").replace("^", "") + ".csv"


def _slice_range(result, symbol):
    """snapshot に含める期間 [開始日, 終了日] を決める。

    開始日は対象記録の株価基準日の最小、終了日は評価済み取引の決済日の最大
    （無ければ株価基準日の最大）。ベンチマークは全銘柄を通した範囲。
    """
    trades = result["trades"]
    targets = result["targets"]
    if symbol == TOPIX_PROXY_SYMBOL:
        base_dates = [str(r.get("株価基準日", "")) for r in targets]
        exit_dates = [t["決済日"] for t in trades
                      if t["状態"] == STATUS_EVALUATED and t["決済日"]]
    else:
        base_dates = [str(r.get("株価基準日", "")) for r in targets
                      if r.get("コード") == symbol]
        exit_dates = [t["決済日"] for t in trades
                      if t["コード"] == symbol and t["状態"] == STATUS_EVALUATED and t["決済日"]]
    base_dates = [d for d in base_dates if d]
    if not base_dates:
        return None
    start = min(base_dates)
    end = max(exit_dates) if exit_dates else max(base_dates)
    return start, end


def write_snapshot(result, snapshot_dir):
    """snapshot（銘柄別CSV＋manifest.json）を書き、manifest dict を返す。"""
    os.makedirs(snapshot_dir, exist_ok=True)
    fetched_at = result["fetched_at"]
    jst = fetched_at.astimezone(timezone(timedelta(hours=9)))
    try:
        import yfinance
        yf_version = yfinance.__version__
    except Exception:
        yf_version = "unknown"

    files_meta = {}
    symbols = list(result["series"].keys())
    if result["topix"]["rows"] and TOPIX_PROXY_SYMBOL not in symbols:
        symbols.append(TOPIX_PROXY_SYMBOL)
    for symbol in symbols:
        if symbol == TOPIX_PROXY_SYMBOL:
            rows_all = result["topix"]["rows"]
        else:
            rows_all = result["series"][symbol]["rows"]
        if not rows_all:
            continue
        rng = _slice_range(result, symbol)
        if rng is None:
            continue
        start, end = rng
        rows = [r for r in rows_all if start <= r["date"] <= end]
        if not rows:
            continue
        data = snapshot_csv_bytes(rows)
        fname = _safe_filename(symbol)
        with open(os.path.join(snapshot_dir, fname), "wb") as f:
            f.write(data)
        files_meta[symbol] = {
            "ファイル名": fname,
            "開始日": rows[0]["date"],
            "終了日": rows[-1]["date"],
            "行数": len(rows),
            "SHA-256": sha256_hex(data),
        }

    manifest = {
        "snapshot_id": result["run_id"],
        "評価実行ID": result["run_id"],
        "取得日時UTC": fetched_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "取得日時JST": jst.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        "データ提供元": "Yahoo Finance",
        "取得手段": f"yfinance {yf_version}",
        "取得パラメータ": {"interval": "1d", "auto_adjust": True},
        "調整済み価格": True,
        "価格の性質": DISCLAIMER,
        "計画バージョン": PLAN_VERSION,
        "スコア仕様バージョン": SCORING_VERSION,
        "対象銘柄": sorted(files_meta.keys()),
        "ファイル": files_meta,
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2,
                                 sort_keys=True) + "\n").encode("utf-8")
    with open(os.path.join(snapshot_dir, "manifest.json"), "wb") as f:
        f.write(manifest_bytes)
    manifest["manifest_SHA256"] = sha256_hex(manifest_bytes)
    return manifest


def verify_snapshot(snapshot_dir):
    """manifest と実ファイルのハッシュ照合。食い違いのリストを返す（空=正常）。"""
    with open(os.path.join(snapshot_dir, "manifest.json"), "rb") as f:
        manifest = json.loads(f.read().decode("utf-8"))
    problems = []
    for symbol, meta in manifest.get("ファイル", {}).items():
        path = os.path.join(snapshot_dir, meta["ファイル名"])
        if not os.path.exists(path):
            problems.append(f"{symbol}: ファイルなし")
            continue
        with open(path, "rb") as f:
            actual = sha256_hex(f.read())
        if actual != meta["SHA-256"]:
            problems.append(f"{symbol}: ハッシュ不一致")
    return problems


def snapshot_fetcher(snapshot_dir):
    """snapshot ディレクトリだけから系列を返す fetcher（ネットワークなし）。"""
    with open(os.path.join(snapshot_dir, "manifest.json"), "rb") as f:
        manifest = json.loads(f.read().decode("utf-8"))
    files = manifest.get("ファイル", {})

    def fetcher(symbol):
        meta = files.get(symbol)
        if meta is None:
            return pd.DataFrame(), "snapshotに含まれていません"
        path = os.path.join(snapshot_dir, meta["ファイル名"])
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        return df, None

    return fetcher


def reevaluate_from_snapshot(records, snapshot_dir, run_id=""):
    """snapshot だけで再評価する。manifest の改変があれば例外を送る。"""
    problems = verify_snapshot(snapshot_dir)
    if problems:
        raise ValueError("snapshotのハッシュ照合に失敗しました: " + "; ".join(problems))
    return evaluate(records, snapshot_fetcher(snapshot_dir), run_id=run_id)


def compare_trades(original, reproduced):
    """2つの trades リストの主要項目の差分リストを返す（空=一致）。"""
    def key(t):
        return (t["スキャンID"], t["コード"], t["期間"])

    def comparable(t):
        out = {}
        for k, v in t.items():
            if isinstance(v, float):
                out[k] = round(v, 6)
            else:
                out[k] = v
        return out

    a = {key(t): comparable(t) for t in original}
    b = {key(t): comparable(t) for t in reproduced}
    diffs = []
    for k in sorted(set(a) | set(b), key=str):
        if k not in a:
            diffs.append(f"{k}: 元にない取引")
        elif k not in b:
            diffs.append(f"{k}: 再評価にない取引")
        elif a[k] != b[k]:
            changed = [f"{f}: {a[k].get(f)} != {b[k].get(f)}"
                       for f in a[k] if a[k].get(f) != b[k].get(f)]
            diffs.append(f"{k}: " + "; ".join(changed))
    return diffs


# ---------------------------------------------------------------
# 実行一式の出力（結果CSV・集計・snapshot・ZIP）
# ---------------------------------------------------------------

def trades_to_csv(trades, manifest_sha256):
    """結果CSV（utf-8-sig、既存モジュールと同じ方式）。各行に manifest の
    SHA-256 と snapshot_id を持たせる（二重記録）。"""
    rows = []
    for t in trades:
        row = {col: t.get(col) for col in RESULT_COLUMNS}
        row["manifest_SHA256"] = manifest_sha256
        rows.append(row)
    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    return df.to_csv(index=False).encode("utf-8-sig")


def _fmt(x, digits=2):
    if x is None:
        return "-"
    if isinstance(x, float):
        if math.isnan(x):
            return "-"
        return f"{x:,.{digits}f}"
    return str(x)


def summary_markdown(summary, manifest, extra_lines=None):
    lines = []
    lines.append("# 検証A 短期フォワード仮想売買評価 結果")
    lines.append("")
    lines.append(f"※ {DISCLAIMER}（価格は auto_adjust=True の調整済み価格）")
    lines.append(f"※ {DD1_LABEL} のみを算出しています。"
                 "DD-1 は取引列の累積損益の落差であり、実際の運用資金の最大ドローダウンではありません。")
    lines.append(f"※ ベースライン定義: {BASELINE_DEFINITION}")
    lines.append("")
    if extra_lines:
        lines.extend(extra_lines)
        lines.append("")
    c = summary["件数"]
    lines.append(f"- 評価実行ID: {summary['評価実行ID']}")
    lines.append(f"- 計画バージョン: {summary['計画バージョン']} / "
                 f"スコア仕様バージョン: {summary['スコア仕様バージョン']}")
    lines.append(f"- 入力記録数: {c['入力記録数']} / 対象外: {c['対象外件数']} / "
                 f"重複除去: {c['重複除去件数']} / 対象記録数: {c['対象記録数']}")
    lines.append(f"- snapshot_id: {manifest['snapshot_id']} / "
                 f"manifest SHA-256: {manifest['manifest_SHA256']}")
    for h in HORIZONS:
        s = summary["期間別"][str(h)]
        b = s["コスト橋渡し"]
        lines.append("")
        lines.append(f"## day{h}（{s['期間区分']}）")
        lines.append(f"- 対象 {s['対象件数']}件 / 評価済み {s['評価済み件数']}件 / "
                     f"評価不能 {s['評価不能件数']}件 / 保留 {s['保留件数']}件")
        lines.append(f"- 勝率（コスト控除後）: {_fmt(s['勝率_コスト控除後'] * 100 if s['勝率_コスト控除後'] is not None else None)}%"
                     f" / 勝率（コスト前）: {_fmt(s['勝率_コスト前'] * 100 if s['勝率_コスト前'] is not None else None)}%")
        lines.append(f"- 平均利益: {_fmt(s['平均利益'])}円 / 平均損失: {_fmt(s['平均損失'])}円 / "
                     f"ペイオフレシオ: {_fmt(s['ペイオフレシオ'])}")
        lines.append(f"- 期待値: {_fmt(s['期待値_円'])}円（{_fmt(s['期待値_pct'])}%） / "
                     f"累積損益: {_fmt(s['累積損益'])}円")
        lines.append(f"- {DD1_LABEL}: {_fmt(s[DD1_LABEL])}円")
        lines.append(f"- コスト橋渡し: グロス {_fmt(b['グロス合計'])}円 → 手数料 {_fmt(b['手数料合計_ケース1'])}円"
                     f" ＋ スリッページ {_fmt(b['スリッページ合計_標準'])}円 → ネット {_fmt(b['ネット合計_主結果'])}円")
        lines.append(f"- 参考: ネット合計（ケース2・手数料0） {_fmt(b['ネット合計_ケース2手数料0'])}円 / "
                     f"ネット合計（ストレス） {_fmt(b['ネット合計_ストレス'])}円")
        bl = s["ベースライン比較"]
        lines.append(f"- 平均超過リターン: 対同一銘柄 {_fmt(bl['平均超過リターン_対同一銘柄pct'])}%pt / "
                     f"対TOPIX代理 {_fmt(bl['平均超過リターン_対TOPIX代理pct'])}%pt"
                     f"（比較可能 {bl['TOPIX代理比較可能件数']}件）")
        fe = s["実行可能性"]
        lines.append(f"- 実行可能性: 購入可能 {fe['購入可能件数']}件 / 単元不足 {fe['単元不足件数']}件"
                     f" / 購入可能割合 {_fmt(fe['購入可能割合'] * 100 if fe['購入可能割合'] is not None else None)}%")
        ev = s["最低証拠条件"]
        lines.append(f"- 最低証拠条件: 評価済み {ev['現在']['評価済み取引']}/100件、"
                     f"コホート {ev['現在']['コホート']}/8件、観測期間 {ev['現在']['観測期間日数']}/60日"
                     f" → {'達成' if ev['達成'] else '未達'}")
        if not ev["達成"]:
            lines.append(f"  - 不足: 評価済み取引 あと{ev['不足']['評価済み取引']}件、"
                         f"コホート あと{ev['不足']['コホート']}件、"
                         f"観測期間 あと{ev['不足']['観測期間日数']}日")
        lines.append(f"- 判定: {s['判定']}")
        if s["判定条件"]:
            for name, ok in s["判定条件"].items():
                lines.append(f"  - {name}: {'満たす' if ok else '満たさない'}")
    lines.append("")
    return "\n".join(lines)


def run_evaluation(records, fetcher, out_dir, run_id=None, note=None,
                     fetched_at=None):
    """評価→snapshot→結果CSV→集計→ZIP→実行サマリ の一式を実行する。

    戻り値: 実行サマリ dict（manifest SHA-256、ZIP SHA-256 を含む）。
    """
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("va-%Y%m%dT%H%M%SZ")
    os.makedirs(out_dir, exist_ok=True)
    result = evaluate(records, fetcher, run_id=run_id, fetched_at=fetched_at)
    for t in result["trades"]:
        t["snapshot_id"] = run_id

    snapshot_dir = os.path.join(out_dir, "snapshot")
    manifest = write_snapshot(result, snapshot_dir)

    results_csv = trades_to_csv(result["trades"], manifest["manifest_SHA256"])
    with open(os.path.join(out_dir, "verification_a_results.csv"), "wb") as f:
        f.write(results_csv)

    summary = summarize(result)
    summary["snapshot_id"] = run_id
    summary["manifest_SHA256"] = manifest["manifest_SHA256"]
    if note:
        summary["注記"] = note
    with open(os.path.join(out_dir, "verification_a_summary.json"), "w",
              encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    extra = [f"※ {note}"] if note else None
    with open(os.path.join(out_dir, "verification_a_summary.md"), "w",
              encoding="utf-8") as f:
        f.write(summary_markdown(summary, manifest, extra_lines=extra))

    # ZIPバックアップ（この時点までの出力一式）
    zip_name = f"{run_id}_backup.zip"
    zip_path = os.path.join(out_dir, zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, fnames in os.walk(out_dir):
            for fn in fnames:
                full = os.path.join(root, fn)
                if full == zip_path:
                    continue
                zf.write(full, os.path.relpath(full, out_dir))
    with open(zip_path, "rb") as f:
        zip_sha = sha256_hex(f.read())

    run_summary = {
        "評価実行ID": run_id,
        "snapshot_id": run_id,
        "計画バージョン": PLAN_VERSION,
        "注記_価格": DISCLAIMER,
        "注記": note,
        "manifest_SHA256": manifest["manifest_SHA256"],
        "ZIPファイル名": zip_name,
        "ZIP_SHA256": zip_sha,
        "件数": result["counts"],
        "判定_day5": summary["期間別"][str(PRIMARY_HORIZON)]["判定"],
        "判定_day10": summary["期間別"][str(HORIZONS[1])]["判定"],
    }
    with open(os.path.join(out_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n")
    run_summary["summary"] = summary
    run_summary["manifest"] = manifest
    return run_summary
