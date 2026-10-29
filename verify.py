"""Stock AI Lab の検証ロジック（過去データでの評価の再現と採点）。

検証仕様バージョン 1（2026-10-29 確定。結果を見てから条件を変えない）:
- 対象は現在の8銘柄（日本株4・米国株4）。
- 評価日は5営業日ごと。評価日までのデータだけを先頭から切り取り、
  その範囲だけで指標計算と評価を行う（未来情報を使わない）。
- 測定窓は短期5・中期20・長期60営業日。
- 実際の方向の小動き判定は短期±1%・中期±2%・長期±3%。
- 予測方向はスコア+2以上を上昇、-2以下を下落、その間を中立とする
  （現在の分析表示の基準そのまま）。
- 中立予測・実際小動き・判定保留・対象外は、的中率の分母に入れない。
- 方向付き騰落率は参考値であり、売買コストを含まない実際の損益ではない。
  中立予測は想定損益の集計に含めない。
- データは5年分取得を基本とし、実際に取得できた期間・件数を表示する。
"""

from datetime import datetime

import pandas as pd

import analysis

SPEC_VERSION = 1

VERIFY_SYMBOLS = [
    "7203.T", "6758.T", "8306.T", "9984.T",
    "AAPL", "MSFT", "NVDA", "SPY",
]

FETCH_PERIOD = "5y"
EVAL_STEP = 5  # 評価日の間隔（営業日）

HORIZONS = ["短期", "中期", "長期"]
WINDOWS = {"短期": 5, "中期": 20, "長期": 60}
FLAT_BANDS = {"短期": 1.0, "中期": 2.0, "長期": 3.0}
MIN_BARS = {
    "短期": analysis.MIN_BARS_SHORT,
    "中期": analysis.MIN_BARS_MID,
    "長期": analysis.MIN_BARS_LONG,
}
# 的中・不的中の合計がこの件数未満なら「精度を判断できない」と表示する目安
MIN_CASES_FOR_JUDGEMENT = 30

RECORD_COLUMNS = [
    "実行日時", "仕様版数", "銘柄", "会社名", "市場", "評価日", "時間軸",
    "スコア", "予測方向", "評価ラベル", "評価日終値", "測定日", "測定日終値",
    "騰落率pct", "実際の方向", "判定", "対象外理由", "方向付き騰落率pct",
    "データ最終日", "取得期間",
]

JUDGED = ["的中", "不的中", "中立予測", "実際小動き", "判定保留", "対象外"]


def direction_from_result(result):
    """評価結果から予測方向を決める。判定できない場合は「判定保留」。"""
    if not result.get("available"):
        return "判定保留"
    score = result.get("score")
    if score is None:
        return "判定保留"
    if score >= 2:
        return "上昇"
    if score <= -2:
        return "下落"
    return "中立"


def actual_direction(forward_pct, horizon):
    """測定窓の騰落率から実際の方向を決める。小動きの幅は時間軸ごとに固定。"""
    band = FLAT_BANDS[horizon]
    if forward_pct > band:
        return "上昇"
    if forward_pct < -band:
        return "下落"
    return "小動き"


def _base_record(symbol, run_at, period_label, last_date):
    return {
        "実行日時": run_at,
        "仕様版数": SPEC_VERSION,
        "銘柄": symbol,
        "会社名": analysis.resolve_company_name(symbol),
        "市場": analysis.market_of(symbol),
        "評価日": "",
        "時間軸": "",
        "スコア": None,
        "予測方向": "",
        "評価ラベル": "",
        "評価日終値": None,
        "測定日": "",
        "測定日終値": None,
        "騰落率pct": None,
        "実際の方向": "",
        "判定": "",
        "対象外理由": "",
        "方向付き騰落率pct": None,
        "データ最終日": str(last_date) if last_date is not None else "",
        "取得期間": period_label,
    }


def verify_symbol(symbol, raw, period_label=FETCH_PERIOD, run_at=""):
    """1銘柄分の過去検証を行う。

    raw は fetch_history で取得した日足（失敗時は空のDataFrame）。
    戻り値: (records, exclusions)
      records: 評価できた地点と、地点ごとの判定保留の記録（辞書のリスト）
      exclusions: 対象外の集計（辞書のリスト：時間軸・理由・件数）
    評価は「評価日までの切り取りデータ」だけで指標計算して行う。
    """
    records = []
    exclusions = []

    def add_exclusion(horizon, reason, count):
        if count > 0:
            exclusions.append(
                {"銘柄": symbol, "時間軸": horizon, "理由": reason, "件数": count}
            )

    if raw is None or raw.empty:
        # 取得失敗は予測記録には入れず、対象外集計として区別して記録する
        exclusions.append({
            "銘柄": symbol, "時間軸": "全体",
            "理由": "データ取得に失敗したため、この銘柄は検証対象外",
            "件数": 1,
        })
        return records, exclusions

    last_date = raw.index[-1].date()
    closes = raw["Close"]
    n = len(raw)
    warmup = {h: 0 for h in HORIZONS}  # 履歴不足で評価を始められなかった地点数
    tail = {h: 0 for h in HORIZONS}  # 測定窓が足りず対象外になった地点数

    # 評価日候補：切り取り本数が EVAL_STEP ごとに増える位置。
    # 候補ごとに一度だけ、切り取りデータで指標計算と全時間軸の評価を行う。
    for end in range(EVAL_STEP, n + 1, EVAL_STEP):
        i = end - 1  # 評価日の行インデックス
        assessed = None
        for horizon in HORIZONS:
            if end < MIN_BARS[horizon]:
                warmup[horizon] += 1
                continue
            window = WINDOWS[horizon]
            if i + window > n - 1:
                tail[horizon] += 1
                continue
            if assessed is None:
                assessed = analysis.assess_all(analysis.add_indicators(raw.iloc[:end]))
            result = assessed[horizon]
            eval_close = float(closes.iloc[i])
            meas_idx = i + window
            meas_close = float(closes.iloc[meas_idx])
            forward_pct = (meas_close / eval_close - 1.0) * 100.0 if eval_close else 0.0

            rec = _base_record(symbol, run_at, period_label, last_date)
            rec["評価日"] = str(raw.index[i].date())
            rec["時間軸"] = horizon
            rec["評価日終値"] = round(eval_close, 3)
            rec["測定日"] = str(raw.index[meas_idx].date())
            rec["測定日終値"] = round(meas_close, 3)
            rec["騰落率pct"] = round(forward_pct, 3)
            rec["実際の方向"] = actual_direction(forward_pct, horizon)

            prediction = direction_from_result(result)
            rec["予測方向"] = prediction
            rec["評価ラベル"] = result.get("label", "")
            rec["スコア"] = result.get("score")

            if prediction == "判定保留":
                rec["判定"] = "判定保留"
            elif prediction == "中立":
                rec["判定"] = "中立予測"
            else:
                rec["方向付き騰落率pct"] = round(
                    forward_pct if prediction == "上昇" else -forward_pct, 3
                )
                if rec["実際の方向"] == "小動き":
                    rec["判定"] = "実際小動き"
                elif prediction == rec["実際の方向"]:
                    rec["判定"] = "的中"
                else:
                    rec["判定"] = "不的中"
            records.append(rec)

    for horizon in HORIZONS:
        add_exclusion(horizon, "履歴不足のため評価開始前の地点", warmup[horizon])
        add_exclusion(horizon, "測定窓の終わりまでデータがない地点", tail[horizon])

    return records, exclusions


def verify_all(symbols, fetcher, period_label=FETCH_PERIOD, run_at=None, progress=None):
    """複数銘柄を検証する。fetcher(symbol) は (DataFrame, error) を返す関数。

    取得に失敗した銘柄は対象外として記録し、他銘柄の検証は続ける。
    progress が渡された場合は、銘柄が終わるごとに progress(完了数, 全体数) を呼ぶ。
    戻り値: (records, exclusions, fetch_info)
      fetch_info: 銘柄ごとの取得結果（本数・最終日・成功可否・理由）
    """
    if run_at is None:
        run_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    records = []
    exclusions = []
    fetch_info = []
    for done, symbol in enumerate(symbols, start=1):
        raw, err = fetcher(symbol)
        if err is not None or raw is None or raw.empty:
            recs, excls = verify_symbol(symbol, pd.DataFrame(), period_label, run_at)
            records.extend(recs)
            exclusions.extend(excls)
            fetch_info.append({
                "銘柄": symbol, "取得": "失敗", "本数": 0,
                "データ最終日": "", "理由": err or "データなし",
            })
        else:
            recs, excls = verify_symbol(symbol, raw, period_label, run_at)
            records.extend(recs)
            exclusions.extend(excls)
            fetch_info.append({
                "銘柄": symbol, "取得": "成功", "本数": len(raw),
                "データ最終日": str(raw.index[-1].date()), "理由": "",
            })
        if progress is not None:
            progress(done, len(symbols))
    return records, exclusions, fetch_info


def aggregate(records, exclusions=None):
    """記録を集計する。的中率の分母は的中＋不的中のみ。

    対象外の件数は、予測記録とは区別された対象外集計（exclusions）から数える。
    戻り値: {"時間軸別": [...], "銘柄別": [...], "実際方向別内訳": [...],
            "取得失敗": [...]}
    """
    df = pd.DataFrame(records)
    by_horizon = []
    by_symbol = []
    breakdown = []
    failed = []
    exclusions = exclusions or []
    for ex in exclusions:
        if ex.get("時間軸") == "全体":
            failed.append(ex)

    def exclusion_count(horizon=None, symbol=None):
        total = 0
        for ex in exclusions:
            if ex.get("時間軸") == "全体":
                continue
            if horizon is not None and ex.get("時間軸") != horizon:
                continue
            if symbol is not None and ex.get("銘柄") != symbol:
                continue
            total += int(ex.get("件数") or 0)
        return total

    if df.empty:
        for horizon in HORIZONS:
            cnt = exclusion_count(horizon=horizon)
            if cnt:
                by_horizon.append({
                    "時間軸": horizon, "的中": 0, "不的中": 0, "的中率pct": None,
                    "判定件数": 0, "中立予測": 0, "中立割合pct": None,
                    "実際小動き": 0, "判定保留": 0,
                    "対象外": cnt, "方向付き騰落率_平均pct": None,
                    "方向付き騰落率_合計pct": None,
                    "精度判断": "件数不足のため精度を判断できない",
                })
        return {
            "時間軸別": by_horizon, "銘柄別": by_symbol,
            "実際方向別内訳": breakdown, "取得失敗": failed,
        }

    def summarize(sub):
        counts = {j: int((sub["判定"] == j).sum()) for j in JUDGED}
        denom = counts["的中"] + counts["不的中"]
        hit_rate = round(counts["的中"] / denom * 100, 1) if denom else None
        # 中立割合は、その時間軸の予測記録の総数（対象外集計は含まない）に対する割合
        neutral_rate = (
            round(counts["中立予測"] / len(sub) * 100, 1) if len(sub) else None
        )
        directional = sub[sub["方向付き騰落率pct"].notna()]
        pnl_mean = (
            round(float(directional["方向付き騰落率pct"].mean()), 3)
            if len(directional)
            else None
        )
        pnl_sum = (
            round(float(directional["方向付き騰落率pct"].sum()), 3)
            if len(directional)
            else None
        )
        return counts, denom, hit_rate, pnl_mean, pnl_sum, neutral_rate

    for horizon in HORIZONS:
        sub = df[df["時間軸"] == horizon]
        excl = exclusion_count(horizon=horizon)
        if sub.empty and not excl:
            continue
        if sub.empty:
            counts = {j: 0 for j in JUDGED}
            denom, hit_rate, pnl_mean, pnl_sum, neutral_rate = 0, None, None, None, None
        else:
            counts, denom, hit_rate, pnl_mean, pnl_sum, neutral_rate = summarize(sub)
        by_horizon.append({
            "時間軸": horizon,
            "的中": counts["的中"], "不的中": counts["不的中"],
            "的中率pct": hit_rate, "判定件数": denom,
            "中立予測": counts["中立予測"], "中立割合pct": neutral_rate,
            "実際小動き": counts["実際小動き"],
            "判定保留": counts["判定保留"], "対象外": excl,
            "方向付き騰落率_平均pct": pnl_mean,
            "方向付き騰落率_合計pct": pnl_sum,
            "精度判断": (
                "判断可能の目安に到達"
                if denom >= MIN_CASES_FOR_JUDGEMENT
                else "件数不足のため精度を判断できない"
            ),
        })

    for symbol in sorted(df["銘柄"].unique()):
        for horizon in HORIZONS:
            sub = df[(df["銘柄"] == symbol) & (df["時間軸"] == horizon)]
            excl = exclusion_count(horizon=horizon, symbol=symbol)
            if sub.empty and not excl:
                continue
            if sub.empty:
                counts = {j: 0 for j in JUDGED}
                denom, hit_rate = 0, None
            else:
                counts, denom, hit_rate, _, _, _ = summarize(sub)
            by_symbol.append({
                "銘柄": symbol, "時間軸": horizon,
                "的中": counts["的中"], "不的中": counts["不的中"],
                "的中率pct": hit_rate, "判定件数": denom,
                "中立予測": counts["中立予測"], "実際小動き": counts["実際小動き"],
                "判定保留": counts["判定保留"], "対象外": excl,
            })

    # 実際に上昇／下落した場面での予測内訳（見逃しの把握用）
    judged_rows = df[df["判定"].isin(["的中", "不的中", "中立予測", "実際小動き"])]
    for horizon in HORIZONS:
        for actual in ["上昇", "下落"]:
            sub = judged_rows[
                (judged_rows["時間軸"] == horizon)
                & (judged_rows["実際の方向"] == actual)
            ]
            if sub.empty:
                continue
            breakdown.append({
                "時間軸": horizon, "実際の方向": actual,
                "上昇予測": int((sub["予測方向"] == "上昇").sum()),
                "中立予測": int((sub["予測方向"] == "中立").sum()),
                "下落予測": int((sub["予測方向"] == "下落").sum()),
            })

    return {
        "時間軸別": by_horizon, "銘柄別": by_symbol,
        "実際方向別内訳": breakdown, "取得失敗": failed,
    }


def record_key(rec):
    """重複判定のキー（全記録で統一）：銘柄・評価日・時間軸・仕様版数。"""
    return (
        rec.get("銘柄"), rec.get("評価日"), rec.get("時間軸"), rec.get("仕様版数"),
    )


def _canonical(rec):
    """実行日時を除いた記録内容の比較用タプル。数値は float にそろえる。"""
    values = []
    for col in RECORD_COLUMNS:
        if col == "実行日時":
            continue
        val = rec.get(col)
        if val is None or val == "":
            values.append(None)
            continue
        try:
            values.append(float(val))
        except (TypeError, ValueError):
            values.append(str(val))
    return tuple(values)


def merge_records(existing, incoming):
    """記録を合算する。重複は除き、競合は黙って採用せず検出する。

    重複：キーが同じで内容（実行日時を除く）も同じ → 新しい方は捨てる。
    競合：キーが同じで内容が異なる → 既存を残し、新しい方は取り込まず、
    競合として返す。呼び出し側は競合を必ず利用者に通知すること。
    戻り値: (merged, conflicts)
    """
    merged = list(existing)
    by_key = {}
    for rec in merged:
        by_key.setdefault(record_key(rec), rec)
    conflicts = []
    for rec in incoming:
        key = record_key(rec)
        if key not in by_key:
            by_key[key] = rec
            merged.append(rec)
            continue
        current = by_key[key]
        if _canonical(current) == _canonical(rec):
            continue  # 完全な重複
        conflicts.append({
            "銘柄": key[0], "評価日": key[1], "時間軸": key[2], "仕様版数": key[3],
            "既存の判定": current.get("判定"), "新規の判定": rec.get("判定"),
            "既存のスコア": current.get("スコア"), "新規のスコア": rec.get("スコア"),
            "既存の実行日時": current.get("実行日時"),
            "新規の実行日時": rec.get("実行日時"),
        })
    return merged, conflicts


def dedupe_records(records):
    """同一リスト内の重複を除く（競合がある場合は最初の記録を残す）。"""
    merged, _ = merge_records([], records)
    return merged


EXCLUSION_COLUMNS = ["銘柄", "時間軸", "理由", "件数"]


def merge_exclusions(existing, incoming):
    """対象外集計を合算する。同じ銘柄・時間軸・理由は合算せず既存を残す。

    対象外の件数は「その実行での地点数」であり、同じ地点を二重に数えない
    ため、件数を足し合わせることはしない。
    """
    merged = list(existing)
    seen = {(e.get("銘柄"), e.get("時間軸"), e.get("理由")) for e in merged}
    for ex in incoming:
        key = (ex.get("銘柄"), ex.get("時間軸"), ex.get("理由"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(ex)
    return merged


def exclusions_to_csv(exclusions):
    df = pd.DataFrame(exclusions, columns=EXCLUSION_COLUMNS)
    return df.to_csv(index=False).encode("utf-8-sig")


def exclusions_from_csv(data_bytes):
    """対象外集計のCSVから読み戻す。形式が違う場合は空リストを返す。"""
    try:
        import io

        df = pd.read_csv(io.BytesIO(data_bytes), dtype=str, keep_default_na=False)
    except Exception:
        return []
    if not all(c in df.columns for c in EXCLUSION_COLUMNS):
        return []
    exclusions = []
    for _, row in df.iterrows():
        try:
            count = int(float(row["件数"]))
        except (TypeError, ValueError):
            continue
        exclusions.append({
            "銘柄": row["銘柄"], "時間軸": row["時間軸"],
            "理由": row["理由"], "件数": count,
        })
    return exclusions


def records_to_csv(records):
    df = pd.DataFrame(records, columns=RECORD_COLUMNS)
    return df.to_csv(index=False).encode("utf-8-sig")


def records_from_csv(data_bytes):
    """CSV（バイト列）から記録を読み戻す。失敗時は空リストを返す。"""
    try:
        import io

        df = pd.read_csv(io.BytesIO(data_bytes), dtype=str, keep_default_na=False)
    except Exception:
        return []
    if not all(c in df.columns for c in RECORD_COLUMNS):
        return []
    records = []
    numeric_cols = {"仕様版数", "スコア", "評価日終値", "測定日終値", "騰落率pct", "方向付き騰落率pct"}
    for _, row in df.iterrows():
        rec = {}
        for col in RECORD_COLUMNS:
            val = row[col]
            if col in numeric_cols:
                if val == "" or val is None:
                    rec[col] = None
                else:
                    try:
                        num = float(val)
                        rec[col] = int(num) if col in ("仕様版数", "スコア") and num == int(num) else num
                    except (TypeError, ValueError):
                        rec[col] = None
            else:
                rec[col] = val
        records.append(rec)
    return records
