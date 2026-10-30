"""フォワード登録と事後評価（第1段階、2026-10-30 設計確定）。

スキャン時点の結果をそのまま記録し、後日、実際の株価と比較する。
遡及型の「検証」（verify.py）が過去の評価日を抽出して測るのに対し、
こちらは「登録した予測を将来の実績で測る」ための仕組み。

設計上の約束:
- 登録はスキャン時点のデータだけで作る（未来情報を使わない）。
- 評価は、基準日より後の足だけを使う。リターンは「評価時に取得した
  1つの調整済み系列」の中で計算する（基準価格は、その系列における
  基準日の終値）。これにより記録後の株式分割・配当調整で歪まない。
  記録時点の終値は人間の確認用として別に保持する。
- 指数比較は、銘柄の基準日・評価日に対応する同じ期間で計算する。
  対応する指数データがない場合は無理に計算せず、理由を残す。
  TOPIX 指数そのものは無料取得できないため、TOPIX連動ETF（1306.T）を
  参考指標として併記する（指数そのものではない）。
- 最大上昇率・最大下落率は、評価用基準価格を分母に、基準日の翌足から
  評価日までの足の高値・安値で計算する（再現可能な定義として固定）。
- 保存は専用CSV（登録・評価の2種）のみ。既存の検証CSV・仮想売買メモ
  CSV・スキャンCSVの形式には触れない。
- 重複・競合の規約は verify.py と同一：キーが同じで内容も同じなら
  重複として捨て、内容が違えば既存を残して競合として返す。
  同一スキャン結果の再登録は重複になり、別途実行したスキャンは
  スキャンID（実行日時）が異なるため別記録として残る。
"""

from datetime import datetime

import pandas as pd

import analysis

# 記録形式の版数。列や意味を変える場合は上げる。
RECORD_VERSION = 1

# 評価する営業日数（基準日の後の足の本数で数える）。
CHECKPOINTS = (1, 5, 10, 20)

NIKKEI_SYMBOL = "^N225"
# TOPIX連動ETF。TOPIX指数そのものではなく、参考指標としての代替。
TOPIX_PROXY_SYMBOL = "1306.T"
TOPIX_PROXY_NAME = "TOPIX連動ETF(1306.T)"

HORIZONS = ("短期", "中期", "長期")

RECORD_COLUMNS = [
    "記録仕様版数", "スキャンID", "スキャン日時", "株価基準日",
    "コード", "会社名", "業種", "区分", "順位",
    "短期スコア", "短期ラベル", "中期スコア", "中期ラベル",
    "長期スコア", "長期ラベル",
    "短期予測方向", "中期予測方向", "長期予測方向",
    "記録時終値", "データ取得状況", "データ最終日", "本数",
    "選定理由", "警告・理由", "スコア仕様バージョン",
    "日経平均_基準値", "日経平均_基準日",
    "TOPIX連動ETF_基準値", "TOPIX連動ETF_基準日",
]

RECORD_NUMERIC = {
    "記録仕様版数", "順位", "短期スコア", "中期スコア", "長期スコア",
    "記録時終値", "本数", "スコア仕様バージョン",
    "日経平均_基準値", "TOPIX連動ETF_基準値",
}

RESULT_COLUMNS = [
    "記録仕様版数", "スキャンID", "コード", "会社名", "評価営業日数",
    "株価基準日", "評価日", "評価用基準価格", "評価価格", "リターンpct",
    "最大上昇率pct", "最大下落率pct",
    "日経平均_基準値", "日経平均_評価値", "日経平均_リターンpct", "日経超過リターンpct",
    "TOPIX連動ETF_基準値", "TOPIX連動ETF_評価値", "TOPIX連動ETF_リターンpct",
    "TOPIX超過リターンpct",
    "状態", "理由", "実行日時",
]

RESULT_NUMERIC = {
    "記録仕様版数", "評価営業日数", "評価用基準価格", "評価価格", "リターンpct",
    "最大上昇率pct", "最大下落率pct",
    "日経平均_基準値", "日経平均_評価値", "日経平均_リターンpct", "日経超過リターンpct",
    "TOPIX連動ETF_基準値", "TOPIX連動ETF_評価値", "TOPIX連動ETF_リターンpct",
    "TOPIX超過リターンpct",
}


def direction_from_score(score):
    """スコアから予測方向を決める（検証仕様 v1 と同じ規約）。

    +2以上は「上昇」、-2以下は「下落」、それ以外は「中立」。
    スコアが無い場合は「判定保留」。
    """
    if score is None:
        return "判定保留"
    try:
        value = float(score)
    except (TypeError, ValueError):
        return "判定保留"
    if value >= 2:
        return "上昇"
    if value <= -2:
        return "下落"
    return "中立"


def _data_status(row):
    """スキャン行からデータ取得状況を表す（区分と理由から機械的に決める）。"""
    if row.get("区分") == "ランキング":
        return "取得成功"
    reason = str(row.get("警告・理由") or "")
    if reason.startswith("取得失敗"):
        return "取得失敗"
    if "データが古い" in reason:
        return "データが古い可能性"
    if "中期評価ができません" in reason:
        return "データ不足"
    return "その他"


def build_records(ranked, excluded, meta, benchmarks=None):
    """スキャン結果（run_scan の戻り値）からフォワード登録レコードを作る。

    benchmarks: {銘柄コード: (終値, 日付文字列)}。指数の基準値。
    取れなかった指数は空欄にし、捏造しない。
    """
    benchmarks = benchmarks or {}
    scan_id = str(meta.get("実行日時", ""))
    nikkei = benchmarks.get(NIKKEI_SYMBOL) or (None, "")
    topix = benchmarks.get(TOPIX_PROXY_SYMBOL) or (None, "")
    records = []
    for row in list(ranked) + list(excluded):
        records.append({
            "記録仕様版数": RECORD_VERSION,
            "スキャンID": scan_id,
            "スキャン日時": scan_id,
            "株価基準日": row.get("データ最終日", ""),
            "コード": row.get("コード", ""),
            "会社名": row.get("会社名", ""),
            "業種": row.get("業種", ""),
            "区分": row.get("区分", ""),
            "順位": row.get("順位"),
            "短期スコア": row.get("短期スコア"),
            "短期ラベル": row.get("短期ラベル", ""),
            "中期スコア": row.get("中期スコア"),
            "中期ラベル": row.get("中期ラベル", ""),
            "長期スコア": row.get("長期スコア"),
            "長期ラベル": row.get("長期ラベル", ""),
            "短期予測方向": direction_from_score(row.get("短期スコア")),
            "中期予測方向": direction_from_score(row.get("中期スコア")),
            "長期予測方向": direction_from_score(row.get("長期スコア")),
            "記録時終値": row.get("終値"),
            "データ取得状況": _data_status(row),
            "データ最終日": row.get("データ最終日", ""),
            "本数": row.get("本数", 0),
            "選定理由": row.get("選定理由", ""),
            "警告・理由": row.get("警告・理由", ""),
            "スコア仕様バージョン": analysis.SCORING_VERSION,
            "日経平均_基準値": nikkei[0],
            "日経平均_基準日": nikkei[1],
            "TOPIX連動ETF_基準値": topix[0],
            "TOPIX連動ETF_基準日": topix[1],
        })
    return records


# ============================================================
# CSV 入出力（verify.py と同じ方式：文字列で読み、数値列だけ型を戻す）
# ============================================================

def _to_csv(records, columns):
    df = pd.DataFrame(records, columns=columns)
    return df.to_csv(index=False).encode("utf-8-sig")


def _from_csv(data_bytes, columns, numeric_cols, int_cols=()):
    try:
        import io

        df = pd.read_csv(io.BytesIO(data_bytes), dtype=str, keep_default_na=False)
    except Exception:
        return []
    if not all(c in df.columns for c in columns):
        return []
    records = []
    for _, row in df.iterrows():
        rec = {}
        for col in columns:
            val = row[col]
            if col in numeric_cols:
                if val == "" or val is None:
                    rec[col] = None
                else:
                    try:
                        num = float(val)
                        rec[col] = int(num) if col in int_cols and num == int(num) else num
                    except (TypeError, ValueError):
                        rec[col] = None
            else:
                rec[col] = val
        records.append(rec)
    return records


def records_to_csv(records):
    return _to_csv(records, RECORD_COLUMNS)


def records_from_csv(data_bytes):
    return _from_csv(
        data_bytes, RECORD_COLUMNS, RECORD_NUMERIC,
        int_cols=("記録仕様版数", "順位", "短期スコア", "中期スコア", "長期スコア",
                  "本数", "スコア仕様バージョン"),
    )


def results_to_csv(results):
    return _to_csv(results, RESULT_COLUMNS)


def results_from_csv(data_bytes):
    return _from_csv(
        data_bytes, RESULT_COLUMNS, RESULT_NUMERIC,
        int_cols=("記録仕様版数", "評価営業日数"),
    )


# ============================================================
# 重複・競合の統合（verify.py と同一の規約）
# ============================================================

def _canonical(rec, columns, ignore_cols):
    values = []
    for col in columns:
        if col in ignore_cols:
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


def _merge(existing, incoming, columns, key_fn, ignore_cols, conflict_info):
    merged = list(existing)
    by_key = {}
    for rec in merged:
        by_key.setdefault(key_fn(rec), rec)
    conflicts = []
    for rec in incoming:
        key = key_fn(rec)
        if key not in by_key:
            by_key[key] = rec
            merged.append(rec)
            continue
        current = by_key[key]
        if _canonical(current, columns, ignore_cols) == _canonical(rec, columns, ignore_cols):
            continue  # 完全な重複
        conflicts.append(conflict_info(key, current, rec))
    return merged, conflicts


def record_key(rec):
    """登録の重複判定キー：スキャンIDと銘柄コード。"""
    return (rec.get("スキャンID"), rec.get("コード"))


def merge_records(existing, incoming):
    """登録レコードを合算する。戻り値: (merged, conflicts)。"""
    def info(key, current, rec):
        return {
            "スキャンID": key[0], "コード": key[1],
            "既存の記録時終値": current.get("記録時終値"),
            "新規の記録時終値": rec.get("記録時終値"),
            "既存の株価基準日": current.get("株価基準日"),
            "新規の株価基準日": rec.get("株価基準日"),
        }

    return _merge(existing, incoming, RECORD_COLUMNS, record_key, (), info)


def result_key(rec):
    """評価結果の重複判定キー：スキャンID・銘柄コード・評価営業日数。"""
    return (rec.get("スキャンID"), rec.get("コード"), rec.get("評価営業日数"))


def merge_results(existing, incoming):
    """評価結果を合算する。実行日時は比較から除く。戻り値: (merged, conflicts)。"""
    def info(key, current, rec):
        return {
            "スキャンID": key[0], "コード": key[1], "評価営業日数": key[2],
            "既存の状態": current.get("状態"), "新規の状態": rec.get("状態"),
            "既存のリターンpct": current.get("リターンpct"),
            "新規のリターンpct": rec.get("リターンpct"),
            "既存の実行日時": current.get("実行日時"),
            "新規の実行日時": rec.get("実行日時"),
        }

    return _merge(existing, incoming, RESULT_COLUMNS, result_key, ("実行日時",), info)


# ============================================================
# 事後評価
# ============================================================

def _date_strings(df):
    """DataFrame のインデックスを日付文字列（YYYY-MM-DD）のリストにする。"""
    out = []
    for ts in df.index:
        try:
            out.append(pd.Timestamp(ts).date().isoformat())
        except Exception:
            out.append(str(ts)[:10])
    return out


def _benchmark_series(fetcher):
    """指数の {日付文字列: 終値} を作る。取れなければ空 dict。"""
    series = {}
    for symbol in (NIKKEI_SYMBOL, TOPIX_PROXY_SYMBOL):
        try:
            df, err = fetcher(symbol)
        except Exception:
            df, err = None, "取得に失敗しました"
        if err is not None or df is None or df.empty or "Close" not in df.columns:
            series[symbol] = {}
            continue
        dates = _date_strings(df)
        closes = df["Close"].tolist()
        series[symbol] = {d: float(c) for d, c in zip(dates, closes)}
    return series


def _pct(new, base):
    return round((float(new) / float(base) - 1.0) * 100.0, 3)


def evaluate_records(records, fetcher, evaluated_at=None, checkpoints=CHECKPOINTS):
    """登録レコードを事後評価する。

    fetcher(symbol) -> (DataFrame, error) 。analysis.fetch_history と同じ契約。
    戻り値: (results, pending, failed)
      results: 評価結果レコード（評価済み／評価不能）
      pending: まだ評価時点に到達していないもの（表示用、保存しない）
      failed: 価格取得に失敗した銘柄（表示用）
    """
    if evaluated_at is None:
        evaluated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    benchmarks = _benchmark_series(fetcher)
    results = []
    pending = []
    failed = []
    cache = {}
    for rec in records:
        if rec.get("区分") != "ランキング":
            continue  # 参考外は記録のみで、評価対象にしない
        symbol = rec.get("コード", "")
        if symbol not in cache:
            try:
                cache[symbol] = fetcher(symbol)
            except Exception:
                cache[symbol] = (None, "取得に失敗しました")
        df, err = cache[symbol]
        base_date = str(rec.get("株価基準日", ""))
        if err is not None or df is None or df.empty:
            failed.append({"コード": symbol, "理由": f"取得失敗：{err or 'データなし'}"})
            continue
        dates = _date_strings(df)
        if base_date not in dates:
            results.append({
                "記録仕様版数": RECORD_VERSION,
                "スキャンID": rec.get("スキャンID", ""), "コード": symbol,
                "会社名": rec.get("会社名", ""), "評価営業日数": None,
                "株価基準日": base_date, "評価日": "",
                "評価用基準価格": None, "評価価格": None, "リターンpct": None,
                "最大上昇率pct": None, "最大下落率pct": None,
                "日経平均_基準値": None, "日経平均_評価値": None,
                "日経平均_リターンpct": None, "日経超過リターンpct": None,
                "TOPIX連動ETF_基準値": None, "TOPIX連動ETF_評価値": None,
                "TOPIX連動ETF_リターンpct": None, "TOPIX超過リターンpct": None,
                "状態": "評価不能",
                "理由": "基準日のデータが取得した価格系列にありません"
                        "（上場廃止・データ改訂の可能性があります）",
                "実行日時": evaluated_at,
            })
            continue
        base_pos = dates.index(base_date)
        base_price = float(df["Close"].iloc[base_pos])
        for n in checkpoints:
            eval_pos = base_pos + n
            if eval_pos >= len(df):
                pending.append({
                    "スキャンID": rec.get("スキャンID", ""), "コード": symbol,
                    "評価営業日数": n,
                    "理由": f"基準日から{n}営業日にまだ到達していません",
                })
                continue
            eval_date = dates[eval_pos]
            eval_price = float(df["Close"].iloc[eval_pos])
            window = df.iloc[base_pos + 1: eval_pos + 1]
            ret = _pct(eval_price, base_price)
            row = {
                "記録仕様版数": RECORD_VERSION,
                "スキャンID": rec.get("スキャンID", ""), "コード": symbol,
                "会社名": rec.get("会社名", ""), "評価営業日数": n,
                "株価基準日": base_date, "評価日": eval_date,
                "評価用基準価格": round(base_price, 3),
                "評価価格": round(eval_price, 3),
                "リターンpct": ret,
                "最大上昇率pct": _pct(float(window["High"].max()), base_price),
                "最大下落率pct": _pct(float(window["Low"].min()), base_price),
                "状態": "評価済み", "理由": "", "実行日時": evaluated_at,
            }
            reasons = []
            for symbol_b, label, c_base, c_eval, c_ret, c_excess in (
                (NIKKEI_SYMBOL, "日経平均", "日経平均_基準値", "日経平均_評価値",
                 "日経平均_リターンpct", "日経超過リターンpct"),
                (TOPIX_PROXY_SYMBOL, "TOPIX連動ETF", "TOPIX連動ETF_基準値",
                 "TOPIX連動ETF_評価値", "TOPIX連動ETF_リターンpct", "TOPIX超過リターンpct"),
            ):
                series = benchmarks.get(symbol_b) or {}
                b_base = series.get(base_date)
                b_eval = series.get(eval_date)
                if b_base is None or b_eval is None:
                    row[c_base] = None
                    row[c_eval] = None
                    row[c_ret] = None
                    row[c_excess] = None
                    reasons.append(f"{label}の対応する期間のデータがないため超過リターンは計算していません")
                else:
                    b_ret = _pct(b_eval, b_base)
                    row[c_base] = round(float(b_base), 3)
                    row[c_eval] = round(float(b_eval), 3)
                    row[c_ret] = b_ret
                    row[c_excess] = round(ret - b_ret, 3)
            row["理由"] = " / ".join(reasons)
            results.append(row)
    return results, pending, failed


# ============================================================
# 集計
# ============================================================

def _mean(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 3)


def summarize(results, records):
    """評価結果を集計する。

    戻り値: {
      "時点別": {評価営業日数: {件数, 平均リターンpct, 上昇割合pct,
                              平均日経超過pct, 平均TOPIX超過pct}},
      "順位帯別": {評価営業日数: {帯: {件数, 平均リターンpct, 上昇割合pct,
                                    平均TOPIX超過pct}}},
      "時間軸別": {(評価営業日数, 時間軸): {方向件数, 的中率pct,
                                          方向調整後平均リターンpct}},
    }
    的中率は方向（上昇／下落）の予測だけを分母にし、中立・判定保留は
    含めない（検証仕様 v1 と同じ考え方）。
    """
    rec_by_key = {record_key(r): r for r in records}
    evaluated = [r for r in results if r.get("状態") == "評価済み"]
    by_point = {}
    bands = {}
    horizons = {}
    for res in evaluated:
        n = res.get("評価営業日数")
        ret = res.get("リターンpct")
        point = by_point.setdefault(n, {"リターン": [], "日経超過": [], "TOPIX超過": [], "上昇": 0})
        point["リターン"].append(ret)
        point["日経超過"].append(res.get("日経超過リターンpct"))
        point["TOPIX超過"].append(res.get("TOPIX超過リターンpct"))
        if ret is not None and ret > 0:
            point["上昇"] += 1
        rec = rec_by_key.get((res.get("スキャンID"), res.get("コード"))) or {}
        rank = rec.get("順位")
        band_map = bands.setdefault(n, {})
        for band_name, limit in (("上位5", 5), ("上位10", 10), ("上位20", 20), ("全体", None)):
            if limit is not None and not (isinstance(rank, (int, float)) and rank <= limit):
                continue
            band = band_map.setdefault(band_name, {"リターン": [], "TOPIX超過": [], "上昇": 0})
            band["リターン"].append(ret)
            band["TOPIX超過"].append(res.get("TOPIX超過リターンpct"))
            if ret is not None and ret > 0:
                band["上昇"] += 1
        for horizon in HORIZONS:
            direction = rec.get(f"{horizon}予測方向")
            if direction not in ("上昇", "下落") or ret is None:
                continue
            cell = horizons.setdefault((n, horizon), {"件数": 0, "的中": 0, "調整後": []})
            sign = 1.0 if direction == "上昇" else -1.0
            cell["件数"] += 1
            if ret * sign > 0:
                cell["的中"] += 1
            cell["調整後"].append(round(ret * sign, 3))

    def _rate(up, total):
        return round(up / total * 100.0, 1) if total else None

    point_out = {}
    for n, p in sorted(by_point.items(), key=lambda kv: (kv[0] is None, kv[0])):
        total = len(p["リターン"])
        point_out[n] = {
            "件数": total,
            "平均リターンpct": _mean(p["リターン"]),
            "上昇割合pct": _rate(p["上昇"], total),
            "平均日経超過pct": _mean(p["日経超過"]),
            "平均TOPIX超過pct": _mean(p["TOPIX超過"]),
        }
    band_out = {}
    for n, band_map in bands.items():
        band_out[n] = {}
        for band_name in ("上位5", "上位10", "上位20", "全体"):
            if band_name not in band_map:
                continue
            b = band_map[band_name]
            total = len(b["リターン"])
            band_out[n][band_name] = {
                "件数": total,
                "平均リターンpct": _mean(b["リターン"]),
                "上昇割合pct": _rate(b["上昇"], total),
                "平均TOPIX超過pct": _mean(b["TOPIX超過"]),
            }
    horizon_out = {}
    for (n, horizon), cell in horizons.items():
        horizon_out[(n, horizon)] = {
            "方向件数": cell["件数"],
            "的中率pct": _rate(cell["的中"], cell["件数"]),
            "方向調整後平均リターンpct": _mean(cell["調整後"]),
        }
    return {"時点別": point_out, "順位帯別": band_out, "時間軸別": horizon_out}
