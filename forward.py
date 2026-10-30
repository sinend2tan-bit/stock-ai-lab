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

V2 並行比較の追加（2026-10-30 最終設計）:
- 登録・結果は同じCSVのまま、スコア仕様バージョン列で V1／V2 を区別する。
  重複キーにも版を含める。結果CSVには版列を、登録CSVには過熱フラグ列を
  末尾に追加した（記録形式の版数を 2 に上げた）。旧形式のCSVは、追加列を
  既定値で補って読み込む（登録：過熱フラグは空欄、結果：版は 1）。
  記録仕様版数そのものは重複判定の内容比較から外し、形式の版上げだけで
  既存データが競合扱いにならないようにする。
- V2 の行は build_records の引数（scoring_version／direction_fn／
  overheat_flags）で作る。既定値は V1 の従来動作。
- compare_versions が V1・V2 の比較（16セルの行列、共通母集団の性能比較、
  運用判断比較、片側登録の切り分け、採用ゲート）を担う。
"""

from datetime import datetime

import pandas as pd

import analysis

# 記録形式の版数。列や意味を変える場合は上げる。
# 2: 結果CSVへスコア仕様バージョン列、登録CSVへ過熱フラグ列を追加。
RECORD_VERSION = 2

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
    "過熱フラグ",
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
    "スコア仕様バージョン",
]

RESULT_NUMERIC = {
    "記録仕様版数", "評価営業日数", "評価用基準価格", "評価価格", "リターンpct",
    "最大上昇率pct", "最大下落率pct",
    "日経平均_基準値", "日経平均_評価値", "日経平均_リターンpct", "日経超過リターンpct",
    "TOPIX連動ETF_基準値", "TOPIX連動ETF_評価値", "TOPIX連動ETF_リターンpct",
    "TOPIX超過リターンpct", "スコア仕様バージョン",
}

# V2 を再取得データで計算した場合に、記録の警告・理由へ残す注記。
# 比較側はこの文字列でフォールバック由来の記録を数える。
FALLBACK_NOTE = "V2計算は再取得データ（基準日検査済み）で実施"

# 採用ゲートの主判定エンドポイントと比較の前提（最終設計で凍結）。
PRIMARY_HORIZON = "中期"
PRIMARY_CHECKPOINT = 20
V1_DIRECTIONS = ("上昇", "下落", "中立", "判定保留")
V2_DIRECTIONS = ("上昇", "下落注意", "中立", "判定保留")
GATE_MIN_COHORTS = 20
GATE_MIN_SPAN_DAYS = 90
GATE_MIN_PAIRED = 200
GATE_MIN_NON_RISING_COHORTS = 5
GATE_CI_LOWER_BOUND = -0.3
GATE_HIT_NONINFERIORITY = 2.0
GATE_SYMBOL_DEPENDENCE = 0.5


def _version_of(rec):
    """レコードのスコア仕様バージョンを int で返す。欠けていれば 1（V1）。"""
    try:
        return int(float(rec.get("スコア仕様バージョン")))
    except (TypeError, ValueError):
        return 1


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


def build_records(ranked, excluded, meta, benchmarks=None, scoring_version=None,
                  direction_fn=None, overheat_flags=None):
    """スキャン結果（run_scan の戻り値）からフォワード登録レコードを作る。

    benchmarks: {銘柄コード: (終値, 日付文字列)}。指数の基準値。
    取れなかった指数は空欄にし、捏造しない。
    scoring_version／direction_fn／overheat_flags は V2 用の引数で、
    既定値では V1 の従来動作になる。overheat_flags は {コード: bool}。
    """
    benchmarks = benchmarks or {}
    if scoring_version is None:
        scoring_version = analysis.SCORING_VERSION
    if direction_fn is None:
        direction_fn = direction_from_score
    overheat_flags = overheat_flags or {}
    scan_id = str(meta.get("実行日時", ""))
    nikkei = benchmarks.get(NIKKEI_SYMBOL) or (None, "")
    topix = benchmarks.get(TOPIX_PROXY_SYMBOL) or (None, "")
    records = []
    for row in list(ranked) + list(excluded):
        code = row.get("コード", "")
        records.append({
            "記録仕様版数": RECORD_VERSION,
            "スキャンID": scan_id,
            "スキャン日時": scan_id,
            "株価基準日": row.get("データ最終日", ""),
            "コード": code,
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
            "短期予測方向": direction_fn(row.get("短期スコア")),
            "中期予測方向": direction_fn(row.get("中期スコア")),
            "長期予測方向": direction_fn(row.get("長期スコア")),
            "記録時終値": row.get("終値"),
            "データ取得状況": _data_status(row),
            "データ最終日": row.get("データ最終日", ""),
            "本数": row.get("本数", 0),
            "選定理由": row.get("選定理由", ""),
            "警告・理由": row.get("警告・理由", ""),
            "スコア仕様バージョン": scoring_version,
            "日経平均_基準値": nikkei[0],
            "日経平均_基準日": nikkei[1],
            "TOPIX連動ETF_基準値": topix[0],
            "TOPIX連動ETF_基準日": topix[1],
            "過熱フラグ": 1 if overheat_flags.get(code) else "",
        })
    return records


# ============================================================
# CSV 入出力（verify.py と同じ方式：文字列で読み、数値列だけ型を戻す）
# ============================================================

def _to_csv(records, columns):
    df = pd.DataFrame(records, columns=columns)
    return df.to_csv(index=False).encode("utf-8-sig")


def _from_csv(data_bytes, columns, numeric_cols, int_cols=(), fill_defaults=None):
    try:
        import io

        df = pd.read_csv(io.BytesIO(data_bytes), dtype=str, keep_default_na=False)
    except Exception:
        return []
    # 旧形式のファイルには後から追加した列がない。既定値が定められた列だけ
    # 補い、それ以外の列が欠けているファイルは従来どおり読み込まない。
    for col, default in (fill_defaults or {}).items():
        if col not in df.columns:
            df[col] = default
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
        fill_defaults={"過熱フラグ": ""},
    )


def results_to_csv(results):
    return _to_csv(results, RESULT_COLUMNS)


def results_from_csv(data_bytes):
    return _from_csv(
        data_bytes, RESULT_COLUMNS, RESULT_NUMERIC,
        int_cols=("記録仕様版数", "評価営業日数", "スコア仕様バージョン"),
        fill_defaults={"スコア仕様バージョン": "1"},
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
    """登録の重複判定キー：スキャンID・銘柄コード・スコア仕様バージョン。"""
    return (rec.get("スキャンID"), rec.get("コード"), _version_of(rec))


def merge_records(existing, incoming):
    """登録レコードを合算する。戻り値: (merged, conflicts)。

    記録仕様版数（ファイル形式の版）は内容比較から外す。形式の版上げだけで
    既存の記録が競合扱いにならないようにするため。
    """
    def info(key, current, rec):
        return {
            "スキャンID": key[0], "コード": key[1], "スコア仕様バージョン": key[2],
            "既存の記録時終値": current.get("記録時終値"),
            "新規の記録時終値": rec.get("記録時終値"),
            "既存の株価基準日": current.get("株価基準日"),
            "新規の株価基準日": rec.get("株価基準日"),
        }

    return _merge(existing, incoming, RECORD_COLUMNS, record_key, ("記録仕様版数",), info)


def result_key(rec):
    """評価結果の重複判定キー：スキャンID・銘柄コード・評価営業日数・版。"""
    return (rec.get("スキャンID"), rec.get("コード"), rec.get("評価営業日数"), _version_of(rec))


def merge_results(existing, incoming):
    """評価結果を合算する。実行日時と記録仕様版数は比較から除く。戻り値: (merged, conflicts)。"""
    def info(key, current, rec):
        return {
            "スキャンID": key[0], "コード": key[1], "評価営業日数": key[2],
            "スコア仕様バージョン": key[3],
            "既存の状態": current.get("状態"), "新規の状態": rec.get("状態"),
            "既存のリターンpct": current.get("リターンpct"),
            "新規のリターンpct": rec.get("リターンpct"),
            "既存の実行日時": current.get("実行日時"),
            "新規の実行日時": rec.get("実行日時"),
        }

    return _merge(existing, incoming, RESULT_COLUMNS, result_key, ("実行日時", "記録仕様版数"), info)


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
                "スコア仕様バージョン": _version_of(rec),
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
                "スコア仕様バージョン": _version_of(rec),
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


def summarize(results, records, version=None):
    """評価結果を集計する。

    version を指定すると、そのスコア仕様バージョンの記録・結果だけに
    絞って集計する（V1・V2 の版別集計用）。既定は全版まとめての従来動作。

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
    if version is not None:
        records = [r for r in records if _version_of(r) == version]
        results = [r for r in results if _version_of(r) == version]
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
        rec = rec_by_key.get((res.get("スキャンID"), res.get("コード"), _version_of(res))) or {}
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


# ============================================================
# V2 用のスキャン行の構築と、再取得データの基準日検査
# ============================================================

def _rank_key(row):
    """V2 の順位付け（scan._score_key と同じ規則：中期→長期→短期→コード）。"""
    def val(score):
        return score if isinstance(score, (int, float)) else -999
    return (-val(row["中期スコア"]), -val(row["長期スコア"]), -val(row["短期スコア"]), row["コード"])


def prepare_v2_frame(raw, base_date, base_close):
    """再取得した株価履歴を検査し、基準日までに切り取って返す。

    戻り値: (frame, reason)。検査に失敗した場合は (None, 理由)。
    検査内容（最終設計の修正3）:
    (a) 基準日の行が存在すること、(b) 切り取り後の最終日が基準日と一致
    すること、(c) 切り取り後の最終終値とスキャン時の終値の相対差が
    2% 以内であること。基準日より後のバーは計算に混入しない。
    """
    if raw is None or len(raw) == 0:
        return None, "V2計算用データが取得できませんでした"
    dates = _date_strings(raw)
    if base_date not in dates:
        return None, "再取得データにスキャン基準日がありません"
    pos = dates.index(base_date)
    truncated = raw.iloc[:pos + 1]
    if base_close:
        try:
            last_close = float(truncated["Close"].iloc[-1])
            base_value = float(base_close)
        except (TypeError, ValueError):
            return None, "基準日の終値を確認できませんでした"
        if base_value and abs(last_close / base_value - 1.0) * 100.0 > 2.0:
            return None, "再取得データの基準日終値がスキャン時と2%以上異なります"
    return truncated, ""


def build_v2_scan_rows(ranked, excluded, assessments, failed=None):
    """V1 のスキャン行と V2 の評価結果から、V2 用のスキャン行を作る。

    assessments: {コード: {"results": assess_all_v2 の戻り値, "overheat": bool}}
    failed: {コード: 理由}（基準日検査などで V2 を計算できなかった銘柄）
    戻り値: (ranked_v2, excluded_v2, overheat_flags, skipped)

    原則（代用の禁止）:
    - V2 の行のスコア・ラベルは、必ず scoring_v2 が計算した値だけを使う。
      V1 のスコアを V2 の結果として流用することはしない。
    - V2 を計算できなかった銘柄（failed、または評価結果が用意できない
      銘柄）は V2 の行自体を作らない。skipped に {コード, 理由} として
      返し、登録側・比較側では「V2未登録（片側登録）」として扱う。
    - V1 から引き継ぐのは、銘柄情報・終値・データ最終日・本数などの
      スキャン時点の事実と、データの古さによる参考外の理由（鮮度は
      バージョンと無関係なデータ側の事情）だけであり、予測の中身は
      引き継がない。鮮度で参考外になる銘柄も、スコアは V2 自身の値を残す。
    """
    failed = failed or {}
    overheat_flags = {}
    ranked_v2 = []
    excluded_v2 = []
    skipped = []

    def base_row(src):
        return {
            "区分": "", "順位": None, "コード": src.get("コード", ""),
            "会社名": src.get("会社名", ""), "業種": src.get("業種", ""),
            "中期スコア": None, "中期ラベル": "", "短期スコア": None, "短期ラベル": "",
            "長期スコア": None, "長期ラベル": "", "終値": src.get("終値"),
            "選定理由": "", "警告・理由": "", "データ最終日": src.get("データ最終日", ""),
            "本数": src.get("本数", 0), "取得日時": src.get("取得日時", ""),
        }

    def place(src):
        code = src.get("コード", "")
        if code in failed:
            skipped.append({"コード": code, "理由": failed[code]})
            return
        entry = assessments.get(code)
        if entry is None:
            reason = str(src.get("警告・理由") or "")
            if not reason.startswith("取得失敗"):
                reason = "V2計算用データが用意できませんでした"
            skipped.append({"コード": code, "理由": reason})
            return
        overheat_flags[code] = bool(entry.get("overheat"))
        results = entry["results"]
        row = base_row(src)
        for horizon in HORIZONS:
            row[f"{horizon}スコア"] = results[horizon]["score"]
            row[f"{horizon}ラベル"] = results[horizon]["label"]
        mid = results["中期"]
        src_reason = str(src.get("警告・理由") or "")
        if src.get("区分") == "参考外" and src_reason.startswith("データが古い可能性があります"):
            # 鮮度による参考外はデータ側の事情として引き継ぐ（スコアは V2 の値）
            row["区分"] = "参考外"
            row["警告・理由"] = src_reason
            excluded_v2.append(row)
            return
        if not mid["available"]:
            row["区分"] = "参考外"
            row["警告・理由"] = "中期評価ができません（データ不足）：" + " ".join(mid["reasons"])
            excluded_v2.append(row)
            return
        row["区分"] = "ランキング"
        row["選定理由"] = " / ".join(mid["reasons"])
        warnings = list(mid["warnings"]) + list(results["短期"]["warnings"])
        if "データ最終日から間隔があります" in src_reason:
            warnings.append("データ最終日から間隔があります（週末・休場の可能性）")
        row["警告・理由"] = " / ".join(warnings)
        ranked_v2.append(row)

    for src in list(ranked) + list(excluded):
        place(src)

    ranked_v2.sort(key=_rank_key)
    for i, row in enumerate(ranked_v2, start=1):
        row["順位"] = i
    return ranked_v2, excluded_v2, overheat_flags, skipped


# ============================================================
# V1・V2 並行比較
# ============================================================

_T_CRIT_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080,
    22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
    29: 2.045, 30: 2.042,
}


def _t_crit(df):
    return _T_CRIT_975.get(df, 1.96)


def _stance_v1(direction):
    if direction == "上昇":
        return 1.0
    if direction == "下落":
        return -1.0
    return 0.0


def _stance_v2(direction):
    return 1.0 if direction == "上昇" else 0.0


def compare_versions(results, records, regime_by_scan=None):
    """V1（版1）と V2（版2）のフォワード記録・結果を比較する。

    前提：評価結果のリターン等は価格のみで決まるため両版で同一であり、
    結果行はどちらか一方の版のものを単位の値として使う（両版の記録が
    そろっている銘柄・スキャンだけを比較対象にする）。片方の版しか
    登録されていない銘柄は比較に入れず、片側登録として別に数える。

    戻り値の主なキー:
      "行列": {(評価営業日数, 時間軸): {(V1方向, V2方向): {件数,
              V1方向付き平均リターンpct, V2方向付き平均リターンpct,
              差分平均pct, 差分合計pct, 超過件数, 差分平均TOPIX超過pct}}}
      "完全性": {(評価営業日数, 時間軸): {対象件数, セル合計件数,
              未分類件数, 差分合計pct, セル差分合計pct, 一致}}
      "片側登録": [{スキャンID, コード, 不足側}], "片側登録件数": {"V1のみ", "V2のみ"}
      "評価待ち": {評価営業日数: 件数}
      "共通母集団": {(評価営業日数, 時間軸): {件数, V1的中率pct, V2的中率pct,
              常時上昇pct, V1平均方向付きTOPIX超過pct, V2平均方向付きTOPIX超過pct,
              ペア差分平均TOPIX超過pct}}
      "見送り": {(評価営業日数, 時間軸): {V2方向: {件数, 回避件数, 回避合計pct,
              機会損失件数, 機会損失合計pct, 純影響pct, 平均最大下落率pct}}}
      "コホート別": {スキャンID: {件数, 差分平均pct}}（主エンドポイント）
      "ゲート": {項目: [{項目, 基準, 実績, 判定}], 充足,
              コホート平均差分pct, 信頼区間95, 銘柄依存, 再取得フォールバック}
    的中率はフォワード集計と同じ規約（方向付きリターンの符号）で数える。
    """
    rec_map = {}
    for rec in records:
        rec_map[(rec.get("スキャンID"), rec.get("コード"), _version_of(rec))] = rec
    res_map = {}
    for res in results:
        if res.get("状態") != "評価済み":
            continue
        res_map[(res.get("スキャンID"), res.get("コード"),
                 res.get("評価営業日数"), _version_of(res))] = res

    pair_keys = set()
    one_sided = []
    one_sided_counts = {"V1のみ": 0, "V2のみ": 0}
    seen_pairs = set()
    for (scan_id, code, version) in rec_map:
        if (scan_id, code) in seen_pairs:
            continue
        seen_pairs.add((scan_id, code))
        has_v1 = (scan_id, code, 1) in rec_map
        has_v2 = (scan_id, code, 2) in rec_map
        if has_v1 and has_v2:
            pair_keys.add((scan_id, code))
        elif has_v1 or has_v2:
            missing = "V2" if has_v1 else "V1"
            if missing == "V2":
                existing = rec_map[(scan_id, code, 1)]
                if existing.get("データ取得状況") == "取得失敗":
                    reason = "株価データの取得に失敗したため、V2も計算対象なし"
                else:
                    reason = "V2の計算または基準日検査に失敗したため未登録（詳細は登録時に表示）"
            else:
                reason = "V1が未登録（V2のみ登録された記録）"
            one_sided.append({
                "スキャンID": scan_id, "コード": code, "不足側": missing, "理由": reason,
            })
            one_sided_counts["V1のみ" if has_v1 else "V2のみ"] += 1

    cells = {}
    completeness = {}
    common = {}
    skip = {}
    pending = {}
    cohort_units = {}
    symbol_units = {}

    for (scan_id, code) in sorted(pair_keys):
        v1 = rec_map[(scan_id, code, 1)]
        v2 = rec_map[(scan_id, code, 2)]
        for n in CHECKPOINTS:
            res = res_map.get((scan_id, code, n, 1)) or res_map.get((scan_id, code, n, 2))
            if res is None:
                pending[n] = pending.get(n, 0) + 1
                continue
            ret = res.get("リターンpct")
            if ret is None:
                pending[n] = pending.get(n, 0) + 1
                continue
            exc = res.get("TOPIX超過リターンpct")
            max_down = res.get("最大下落率pct")
            for horizon in HORIZONS:
                d1 = v1.get(f"{horizon}予測方向")
                d2 = v2.get(f"{horizon}予測方向")
                key = (n, horizon)
                comp = completeness.setdefault(key, {
                    "対象件数": 0, "セル合計件数": 0, "未分類件数": 0,
                    "差分合計pct": 0.0, "セル差分合計pct": 0.0,
                })
                comp["対象件数"] += 1
                # 全体の差分はセル集計とは別経路で積み上げる（完全性の検算用）
                comp["差分合計pct"] += (_stance_v2(d2) - _stance_v1(d1)) * ret
                if d1 not in V1_DIRECTIONS or d2 not in V2_DIRECTIONS:
                    comp["未分類件数"] += 1
                    continue
                s1 = _stance_v1(d1)
                s2 = _stance_v2(d2)
                diff = (s2 - s1) * ret
                comp["セル合計件数"] += 1
                comp["セル差分合計pct"] += diff
                cell = cells.setdefault(key, {}).setdefault((d1, d2), {
                    "件数": 0, "V1方向付き合計": 0.0, "V2方向付き合計": 0.0,
                    "差分合計pct": 0.0, "超過件数": 0, "差分TOPIX超過合計": 0.0,
                })
                cell["件数"] += 1
                cell["V1方向付き合計"] += s1 * ret
                cell["V2方向付き合計"] += s2 * ret
                cell["差分合計pct"] += diff
                if exc is not None:
                    cell["超過件数"] += 1
                    cell["差分TOPIX超過合計"] += (s2 - s1) * exc
                # 共通母集団（両版とも方向の立場がある単位）
                if s1 != 0.0 and s2 != 0.0:
                    c = common.setdefault(key, {
                        "件数": 0, "V1的中": 0, "V2的中": 0, "上昇": 0,
                        "超過件数": 0, "V1超過合計": 0.0, "V2超過合計": 0.0,
                    })
                    c["件数"] += 1
                    if ret * s1 > 0:
                        c["V1的中"] += 1
                    if ret * s2 > 0:
                        c["V2的中"] += 1
                    if ret > 0:
                        c["上昇"] += 1
                    if exc is not None:
                        c["超過件数"] += 1
                        c["V1超過合計"] += s1 * exc
                        c["V2超過合計"] += s2 * exc
                # 見送り（V1 が下落、V2 が立場なし）。セル（V2方向）別に分ける
                if d1 == "下落" and s2 == 0.0:
                    sk = skip.setdefault(key, {}).setdefault(d2, {
                        "件数": 0, "回避件数": 0, "回避合計pct": 0.0,
                        "機会損失件数": 0, "機会損失合計pct": 0.0, "最大下落": [],
                    })
                    sk["件数"] += 1
                    if ret > 0:
                        sk["回避件数"] += 1
                        sk["回避合計pct"] += ret
                    elif ret < 0:
                        sk["機会損失件数"] += 1
                        sk["機会損失合計pct"] += -ret
                    if max_down is not None:
                        sk["最大下落"].append(max_down)
                # 採用ゲートの主エンドポイント（コホート・銘柄別の差分）
                if horizon == PRIMARY_HORIZON and n == PRIMARY_CHECKPOINT and exc is not None:
                    d_exc = (s2 - s1) * exc
                    cohort_units.setdefault(scan_id, []).append(d_exc)
                    su = symbol_units.setdefault(code, {"件数": 0, "差分合計": 0.0})
                    su["件数"] += 1
                    su["差分合計"] += d_exc

    matrix_out = {}
    for key, cell_map in cells.items():
        matrix_out[key] = {}
        for combo, cell in cell_map.items():
            cnt = cell["件数"]
            matrix_out[key][combo] = {
                "件数": cnt,
                "V1方向付き平均リターンpct": round(cell["V1方向付き合計"] / cnt, 3) if cnt else None,
                "V2方向付き平均リターンpct": round(cell["V2方向付き合計"] / cnt, 3) if cnt else None,
                "差分平均pct": round(cell["差分合計pct"] / cnt, 3) if cnt else None,
                "差分合計pct": round(cell["差分合計pct"], 3),
                "超過件数": cell["超過件数"],
                "差分平均TOPIX超過pct": (
                    round(cell["差分TOPIX超過合計"] / cell["超過件数"], 3) if cell["超過件数"] else None
                ),
            }
    completeness_out = {}
    for key, comp in completeness.items():
        completeness_out[key] = {
            "対象件数": comp["対象件数"],
            "セル合計件数": comp["セル合計件数"],
            "未分類件数": comp["未分類件数"],
            "差分合計pct": round(comp["差分合計pct"], 3),
            "セル差分合計pct": round(comp["セル差分合計pct"], 3),
            "一致": (
                comp["セル合計件数"] + comp["未分類件数"] == comp["対象件数"]
                and abs(comp["差分合計pct"] - comp["セル差分合計pct"]) < 1e-6
            ),
        }
    common_out = {}
    for key, c in common.items():
        cnt = c["件数"]
        exc_cnt = c["超過件数"]
        v1_mean = c["V1超過合計"] / exc_cnt if exc_cnt else None
        v2_mean = c["V2超過合計"] / exc_cnt if exc_cnt else None
        common_out[key] = {
            "件数": cnt,
            "V1的中率pct": round(c["V1的中"] / cnt * 100.0, 1) if cnt else None,
            "V2的中率pct": round(c["V2的中"] / cnt * 100.0, 1) if cnt else None,
            "常時上昇pct": round(c["上昇"] / cnt * 100.0, 1) if cnt else None,
            "V1平均方向付きTOPIX超過pct": round(v1_mean, 3) if v1_mean is not None else None,
            "V2平均方向付きTOPIX超過pct": round(v2_mean, 3) if v2_mean is not None else None,
            "ペア差分平均TOPIX超過pct": (
                round(v2_mean - v1_mean, 3) if (v1_mean is not None and v2_mean is not None) else None
            ),
        }
    skip_out = {}
    for key, by_dir in skip.items():
        skip_out[key] = {}
        for d2, sk in by_dir.items():
            skip_out[key][d2] = {
                "件数": sk["件数"],
                "回避件数": sk["回避件数"],
                "回避合計pct": round(sk["回避合計pct"], 3),
                "機会損失件数": sk["機会損失件数"],
                "機会損失合計pct": round(sk["機会損失合計pct"], 3),
                "純影響pct": round(sk["回避合計pct"] - sk["機会損失合計pct"], 3),
                "平均最大下落率pct": _mean(sk["最大下落"]),
            }

    # --- 採用ゲート（主エンドポイント：中期・20営業日） ---
    cohort_out = {}
    cohort_means = []
    for scan_id, values in sorted(cohort_units.items()):
        mean_v = sum(values) / len(values)
        cohort_out[scan_id] = {"件数": len(values), "差分平均pct": round(mean_v, 3)}
        cohort_means.append(mean_v)
    n_cohorts = len(cohort_means)
    cohort_mean = sum(cohort_means) / n_cohorts if n_cohorts else None
    ci = None
    if n_cohorts >= 2:
        var = sum((v - cohort_mean) ** 2 for v in cohort_means) / (n_cohorts - 1)
        sd = var ** 0.5
        half = _t_crit(n_cohorts - 1) * sd / (n_cohorts ** 0.5)
        ci = (round(cohort_mean - half, 3), round(cohort_mean + half, 3))
    cohort_dates = []
    for scan_id in cohort_out:
        try:
            cohort_dates.append(datetime.strptime(str(scan_id)[:10], "%Y-%m-%d").date())
        except (TypeError, ValueError):
            pass
    span_days = (max(cohort_dates) - min(cohort_dates)).days if len(cohort_dates) >= 2 else 0
    primary_common = common_out.get((PRIMARY_CHECKPOINT, PRIMARY_HORIZON), {})
    if regime_by_scan is None:
        non_rising = None
    else:
        non_rising = sum(
            1 for s in cohort_out if regime_by_scan.get(s) not in (None, "上昇相場")
        )
    total_symbol_diff = sum(su["差分合計"] for su in symbol_units.values())
    top_symbol = None
    top_share = None
    loo_mean = None
    dependence_warning = False
    if symbol_units and total_symbol_diff > 0:
        top_symbol = max(symbol_units, key=lambda c: symbol_units[c]["差分合計"])
        top_share = symbol_units[top_symbol]["差分合計"] / total_symbol_diff
        dependence_warning = top_share > GATE_SYMBOL_DEPENDENCE
        rest_units = sum(su["件数"] for c, su in symbol_units.items() if c != top_symbol)
        if rest_units:
            loo_mean = round(
                (total_symbol_diff - symbol_units[top_symbol]["差分合計"]) / rest_units, 3
            )
    fallback_scans = sorted({
        rec.get("スキャンID") for rec in records
        if _version_of(rec) == 2 and FALLBACK_NOTE in str(rec.get("警告・理由") or "")
    })

    def gate_item(name, criterion, actual, met):
        return {"項目": name, "基準": criterion, "実績": actual, "判定": met}

    v1_hit = primary_common.get("V1的中率pct")
    v2_hit = primary_common.get("V2的中率pct")
    always_up = primary_common.get("常時上昇pct")
    items = [
        gate_item("コホート数（スキャン回数）", f"{GATE_MIN_COHORTS}回以上",
                  n_cohorts, n_cohorts >= GATE_MIN_COHORTS),
        gate_item("蓄積期間", f"{GATE_MIN_SPAN_DAYS}日以上",
                  span_days, span_days >= GATE_MIN_SPAN_DAYS),
        gate_item("主エンドポイントのペア判定件数", f"{GATE_MIN_PAIRED}件以上",
                  primary_common.get("件数", 0),
                  primary_common.get("件数", 0) >= GATE_MIN_PAIRED),
        gate_item("非・上昇相場のコホート数", f"{GATE_MIN_NON_RISING_COHORTS}回以上",
                  non_rising,
                  None if non_rising is None else non_rising >= GATE_MIN_NON_RISING_COHORTS),
        gate_item("コホート平均のペア差分（TOPIX超過）", "プラス",
                  round(cohort_mean, 3) if cohort_mean is not None else None,
                  None if cohort_mean is None else cohort_mean > 0),
        gate_item("信頼区間95%の下限", f"{GATE_CI_LOWER_BOUND}%を上回る",
                  ci[0] if ci else None,
                  None if ci is None else ci[0] > GATE_CI_LOWER_BOUND),
        gate_item("的中率の非劣性（V2 が V1 − 2pt 以内）", "V1 − 2pt 以上",
                  (v1_hit, v2_hit),
                  None if (v1_hit is None or v2_hit is None)
                  else v2_hit >= v1_hit - GATE_HIT_NONINFERIORITY),
        gate_item("常時上昇との差（共通母集団）", "V2 の差が V1 より小さい",
                  (v1_hit, v2_hit, always_up),
                  None if (v1_hit is None or v2_hit is None or always_up is None)
                  else abs(v2_hit - always_up) < abs(v1_hit - always_up)),
    ]
    gate = {
        "項目": items,
        "充足": all(item["判定"] is True for item in items),
        "主エンドポイント": f"{PRIMARY_HORIZON}・{PRIMARY_CHECKPOINT}営業日",
        "コホート平均差分pct": round(cohort_mean, 3) if cohort_mean is not None else None,
        "信頼区間95": ci,
        "銘柄依存": {
            "最多銘柄": top_symbol,
            "寄与率": round(top_share, 3) if top_share is not None else None,
            "依存警告": dependence_warning,
            "最多銘柄を除いた差分平均pct": loo_mean,
        },
        "再取得フォールバック": {"コホート数": len(fallback_scans), "スキャンID": fallback_scans},
    }

    return {
        "行列": matrix_out,
        "完全性": completeness_out,
        "片側登録": one_sided,
        "片側登録件数": one_sided_counts,
        "評価待ち": pending,
        "共通母集団": common_out,
        "見送り": skip_out,
        "コホート別": cohort_out,
        "ゲート": gate,
    }


# ============================================================
# V2 未登録理由の保存（既存CSVとは別の小さなCSVで保持する）
# ============================================================

SKIPPED_COLUMNS = ["スキャンID", "コード", "理由"]


def skipped_to_csv(entries):
    df = pd.DataFrame(entries, columns=SKIPPED_COLUMNS)
    return df.to_csv(index=False).encode("utf-8-sig")


def skipped_from_csv(data_bytes):
    """未登録理由CSVを読む。形式が合わなければ空リストを返す。"""
    import io

    try:
        text = data_bytes.decode("utf-8-sig")
    except (UnicodeDecodeError, AttributeError):
        return []
    try:
        df = pd.read_csv(io.StringIO(text), dtype=str).fillna("")
    except Exception:
        return []
    if not all(c in df.columns for c in SKIPPED_COLUMNS):
        return []
    return [{c: str(row[c]) for c in SKIPPED_COLUMNS} for _, row in df.iterrows()]


def merge_skipped(existing, new_entries):
    """(スキャンID, コード) で重複を除いて統合する。既存の理由を優先する。"""
    seen = {(e.get("スキャンID"), e.get("コード")) for e in existing}
    merged = list(existing)
    for entry in new_entries:
        key = (entry.get("スキャンID"), entry.get("コード"))
        if key in seen:
            continue
        seen.add(key)
        merged.append({c: entry.get(c, "") for c in SKIPPED_COLUMNS})
    return merged


def caution_reference(results, records):
    """V2 の「下落注意」参考グループの集計（的中率には含めない）。

    戻り値: {(評価営業日数, 時間軸): {件数, 平均リターンpct, 平均最大下落率pct}}
    V2 の記録でその時間軸の方向が「下落注意」の単位だけを対象に、
    実際にその後どう動いたか（平均リターンと平均の最大下落）を出す。
    """
    v2_dirs = {}
    for rec in records:
        if _version_of(rec) != 2:
            continue
        v2_dirs[(rec.get("スキャンID"), rec.get("コード"))] = {
            h: rec.get(f"{h}予測方向") for h in HORIZONS
        }
    groups = {}
    for res in results:
        if res.get("状態") != "評価済み" or _version_of(res) != 2:
            continue
        dirs = v2_dirs.get((res.get("スキャンID"), res.get("コード")))
        if not dirs:
            continue
        ret = res.get("リターンpct")
        if ret is None:
            continue
        for horizon in HORIZONS:
            if dirs.get(horizon) != "下落注意":
                continue
            key = (res.get("評価営業日数"), horizon)
            groups.setdefault(key, []).append((ret, res.get("最大下落率pct")))
    return {
        key: {
            "件数": len(vals),
            "平均リターンpct": _mean([v[0] for v in vals]),
            "平均最大下落率pct": _mean([v[1] for v in vals]),
        }
        for key, vals in groups.items()
    }
