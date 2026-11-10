"""株価データの品質診断（読み取り専用・独立モジュール）。

- 入力DataFrame（analysis.fetch_history / scan._shape が扱う形の
  Open / High / Low / Close / Volume と日付インデックス）を読み取り、
  診断結果だけを返す。入力は変更しない（非破壊）。
- 補正・補間・削除はしない。異常があってもデータはそのまま返すのみで、
  既存のスコア・ランキング・採用ゲート・銘柄ユニバース・データ取得には
  一切接続しない。既存モジュール（analysis / scan / scoring_v2 等）にも
  依存しない独立モジュールとする。
- 日付間隔の疑わしさは、公式休場日カレンダーがない段階では異常と確定せず
  「要確認」の警告に留める（週末またぎは異常にしない）。
- 未来日付の判定基準は、呼び出し側が as_of（基準日）か market（市場の
  タイムゾーン指定）のどちらかを必ず明示する。実行環境のローカル日付で
  既定判定はしない（市場現地日付とずれるため）。
- 入力契約は「単一銘柄の OHLCV」である。列名の重複（複数銘柄の
  MultiIndex を先頭レベルだけ平坦化した場合など）は列の特定が曖昧に
  なるため契約外とし、diagnose_history は ValueError にする。
- duplicate_row_count は「重複日付に関係する全行数」（各重複日付の
  出現回数の合計）であり、「余分な重複行数」（出現回数−1 の合計）
  ではない。
- anomaly_count には必須列の欠落数（len(missing_columns)）も含める。
  必須列が欠けた入力は構造が不完全であり、異常として数える。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
PRICE_COLUMNS = ("Open", "High", "Low", "Close")

# 市場の現地タイムゾーン（analysis.MARKET_TIMEZONES と同じ表記・同じ値）
MARKET_TIMEZONES = {
    "JP": ZoneInfo("Asia/Tokyo"),
    "US": ZoneInfo("America/New_York"),
}

# 重要度の区分（診断結果の分類であり、売買判定や除外判定ではない）
SEVERITY_ANOMALY = "異常候補"
SEVERITY_WARNING = "警告候補"
SEVERITY_REVIEW = "要確認"


def _date_label(parsed_ts, raw_label) -> str:
    """行の日付ラベル。パースできた日付は YYYY-MM-DD、できなければ元の表記。"""
    if parsed_ts is not None and not pd.isna(parsed_ts):
        return parsed_ts.strftime("%Y-%m-%d")
    return str(raw_label)


def _naive(ts: pd.Timestamp) -> pd.Timestamp:
    """タイムゾーン付きTimestampを比較用にnaiveへ変換する。"""
    if ts.tzinfo is not None:
        return ts.tz_localize(None)
    return ts


@dataclass
class DataQualityReport:
    """diagnose_history の診断結果。値はすべて入力から読み取った事実のみ。"""

    row_count: int = 0  # 元の観測本数（整形前の行数）
    columns: list = field(default_factory=list)
    missing_columns: list = field(default_factory=list)
    index_is_datetime: bool = False
    # 日付系列
    nat_count: int = 0
    nat_rows: list = field(default_factory=list)
    duplicate_dates: list = field(default_factory=list)  # {date, count, rows}
    duplicate_row_count: int = 0  # 重複日付に関係する全行数（出現回数の合計。余分な行数ではない）
    monotonic_increasing: bool = True
    out_of_order: list = field(default_factory=list)  # {row, date, previous_date}
    future_dates: list = field(default_factory=list)  # {row, date}
    review_gaps: list = field(default_factory=list)  # {from_date, to_date, skipped_weekdays}
    # 価格の整合性（種類別の該当行・項目）
    price_issues: dict = field(default_factory=dict)
    # 出来高（ゼロは警告候補、負値・非有限値は異常候補）
    volume_issues: dict = field(default_factory=dict)
    # 欠損の可視化
    missing_per_column: dict = field(default_factory=dict)
    rows_with_missing: int = 0
    rows_dropped_by_dropna: int = 0  # 既存 dropna() が除外する行数
    row_count_after_dropna: int = 0
    last_row_missing: bool = False
    consecutive_missing_runs: list = field(default_factory=list)  # {start_row, length}

    @property
    def anomaly_count(self) -> int:
        total = (
            len(self.missing_columns)
            + self.nat_count
            + self.duplicate_row_count
            + len(self.out_of_order)
            + len(self.future_dates)
        )
        total += sum(len(v) for v in self.price_issues.values())
        total += len(self.volume_issues.get("negative", []))
        total += len(self.volume_issues.get("non_finite", []))
        return total

    @property
    def warning_count(self) -> int:
        return len(self.volume_issues.get("zero", []))

    @property
    def review_count(self) -> int:
        return len(self.review_gaps)

    @property
    def has_anomalies(self) -> bool:
        return self.anomaly_count > 0

    def to_dict(self) -> dict:
        """辞書化する。入れ子のリスト・辞書は複製して返すため、戻り値を
        変更してもレポート内部の状態には影響しない。"""
        return {
            "row_count": self.row_count,
            "columns": list(self.columns),
            "missing_columns": list(self.missing_columns),
            "index_is_datetime": self.index_is_datetime,
            "nat_count": self.nat_count,
            "nat_rows": list(self.nat_rows),
            "duplicate_dates": copy.deepcopy(self.duplicate_dates),
            "duplicate_row_count": self.duplicate_row_count,
            "monotonic_increasing": self.monotonic_increasing,
            "out_of_order": copy.deepcopy(self.out_of_order),
            "future_dates": copy.deepcopy(self.future_dates),
            "review_gaps": copy.deepcopy(self.review_gaps),
            "price_issues": copy.deepcopy(self.price_issues),
            "volume_issues": copy.deepcopy(self.volume_issues),
            "missing_per_column": dict(self.missing_per_column),
            "rows_with_missing": self.rows_with_missing,
            "rows_dropped_by_dropna": self.rows_dropped_by_dropna,
            "row_count_after_dropna": self.row_count_after_dropna,
            "last_row_missing": self.last_row_missing,
            "consecutive_missing_runs": copy.deepcopy(self.consecutive_missing_runs),
            "anomaly_count": self.anomaly_count,
            "warning_count": self.warning_count,
            "review_count": self.review_count,
        }


def _parse_index(idx: pd.Index):
    """インデックスを日付として解釈する。(DatetimeIndex, 解釈できたか) を返す。"""
    if isinstance(idx, pd.DatetimeIndex):
        return idx, True
    if idx.dtype == object or pd.api.types.is_string_dtype(idx):
        parsed = pd.to_datetime(idx, errors="coerce")
        return parsed, bool(parsed.notna().any())
    return pd.DatetimeIndex([pd.NaT] * len(idx)), False


def _numeric(view: pd.DataFrame, column: str) -> np.ndarray:
    """列を float 配列として読み取る（数値化できない値は NaN になる）。"""
    return pd.to_numeric(view[column], errors="coerce").to_numpy(dtype=float)


def _reference_date(as_of, market, now_utc) -> pd.Timestamp:
    """未来日付の判定基準日（市場現地日付）を決める。

    as_of が指定されていればそれを基準日とする。as_of が無い場合は
    market（"JP" / "US"）の現地タイムゾーンでの現在日付を使う。
    どちらも無い場合は、実行環境の日付で曖昧に判定せず ValueError にする。
    now_utc はテスト用に現在時刻（UTC）を注入するためのものである。
    """
    if as_of is not None:
        return _naive(pd.Timestamp(as_of)).normalize()
    if market is None:
        raise ValueError(
            "未来日付の判定には as_of（基準日）か market（市場）の指定が必要です"
        )
    if market not in MARKET_TIMEZONES:
        raise ValueError(
            f"market は {sorted(MARKET_TIMEZONES)} のいずれかを指定してください: {market!r}"
        )
    tz = MARKET_TIMEZONES[market]
    if now_utc is None:
        now = datetime.now(timezone.utc)
    else:
        now = pd.Timestamp(now_utc).to_pydatetime()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
    return pd.Timestamp(now.astimezone(tz).date())


def diagnose_history(
    df, *, as_of=None, market=None, now_utc=None, required_columns=REQUIRED_COLUMNS
) -> DataQualityReport:
    """株価DataFrameの品質を診断し、結果だけを返す。入力は変更しない。

    引数:
        df: 日付インデックスと OHLCV 列を持つDataFrame（整形前・整形後どちらでも可）。
            単一銘柄が前提で、列名が重複している場合は ValueError になる。
        as_of: 未来日付の判定基準日。指定した場合はこれを優先する。
        market: as_of が無い場合に、市場現地日付を求めるための市場（"JP" / "US"）。
            as_of と market のどちらも指定しない場合は ValueError になる
            （実行環境のローカル日付では判定しない）。
        now_utc: market 指定時に使う現在時刻（UTC）。テスト用の注入点。
        required_columns: 必要列（既定は analysis.NEEDED_COLUMNS と同じ5列）。

    戻り値:
        DataQualityReport。診断は読み取りのみで、補正・補間・削除はしない。
    """
    report = DataQualityReport()
    if df is None:
        report.missing_columns = list(required_columns)
        return report

    # 入力を変更しないため、以降はコピーに対して読み取りを行う。
    view = df.copy()
    if isinstance(view.columns, pd.MultiIndex):
        view.columns = view.columns.get_level_values(0)
    # 入力契約は単一銘柄の OHLCV。列名が重複していると列の特定が曖昧に
    # なる（複数銘柄の MultiIndex を平坦化した場合など）ため、契約外として
    # 明示的にエラーにする（黙って誤診断しない）。
    if view.columns.duplicated().any():
        dupes = sorted({str(c) for c in view.columns[view.columns.duplicated()]})
        raise ValueError(
            "入力は単一銘柄のOHLCVが前提です。重複した列名があります: "
            + ", ".join(dupes)
        )
    report.row_count = len(view)
    report.columns = [str(c) for c in view.columns]
    present = [c for c in required_columns if c in view.columns]
    report.missing_columns = [c for c in required_columns if c not in view.columns]
    n = report.row_count

    parsed, report.index_is_datetime = _parse_index(view.index)
    labels = [
        _date_label(parsed[i] if i < len(parsed) else None, view.index[i])
        for i in range(n)
    ]

    # ---- 日付系列 ----
    valid = []  # (位置, naive Timestamp) NaT を除いた系列（元の並び順）
    for i in range(n):
        ts = parsed[i]
        if pd.isna(ts):
            report.nat_rows.append(i)
        else:
            valid.append((i, _naive(ts)))
    report.nat_count = len(report.nat_rows)

    # 重複日付
    by_date: dict = {}
    for i, ts in valid:
        by_date.setdefault(ts, []).append(i)
    for ts in sorted(by_date):
        rows = by_date[ts]
        if len(rows) > 1:
            report.duplicate_dates.append(
                {"date": ts.strftime("%Y-%m-%d"), "count": len(rows), "rows": list(rows)}
            )
            report.duplicate_row_count += len(rows)
    report.duplicate_dates.sort(key=lambda d: d["date"])

    # 昇順でない並び（直前の有効日付より前になっている行）
    prev = None
    for i, ts in valid:
        if prev is not None and ts < prev:
            report.out_of_order.append(
                {
                    "row": i,
                    "date": ts.strftime("%Y-%m-%d"),
                    "previous_date": prev.strftime("%Y-%m-%d"),
                }
            )
        prev = ts
    report.monotonic_increasing = not report.out_of_order

    # 未来日付（基準日は as_of か market の明示が必須。既定では判定しない）
    as_of_ts = _reference_date(as_of, market, now_utc)
    for i, ts in valid:
        if ts.normalize() > as_of_ts:
            report.future_dates.append({"row": i, "date": ts.strftime("%Y-%m-%d")})

    # 日付間隔の疑わしさ（平日を飛ばしている間隔のみ「要確認」。異常確定はしない）
    for (i1, ts1), (i2, ts2) in zip(valid, valid[1:]):
        if ts2 <= ts1:
            continue
        d1 = np.datetime64(ts1.date(), "D")
        d2 = np.datetime64(ts2.date(), "D")
        skipped = int(np.busday_count(d1 + 1, d2))
        if skipped >= 1:
            report.review_gaps.append(
                {
                    "from_date": ts1.strftime("%Y-%m-%d"),
                    "to_date": ts2.strftime("%Y-%m-%d"),
                    "skipped_weekdays": skipped,
                }
            )

    # ---- 欠損の可視化（既存 dropna() と同じ基準：必要列のいずれかが NaN） ----
    missing_row_mask = np.zeros(n, dtype=bool)
    for col in present:
        col_isna = view[col].isna().to_numpy()
        report.missing_per_column[col] = int(col_isna.sum())
        missing_row_mask |= col_isna
    report.rows_with_missing = int(missing_row_mask.sum())
    report.rows_dropped_by_dropna = report.rows_with_missing
    report.row_count_after_dropna = n - report.rows_with_missing
    report.last_row_missing = bool(n > 0 and missing_row_mask[-1])
    run_start = None
    for i in range(n + 1):
        in_run = i < n and missing_row_mask[i]
        if in_run and run_start is None:
            run_start = i
        elif not in_run and run_start is not None:
            report.consecutive_missing_runs.append(
                {"start_row": run_start, "length": i - run_start}
            )
            run_start = None

    # ---- 価格の整合性 ----
    price_issues = {
        "non_finite": [],
        "non_positive": [],
        "high_below_open_close": [],
        "low_above_open_close": [],
        "high_below_low": [],
    }
    arrays = {}
    for col in PRICE_COLUMNS:
        if col not in view.columns:
            continue
        arr = _numeric(view, col)
        arrays[col] = arr
        for i in range(n):
            v = arr[i]
            if not np.isfinite(v):
                kind = "NaN" if np.isnan(v) else ("inf" if v > 0 else "-inf")
                price_issues["non_finite"].append(
                    {"row": i, "date": labels[i], "column": col, "detail": kind}
                )
            elif v <= 0:
                price_issues["non_positive"].append(
                    {"row": i, "date": labels[i], "column": col, "detail": f"value={v}"}
                )
    # High/Low と Open/Close の比較は、利用可能な有限値を個別に使う。
    # Open と Close の両方が有限値でなくても、片方だけでも比較できる場合は判定する
    # （欠損値そのものは上の non_finite で別途記録される）。
    if "High" in arrays:
        high = arrays["High"]
        for i in range(n):
            h = high[i]
            if not np.isfinite(h):
                continue
            refs = [
                (name, arrays[name][i])
                for name in ("Open", "Close")
                if name in arrays and np.isfinite(arrays[name][i])
            ]
            if not refs:
                continue
            ref_max = max(v for _, v in refs)
            if h < ref_max:
                if len(refs) == 2:
                    detail = f"High={h} < max(Open, Close)={ref_max}"
                else:
                    detail = f"High={h} < {refs[0][0]}={ref_max}"
                price_issues["high_below_open_close"].append(
                    {"row": i, "date": labels[i], "column": "High", "detail": detail}
                )
    if "Low" in arrays:
        low_arr = arrays["Low"]
        for i in range(n):
            low_v = low_arr[i]
            if not np.isfinite(low_v):
                continue
            refs = [
                (name, arrays[name][i])
                for name in ("Open", "Close")
                if name in arrays and np.isfinite(arrays[name][i])
            ]
            if not refs:
                continue
            ref_min = min(v for _, v in refs)
            if low_v > ref_min:
                if len(refs) == 2:
                    detail = f"Low={low_v} > min(Open, Close)={ref_min}"
                else:
                    detail = f"Low={low_v} > {refs[0][0]}={ref_min}"
                price_issues["low_above_open_close"].append(
                    {"row": i, "date": labels[i], "column": "Low", "detail": detail}
                )
    if "High" in arrays and "Low" in arrays:
        for i in range(n):
            h, low = arrays["High"][i], arrays["Low"][i]
            if np.isfinite(h) and np.isfinite(low) and h < low:
                price_issues["high_below_low"].append(
                    {"row": i, "date": labels[i], "column": "High/Low",
                     "detail": f"High={h} < Low={low}"}
                )
    report.price_issues = price_issues

    # ---- 出来高 ----
    volume_issues = {"zero": [], "negative": [], "non_finite": []}
    if "Volume" in view.columns:
        vol = _numeric(view, "Volume")
        for i in range(n):
            v = vol[i]
            if not np.isfinite(v):
                kind = "NaN" if np.isnan(v) else ("inf" if v > 0 else "-inf")
                volume_issues["non_finite"].append(
                    {"row": i, "date": labels[i], "column": "Volume", "detail": kind}
                )
            elif v < 0:
                volume_issues["negative"].append(
                    {"row": i, "date": labels[i], "column": "Volume", "detail": f"value={v}"}
                )
            elif v == 0:
                volume_issues["zero"].append(
                    {"row": i, "date": labels[i], "column": "Volume", "detail": "value=0"}
                )
    report.volume_issues = volume_issues

    return report
