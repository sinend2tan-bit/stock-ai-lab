"""Stock AI Lab 手動実行用デイリースキャン CLI（段階3で追加）。

既存の scan.run_scan / scan.scan_to_csv をそのまま使う薄いラッパーで、
分析・スコア・ランキングの条件は一切変更しない。行うのは次の判定と保存だけ。

- 日本時間（Asia/Tokyo）の当日を「期待データ日」とする。
- 対象60銘柄すべてのデータ日（スキャン結果の「データ最終日」）が
  期待データ日と一致した場合のみ、完成版ランキングCSVを書き出す。
- 1銘柄でも欠損・古いデータ・取得失敗・日付不正があれば完成版CSVは出さず、
  状態JSONだけを書き出す。
- 出力ファイル名は期待データ日で決定的に決まるため、同日の再実行では
  同じファイルを上書きし、重複ファイルは増えない。
- 保存は一時ファイル経由の原子的な置換で行う。完成版CSVの生成・書き込みに
  失敗した場合は、状態=error とエラー内容を状態JSONに記録する。状態JSON
  自体の書き込みに失敗した場合は、標準エラーに明示して非0で終了する。

使い方: python run_daily_scan.py [--out-dir output]
終了コード: 0 = 完成版CSVを出した、1 = 状態JSONのみ（不完全またはエラー）。
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import analysis
import scan

JST = ZoneInfo("Asia/Tokyo")
DATE_FORMAT = "%Y-%m-%d"
NO_DATE_KEY = "なし"


class StatusWriteError(RuntimeError):
    """状態JSONの書き込みに失敗したことを表す。呼び出し側は標準エラーへ
    明示して非0で終了する（状態を残す手段自体が失われたため）。"""


def _write_bytes_atomic(path, data):
    """一時ファイルに書いてから os.replace で原子的に置き換える。

    書き込み途中の失敗で既存ファイルが壊れた状態にならないようにする。
    失敗時は一時ファイルを残さない。
    """
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_bytes(data)
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _parse_data_date(raw):
    """スキャン結果の「データ最終日」を date に解釈する。解釈できなければ None。"""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, DATE_FORMAT).date()
    except ValueError:
        return None


def assess_rows(universe, ranked, excluded, expected_date):
    """スキャン結果の行から、銘柄ごとのデータ日別件数と失敗理由を集計する。

    戻り値: (counts, failures)
      counts: {データ日文字列: 件数}（データ日がない銘柄は「なし」に数える）
      failures: [{"コード", "データ日", "理由"}]（期待データ日に一致しない銘柄）
    """
    rows = {row["コード"]: row for row in list(ranked) + list(excluded)}
    counts = {}
    failures = []
    for symbol, _name, _sector in universe:
        row = rows.get(symbol)
        if row is None:
            counts[NO_DATE_KEY] = counts.get(NO_DATE_KEY, 0) + 1
            failures.append({
                "コード": symbol, "データ日": None,
                "理由": "スキャン結果に含まれていません",
            })
            continue
        raw = row.get("データ最終日") or ""
        parsed = _parse_data_date(raw)
        if parsed is not None:
            key = parsed.isoformat()
        else:
            key = str(raw).strip() or NO_DATE_KEY
        counts[key] = counts.get(key, 0) + 1
        if parsed is not None and parsed == expected_date:
            continue
        warning = (row.get("警告・理由") or "").strip()
        if parsed is None:
            if str(raw).strip():
                reason = f"データ日を解釈できません（{raw}）"
                data_date = str(raw).strip()
            else:
                reason = warning or "データがありません"
                data_date = None
        else:
            reason = f"データ日が期待データ日と一致しません（{parsed.isoformat()}）"
            if warning:
                reason += f"：{warning}"
            data_date = parsed.isoformat()
        failures.append({"コード": symbol, "データ日": data_date, "理由": reason})
    return dict(sorted(counts.items())), failures


def execute(out_dir, batch_fetcher=None, single_fetcher=None, now=None,
            universe=None, scan_runner=None):
    """スキャンを実行し、完成版CSV（条件を満たす場合のみ）と状態JSONを書き出す。

    取得関数とスキャン実行関数は差し替え可能（テストはモックを注入する）。
    戻り値: 状態 dict（状態JSONに書いた内容）。
    """
    out_dir = Path(out_dir)
    if universe is None:
        universe = scan.UNIVERSE_JP
    if batch_fetcher is None:
        batch_fetcher = lambda symbols: scan.fetch_batch(symbols, scan.SCAN_PERIOD)
    if single_fetcher is None:
        single_fetcher = lambda symbol: analysis.fetch_history(symbol, scan.SCAN_PERIOD)
    if scan_runner is None:
        scan_runner = scan.run_scan
    if now is None:
        now = datetime.now(JST)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=JST)
    else:
        now = now.astimezone(JST)
    expected = now.date()
    expected_str = expected.isoformat()
    run_at = now.strftime("%Y-%m-%d %H:%M:%S")
    status_name = f"daily_scan_status_{expected_str}.json"
    status = {
        "実行日時": run_at,
        "タイムゾーン": "Asia/Tokyo",
        "期待データ日": expected_str,
        "状態": "error",
        "対象数": len(universe),
        "データ日別件数": {},
        "失敗銘柄": [],
        "完成版CSV": None,
        "状態ファイル": status_name,
        "エラー": None,
    }
    ranked = excluded = None
    try:
        ranked, excluded, _meta = scan_runner(
            universe, batch_fetcher, single_fetcher, run_at=run_at
        )
    except Exception as exc:
        status["エラー"] = f"{type(exc).__name__}: {exc}"
    if ranked is not None:
        counts, failures = assess_rows(universe, ranked, excluded, expected)
        status["データ日別件数"] = counts
        status["失敗銘柄"] = failures
        if not failures:
            csv_name = f"daily_scan_ranking_{expected_str}.csv"
            try:
                csv_bytes = scan.scan_to_csv(ranked, excluded)
                out_dir.mkdir(parents=True, exist_ok=True)
                _write_bytes_atomic(out_dir / csv_name, csv_bytes)
            except Exception as exc:
                status["状態"] = "error"
                status["エラー"] = (
                    f"完成版CSVの保存に失敗しました: {type(exc).__name__}: {exc}"
                )
            else:
                status["状態"] = "complete"
                status["完成版CSV"] = csv_name
        else:
            status["状態"] = "incomplete"
    status_path = out_dir / status_name
    payload = (
        json.dumps(status, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_bytes_atomic(status_path, payload)
    except Exception as exc:
        raise StatusWriteError(
            f"状態JSONの書き込みに失敗しました（{status_path}）: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Stock AI Lab 手動デイリースキャン"
                    "（完成版CSVは全銘柄のデータ日が当日の場合のみ出力）"
    )
    parser.add_argument("--out-dir", default="output",
                        help="出力先ディレクトリ（既定: output）")
    args = parser.parse_args(argv)
    try:
        status = execute(args.out_dir)
    except StatusWriteError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status["状態"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
