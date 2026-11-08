"""手動デイリースキャン CLI（run_daily_scan.py）の単体テスト。ネットワークは使わない。

取得関数はすべてモックに差し替え、scan.run_scan 本体を通して検証する。
期待データ日は run_daily_scan.execute の now 引数で固定する（2026-11-09 = 月曜）。
"""

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import run_daily_scan
import scan

JST = ZoneInfo("Asia/Tokyo")
NOW = datetime(2026, 11, 9, 15, 30, 0, tzinfo=JST)
EXPECTED = "2026-11-09"
STALE_END = pd.Timestamp("2026-10-26").date()   # 期待データ日の14暦日前
OLD_END = pd.Timestamp("2026-10-30").date()     # 全銘柄ぶんの「古い」データ日
SYMBOLS = [u[0] for u in scan.UNIVERSE_JP]


def make_frame(end, n=300):
    idx = pd.bdate_range(end=end, periods=n)
    closes = [100.0 + 0.2 * k for k in range(n)]
    return pd.DataFrame(
        {
            "Open": closes, "High": closes, "Low": closes,
            "Close": closes, "Volume": [1000] * n,
        },
        index=idx,
    )


TODAY_FRAME = make_frame(NOW.date())


def batch_with(frames):
    def _batch(symbols):
        return {s: (frames[s], None) for s in symbols}
    return _batch


def single_with(frames):
    def _single(symbol):
        return frames[symbol], None
    return _single


def all_today_frames():
    return {s: TODAY_FRAME for s in SYMBOLS}


def read_status(out_dir):
    path = out_dir / f"daily_scan_status_{EXPECTED}.json"
    return json.loads(path.read_text(encoding="utf-8"))


# --- 全銘柄が当日データ：完成版CSVと状態JSONを出す ---

def test_all_60_today_writes_csv_and_complete_status(tmp_path):
    frames = all_today_frames()
    status = run_daily_scan.execute(
        tmp_path, batch_with(frames), single_with(frames), now=NOW
    )
    assert status["状態"] == "complete"
    assert status["期待データ日"] == EXPECTED
    assert status["対象数"] == 60
    assert status["データ日別件数"] == {EXPECTED: 60}
    assert status["失敗銘柄"] == []
    csv_path = tmp_path / f"daily_scan_ranking_{EXPECTED}.csv"
    assert csv_path.exists()
    assert status["完成版CSV"] == csv_path.name
    text = csv_path.read_bytes().decode("utf-8-sig")
    assert "ランキング" in text
    for code in ("7203.T", "9984.T", "1605.T"):
        assert code in text
    # 状態JSONがファイルにも同じ内容で残っていること
    saved = read_status(tmp_path)
    assert saved["状態"] == "complete"
    assert saved["実行日時"] == "2026-11-09 15:30:00"
    assert saved["データ日別件数"] == {EXPECTED: 60}


# --- 1銘柄だけ古い：完成版CSVは出さず、状態JSONだけ ---

def test_one_stale_symbol_writes_only_status_json(tmp_path):
    frames = all_today_frames()
    frames["7203.T"] = make_frame(STALE_END)
    status = run_daily_scan.execute(
        tmp_path, batch_with(frames), single_with(frames), now=NOW
    )
    assert status["状態"] == "incomplete"
    assert not (tmp_path / f"daily_scan_ranking_{EXPECTED}.csv").exists()
    assert (tmp_path / f"daily_scan_status_{EXPECTED}.json").exists()
    assert status["データ日別件数"] == {EXPECTED: 59, "2026-10-26": 1}
    assert [f["コード"] for f in status["失敗銘柄"]] == ["7203.T"]
    assert status["失敗銘柄"][0]["データ日"] == "2026-10-26"
    assert status["失敗銘柄"][0]["理由"] != ""


# --- 全銘柄に当日データなし ---

def test_no_symbol_has_todays_data(tmp_path):
    old = make_frame(OLD_END)
    frames = {s: old for s in SYMBOLS}
    status = run_daily_scan.execute(
        tmp_path, batch_with(frames), single_with(frames), now=NOW
    )
    assert status["状態"] == "incomplete"
    assert not (tmp_path / f"daily_scan_ranking_{EXPECTED}.csv").exists()
    assert status["データ日別件数"] == {"2026-10-30": 60}
    assert len(status["失敗銘柄"]) == 60


# --- 取得失敗：理由つきで記録される ---

def test_fetch_failure_recorded_with_reason(tmp_path):
    frames = all_today_frames()

    def batch(symbols):
        out = {}
        for s in symbols:
            if s == "8306.T":
                out[s] = (pd.DataFrame(), "一括取得でデータが取れませんでした")
            else:
                out[s] = (frames[s], None)
        return out

    def single(symbol):
        if symbol == "8306.T":
            return pd.DataFrame(), "データが取得できませんでした"
        return frames[symbol], None

    status = run_daily_scan.execute(tmp_path, batch, single, now=NOW)
    assert status["状態"] == "incomplete"
    assert not (tmp_path / f"daily_scan_ranking_{EXPECTED}.csv").exists()
    failures = {f["コード"]: f for f in status["失敗銘柄"]}
    assert list(failures) == ["8306.T"]
    assert failures["8306.T"]["データ日"] is None
    assert "取得失敗" in failures["8306.T"]["理由"]
    assert status["データ日別件数"] == {EXPECTED: 59, "なし": 1}


# --- 日付不正：解釈できないデータ日は失敗として扱い、落ちない ---

def test_invalid_date_in_scan_rows(tmp_path):
    def stub_runner(universe, batch_fetcher, single_fetcher, run_at=None):
        ranked = []
        for symbol, _name, _sector in universe:
            ranked.append({
                "コード": symbol,
                "データ最終日": "2026-13-40" if symbol == "7203.T" else EXPECTED,
                "警告・理由": "",
            })
        return ranked, [], {}

    status = run_daily_scan.execute(
        tmp_path, batch_with(all_today_frames()), single_with(all_today_frames()),
        now=NOW, scan_runner=stub_runner,
    )
    assert status["状態"] == "incomplete"
    assert not (tmp_path / f"daily_scan_ranking_{EXPECTED}.csv").exists()
    assert status["データ日別件数"] == {EXPECTED: 59, "2026-13-40": 1}
    assert [f["コード"] for f in status["失敗銘柄"]] == ["7203.T"]
    assert "解釈できません" in status["失敗銘柄"][0]["理由"]


# --- 内部例外：状態JSONにエラーを記録し、完成版CSVは出さない ---

def test_internal_exception_writes_error_status(tmp_path):
    def batch_boom(symbols):
        raise RuntimeError("boom")

    status = run_daily_scan.execute(
        tmp_path, batch_boom, single_with(all_today_frames()), now=NOW
    )
    assert status["状態"] == "error"
    assert "boom" in status["エラー"]
    assert not (tmp_path / f"daily_scan_ranking_{EXPECTED}.csv").exists()
    saved = read_status(tmp_path)
    assert saved["状態"] == "error"
    assert saved["対象数"] == 60
    assert saved["期待データ日"] == EXPECTED


# --- 同日再実行：同じファイルを上書きし、重複ファイルが増えない ---

def test_same_day_rerun_overwrites_same_files(tmp_path):
    frames = all_today_frames()
    first = run_daily_scan.execute(
        tmp_path, batch_with(frames), single_with(frames), now=NOW
    )
    csv_bytes_first = (tmp_path / first["完成版CSV"]).read_bytes()
    names_first = sorted(p.name for p in tmp_path.iterdir())

    second = run_daily_scan.execute(
        tmp_path, batch_with(frames), single_with(frames), now=NOW
    )
    names_second = sorted(p.name for p in tmp_path.iterdir())
    assert names_second == names_first
    assert names_second == [
        f"daily_scan_ranking_{EXPECTED}.csv",
        f"daily_scan_status_{EXPECTED}.json",
    ]
    assert (tmp_path / second["完成版CSV"]).read_bytes() == csv_bytes_first
    assert second["状態"] == "complete"
