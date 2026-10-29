"""Stock AI Lab のスキャン機能（主要な日本株を一括で評価し、中期スコア順に並べる）。

設計（2026-10-29 確定）:
- 対象は UNIVERSE_JP の約60銘柄。リストはコード・会社名・業種の組で持ち、
  日経225などへの拡大はリストの差し替えだけで行えるようにする。
- ランキングの主軸は中期評価。短期・長期は補助情報として表示する。
- ランキングに入れるのは、中期評価ができて、データが「古い可能性」でない
  銘柄だけ。新旧判定が「間隔あり（週末・休場で通常）」の銘柄は、警告を付けて
  ランキングに入れる。取得失敗・データ不足・古いデータは、理由つきで
  「参考外」に分ける。
- 取得は複数銘柄の一括取得（25銘柄ずつ）を使い、取れなかった銘柄だけ
  既存の1銘柄取得（再試行つき）で取り直す。
- スキャンの上位は「条件がそろった銘柄」であり、将来の値上がりの予測ではない。
- 分析ロジック（analysis.py）と検証（verify.py）には手を入れない。
"""

import time
from datetime import datetime

import pandas as pd
import yfinance as yf

import analysis

SCAN_PERIOD = "2y"
CHUNK_SIZE = 25
CHUNK_PAUSE_SECONDS = 1.0

# スキャン対象（コード, 会社名, 業種）。2026-10-29 に全銘柄の実データ取得で
# 上場とティッカー形式を確認済み。9613 は上場廃止のため含めない（4307 に差し替え）。
UNIVERSE_JP = [
    # 自動車・輸送用機器
    ("7203.T", "トヨタ自動車", "自動車・輸送用機器"),
    ("7267.T", "本田技研工業", "自動車・輸送用機器"),
    ("7269.T", "スズキ", "自動車・輸送用機器"),
    ("6902.T", "デンソー", "自動車・輸送用機器"),
    ("5108.T", "ブリヂストン", "自動車・輸送用機器"),
    # 電気機器・半導体
    ("6758.T", "ソニーグループ", "電気機器・半導体"),
    ("6501.T", "日立製作所", "電気機器・半導体"),
    ("8035.T", "東京エレクトロン", "電気機器・半導体"),
    ("6857.T", "アドバンテスト", "電気機器・半導体"),
    ("6920.T", "レーザーテック", "電気機器・半導体"),
    ("6146.T", "ディスコ", "電気機器・半導体"),
    ("6723.T", "ルネサスエレクトロニクス", "電気機器・半導体"),
    ("6981.T", "村田製作所", "電気機器・半導体"),
    ("6594.T", "ニデック", "電気機器・半導体"),
    ("6954.T", "ファナック", "電気機器・半導体"),
    ("6861.T", "キーエンス", "電気機器・半導体"),
    ("7751.T", "キヤノン", "電気機器・半導体"),
    ("6752.T", "パナソニック ホールディングス", "電気機器・半導体"),
    ("6503.T", "三菱電機", "電気機器・半導体"),
    # 機械・重工
    ("6367.T", "ダイキン工業", "機械・重工"),
    ("6301.T", "小松製作所", "機械・重工"),
    ("7011.T", "三菱重工業", "機械・重工"),
    # 化学・素材
    ("4063.T", "信越化学工業", "化学・素材"),
    ("4452.T", "花王", "化学・素材"),
    ("4901.T", "富士フイルムホールディングス", "化学・素材"),
    ("5401.T", "日本製鉄", "化学・素材"),
    # 医薬品・ヘルスケア
    ("4502.T", "武田薬品工業", "医薬品・ヘルスケア"),
    ("4503.T", "アステラス製薬", "医薬品・ヘルスケア"),
    ("4568.T", "第一三共", "医薬品・ヘルスケア"),
    ("4519.T", "中外製薬", "医薬品・ヘルスケア"),
    ("7741.T", "HOYA", "医薬品・ヘルスケア"),
    ("2413.T", "エムスリー", "医薬品・ヘルスケア"),
    # 銀行・金融・保険
    ("8306.T", "三菱UFJフィナンシャル・グループ", "銀行・金融・保険"),
    ("8316.T", "三井住友フィナンシャルグループ", "銀行・金融・保険"),
    ("8411.T", "みずほフィナンシャルグループ", "銀行・金融・保険"),
    ("8766.T", "東京海上ホールディングス", "銀行・金融・保険"),
    ("8591.T", "オリックス", "銀行・金融・保険"),
    ("8604.T", "野村ホールディングス", "銀行・金融・保険"),
    # 商社
    ("8058.T", "三菱商事", "商社"),
    ("8031.T", "三井物産", "商社"),
    ("8001.T", "伊藤忠商事", "商社"),
    # 情報・通信・IT
    ("9432.T", "日本電信電話", "情報・通信・IT"),
    ("9433.T", "KDDI", "情報・通信・IT"),
    ("9434.T", "ソフトバンク", "情報・通信・IT"),
    ("9984.T", "ソフトバンクグループ", "情報・通信・IT"),
    ("4307.T", "野村総合研究所", "情報・通信・IT"),
    ("4689.T", "LINEヤフー", "情報・通信・IT"),
    ("4755.T", "楽天グループ", "情報・通信・IT"),
    # サービス・ゲーム・レジャー
    ("6098.T", "リクルートホールディングス", "サービス・ゲーム・レジャー"),
    ("7974.T", "任天堂", "サービス・ゲーム・レジャー"),
    ("4661.T", "オリエンタルランド", "サービス・ゲーム・レジャー"),
    # 小売・食品・日用品
    ("9983.T", "ファーストリテイリング", "小売・食品・日用品"),
    ("3382.T", "セブン&アイ・ホールディングス", "小売・食品・日用品"),
    ("8267.T", "イオン", "小売・食品・日用品"),
    ("2914.T", "日本たばこ産業", "小売・食品・日用品"),
    ("2802.T", "味の素", "小売・食品・日用品"),
    # 不動産・運輸・エネルギー
    ("8801.T", "三井不動産", "不動産・運輸・エネルギー"),
    ("9020.T", "東日本旅客鉄道", "不動産・運輸・エネルギー"),
    ("9202.T", "ANAホールディングス", "不動産・運輸・エネルギー"),
    ("1605.T", "INPEX", "不動産・運輸・エネルギー"),
]

SCAN_COLUMNS = [
    "区分", "順位", "コード", "会社名", "業種",
    "中期スコア", "中期ラベル", "短期スコア", "短期ラベル",
    "長期スコア", "長期ラベル", "終値", "選定理由", "警告・理由",
    "データ最終日", "本数", "取得日時",
]


def _shape(df):
    """一括取得の結果を、analysis.fetch_history と同じ形に整える。"""
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        out = df.copy()
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = out.columns.get_level_values(0)
        if not all(c in out.columns for c in analysis.NEEDED_COLUMNS):
            return pd.DataFrame()
        return out[analysis.NEEDED_COLUMNS].dropna().copy()
    except Exception:
        return pd.DataFrame()


def fetch_batch(symbols, period=SCAN_PERIOD):
    """複数銘柄を一括取得する。戻り値: {symbol: (DataFrame, error)}。

    CHUNK_SIZE ずつ yf.download でまとめて取得する。ここで取れなかった銘柄は
    呼び出し側が個別取得で取り直す前提で、空データと理由を返す。
    """
    results = {}
    symbols = list(symbols)
    for start in range(0, len(symbols), CHUNK_SIZE):
        chunk = symbols[start:start + CHUNK_SIZE]
        try:
            data = yf.download(
                chunk, period=period, auto_adjust=True,
                progress=False, threads=True, group_by="ticker",
            )
        except Exception as exc:
            for sym in chunk:
                results[sym] = (pd.DataFrame(), f"一括取得でエラーが発生しました（{type(exc).__name__}）")
            data = None
        if data is not None and not data.empty:
            for sym in chunk:
                try:
                    sub = data[sym] if len(chunk) > 1 else data
                except Exception:
                    sub = pd.DataFrame()
                shaped = _shape(sub)
                if shaped.empty:
                    results[sym] = (pd.DataFrame(), "一括取得でデータが取れませんでした")
                else:
                    results[sym] = (shaped, None)
        else:
            for sym in chunk:
                if sym not in results:
                    results[sym] = (pd.DataFrame(), "一括取得でデータが取れませんでした")
        if start + CHUNK_SIZE < len(symbols):
            time.sleep(CHUNK_PAUSE_SECONDS)
    return results


def _score_key(row):
    """並び順：中期→長期→短期のスコア降順、最後はコード順（決定的にする）。"""
    def val(score):
        return score if isinstance(score, (int, float)) else -999
    return (-val(row["中期スコア"]), -val(row["長期スコア"]), -val(row["短期スコア"]), row["コード"])


def run_scan(universe, batch_fetcher, single_fetcher, run_at=None, progress=None):
    """スキャンを実行する。

    universe: (コード, 会社名, 業種) のリスト。
    batch_fetcher(symbols) -> {symbol: (df, err)}、single_fetcher(symbol) -> (df, err)。
    一括取得で取れなかった銘柄は single_fetcher で取り直す。
    戻り値: (ranked, excluded, meta)
    """
    if run_at is None:
        run_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    symbols = [u[0] for u in universe]
    batch = batch_fetcher(symbols)
    ranked = []
    excluded = []
    for done, (symbol, name, sector) in enumerate(universe, start=1):
        raw, err = batch.get(symbol, (pd.DataFrame(), "一括取得の対象になりませんでした"))
        if err is not None or raw is None or raw.empty:
            raw, err = single_fetcher(symbol)
        base = {
            "区分": "", "順位": None, "コード": symbol, "会社名": name, "業種": sector,
            "中期スコア": None, "中期ラベル": "", "短期スコア": None, "短期ラベル": "",
            "長期スコア": None, "長期ラベル": "", "終値": None,
            "選定理由": "", "警告・理由": "", "データ最終日": "", "本数": 0,
            "取得日時": run_at,
        }
        if err is not None or raw is None or raw.empty:
            base["区分"] = "参考外"
            base["警告・理由"] = f"取得失敗：{err or 'データなし'}"
            excluded.append(base)
        else:
            df = analysis.add_indicators(raw)
            results = analysis.assess_all(df)
            fresh = analysis.freshness_check(df, symbol)
            mid = results["中期"]
            base["終値"] = round(float(df["Close"].iloc[-1]), 3)
            base["データ最終日"] = str(fresh["last_date"])
            base["本数"] = fresh["rows"]
            base["短期スコア"] = results["短期"]["score"]
            base["短期ラベル"] = results["短期"]["label"]
            base["中期スコア"] = mid["score"]
            base["中期ラベル"] = mid["label"]
            base["長期スコア"] = results["長期"]["score"]
            base["長期ラベル"] = results["長期"]["label"]
            if not mid["available"]:
                base["区分"] = "参考外"
                base["警告・理由"] = "中期評価ができません（データ不足）：" + " ".join(mid["reasons"])
                excluded.append(base)
            elif fresh["status"] in ("stale", "no_data"):
                # 「古い可能性」のデータはランキングの信頼性を損なうため参考外にする。
                # 「間隔あり（check）」は週末・休場で通常に起こるため、警告を付けて
                # ランキングに入れる（freshのみに絞ると週末に全銘柄が消えてしまう）。
                base["区分"] = "参考外"
                base["警告・理由"] = f"データが古い可能性があります：{fresh['message']}"
                excluded.append(base)
            else:
                base["区分"] = "ランキング"
                base["選定理由"] = " / ".join(mid["reasons"])
                warnings = list(mid["warnings"]) + list(results["短期"]["warnings"])
                if fresh["status"] == "check":
                    warnings.append("データ最終日から間隔があります（週末・休場の可能性）")
                base["警告・理由"] = " / ".join(warnings)
                ranked.append(base)
        if progress is not None:
            progress(done, len(universe))
    ranked.sort(key=_score_key)
    for i, row in enumerate(ranked, start=1):
        row["順位"] = i
    meta = {
        "実行日時": run_at, "対象銘柄数": len(universe),
        "ランキング件数": len(ranked), "参考外件数": len(excluded),
        "取得期間": SCAN_PERIOD,
    }
    return ranked, excluded, meta


def scan_to_csv(ranked, excluded):
    df = pd.DataFrame(ranked + excluded, columns=SCAN_COLUMNS)
    return df.to_csv(index=False).encode("utf-8-sig")
