import io
import zipfile
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import analysis
import forward
import scan
import scoring_v2
import verify

st.set_page_config(page_title="Stock AI Lab", page_icon="📈", layout="wide")

st.title("Stock AI Lab | 株式分析・検証ツール")
st.caption("日本株・米国株 / 短期・中期・長期の比較 / 自動発注なし")

st.warning(
    "試作版です。投資助言・利益予測ではありません。データは遅延・欠落・取得失敗の可能性があります。"
    "日中のリアルタイム・デイトレード用途には使わず、まずは分析と仮想検証に使用してください。"
    "表示するスコアはテクニカル条件の一致度であり、勝率や利益の確率ではありません。"
)

DEFAULT_JP = "7203.T, 6758.T, 8306.T, 9984.T"
DEFAULT_US = "AAPL, MSFT, NVDA, SPY"


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_company_name(symbol):
    """会社名を表示専用に取得する（1日キャッシュ）。失敗したら空文字を返し、
    コードのみの表示に切り替える。分析には使わない。"""
    try:
        import yfinance as yf

        info = yf.Ticker(symbol).info
        if isinstance(info, dict):
            return info.get("shortName") or info.get("longName") or ""
    except Exception:
        pass
    return ""


def company_name(symbol):
    """会社名を返す。初期銘柄は内蔵の対応表を使い、余計な通信をしない。"""
    name = analysis.resolve_company_name(symbol)
    if name:
        return name
    return analysis.resolve_company_name(symbol, fetch_company_name(symbol))

with st.sidebar:
    st.header("設定")
    market = st.selectbox("市場", ["日本株", "米国株"])
    default_symbols = DEFAULT_JP if market == "日本株" else DEFAULT_US
    symbols_text = st.text_area(
        "銘柄コード（カンマ区切り）",
        value=default_symbols,
        key=f"symbols_text_{market}",
        help="日本株は例：7203.T、米国株は例：AAPL。銘柄コードはご自身でも確認してください。",
    )
    sidebar_symbols = list(dict.fromkeys(
        [s.strip().upper() for s in symbols_text.split(",") if s.strip()]
    ))
    if sidebar_symbols:
        selected_symbol = st.selectbox(
            "詳しく分析する銘柄",
            sidebar_symbols,
            key=f"selected_symbol_{market}",
            format_func=lambda s: analysis.select_label(s, company_name(s)),
        )
    else:
        selected_symbol = ""
        st.caption("詳しく分析する銘柄は、銘柄コードを入力すると選べます。")
    period = st.selectbox(
        "取得期間", ["2y", "1y", "5y"], index=0,
        help="長期分析には日足250本以上が必要です。原則として2y（2年分）を選んでください。",
    )
    refresh = st.button("データを取得・更新", type="primary")
    st.caption("無料データ取得は制限・仕様変更により動かない場合があります。")

symbols = list(dict.fromkeys([s.strip().upper() for s in symbols_text.split(",") if s.strip()]))


@st.cache_data(ttl=900, show_spinner=False)
def get_history(symbol, period):
    """取得結果（データとエラー）を15分キャッシュする。更新ボタンで消去できる。"""
    return analysis.fetch_history(symbol, period)


if refresh:
    st.cache_data.clear()


def show_freshness(df, symbol):
    """データの最終日・本数・新旧の暫定判定を表示する。古い場合は目立つ警告にする。"""
    fresh = analysis.freshness_check(df, symbol)
    st.write(f"データ最終日：{fresh['last_date']} / 取得本数：{fresh['rows']}本")
    if fresh["status"] == "stale":
        st.error("注意：" + fresh["message"])
    elif fresh["status"] == "check":
        st.warning(fresh["message"])
    else:
        st.success(fresh["message"])
    st.caption(fresh["note"])
    return fresh


def _regime_at(indicators, base_date):
    """基準日時点の指数の地合いを返す（V1・V2比較の採用ゲート用）。

    判定（固定）：基準日に終値がEMA200以上かつ直近60営業日の騰落が
    0%以上なら「上昇相場」、それ以外は「非上昇相場」。
    EMA200がまだ無い、または基準日の行が無い場合は None（未判定）。
    """
    if indicators is None or indicators.empty or not base_date:
        return None
    dates = [d.date().isoformat() for d in indicators.index]
    if base_date not in dates:
        return None
    pos = dates.index(base_date)
    if pos < 60:
        return None
    row = indicators.iloc[pos]
    ema200 = row.get("EMA200")
    close = row.get("Close")
    if ema200 is None or pd.isna(ema200) or close is None or pd.isna(close):
        return None
    past_close = indicators["Close"].iloc[pos - 60]
    if past_close is None or pd.isna(past_close) or float(past_close) == 0:
        return None
    ret60 = (float(close) / float(past_close) - 1.0) * 100.0
    return "上昇相場" if (float(close) >= float(ema200) and ret60 >= 0) else "非上昇相場"


tab_overview, tab_detail, tab_journal, tab_verify, tab_scan, tab_notes = st.tabs(
    ["銘柄一覧", "個別分析", "仮想売買メモ", "検証", "スキャン", "使い方・注意"]
)

with tab_overview:
    st.subheader("ウォッチリスト")
    st.write("スコアはテクニカル条件の一致度であり、勝率や利益確率ではありません。条件がそろわない場合は中立または判定保留になります。")
    if not symbols:
        st.info("左側で銘柄コードを入力してください。")
    else:
        rows = []
        progress = st.progress(0)
        status = st.empty()
        for i, sym in enumerate(symbols):
            status.write(f"取得中：{sym}")
            raw, err = get_history(sym, period)
            if err is not None or raw.empty:
                rows.append({
                    "銘柄": sym, "会社名": company_name(sym), "状態": "取得失敗", "終値": None,
                    "データ最終日": None, "本数": 0,
                    "短期": "—", "中期": "—", "長期": "—",
                    "備考": err or "データなし",
                })
            else:
                df = analysis.add_indicators(raw)
                results = analysis.assess_all(df)
                fresh = analysis.freshness_check(df, sym)
                note = ""
                if fresh["status"] == "stale":
                    note = "データが古い可能性あり"
                elif fresh["status"] == "check":
                    note = "最終日から間隔あり"
                rows.append({
                    "銘柄": sym, "会社名": company_name(sym), "状態": "取得成功",
                    "終値": round(float(df["Close"].iloc[-1]), 3),
                    "データ最終日": str(fresh["last_date"]),
                    "本数": fresh["rows"],
                    "短期": results["短期"]["label"] if results["短期"]["available"] else "判定保留",
                    "中期": results["中期"]["label"] if results["中期"]["available"] else "判定保留",
                    "長期": results["長期"]["label"] if results["長期"]["available"] else "判定不可",
                    "備考": note,
                })
            progress.progress((i + 1) / len(symbols))
        status.empty()
        progress.empty()
        table = pd.DataFrame(rows)
        st.dataframe(table, width="stretch", hide_index=True)
        st.download_button(
            "一覧をCSVで保存",
            data=table.to_csv(index=False).encode("utf-8-sig"),
            file_name="stock_ai_watchlist.csv",
            mime="text/csv",
        )
        st.caption("データの新旧は暦日による暫定チェックです。公式の営業日カレンダーとは照合していません。個別分析タブで銘柄ごとの詳細と警告を確認してください。")

with tab_detail:
    st.subheader("個別銘柄のテクニカル分析")
    if not selected_symbol:
        st.info("左側に銘柄コードを入力してください。")
    else:
        st.markdown(f"**対象銘柄：{analysis.display_label(selected_symbol, company_name(selected_symbol))}**")
        raw, err = get_history(selected_symbol, period)
        if err is not None or raw.empty:
            st.error("データを取得できませんでした。" + (err or ""))
            st.write("銘柄コードを確認するか、時間をおいて「データを取得・更新」を押してください。取得できない場合は、その旨を表示するだけで、有料データへは切り替えません。")
        else:
            df = analysis.add_indicators(raw)
            show_freshness(df, selected_symbol)
            latest = df.iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("最新終値（調整後）", f"{latest['Close']:.3f}")
            c2.metric("RSI(14)", "—" if pd.isna(latest["RSI14"]) else f"{latest['RSI14']:.1f}")
            c3.metric("EMA20", f"{latest['EMA20']:.3f}")
            c4.metric("データ最終日", str(df.index[-1].date()))
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="価格"
            ))
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20"))
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50"))
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], name="EMA200"))
            fig.update_layout(height=500, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, width="stretch")
            results = analysis.assess_all(df)
            cols = st.columns(3)
            for col, horizon in zip(cols, ["短期", "中期", "長期"]):
                result = results[horizon]
                with col:
                    st.markdown(f"**{horizon}の条件評価**")
                    if result["available"]:
                        st.metric("条件スコア", result["score"])
                    else:
                        st.metric("条件スコア", "—")
                    st.write(result["label"])
                    for reason in result["reasons"]:
                        st.write(f"• {reason}")
                    for warning in result["warnings"]:
                        st.warning(warning)
            st.download_button(
                "分析データをCSVで保存",
                data=df.to_csv(index=True).encode("utf-8-sig"),
                file_name=f"{selected_symbol.replace('.', '_')}_analysis.csv",
                mime="text/csv",
            )
            st.caption("短期判定も日足データに基づきます。1分足・5分足のリアルタイムデイトレード判定ではありません。")

with tab_journal:
    st.subheader("仮想売買メモ")
    st.write("入力した内容は、この画面のセッション内で扱います。永続保存は保証されません。定期的にCSVでダウンロードしてください。")
    if "journal" not in st.session_state:
        st.session_state.journal = pd.DataFrame(columns=[
            "日時", "銘柄", "スタイル", "方向", "想定エントリー", "想定損切り",
            "想定利確", "終了価格", "メモ", "結果"
        ])
    with st.form("journal_form"):
        j1, j2 = st.columns(2)
        symbol = j1.text_input("銘柄コード", value=selected_symbol or "AAPL")
        style = j2.selectbox("スタイル", ["短期", "中期", "長期"])
        direction = j1.selectbox("方向", ["買い", "売り", "見送り"])
        entry = j2.number_input("想定エントリー価格", min_value=0.0, value=0.0, step=0.1, format="%.3f")
        stop = j1.number_input("想定損切り価格", min_value=0.0, value=0.0, step=0.1, format="%.3f")
        target = j2.number_input("想定利確価格", min_value=0.0, value=0.0, step=0.1, format="%.3f")
        exit_price = j1.number_input("終了価格（未確定なら0）", min_value=0.0, value=0.0, step=0.1, format="%.3f")
        note = st.text_area("判断理由・後から分かったこと")
        submitted = st.form_submit_button("メモを追加")
    if submitted:
        result = ""
        if entry > 0 and exit_price > 0 and direction in ["買い", "売り"]:
            pnl = (exit_price - entry) if direction == "買い" else (entry - exit_price)
            result = f"価格差 {pnl:.3f}（コスト除外）"
        new_row = {
            "日時": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S"),
            "銘柄": symbol, "スタイル": style, "方向": direction,
            "想定エントリー": entry, "想定損切り": stop, "想定利確": target,
            "終了価格": exit_price, "メモ": note, "結果": result
        }
        st.session_state.journal = pd.concat([st.session_state.journal, pd.DataFrame([new_row])], ignore_index=True)
        st.success("このセッションにメモを追加しました。CSVで保存してください。")
    st.dataframe(st.session_state.journal, width="stretch", hide_index=True)
    st.download_button(
        "仮想売買メモをCSVで保存",
        data=st.session_state.journal.to_csv(index=False).encode("utf-8-sig"),
        file_name="stock_paper_journal.csv",
        mime="text/csv",
    )

with tab_verify:
    st.subheader("過去データでの検証")
    st.write(
        "過去の評価日時点で分析したと仮定し、その後の値動きで採点します。"
        "評価は評価日までのデータだけで再現し、未来の価格は計算に使いません。"
        f"検証仕様の版数は {verify.SPEC_VERSION} で、条件は固定です（測定窓：短期5・中期20・長期60営業日、"
        "小動きの幅：短期±1%・中期±2%・長期±3%、評価日は5営業日ごと、データは5年分）。"
    )
    st.write(
        "方向付き騰落率は参考値で、売買コストを含まないため実際の売買損益とは異なります。"
        "中立予測は想定損益の集計に含めません。評価日を5営業日ごとに取るため測定窓が重なり、"
        "検証件数は独立した試行の数ではありません。価格は分割・配当の調整済みで、当時の表示そのままではありません。"
        "的中・不的中の合計が少ない時間軸は、精度を判断できません。"
    )

    if "verify_records" not in st.session_state:
        st.session_state.verify_records = []
    if "verify_exclusions" not in st.session_state:
        st.session_state.verify_exclusions = []
    if "verify_fetch_info" not in st.session_state:
        st.session_state.verify_fetch_info = []

    if "verify_conflicts" not in st.session_state:
        st.session_state.verify_conflicts = []

    run_verify = st.button("検証を実行（対象8銘柄・5年分）", type="primary")
    if run_verify:
        progress_bar = st.progress(0, text="データ取得と検証計算を行っています。")
        records, exclusions, fetch_info = verify.verify_all(
            verify.VERIFY_SYMBOLS,
            lambda s: analysis.fetch_history(s, verify.FETCH_PERIOD),
            progress=lambda done, total: progress_bar.progress(
                done / total, text=f"検証中：{done}/{total} 銘柄"
            ),
        )
        progress_bar.empty()
        merged, conflicts = verify.merge_records(st.session_state.verify_records, records)
        st.session_state.verify_records = merged
        st.session_state.verify_conflicts = conflicts
        st.session_state.verify_exclusions = verify.merge_exclusions(
            st.session_state.verify_exclusions, exclusions
        )
        st.session_state.verify_fetch_info = fetch_info
        st.success(f"検証を実行しました。記録は合計 {len(merged)} 件です（重複は除いています）。")

    uploaded = st.file_uploader(
        "保存した検証記録CSVを読み込む（復元・合算）", type="csv", key="verify_csv_upload"
    )
    if uploaded is not None:
        loaded = verify.records_from_csv(uploaded.getvalue())
        if loaded:
            before = len(st.session_state.verify_records)
            merged, conflicts = verify.merge_records(st.session_state.verify_records, loaded)
            st.session_state.verify_records = merged
            st.session_state.verify_conflicts = conflicts
            st.success(f"CSVから {len(loaded)} 件を読み込み、{len(merged) - before} 件を追加しました（重複は除いています）。")
        else:
            st.error("このCSVは検証記録の形式ではありませんでした。記録は変更していません。")

    uploaded_ex = st.file_uploader(
        "保存した対象外集計CSVを読み込む（復元）", type="csv", key="verify_excl_upload"
    )
    if uploaded_ex is not None:
        loaded_ex = verify.exclusions_from_csv(uploaded_ex.getvalue())
        if loaded_ex:
            before = len(st.session_state.verify_exclusions)
            st.session_state.verify_exclusions = verify.merge_exclusions(
                st.session_state.verify_exclusions, loaded_ex
            )
            st.success(
                f"対象外集計を {len(st.session_state.verify_exclusions) - before} 件追加しました。"
                "対象外の件数は実行ごとの地点数のため、同じ条件のものは合算しません（二重計上防止）。"
            )
        else:
            st.error("このCSVは対象外集計の形式ではありませんでした。対象外集計は変更していません。")

    if st.session_state.verify_conflicts:
        st.error(
            "同じキー（銘柄・評価日・時間軸・仕様版数）で内容の異なる記録がありました。"
            "競合した新しい記録は取り込んでいません。既存の記録を残しています。"
        )
        st.dataframe(
            pd.DataFrame(st.session_state.verify_conflicts),
            width="stretch", hide_index=True,
        )

    if st.session_state.verify_fetch_info:
        st.markdown("**今回の取得結果（実際に取得できた期間と件数）**")
        st.dataframe(
            pd.DataFrame(st.session_state.verify_fetch_info),
            width="stretch", hide_index=True,
        )
    if st.session_state.verify_exclusions:
        st.markdown("**対象外の集計（成功や的中には数えていません）**")
        st.dataframe(
            pd.DataFrame(st.session_state.verify_exclusions),
            width="stretch", hide_index=True,
        )

    v_records = st.session_state.verify_records
    if v_records or st.session_state.verify_exclusions:
        summary = verify.aggregate(v_records, st.session_state.verify_exclusions)
        st.markdown("**時間軸別の集計（的中率の分母は的中＋不的中のみ）**")
        horizon_df = pd.DataFrame(summary["時間軸別"])
        horizon_df = horizon_df.astype(object).where(pd.notna(horizon_df), None)
        st.dataframe(horizon_df, width="stretch", hide_index=True)
        for row in summary["時間軸別"]:
            if row["精度判断"].startswith("件数不足"):
                st.warning(
                    f"{row['時間軸']}は的中・不的中の合計が {row['判定件数']} 件のため、精度を判断できません。"
                )
        if summary["実際方向別内訳"]:
            st.markdown("**実際に上昇・下落した場面での予測内訳（見逃しの把握用）**")
            st.dataframe(
                pd.DataFrame(summary["実際方向別内訳"]), width="stretch", hide_index=True
            )
        with st.expander("銘柄別の集計"):
            st.dataframe(pd.DataFrame(summary["銘柄別"]), width="stretch", hide_index=True)
        if v_records:
            with st.expander(f"検証記録の明細（{len(v_records)}件）"):
                st.dataframe(
                    pd.DataFrame(v_records, columns=verify.RECORD_COLUMNS),
                    width="stretch", hide_index=True,
                )
            st.download_button(
                "検証記録をCSVで保存",
                data=verify.records_to_csv(v_records),
                file_name="stock_ai_verification.csv",
                mime="text/csv",
            )
        if st.session_state.verify_exclusions:
            st.download_button(
                "対象外集計をCSVで保存",
                data=verify.exclusions_to_csv(st.session_state.verify_exclusions),
                file_name="stock_ai_verification_exclusions.csv",
                mime="text/csv",
            )
        if st.button("この画面の検証記録を消去"):
            st.session_state.verify_records = []
            st.session_state.verify_exclusions = []
            st.session_state.verify_fetch_info = []
            st.session_state.verify_conflicts = []
            st.rerun()
    else:
        st.info("まだ検証記録がありません。「検証を実行」を押すか、保存したCSVを読み込んでください。記録はこの画面を開いている間だけ保持されるため、CSVで保存してください。")

with tab_scan:
    st.subheader("主要な日本株のスキャン")
    st.write(
        f"主要な日本株 {len(scan.UNIVERSE_JP)} 銘柄を一括で取得し、中期評価のスコア順に並べます。"
        "短期・長期の評価は補助情報です。ランキングに入るのは、中期評価ができて、"
        "データが古い可能性でない銘柄だけです（週末・休場による間隔は、警告を付けて含めます）。"
        "取得に失敗した銘柄や、データが古い・不足している銘柄は、理由つきで「参考外」として分けて表示します。"
    )
    st.write(
        "上位の銘柄は「条件がそろった銘柄」であって、将来の値上がりを予測するものではありません。"
        "選定理由と警告を確認し、最終判断はご自身で行ってください。取得結果は15分キャッシュされます。"
    )

    if "scan_ranked" not in st.session_state:
        st.session_state.scan_ranked = []
    if "scan_excluded" not in st.session_state:
        st.session_state.scan_excluded = []
    if "scan_meta" not in st.session_state:
        st.session_state.scan_meta = None
    if "scan_frames" not in st.session_state:
        st.session_state.scan_frames = {}

    @st.cache_data(ttl=900, show_spinner=False)
    def get_batch(symbols_key, period):
        return scan.fetch_batch(list(symbols_key), period)

    force_rescan = st.checkbox("キャッシュを使わず最新のデータで取り直す", key="scan_force")
    if st.button("スキャンを実行", type="primary"):
        if force_rescan:
            st.cache_data.clear()
        progress_bar = st.progress(0, text="データ取得と評価を行っています。")
        # V2 の計算には、スキャンが実際に使ったものと同一の株価データが必要なため、
        # 取得したデータをそのまま控えておく（V1 と V2 を同じデータで計算する）
        captured_frames = {}

        def batch_fetcher(syms):
            batch = get_batch(tuple(syms), scan.SCAN_PERIOD)
            for sym, (df, err) in batch.items():
                if err is None and df is not None and not df.empty:
                    captured_frames[sym] = df.copy()
            return batch

        def single_fetcher(symbol):
            df, err = analysis.fetch_history(symbol, scan.SCAN_PERIOD)
            if err is None and df is not None and not df.empty:
                captured_frames[symbol] = df.copy()
            return df, err

        ranked, excluded, meta = scan.run_scan(
            scan.UNIVERSE_JP,
            batch_fetcher,
            single_fetcher,
            progress=lambda done, total: progress_bar.progress(
                done / total, text=f"スキャン中：{done}/{total} 銘柄"
            ),
        )
        progress_bar.empty()
        st.session_state.scan_ranked = ranked
        st.session_state.scan_excluded = excluded
        st.session_state.scan_meta = meta
        st.session_state.scan_frames = captured_frames
        st.success(
            f"スキャンが終わりました。対象 {meta['対象銘柄数']} 銘柄のうち、"
            f"ランキング {meta['ランキング件数']} 件、参考外 {meta['参考外件数']} 件です。"
        )

    if st.session_state.scan_meta:
        meta = st.session_state.scan_meta
        st.caption(f"実行日時：{meta['実行日時']} / 取得期間：{meta['取得期間']}")
        ranked_df = pd.DataFrame(st.session_state.scan_ranked, columns=scan.SCAN_COLUMNS)
        ranked_df = ranked_df.astype(object).where(pd.notna(ranked_df), None)
        st.dataframe(ranked_df, width="stretch", hide_index=True)
        if st.session_state.scan_excluded:
            st.markdown("**参考外（ランキングには入れていません）**")
            excluded_df = pd.DataFrame(st.session_state.scan_excluded, columns=scan.SCAN_COLUMNS)
            excluded_df = excluded_df.astype(object).where(pd.notna(excluded_df), None)
            st.dataframe(excluded_df, width="stretch", hide_index=True)
        st.download_button(
            "スキャン結果をCSVで保存",
            data=scan.scan_to_csv(st.session_state.scan_ranked, st.session_state.scan_excluded),
            file_name="stock_ai_scan.csv",
            mime="text/csv",
        )
    else:
        st.info("まだスキャン結果がありません。「スキャンを実行」を押してください。結果はこの画面を開いている間だけ保持されるため、必要ならCSVで保存してください。")

    st.divider()
    st.subheader("フォワード登録と事後評価")
    st.write(
        "スキャン結果をその時点のまま登録しておき、1・5・10・20営業日後に実際の値動きと比較します。"
        "登録した予測を後から測るための仕組みで、過去のデータを測る「検証」タブとは別です。"
        "登録は全60銘柄（参考外を含む）が対象で、評価は順位が付いた銘柄だけを対象にします。"
        "登録時は現行仕様（V1）と新仕様（V2）の両方で記録します。V2は比較評価のための並行記録で、"
        "画面の表示や順位づけはこれまでどおりV1のままです。"
    )
    st.caption(
        "指数比較には日経平均とTOPIX連動ETF（1306.T）を使います。"
        "TOPIX指数そのものは無料で取得できないため、1306.T はTOPIXの代替の参考指標です（指数そのものではありません）。"
    )

    if "fwd_records" not in st.session_state:
        st.session_state.fwd_records = []
    if "fwd_results" not in st.session_state:
        st.session_state.fwd_results = []
    if "fwd_conflicts" not in st.session_state:
        st.session_state.fwd_conflicts = []
    if "fwd_result_conflicts" not in st.session_state:
        st.session_state.fwd_result_conflicts = []
    if "fwd_records_unsaved" not in st.session_state:
        st.session_state.fwd_records_unsaved = False
    if "fwd_results_unsaved" not in st.session_state:
        st.session_state.fwd_results_unsaved = False
    if "fwd_eval_notes" not in st.session_state:
        st.session_state.fwd_eval_notes = []
    if "fwd_register_note" not in st.session_state:
        st.session_state.fwd_register_note = ""
    if "fwd_v2_skipped" not in st.session_state:
        st.session_state.fwd_v2_skipped = []
    if "fwd_v2_skipped_unsaved" not in st.session_state:
        st.session_state.fwd_v2_skipped_unsaved = False

    def _clear_fwd_records_unsaved():
        st.session_state.fwd_records_unsaved = False

    def _clear_fwd_results_unsaved():
        st.session_state.fwd_results_unsaved = False

    def _clear_fwd_skipped_unsaved():
        st.session_state.fwd_v2_skipped_unsaved = False

    def _version_count(records, version):
        count = 0
        for r in records:
            try:
                v = int(float(r.get("スコア仕様バージョン") or 1))
            except (TypeError, ValueError):
                v = 1
            if v == version:
                count += 1
        return count

    if (
        st.session_state.fwd_records_unsaved
        or st.session_state.fwd_results_unsaved
        or st.session_state.fwd_v2_skipped_unsaved
    ):
        st.warning(
            "未保存のフォワードデータがあります。フォワード登録CSV・フォワード評価CSV・"
            "V2未登録の理由CSVをダウンロードして保存してください"
            "（「使い方・注意」タブの「全データのバックアップ」でもまとめて保存できます）。"
            "画面を閉じると消えます。"
        )

    if st.session_state.scan_meta:
        if st.button("このスキャン結果をフォワード登録する（V1・V2の両方）"):
            with st.spinner("V1・V2の両方で登録しています。"):
                benchmarks = {}
                for b_symbol in (forward.NIKKEI_SYMBOL, forward.TOPIX_PROXY_SYMBOL):
                    b_df, b_err = analysis.fetch_history(b_symbol, "3mo")
                    if b_err is None and b_df is not None and not b_df.empty:
                        benchmarks[b_symbol] = (
                            round(float(b_df["Close"].iloc[-1]), 3),
                            pd.Timestamp(b_df.index[-1]).date().isoformat(),
                        )
                ranked = st.session_state.scan_ranked
                excluded = st.session_state.scan_excluded
                meta = st.session_state.scan_meta
                v1_records = forward.build_records(ranked, excluded, meta, benchmarks)

                # V2 は、スキャンが実際に使ったデータ（控えた取得結果）から計算する。
                # 控えがない銘柄だけ、基準日検査を通した再取得で代替する。
                frames = st.session_state.get("scan_frames") or {}
                assessments, failed, fallback_codes = {}, {}, set()
                for row in list(ranked) + list(excluded):
                    code = row["コード"]
                    if code in frames:
                        ind = analysis.add_indicators(frames[code])
                        assessments[code] = {
                            "results": scoring_v2.assess_all_v2(ind),
                            "overheat": scoring_v2.overheat_flag(ind),
                        }
                    elif row.get("データ最終日"):
                        raw, err = analysis.fetch_history(code, scan.SCAN_PERIOD)
                        if err is not None or raw is None or raw.empty:
                            failed[code] = f"V2再取得に失敗しました：{err or 'データなし'}"
                        else:
                            checked, reason = forward.prepare_v2_frame(
                                raw, row.get("データ最終日"), row.get("終値")
                            )
                            if checked is None:
                                failed[code] = reason
                            else:
                                ind = analysis.add_indicators(checked)
                                assessments[code] = {
                                    "results": scoring_v2.assess_all_v2(ind),
                                    "overheat": scoring_v2.overheat_flag(ind),
                                }
                                fallback_codes.add(code)
                    # データ最終日がない銘柄（取得失敗）は、build_v2_scan_rows が
                    # 理由つきで未登録（skipped）として返す
                ranked_v2, excluded_v2, overheat_flags, skipped = (
                    forward.build_v2_scan_rows(ranked, excluded, assessments, failed)
                )
                for row in ranked_v2 + excluded_v2:
                    if row["コード"] in fallback_codes:
                        note = forward.FALLBACK_NOTE
                        row["警告・理由"] = (
                            (row["警告・理由"] + " / " + note) if row["警告・理由"] else note
                        )
                v2_records = forward.build_records(
                    ranked_v2, excluded_v2, meta, benchmarks,
                    scoring_version=2,
                    direction_fn=scoring_v2.direction_from_score_v2,
                    overheat_flags=overheat_flags,
                )

                merged, conflicts1 = forward.merge_records(
                    st.session_state.fwd_records, v1_records
                )
                added_v1 = len(merged) - len(st.session_state.fwd_records)
                merged, conflicts2 = forward.merge_records(merged, v2_records)
                added_v2 = len(merged) - (len(st.session_state.fwd_records) + added_v1)
                conflicts = conflicts1 + conflicts2
                st.session_state.fwd_records = merged
                st.session_state.fwd_conflicts = conflicts
                if added_v1 or added_v2:
                    st.session_state.fwd_records_unsaved = True
                if skipped:
                    new_skipped = [
                        {"スキャンID": meta["実行日時"], "コード": s["コード"], "理由": s["理由"]}
                        for s in skipped
                    ]
                    before_skipped = len(st.session_state.fwd_v2_skipped)
                    st.session_state.fwd_v2_skipped = forward.merge_skipped(
                        st.session_state.fwd_v2_skipped, new_skipped
                    )
                    if len(st.session_state.fwd_v2_skipped) > before_skipped:
                        st.session_state.fwd_v2_skipped_unsaved = True
                note_text = (
                    f"フォワード登録：V1 {added_v1} 件・V2 {added_v2} 件を追加しました"
                    f"（重複 V1 {len(v1_records) - added_v1 - len(conflicts1)} 件・"
                    f"V2 {len(v2_records) - added_v2 - len(conflicts2)} 件、"
                    f"競合 {len(conflicts)} 件）。"
                )
                if skipped:
                    note_text += (
                        f"V2で計算できなかった銘柄が {len(skipped)} 件あります"
                        "（V2未登録として扱い、理由は下の「V2未登録の理由」に残しました）。"
                    )
                if fallback_codes:
                    note_text += (
                        f"V2のうち {len(fallback_codes)} 件は、スキャン時のデータの控えが"
                        "なかったため再取得データ（基準日検査済み）で計算しました。"
                    )
                st.session_state.fwd_register_note = note_text
            # 未保存の警告を同じ表示で出すため、メッセージを残して再描画する
            st.rerun()
        if st.session_state.fwd_register_note:
            st.success(st.session_state.fwd_register_note)

    fwd_records = st.session_state.fwd_records
    if fwd_records:
        scan_count = len({r["スキャンID"] for r in fwd_records})
        st.write(
            f"登録済み：{len(fwd_records)} 件（スキャン {scan_count} 回分／"
            f"V1 {_version_count(fwd_records, 1)} 件・V2 {_version_count(fwd_records, 2)} 件）"
        )
        with st.expander("登録レコードを見る"):
            rec_df = pd.DataFrame(fwd_records, columns=forward.RECORD_COLUMNS)
            rec_df = rec_df.astype(object).where(pd.notna(rec_df), None)
            st.dataframe(rec_df, width="stretch", hide_index=True)
        st.download_button(
            "フォワード登録CSVを保存",
            data=forward.records_to_csv(fwd_records),
            file_name="stock_ai_forward_records.csv",
            mime="text/csv",
            on_click=_clear_fwd_records_unsaved,
        )
    up_records = st.file_uploader(
        "フォワード登録CSVを読み込む（復元）", type="csv", key="fwd_records_upload"
    )
    if up_records is not None:
        loaded = forward.records_from_csv(up_records.getvalue())
        if loaded:
            before = len(st.session_state.fwd_records)
            merged, conflicts = forward.merge_records(
                st.session_state.fwd_records, loaded
            )
            st.session_state.fwd_records = merged
            st.session_state.fwd_conflicts = conflicts
            st.success(
                f"登録CSVを読み込みました：{len(merged) - before} 件を追加しました"
                f"（競合 {len(conflicts)} 件）。"
            )
        else:
            st.error("登録CSVの形式が違うため、読み込めませんでした。")
    if st.session_state.fwd_conflicts:
        st.warning("登録の競合があります（既存の記録を残し、新しい方は取り込んでいません）。")
        st.dataframe(
            pd.DataFrame(st.session_state.fwd_conflicts),
            width="stretch", hide_index=True,
        )

    if st.session_state.fwd_v2_skipped:
        with st.expander(f"V2未登録の理由（{len(st.session_state.fwd_v2_skipped)} 件）"):
            st.write(
                "V2で計算できなかったため、V2の記録を作らなかった銘柄です。"
                "比較では「片側登録（V2のみ不足）」として扱われます。"
            )
            skipped_df = pd.DataFrame(
                st.session_state.fwd_v2_skipped, columns=forward.SKIPPED_COLUMNS
            )
            st.dataframe(skipped_df, width="stretch", hide_index=True)
        st.download_button(
            "V2未登録の理由をCSVで保存",
            data=forward.skipped_to_csv(st.session_state.fwd_v2_skipped),
            file_name="stock_ai_v2_skipped.csv",
            mime="text/csv",
            on_click=_clear_fwd_skipped_unsaved,
        )
    up_skipped = st.file_uploader(
        "V2未登録の理由CSVを読み込む（復元）", type="csv", key="fwd_v2_skipped_upload"
    )
    if up_skipped is not None:
        loaded_skipped = forward.skipped_from_csv(up_skipped.getvalue())
        if loaded_skipped:
            before_skipped = len(st.session_state.fwd_v2_skipped)
            st.session_state.fwd_v2_skipped = forward.merge_skipped(
                st.session_state.fwd_v2_skipped, loaded_skipped
            )
            st.success(
                "V2未登録の理由を読み込みました："
                f"{len(st.session_state.fwd_v2_skipped) - before_skipped} 件を追加しました。"
            )
        else:
            st.error("V2未登録の理由CSVの形式が違うため、読み込めませんでした。")

    if fwd_records:
        if st.button("事後評価を実行"):
            symbols = sorted(
                {r["コード"] for r in fwd_records if r.get("区分") == "ランキング"}
            )
            batch = scan.fetch_batch(symbols, scan.SCAN_PERIOD)

            def fwd_fetcher(symbol):
                if symbol in batch:
                    return batch[symbol]
                return analysis.fetch_history(symbol, scan.SCAN_PERIOD)

            with st.spinner("評価データを取得しています。"):
                new_results, pending, failed = forward.evaluate_records(
                    fwd_records, fwd_fetcher
                )
            before = len(st.session_state.fwd_results)
            merged, conflicts = forward.merge_results(
                st.session_state.fwd_results, new_results
            )
            st.session_state.fwd_results = merged
            st.session_state.fwd_result_conflicts = conflicts
            added = len(merged) - before
            if added:
                st.session_state.fwd_results_unsaved = True
            unevaluable = sum(
                1 for r in new_results if r.get("状態") == "評価不能"
            )
            notes = [
                f"評価結果：{added} 件を追加しました"
                f"（うち評価不能 {unevaluable} 件、競合 {len(conflicts)} 件）。"
            ]
            if pending:
                notes.append(
                    f"まだ評価時点に到達していないものが {len(pending)} 件あります。"
                    "到達後に再度「事後評価を実行」を押してください。"
                )
            for item in failed:
                notes.append(f"取得失敗：{item['コード']}（{item['理由']}）")
            st.session_state.fwd_eval_notes = notes
            # 未保存の警告を同じ表示で出すため再描画する（メモは残る）
            st.rerun()
        for note in st.session_state.fwd_eval_notes:
            st.info(note)

    fwd_results = st.session_state.fwd_results
    if fwd_results:
        st.download_button(
            "フォワード評価CSVを保存",
            data=forward.results_to_csv(fwd_results),
            file_name="stock_ai_forward_results.csv",
            mime="text/csv",
            on_click=_clear_fwd_results_unsaved,
        )
    up_results = st.file_uploader(
        "フォワード評価CSVを読み込む（復元）", type="csv", key="fwd_results_upload"
    )
    if up_results is not None:
        loaded = forward.results_from_csv(up_results.getvalue())
        if loaded:
            before = len(st.session_state.fwd_results)
            merged, conflicts = forward.merge_results(
                st.session_state.fwd_results, loaded
            )
            st.session_state.fwd_results = merged
            st.session_state.fwd_result_conflicts = conflicts
            st.success(
                f"評価CSVを読み込みました：{len(merged) - before} 件を追加しました"
                f"（競合 {len(conflicts)} 件）。"
            )
        else:
            st.error("評価CSVの形式が違うため、読み込めませんでした。")
    if st.session_state.fwd_result_conflicts:
        st.warning("評価結果の競合があります（既存の結果を残し、新しい方は取り込んでいません）。")
        st.dataframe(
            pd.DataFrame(st.session_state.fwd_result_conflicts),
            width="stretch", hide_index=True,
        )

    if fwd_records and fwd_results:
        summary = forward.summarize(fwd_results, fwd_records)
        if summary["時点別"]:
            st.markdown("**時点別の成績（順位が付いた全銘柄）**")
            point_df = pd.DataFrame(
                [{"評価営業日数": n, **vals} for n, vals in summary["時点別"].items()]
            )
            point_df = point_df.astype(object).where(pd.notna(point_df), None)
            st.dataframe(point_df, width="stretch", hide_index=True)
        if summary["順位帯別"]:
            st.markdown("**中期順位の上位帯別の成績**")
            band_rows = []
            for n, band_map in summary["順位帯別"].items():
                for band_name in ("上位5", "上位10", "上位20", "全体"):
                    if band_name in band_map:
                        band_rows.append(
                            {"評価営業日数": n, "順位帯": band_name, **band_map[band_name]}
                        )
            band_df = pd.DataFrame(band_rows)
            band_df = band_df.astype(object).where(pd.notna(band_df), None)
            st.dataframe(band_df, width="stretch", hide_index=True)
        if summary["時間軸別"]:
            st.markdown("**時間軸別の成績（的中率は方向の予測だけが分母）**")
            horizon_rows = [
                {"評価営業日数": n, "時間軸": horizon, **vals}
                for (n, horizon), vals in sorted(
                    summary["時間軸別"].items(), key=lambda kv: (kv[0][0], kv[0][1])
                )
            ]
            horizon_df = pd.DataFrame(horizon_rows)
            horizon_df = horizon_df.astype(object).where(pd.notna(horizon_df), None)
            st.dataframe(horizon_df, width="stretch", hide_index=True)
            st.caption(
                "的中率は中立・判定保留を含めない分母で計算しています。"
                "件数が少ないうちは精度を判断できません。蓄積してから評価してください。"
            )
        unevaluable_count = sum(
            1 for r in fwd_results if r.get("状態") == "評価不能"
        )
        if unevaluable_count:
            st.info(f"評価不能の記録が {unevaluable_count} 件あります（理由は評価CSVの「理由」欄に残しています）。")

    if fwd_records:
        with st.expander("V1・V2比較（新仕様の並行評価）"):
            if _version_count(fwd_records, 2) == 0:
                st.info(
                    "V2の登録がまだありません。スキャン結果をフォワード登録すると、"
                    "V1とV2の両方で記録されます。"
                )
            else:
                # 地合い（採用ゲート用）：日経平均の基準日時点の状態をスキャンごとに求める
                regime_by_scan = None
                nikkei_df, nikkei_err = get_history(forward.NIKKEI_SYMBOL, "2y")
                if nikkei_err is None and nikkei_df is not None and not nikkei_df.empty:
                    nikkei_ind = analysis.add_indicators(nikkei_df)
                    regime_by_scan = {}
                    for r in fwd_records:
                        scan_id = r["スキャンID"]
                        if scan_id not in regime_by_scan:
                            regime_by_scan[scan_id] = _regime_at(
                                nikkei_ind, r.get("株価基準日", "")
                            )
                comp = forward.compare_versions(
                    st.session_state.fwd_results, fwd_records,
                    regime_by_scan=regime_by_scan,
                )
                st.caption(
                    "これは方向判定を仮想的な建玉として評価する比較で、実際の売買損益ではありません。"
                    "V1の下落側は空売りを含む仮想評価であり、実際に利用可能な売買方法とは区別してください。"
                    "V2は並行評価中です。この画面は情報表示のみで、仕様を自動で切り替えることはありません。"
                )
                pending_total = sum(comp["評価待ち"].values())
                if pending_total:
                    st.write(
                        f"評価待ち：{pending_total} 件"
                        "（評価時点に到達したら「事後評価を実行」を押してください）"
                    )
                one_sided = comp["片側登録件数"]
                if one_sided["V1のみ"] or one_sided["V2のみ"]:
                    st.write(
                        f"片側登録：V1のみ {one_sided['V1のみ']} 件・V2のみ {one_sided['V2のみ']} 件"
                        "（両版そろっていないため、比較の対象外です）"
                    )
                    with st.expander("片側登録の明細（理由つき）"):
                        st.dataframe(
                            pd.DataFrame(comp["片側登録"]),
                            width="stretch", hide_index=True,
                        )
                for version, label in ((1, "V1（現行仕様）"), (2, "V2（新仕様）")):
                    v_summary = forward.summarize(
                        st.session_state.fwd_results, fwd_records, version=version
                    )
                    if v_summary["時点別"]:
                        st.markdown(f"**{label}の時点別の成績**")
                        v_df = pd.DataFrame(
                            [
                                {"評価営業日数": n, **vals}
                                for n, vals in v_summary["時点別"].items()
                            ]
                        )
                        v_df = v_df.astype(object).where(pd.notna(v_df), None)
                        st.dataframe(v_df, width="stretch", hide_index=True)
                if comp["共通母集団"]:
                    st.markdown("**予測性能比較（両版とも方向判定した共通母集団）**")
                    common_rows = [
                        {"評価営業日数": n, "時間軸": h, **vals}
                        for (n, h), vals in sorted(
                            comp["共通母集団"].items(), key=lambda kv: (kv[0][0], kv[0][1])
                        )
                        if vals["件数"]
                    ]
                    if common_rows:
                        common_df = pd.DataFrame(common_rows)
                        common_df = common_df.astype(object).where(
                            pd.notna(common_df), None
                        )
                        st.dataframe(common_df, width="stretch", hide_index=True)
                    st.caption(
                        "V2の方向判定は「上昇」のみのため、共通母集団ではV2の的中率は"
                        "常時上昇と一致します。実質的な差は「ペア差分平均TOPIX超過pct」で見てください。"
                    )
                caution = forward.caution_reference(
                    st.session_state.fwd_results, fwd_records
                )
                if caution:
                    st.markdown("**V2「下落注意」の参考グループ（的中率には含めません）**")
                    caution_rows = [
                        {"評価営業日数": n, "時間軸": h, **vals}
                        for (n, h), vals in sorted(
                            caution.items(), key=lambda kv: (kv[0][0], kv[0][1])
                        )
                    ]
                    caution_df = pd.DataFrame(caution_rows)
                    caution_df = caution_df.astype(object).where(
                        pd.notna(caution_df), None
                    )
                    st.dataframe(caution_df, width="stretch", hide_index=True)
                    st.caption("「下落注意」にした銘柄が、その後実際にどう動いたかです。")
                st.markdown(
                    f"**運用判断比較（{forward.PRIMARY_HORIZON}・"
                    f"{forward.PRIMARY_CHECKPOINT}営業日）**"
                )
                primary_key = (forward.PRIMARY_CHECKPOINT, forward.PRIMARY_HORIZON)
                matrix = comp["行列"].get(primary_key)
                if matrix:
                    matrix_rows = []
                    for d1 in forward.V1_DIRECTIONS:
                        row = {"V1の判定": d1}
                        for d2 in forward.V2_DIRECTIONS:
                            cell = matrix.get((d1, d2))
                            row[d2] = (
                                f"{cell['件数']}件（差分 {cell['差分合計pct']:+.1f}%）"
                                if cell else "－"
                            )
                        matrix_rows.append(row)
                    st.dataframe(
                        pd.DataFrame(matrix_rows), width="stretch", hide_index=True
                    )
                    comp_info = comp["完全性"][primary_key]
                    st.caption(
                        f"対象 {comp_info['対象件数']} 件／セル合計 {comp_info['セル合計件数']} 件／"
                        f"未分類 {comp_info['未分類件数']} 件／件数と差分の一致："
                        f"{'はい' if comp_info['一致'] else 'いいえ'}。"
                        "差分はV2の仮想損益からV1の仮想損益を引いたもの（プラスはV2優位）です。"
                        "TOPIX連動ETF超過ベースの差分は「全組み合わせ」の表にあります。"
                    )
                    with st.expander("運用判断比較の全組み合わせを見る"):
                        all_rows = []
                        for (n, h), cells in sorted(
                            comp["行列"].items(), key=lambda kv: (kv[0][0], kv[0][1])
                        ):
                            for (d1, d2), cell in cells.items():
                                all_rows.append(
                                    {
                                        "評価営業日数": n, "時間軸": h,
                                        "V1の判定": d1, "V2の判定": d2, **cell,
                                    }
                                )
                        all_df = pd.DataFrame(all_rows)
                        all_df = all_df.astype(object).where(pd.notna(all_df), None)
                        st.dataframe(all_df, width="stretch", hide_index=True)
                else:
                    st.info(
                        "主エンドポイント（中期・20営業日）の評価済みデータがまだありません。"
                        "事後評価が進むと、ここに16通りの組み合わせ表が出ます。"
                    )
                skip = comp["見送り"].get(primary_key)
                if skip:
                    st.markdown("**見送りの内訳（V1が「下落」だった銘柄）**")
                    skip_rows = [
                        {"V2の判定": d2, **vals} for d2, vals in skip.items()
                    ]
                    skip_df = pd.DataFrame(skip_rows)
                    skip_df = skip_df.astype(object).where(pd.notna(skip_df), None)
                    st.dataframe(skip_df, width="stretch", hide_index=True)
                    st.caption(
                        "「下落注意」は設計どおりの見送り、「中立」「判定保留」はスコア変化による見送りです。"
                        "回避できた下落（回避）と、見送ったことで逃した上昇（機会損失）の両方を表示します。"
                    )
                gate = comp["ゲート"]
                st.markdown("**採用ゲート（情報表示のみ）**")
                gate_rows = []
                for item in gate["項目"]:
                    mark = "－" if item["判定"] is None else ("○" if item["判定"] else "×")
                    gate_rows.append(
                        {
                            "項目": item["項目"], "基準": item["基準"],
                            "実績": item["実績"], "判定": mark,
                        }
                    )
                st.dataframe(
                    pd.DataFrame(gate_rows), width="stretch", hide_index=True
                )
                if gate["コホート平均差分pct"] is not None:
                    ci = gate["信頼区間95"]
                    ci_text = (
                        f"（95%信頼区間 {ci[0]:+.2f}% ～ {ci[1]:+.2f}%）" if ci else ""
                    )
                    st.write(
                        f"主エンドポイントのコホート平均差分：{gate['コホート平均差分pct']:+.2f}%"
                        f"{ci_text}（スキャン回ごとに平均した値で、件数の多さに引きずられない集計です）"
                    )
                dep = gate["銘柄依存"]
                if dep["依存警告"]:
                    st.warning(
                        f"注意：差分の多くが銘柄 {dep['最多銘柄']} に偏っています"
                        f"（寄与率 {dep['寄与率'] * 100:.0f}%）。"
                        f"その銘柄を除いた差分平均は {dep['最多銘柄を除いた差分平均pct']}% です。"
                    )
                fb = gate["再取得フォールバック"]
                if fb["コホート数"]:
                    st.write(
                        f"V2を再取得データ（基準日検査済み）で計算したスキャン：{fb['コホート数']} 回分"
                        "（スキャン時と同一のデータではないため、解釈に注意してください）"
                    )
                if gate["充足"]:
                    st.success(
                        "採用候補の条件を満たしています。ただし、採用するかどうかは別途の判断です"
                        "（自動では切り替えません）。"
                    )
                else:
                    st.info(
                        "採用ゲートはまだ条件を満たしていません。引き続きフォワードで蓄積してください。"
                    )

with tab_notes:
    st.subheader("使い方")
    st.markdown("""
1. 左側で市場を選び、銘柄コードを入力します。
2. 「銘柄一覧」で候補を比較します。
3. 「個別分析」でチャートと短期・中期・長期の条件を確認します。
4. 仮想売買を行い、判断根拠と結果をCSVに保存します。
5. 十分な期間・件数を検証するまで、実資金での売買判断に使用しないでください。

### 銘柄コード例
- 日本株：`7203.T`（トヨタ）、`6758.T`（ソニーグループ）、`8306.T`（三菱UFJ）
- 米国株：`AAPL`、`MSFT`、`NVDA`、`SPY`

### 短期・中期・長期の判定方法（ルールベース）
- 短期（最低30本）：EMA5とEMA20の位置関係、直近5営業日の値動き、RSI14の勢いと過熱・売られ過ぎの注意
- 中期（最低80本）：EMA20とEMA50の位置関係、終値とEMA20の位置関係、直近20・60営業日の値動き、押し目・戻りの参考情報
- 長期（最低250本かつEMA200が有効な場合のみ）：EMA50とEMA200の位置関係、終値とEMA200の位置関係、直近60・120営業日の値動き
- 本数が不足する区分はスコアを出さず、判定保留または「データ不足のため長期判定不可」と表示します
- スコアは条件の一致度であり、学習済みAIの予測確率や勝率ではありません。条件が一致しなければ中立または判定保留になります

### 現バージョンの制約
- 無料のYahoo Finance系データを使うため、遅延・欠落・取得制限・仕様変更があります。データの利用は個人利用を前提とし、再配布や商用利用の前に権利・規約を確認してください。
- データの新旧は暦日による暫定チェックで、公式の営業日カレンダーとは照合していません。古い可能性がある場合は警告を表示します。
- 短期スコアは日足のテクニカル条件に基づきます。リアルタイムのデイトレード用シグナルではありません。
- 自動発注、証券口座への接続、売買注文機能はありません。
- 仮想売買メモは永続保存されません。CSVを定期的にダウンロードして保管してください。
- フォワード登録・評価もCSVで保存・復元します。登録や評価の後は、未保存の警告が出ている間にCSVを保存してください。下の「全データのバックアップ」で、検証・メモ・フォワードのCSVを1つのZIPにまとめて保存できます。
- スキャンのフォワード登録では、現行仕様（V1）に加えて新仕様（V2）も並行して記録し、「スキャン」タブの「V1・V2比較」で比べられます。V2は値動きの大きさで条件をならし、弱い銘柄は「下落注意」として方向判定から外す仕様です。V2は評価中のため、画面の表示や順位づけはV1のままです。十分なフォワード検証が終わるまで、自動で切り替えることはありません。V2で計算できなかった銘柄は記録を作らず、「V2未登録の理由」として理由を残します。
""")
    st.subheader("全データのバックアップ")
    st.write(
        "検証の記録と対象外集計、仮想売買メモ、フォワード登録と評価結果、V2未登録の理由、最新のスキャン結果を、"
        "1つのZIPにまとめて保存できます（データがあるものだけが含まれます）。"
    )
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if st.session_state.get("verify_records"):
            zf.writestr(
                "stock_ai_verify_records.csv",
                verify.records_to_csv(st.session_state.verify_records),
            )
        if st.session_state.get("verify_exclusions"):
            zf.writestr(
                "stock_ai_verify_exclusions.csv",
                verify.exclusions_to_csv(st.session_state.verify_exclusions),
            )
        journal_df = st.session_state.get("journal")
        if journal_df is not None and not journal_df.empty:
            zf.writestr(
                "stock_ai_journal.csv",
                journal_df.to_csv(index=False).encode("utf-8-sig"),
            )
        if st.session_state.get("fwd_records"):
            zf.writestr(
                "stock_ai_forward_records.csv",
                forward.records_to_csv(st.session_state.fwd_records),
            )
        if st.session_state.get("fwd_results"):
            zf.writestr(
                "stock_ai_forward_results.csv",
                forward.results_to_csv(st.session_state.fwd_results),
            )
        if st.session_state.get("fwd_v2_skipped"):
            zf.writestr(
                "stock_ai_v2_skipped.csv",
                forward.skipped_to_csv(st.session_state.fwd_v2_skipped),
            )
        if st.session_state.get("scan_meta"):
            zf.writestr(
                "stock_ai_scan.csv",
                scan.scan_to_csv(
                    st.session_state.scan_ranked, st.session_state.scan_excluded
                ),
            )
    st.download_button(
        "全データをZIPで保存",
        data=zip_buf.getvalue(),
        file_name="stock_ai_lab_backup.zip",
        mime="application/zip",
    )
    st.markdown("データ取得ライブラリ： [yfinance documentation](https://ranaroussi.github.io/yfinance/)")

st.caption("Stock AI Lab | 分析補助専用・自動発注なし")
