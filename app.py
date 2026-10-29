import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

import analysis
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


tab_overview, tab_detail, tab_journal, tab_verify, tab_notes = st.tabs(
    ["銘柄一覧", "個別分析", "仮想売買メモ", "検証", "使い方・注意"]
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
""")
    st.markdown("データ取得ライブラリ： [yfinance documentation](https://ranaroussi.github.io/yfinance/)")

st.caption("Stock AI Lab | 分析補助専用・自動発注なし")
