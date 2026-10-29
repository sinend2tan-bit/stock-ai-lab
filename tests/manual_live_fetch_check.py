"""実データ取得の手動確認スクリプト（単体テストには含めない）。

実際に Yahoo Finance へアクセスするため、通信障害や取得制限で結果が
変わることがあります。通常の単体テスト（test_analysis.py）は通信を
モック化しており、このスクリプトなしで実行できます。

実行方法:
    python tests/manual_live_fetch_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis


def main():
    targets = [("7203.T", "2y"), ("6758.T", "2y"), ("AAPL", "2y"),
               ("MSFT", "2y"), ("INVALIDXYZ123", "1y")]
    for symbol, period in targets:
        df, err = analysis.fetch_history(symbol, period=period)
        if err is not None:
            print(f"{symbol} ({period}): 取得失敗 - {err}")
            continue
        df = analysis.add_indicators(df)
        fresh = analysis.freshness_check(df, symbol)
        results = analysis.assess_all(df)
        labels = {k: (v["label"] if v["available"] else "判定不可") for k, v in results.items()}
        print(
            f"{symbol} ({period}): rows={len(df)}, last={fresh['last_date']}, "
            f"freshness={fresh['status']}, 短期={labels['短期']}, 中期={labels['中期']}, 長期={labels['長期']}"
        )


if __name__ == "__main__":
    main()
