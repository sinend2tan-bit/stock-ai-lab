"""検証A（短期フォワード仮想売買評価）の CLI ランナー。

使い方:
    python run_verification_a.py <フォワード登録CSVのパス> <出力ディレクトリ> [--run-id ID] [--note 注記]

登録CSVは forward.records_from_csv で読み、価格系列は analysis.fetch_history
（Yahoo Finance / yfinance、無料データのみ）で評価時に再取得する。
出力: 結果CSV、集計（JSON/MD）、snapshot（銘柄別CSV＋manifest.json）、
ZIPバックアップ、実行サマリ（run_summary.json）。
main への反映やアプリ変更は行わない、評価専用の独立ツール。

注意: 本評価は調整済み価格に基づく理論評価であり、実際の約定損益とは異なる。
"""

import argparse
import json
import sys

import analysis
import forward
import verification_a


def main(argv=None):
    parser = argparse.ArgumentParser(description="検証A 短期フォワード仮想売買評価")
    parser.add_argument("records_csv", help="フォワード登録CSVのパス")
    parser.add_argument("out_dir", help="出力ディレクトリ")
    parser.add_argument("--run-id", default=None, help="評価実行ID（省略時は日時から生成）")
    parser.add_argument("--note", default=None, help="出力に残す注記")
    args = parser.parse_args(argv)

    try:
        with open(args.records_csv, "rb") as f:
            records = forward.records_from_csv(f.read())
    except OSError as exc:
        # 存在しないパス等の指定ミスでトレースバックを出さず、原因を明示する。
        print(
            f"指定したファイルを開けませんでした: {args.records_csv} ({exc})",
            file=sys.stderr,
        )
        return 1
    if not records:
        print("登録CSVを読み込めませんでした（列構成が想定と異なる可能性があります）",
              file=sys.stderr)
        return 1

    run_summary = verification_a.run_evaluation(
        records, analysis.fetch_history, args.out_dir,
        run_id=args.run_id, note=args.note,
    )
    printable = {k: v for k, v in run_summary.items()
                 if k not in ("summary", "manifest")}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
