"""時系列評価の準備モジュール（準備段階・評価の実行はしない）。

- 時系列順の学習/検証/最終テスト分割 (ランダム分割はしない)。
- point-in-time 検査: 予測日より後に利用可能になった情報を特徴量に
  含めていないことを assert する。
- 前方リターンの足場: エントリーは翌営業日の始値、出口は保有期間後の
  終値という前提を明記した純粋関数。
- 分類指標 (バランス精度・クラス別 precision/recall) と、
  固定モデル vs 候補モデルの比較インターフェース。

いずれも合成データで検証するための準備であり、実データでの学習・評価は
別工程 (データ取得と品質確認の後、別途承認) とする。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable, Mapping, Sequence


def chronological_split(
    items: Sequence[Any],
    date_of: Callable[[Any], date],
    train_until: date,
    validation_until: date,
    embargo_days: int = 0,
) -> tuple[list[Any], list[Any], list[Any]]:
    """日付順に train / validation / test へ分割する。

    - train: date <= train_until
    - validation: train_until < date <= validation_until
      (embargo_days > 0 のとき境界直後の期間を validation から除く)
    - test: date > validation_until
      (embargo_days > 0 のとき境界直後の期間を test から除く)
    3集合は素であり、元の並び順を保つ。
    """
    if validation_until <= train_until:
        raise ValueError("validation_until must be after train_until")
    embargo = timedelta(days=embargo_days)
    train: list[Any] = []
    validation: list[Any] = []
    test: list[Any] = []
    for item in items:
        day = date_of(item)
        if day <= train_until:
            train.append(item)
        elif day <= validation_until:
            if day > train_until + embargo:
                validation.append(item)
        else:
            if day > validation_until + embargo:
                test.append(item)
    return train, validation, test


def assert_point_in_time(
    available_dates: Sequence[date],
    prediction_date: date,
) -> None:
    """特徴量の利用可能日がすべて予測日以前であることを assert する。

    財務データでは開示日 (DisclosedDate) を available date として渡すこと。
    後日修正された値を「当時知っていた」ことにしないための検査。
    """
    future = [d for d in available_dates if d > prediction_date]
    if future:
        raise ValueError(
            f"point-in-time violation: {len(future)} feature(s) available after "
            f"prediction date {prediction_date.isoformat()} "
            f"(earliest: {min(future).isoformat()})"
        )


def forward_returns(
    bars: Sequence[Mapping[str, Any]],
    horizon: int,
) -> list[dict[str, Any]]:
    """単一銘柄の日足 (日付昇順) から前方リターンを計算する足場。

    前提 (パイロット設計どおり明記):
    - エントリー: 予測日の翌営業日の始値 (Open が無ければ Close で代替)。
    - 出口: エントリー日を 1 日目として horizon 営業日目の終値。
    - 価格は呼び出し側が選んだ系列 (生 or 調整済み) をそのまま使う。
    期間が足りない行は結果に含めない。
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    results: list[dict[str, Any]] = []
    for index in range(len(bars)):
        entry_index = index + 1
        exit_index = entry_index + horizon - 1
        if exit_index >= len(bars):
            break
        entry_bar = bars[entry_index]
        entry_price = entry_bar.get("Open")
        if entry_price is None:
            entry_price = entry_bar.get("Close")
        exit_price = bars[exit_index].get("Close")
        if not entry_price or exit_price is None:
            continue
        results.append(
            {
                "date": bars[index].get("Date"),
                "entry_date": entry_bar.get("Date"),
                "exit_date": bars[exit_index].get("Date"),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "forward_return": exit_price / entry_price - 1.0,
            }
        )
    return results


def classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[Any] = (-1, 0, 1),
) -> dict[str, Any]:
    """バランス精度とクラス別 precision/recall を計算する (sklearn 不要)。"""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    per_label: dict[Any, dict[str, float]] = {}
    recalls: list[float] = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "support": tp + fn}
        recalls.append(recall)
    balanced_accuracy = sum(recalls) / len(recalls) if recalls else 0.0
    accuracy = (
        sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        if y_true
        else 0.0
    )
    return {
        "n": len(y_true),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "per_label": per_label,
    }


@dataclass
class EvaluationResult:
    """1モデルの評価結果を保持するインターフェース (準備段階)。"""

    model_name: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    n_evaluated: int = 0
    note: str = ""


def compare_evaluations(
    baseline: EvaluationResult,
    candidate: EvaluationResult,
) -> dict[str, float]:
    """固定モデル (baseline) と候補モデルの共通指標の差分を返す。

    採用判断そのものは行わない。未使用期間での比較と別途承認が前提。
    """
    deltas: dict[str, float] = {}
    for name, base_value in baseline.metrics.items():
        if name in candidate.metrics:
            deltas[name] = candidate.metrics[name] - base_value
    return deltas
