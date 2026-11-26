"""jquants_lab.evaluation の単体テスト。合成データのみ。評価の準備機能を検証する。"""

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jquants_lab.evaluation import (
    EvaluationResult,
    assert_point_in_time,
    chronological_split,
    classification_metrics,
    compare_evaluations,
    forward_returns,
)
from jquants_lab.synthetic import business_days, make_daily_bars


def _dated_items():
    days = business_days(date(2024, 1, 9), 30)
    return [{"date": d, "value": i} for i, d in enumerate(days)]


def test_chronological_split_is_disjoint_and_ordered():
    items = _dated_items()
    train, validation, test = chronological_split(
        items,
        date_of=lambda item: item["date"],
        train_until=date(2024, 1, 19),
        validation_until=date(2024, 1, 29),
    )
    assert train and validation and test
    assert max(i["date"] for i in train) <= date(2024, 1, 19)
    assert all(date(2024, 1, 19) < i["date"] <= date(2024, 1, 29) for i in validation)
    assert min(i["date"] for i in test) > date(2024, 1, 29)
    total = [i["value"] for i in train + validation + test]
    assert sorted(total) == list(range(30))


def test_chronological_split_embargo_drops_boundary_items():
    items = _dated_items()
    train, validation, test = chronological_split(
        items,
        date_of=lambda item: item["date"],
        train_until=date(2024, 1, 19),
        validation_until=date(2024, 1, 29),
        embargo_days=3,
    )
    assert all(i["date"] > date(2024, 1, 22) for i in validation)
    assert all(i["date"] > date(2024, 2, 1) for i in test)
    assert train  # train 側は embargo の影響を受けない


def test_chronological_split_rejects_reversed_bounds():
    with pytest.raises(ValueError):
        chronological_split(
            _dated_items(),
            date_of=lambda item: item["date"],
            train_until=date(2024, 1, 29),
            validation_until=date(2024, 1, 19),
        )


def test_point_in_time_assertion():
    assert_point_in_time([date(2024, 2, 9)], date(2024, 2, 13))  # 開示後はOK
    with pytest.raises(ValueError):
        # 開示日より前の予測日に財務を使うのはリーク
        assert_point_in_time([date(2024, 2, 9)], date(2024, 2, 8))


def test_forward_returns_next_open_entry():
    bars = [
        {"Date": "2024-01-09", "Open": 100.0, "Close": 100.0},
        {"Date": "2024-01-10", "Open": 110.0, "Close": 120.0},
        {"Date": "2024-01-11", "Open": 120.0, "Close": 132.0},
    ]
    results = forward_returns(bars, horizon=1)
    assert len(results) == 2
    first = results[0]
    assert first["entry_date"] == "2024-01-10"
    assert first["entry_price"] == 110.0
    assert first["exit_price"] == 120.0
    assert first["forward_return"] == pytest.approx(120.0 / 110.0 - 1.0)


def test_forward_returns_on_synthetic_bars():
    bars = [
        record
        for record in make_daily_bars(codes=("7203",), days=10)
    ]
    results = forward_returns(bars, horizon=5)
    assert len(results) == 10 - 5
    assert all("forward_return" in row for row in results)


def test_classification_metrics_known_values():
    y_true = [1, 1, 0, -1, -1, 0]
    y_pred = [1, 0, 0, -1, 1, 0]
    metrics = classification_metrics(y_true, y_pred)
    assert metrics["n"] == 6
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    # recalls: label 1 -> 1/2, label 0 -> 2/2, label -1 -> 1/2
    assert metrics["balanced_accuracy"] == pytest.approx((0.5 + 1.0 + 0.5) / 3)


def test_compare_evaluations_returns_deltas_only_for_shared_metrics():
    baseline = EvaluationResult("fixed", {"balanced_accuracy": 0.40, "n": 100.0})
    candidate = EvaluationResult("candidate", {"balanced_accuracy": 0.45})
    deltas = compare_evaluations(baseline, candidate)
    assert deltas == {"balanced_accuracy": pytest.approx(0.05)}
