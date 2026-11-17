"""Shared quality assessment contract for individual analysis and watchlist."""

import pandas as pd

import analysis_quality


def frame():
    idx = pd.bdate_range("2026-10-01", periods=6)
    close = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    return pd.DataFrame(
        {
            "Open": close,
            "High": [v + 1 for v in close],
            "Low": [v - 1 for v in close],
            "Close": close,
            "Volume": [1000] * len(close),
        },
        index=idx,
    )


def test_clean_price_data_is_eligible():
    report, reasons, error = analysis_quality.assess(frame(), "7203.T")
    assert error is None
    assert report is not None
    assert reasons == []


def test_volume_only_missing_is_warning_not_score_blocker():
    data = frame()
    data.iloc[2, data.columns.get_loc("Volume")] = float("nan")
    report, reasons, error = analysis_quality.assess(data, "7203.T")
    assert error is None
    assert report is not None
    assert reasons == []
    assert report.volume_issues["non_finite"]


def test_ohlc_contradiction_blocks_score():
    data = frame()
    data.iloc[2, data.columns.get_loc("High")] = 90.0
    report, reasons, error = analysis_quality.assess(data, "7203.T")
    assert error is None
    assert report is not None
    assert any("OHLC価格異常" in reason for reason in reasons)


def test_diagnostic_failure_is_not_treated_as_clean(monkeypatch):
    def fail(_symbol):
        raise RuntimeError("market mapping unavailable")

    monkeypatch.setattr("analysis_quality.analysis.market_of", fail)
    report, reasons, error = analysis_quality.assess(frame(), "7203.T")
    assert report is None
    assert reasons == []
    assert error.startswith("RuntimeError:")
