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

def test_scoring_frame_keeps_price_bars_when_volume_is_missing():
    clean = frame()
    raw = frame()
    raw.iloc[2, raw.columns.get_loc("Volume")] = float("nan")
    score_frame = analysis_quality.scoring_frame(raw)
    assert score_frame.index.equals(clean.index)
    assert score_frame["Close"].equals(clean["Close"])
    assert pd.isna(score_frame.iloc[2]["Volume"])

    clean_indicators = analysis.add_indicators(analysis_quality.scoring_frame(clean))
    raw_indicators = analysis.add_indicators(score_frame)
    for column in ["EMA5", "EMA20", "EMA50", "EMA200", "RSI14", "MACD", "Return_1d_pct"]:
        pd.testing.assert_series_equal(clean_indicators[column], raw_indicators[column], check_names=True)
    clean_scores = analysis.assess_all(clean_indicators)
    raw_scores = analysis.assess_all(raw_indicators)
    for horizon in ["短期", "中期", "長期"]:
        assert clean_scores[horizon]["score"] == raw_scores[horizon]["score"]
        assert clean_scores[horizon]["label"] == raw_scores[horizon]["label"]


def test_scoring_frame_omits_interior_ohlc_na_but_keeps_later_bars():
    raw = frame()
    raw.iloc[2, raw.columns.get_loc("Close")] = float("nan")
    score_frame = analysis_quality.scoring_frame(raw)
    assert len(score_frame) == len(raw) - 1
    assert raw.index[-1] in score_frame.index
