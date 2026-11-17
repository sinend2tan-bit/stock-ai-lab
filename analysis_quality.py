"""Shared fail-closed quality assessment for user-facing stock analysis."""

from __future__ import annotations

import analysis
import data_quality
import numpy as np
import pandas as pd


def assess(raw, symbol):
    """Return (report, blocking_reasons, diagnostic_error).

    Diagnostic failure is distinct from a clean report and must never be treated
    as permission to publish a score.
    """
    if raw is None:
        return None, [], "取得データがありません"
    if not isinstance(raw, pd.DataFrame):
        return None, [], "取得データがDataFrameではありません"
    if not isinstance(raw.index, pd.DatetimeIndex):
        return None, [], "日付indexがDatetimeIndexではありません"
    if raw.columns.duplicated().any():
        return None, [], "取得データに重複列があります"
    try:
        market = analysis.market_of(symbol)
        report = data_quality.diagnose_history(
            raw,
            market=market,
            holidays_by_year=analysis.MARKET_HOLIDAYS.get(market),
        )
        reasons = list(report.blocking_reasons())
        # Historical isolated OHLC NA rows are warnings and are excluded from
        # indicator input; latest-row missing price and structural defects block.
        return report, list(dict.fromkeys(reasons)), None
    except Exception as exc:
        return None, [], f"{type(exc).__name__}: {str(exc)[:120]}"

def scoring_frame(raw):
    """Build indicator input from valid OHLC rows while preserving Volume NA."""
    price_columns = ["Open", "High", "Low", "Close"]
    output_columns = price_columns + ["Volume"]
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame(columns=output_columns)
    if raw.columns.duplicated().any() or any(col not in raw.columns for col in price_columns):
        return pd.DataFrame(columns=output_columns)
    numeric = raw[price_columns].apply(pd.to_numeric, errors="coerce")
    valid = numeric.notna().all(axis=1) & np.isfinite(numeric).all(axis=1)
    valid &= (numeric > 0).all(axis=1)
    frame = raw.loc[valid].copy()
    for column in price_columns:
        frame[column] = numeric.loc[valid, column]
    # Volume is not used by the current price-only indicator/score set.
    # Preserve values if available; an absent column stays unknown, never zero.
    if "Volume" in frame.columns:
        frame["Volume"] = pd.to_numeric(frame["Volume"], errors="coerce")
    else:
        frame["Volume"] = np.nan
    return frame[output_columns]
