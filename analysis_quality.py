"""Shared fail-closed quality assessment for user-facing stock analysis."""

from __future__ import annotations

import analysis
import data_quality
import numpy as np


def assess(raw, symbol):
    """Return (report, blocking_reasons, diagnostic_error).

    Diagnostic failure is distinct from a clean report and must never be treated
    as permission to publish a score.
    """
    if raw is None:
        return None, [], "取得データがありません"
    try:
        market = analysis.market_of(symbol)
        report = data_quality.diagnose_history(
            raw,
            market=market,
            holidays_by_year=analysis.MARKET_HOLIDAYS.get(market),
        )
        reasons = list(report.blocking_reasons())
        if report.missing_columns:
            reasons.append("必須列欠落：" + "、".join(report.missing_columns))
        # Historical isolated OHLC NA rows are warnings and are excluded from
        # indicator input; latest-row missing price and structural defects block.
        return report, list(dict.fromkeys(reasons)), None
    except Exception as exc:
        return None, [], f"{type(exc).__name__}: {str(exc)[:120]}"

def scoring_frame(raw):
    """Build indicator input from valid OHLC rows while preserving Volume NA."""
    required = ["Open", "High", "Low", "Close", "Volume"]
    price_columns = ["Open", "High", "Low", "Close"]
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame(columns=required)
    if raw.columns.duplicated().any() or any(col not in raw.columns for col in required):
        return pd.DataFrame(columns=required)
    numeric = raw[price_columns].apply(pd.to_numeric, errors="coerce")
    valid = numeric.notna().all(axis=1) & np.isfinite(numeric).all(axis=1)
    valid &= (numeric > 0).all(axis=1)
    frame = raw.loc[valid, required].copy()
    for column in price_columns:
        frame[column] = numeric.loc[valid, column]
    frame["Volume"] = pd.to_numeric(frame["Volume"], errors="coerce")
    return frame
