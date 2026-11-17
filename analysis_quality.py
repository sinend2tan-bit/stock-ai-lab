"""Shared fail-closed quality assessment for user-facing stock analysis."""

from __future__ import annotations

import analysis
import data_quality


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
        # Historical isolated OHLC NA rows are excluded from the clean indicator
        # series and remain visible as warnings. Only the latest-row missing price,
        # confirmed session gaps, structural date defects, and invalid prices block.
        return report, report.blocking_reasons(), None
    except Exception as exc:
        return None, [], f"{type(exc).__name__}: {str(exc)[:120]}"
