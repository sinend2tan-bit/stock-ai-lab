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
        reasons = list(report.blocking_reasons())
        # An interior OHLC NA is not just a harmless dropna candidate: removing
        # that dated bar changes the indicator input sequence. Hold scores until
        # the source row is corrected or independently verified.
        price_columns = {"Open", "High", "Low", "Close"}
        missing_price_rows = sorted({
            item.get("row")
            for item in report.missing_cells
            if price_columns.intersection(item.get("columns", []))
        })
        if missing_price_rows:
            reasons.append(
                f"履歴中のOHLC欠損 {len(missing_price_rows)}行"
            )
        return report, list(dict.fromkeys(reasons)), None
    except Exception as exc:
        return None, [], f"{type(exc).__name__}: {str(exc)[:120]}"
