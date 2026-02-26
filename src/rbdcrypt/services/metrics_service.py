from __future__ import annotations

from dataclasses import dataclass

from ..storage.repositories import Repositories


@dataclass(slots=True)
class MetricsService:
    repos: Repositories

    def analyze_summary(self) -> dict[str, object]:
        trade_summary = self.repos.trades.summary()
        active_positions = self.repos.positions.count_active()
        portfolio = self.repos.runtime_state.get_json("portfolio") or {}
        last_scan = self.repos.runtime_state.get_json("last_scan") or {}
        total_trades = int(trade_summary["total_trades"])
        wins = int(trade_summary["wins"])
        win_rate = (wins / total_trades) * 100.0 if total_trades else 0.0
        return {
            "trades": trade_summary,
            "win_rate_pct": round(win_rate, 2),
            "active_positions": active_positions,
            "portfolio": portfolio,
            "last_scan": last_scan,
        }
