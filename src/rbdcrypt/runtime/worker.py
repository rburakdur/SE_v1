from __future__ import annotations

import time
import traceback
from dataclasses import dataclass

from ..core.scheduler import IntervalScheduler
from ..models.error_event import ErrorEvent
from .app import RuntimeContainer


@dataclass(slots=True)
class RuntimeWorker:
    runtime: RuntimeContainer

    def run(self, *, one_shot: bool = False, max_iterations: int | None = None) -> None:
        scheduler = IntervalScheduler(interval_sec=self.runtime.settings.runtime.scan_interval_sec)
        self.runtime.trade_service.recover_active_positions()
        if self.runtime.settings.notifications.notify_on_startup:
            self.runtime.notification_service.on_engine_start(
                pnl_pct=self._portfolio_pnl_pct(),
                active_positions=self.runtime.repos.positions.count_active(),
                scanned_count=0,
            )
        iterations = 0
        while True:
            self._run_cycle()
            iterations += 1
            if one_shot or self.runtime.settings.runtime.one_shot:
                return
            if max_iterations is not None and iterations >= max_iterations:
                return
            scheduler.sleep_until_next()

    def _run_cycle(self) -> None:
        loop_errors = 0
        while True:
            try:
                scan_result = self.runtime.scan_service.scan_once()
                trade_result = self.runtime.trade_service.handle_cycle(
                    signals=scan_result.signals,
                    prices_by_symbol=scan_result.prices_by_symbol,
                    symbol_states=scan_result.symbol_states,
                    scanned_count=scan_result.scanned_symbols,
                )
                active_positions = self.runtime.repos.positions.count_active()
                self.runtime.logger.info(
                    "cycle_completed",
                    extra={
                        "event": {
                            "scanned": scan_result.scanned_symbols,
                            "scan_errors": scan_result.error_count,
                            "opened": trade_result.opened,
                            "closed": trade_result.closed,
                            "active_positions": active_positions,
                        }
                    },
                )
                self.runtime.notification_service.on_cycle_completed(
                    pnl_pct=self._portfolio_pnl_pct(),
                    active_positions=active_positions,
                    scanned_count=scan_result.scanned_symbols,
                    opened=trade_result.opened,
                    closed=trade_result.closed,
                    scan_errors=scan_result.error_count,
                )
                return
            except Exception as exc:
                loop_errors += 1
                self.runtime.repos.errors.insert(
                    ErrorEvent(
                        source="runtime.worker",
                        error_type=exc.__class__.__name__,
                        message=str(exc),
                        traceback_single_line=traceback.format_exc().replace("\n", "\\n"),
                    )
                )
                self.runtime.logger.error(
                    "cycle_error",
                    extra={"event": {"type": exc.__class__.__name__, "msg": str(exc), "attempt": loop_errors}},
                )
                if self.runtime.settings.notifications.notify_on_runtime_error:
                    self.runtime.notification_service.on_error(
                        source="runtime.worker",
                        error=exc,
                        symbol="-",
                        pnl_pct=self._portfolio_pnl_pct(),
                        active_positions=self.runtime.repos.positions.count_active(),
                        scanned_count=0,
                    )
                if loop_errors >= self.runtime.settings.runtime.max_loop_errors_before_sleep:
                    time.sleep(self.runtime.settings.runtime.loop_error_sleep_sec)
                    loop_errors = 0

    def _portfolio_pnl_pct(self) -> float:
        portfolio = self.runtime.repos.runtime_state.get_json("portfolio") or {}
        starting = float(self.runtime.settings.balance.starting_balance)
        if starting <= 0:
            return 0.0
        balance = float(portfolio.get("balance", starting))
        return ((balance - starting) / starting) * 100.0
