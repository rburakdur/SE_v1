from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime

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
            try:
                self.runtime.notifier.notify(
                    "rbdcrypt: bot started",
                    (
                        f"env={self.runtime.settings.env} "
                        f"interval={self.runtime.settings.binance.interval} "
                        f"workers={self.runtime.settings.runtime.worker_count} "
                        f"max_symbols={self.runtime.settings.runtime.max_symbols} "
                        f"notify_topic={self.runtime.settings.notifications.topic} "
                        f"detail={self.runtime.settings.notifications.detail_level} "
                        f"started_at={datetime.now(tz=UTC).isoformat()}"
                    ),
                    priority=4,
                    tags="rocket",
                )
            except Exception as exc:
                self.runtime.logger.error(
                    "ntfy_error",
                    extra={"event": {"source": "runtime.worker.startup", "msg": str(exc)}},
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
                )
                self.runtime.logger.info(
                    "cycle_completed",
                    extra={
                        "event": {
                            "scanned": scan_result.scanned_symbols,
                            "scan_errors": scan_result.error_count,
                            "opened": trade_result.opened,
                            "closed": trade_result.closed,
                            "active_positions": self.runtime.repos.positions.count_active(),
                        }
                    },
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
                    try:
                        self.runtime.notifier.notify(
                            "rbdcrypt: cycle error",
                            f"{exc.__class__.__name__}: {exc} (attempt={loop_errors})",
                            priority=5,
                            tags="rotating_light",
                        )
                    except Exception as ntfy_exc:
                        self.runtime.logger.error(
                            "ntfy_error",
                            extra={"event": {"source": "runtime.worker", "msg": str(ntfy_exc)}},
                        )
                if loop_errors >= self.runtime.settings.runtime.max_loop_errors_before_sleep:
                    time.sleep(self.runtime.settings.runtime.loop_error_sleep_sec)
                    loop_errors = 0
