from __future__ import annotations

import shutil
import tarfile
import time
import traceback
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path

from ..core.scheduler import IntervalScheduler
from ..models.error_event import ErrorEvent
from ..notifications.notification_service import OpenPositionSnapshot, PerformanceSnapshot
from .app import RuntimeContainer


@dataclass(slots=True)
class RuntimeWorker:
    runtime: RuntimeContainer

    def run(self, *, one_shot: bool = False, max_iterations: int | None = None) -> None:
        scheduler = IntervalScheduler(interval_sec=self.runtime.settings.runtime.scan_interval_sec)
        self.runtime.trade_service.recover_active_positions()
        if self.runtime.settings.notifications.notify_on_startup:
            perf = self._performance_snapshot(pending_signals=0)
            self.runtime.notification_service.on_engine_start(
                performance=perf,
                open_positions=self._open_position_snapshots(),
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
                    chart_points_by_symbol=scan_result.chart_points_by_symbol,
                    symbol_states=scan_result.symbol_states,
                    scanned_count=scan_result.scanned_symbols,
                )
                perf = self._performance_snapshot(pending_signals=trade_result.max_pos_blocked)
                self.runtime.logger.info(
                    "cycle_completed",
                    extra={
                        "event": {
                            "scanned": scan_result.scanned_symbols,
                            "scan_errors": scan_result.error_count,
                            "opened": trade_result.opened,
                            "closed": trade_result.closed,
                            "active_positions": perf.active_positions,
                        }
                    },
                )
                self.runtime.notification_service.on_cycle_completed(
                    performance=perf,
                    open_positions=self._open_position_snapshots(),
                    opened=trade_result.opened,
                    closed=trade_result.closed,
                    scan_errors=scan_result.error_count,
                )
                self.runtime.notification_service.process_ntfy_commands(
                    export_logs_bundle=self._export_logs_bundle,
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
                    perf = self._performance_snapshot(pending_signals=0)
                    self.runtime.notification_service.on_error(
                        source="runtime.worker",
                        error=exc,
                        symbol="-",
                        pnl_pct=perf.pnl_pct_cumulative,
                        active_positions=perf.active_positions,
                        scanned_count=0,
                    )
                if loop_errors >= self.runtime.settings.runtime.max_loop_errors_before_sleep:
                    time.sleep(self.runtime.settings.runtime.loop_error_sleep_sec)
                    loop_errors = 0

    def _performance_snapshot(self, *, pending_signals: int) -> PerformanceSnapshot:
        now = self.runtime.clock.now()
        portfolio = self.runtime.repos.runtime_state.get_json("portfolio") or {}
        starting = float(self.runtime.settings.balance.starting_balance)
        balance = float(portfolio.get("balance", starting))
        realized_pnl_quote = float(portfolio.get("realized_pnl", balance - starting))
        pnl_pct = ((realized_pnl_quote / starting) * 100.0) if starting > 0 else 0.0
        summary = self.runtime.repos.trades.summary()
        total_trades = int(summary["total_trades"])
        wins = int(summary["wins"])
        losses = max(0, total_trades - wins)
        win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
        day_start = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        with self.runtime.db.read_only() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(pnl_quote), 0.0) AS day_pnl_quote FROM trades_closed WHERE closed_at >= ?",
                (day_start.isoformat(),),
            ).fetchone()
        day_pnl_quote = float(row["day_pnl_quote"] if row else 0.0)
        day_pnl_pct = ((day_pnl_quote / starting) * 100.0) if starting > 0 else 0.0
        active = self.runtime.repos.positions.count_active()
        return PerformanceSnapshot(
            balance=balance,
            realized_pnl_quote=realized_pnl_quote,
            pnl_pct_cumulative=pnl_pct,
            day_pnl_quote=day_pnl_quote,
            day_pnl_pct=day_pnl_pct,
            total_trades=total_trades,
            wins=wins,
            losses=losses,
            win_rate_pct=win_rate,
            active_positions=active,
            max_positions=int(self.runtime.settings.risk.max_active_positions),
            pending_signals=max(0, int(pending_signals)),
        )

    def _open_position_snapshots(self) -> list[OpenPositionSnapshot]:
        now = self.runtime.clock.now()
        snapshots: list[OpenPositionSnapshot] = []
        for pos in self.runtime.repos.positions.list_active():
            hold_min = max(0.0, (now - pos.opened_at).total_seconds() / 60.0)
            tp_target_pct, sl_risk_pct = self._tp_sl_targets_pct(pos.entry_price, pos.current_tp, pos.current_sl, pos.side.value)
            snapshots.append(
                OpenPositionSnapshot(
                    symbol=pos.symbol,
                    side=pos.side.value,
                    entry_price=pos.entry_price,
                    tp_price=pos.current_tp,
                    sl_price=pos.current_sl,
                    tp_target_pct=tp_target_pct,
                    sl_risk_pct=sl_risk_pct,
                    current_pnl_pct=float(pos.current_pnl_pct) * 100.0,
                    hold_minutes=hold_min,
                )
            )
        snapshots.sort(key=lambda p: p.hold_minutes, reverse=True)
        return snapshots

    def _export_logs_bundle(self) -> Path:
        now = self.runtime.clock.now().astimezone(UTC)
        ts = now.strftime("%Y%m%d_%H%M%S")
        export_root = self.runtime.settings.data_dir / "exports"
        export_dir = export_root / f"ntfy_analysis_{ts}"
        export_dir.mkdir(parents=True, exist_ok=True)

        tables = [
            "signals",
            "signal_decisions",
            "market_context",
            "positions_active",
            "trades_closed",
            "errors",
            "heartbeats",
            "runtime_state",
        ]
        for table in tables:
            self.runtime.repos.maintenance.export_csv(
                table=table,
                out_path=export_dir / f"{table}.csv",
            )

        log_path = self.runtime.settings.logging.log_path
        if log_path.is_file():
            shutil.copy2(log_path, export_dir / log_path.name)
        db_path = self.runtime.settings.storage.db_path
        if db_path.is_file():
            shutil.copy2(db_path, export_dir / db_path.name)

        archive_path = export_root / f"ntfy_analysis_{ts}.tar.gz"
        with tarfile.open(archive_path, mode="w:gz") as tf:
            tf.add(export_dir, arcname=export_dir.name)
        return archive_path

    @staticmethod
    def _tp_sl_targets_pct(entry: float, tp: float, sl: float, side: str) -> tuple[float, float]:
        if entry <= 0:
            return 0.0, 0.0
        side_norm = side.lower()
        if side_norm == "short":
            tp_target = ((entry - tp) / entry) * 100.0
            sl_risk = ((sl - entry) / entry) * 100.0
            return tp_target, sl_risk
        tp_target = ((tp - entry) / entry) * 100.0
        sl_risk = ((entry - sl) / entry) * 100.0
        return tp_target, sl_risk
