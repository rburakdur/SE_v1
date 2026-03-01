from __future__ import annotations

import csv
import json
import shutil
import sqlite3
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from ..core.scheduler import IntervalScheduler
from ..models.error_event import ErrorEvent
from ..notifications.notification_service import OpenPositionSnapshot, PerformanceSnapshot
from .app import RuntimeContainer


@dataclass(slots=True)
class RuntimeWorker:
    runtime: RuntimeContainer
    _MAX_NTFY_UPLOAD_BYTES = 10 * 1024 * 1024
    _UPLOAD_CHUNK_BYTES = 9 * 1024 * 1024

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

    def _export_logs_bundle(self, command: str) -> list[Path]:
        now = self.runtime.clock.now().astimezone(UTC)
        ts = now.strftime("%Y%m%d_%H%M%S")
        command_norm = command.strip().lower() if command else "log"
        export_root = self.runtime.settings.data_dir / "exports"
        export_dir = export_root / f"ntfy_{command_norm}_{ts}"
        export_dir.mkdir(parents=True, exist_ok=True)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        logs_dir = export_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        base_name = self.runtime.settings.logging.log_file
        log_files = sorted(
            self.runtime.settings.logging.log_dir.glob(f"{base_name}*"),
            key=lambda p: p.name,
        )
        if command_norm == "log-all":
            for src in log_files:
                if src.is_file():
                    dst = logs_dir / src.name
                    shutil.copy2(src, dst)
        else:
            out_file = logs_dir / "log_recent_2days.jsonl"
            with out_file.open("w", encoding="utf-8") as out:
                for src in log_files:
                    if not src.is_file():
                        continue
                    with src.open("r", encoding="utf-8", errors="replace") as f:
                        for line in f:
                            ts_line = self._parse_log_time(line)
                            if ts_line is None:
                                continue
                            if start <= ts_line <= now:
                                out.write(line)
        self._export_db_csvs(
            export_dir=export_dir,
            command_norm=command_norm,
            start=start,
            end=now,
        )
        meta = {
            "generated_at_utc": now.isoformat(),
            "command": command_norm,
            "topic": self.runtime.settings.notifications.topic,
        }
        meta_path = export_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        archive_path = export_root / f"ntfy_{command_norm}_{ts}.zip"
        self._create_zip_archive(source_dir=export_dir, archive_path=archive_path)
        return self._split_for_ntfy_upload(archive_path)

    @staticmethod
    def _create_zip_archive(*, source_dir: Path, archive_path: Path) -> None:
        with ZipFile(archive_path, mode="w", compression=ZIP_DEFLATED, compresslevel=6) as zf:
            for item in sorted(source_dir.rglob("*")):
                if not item.is_file():
                    continue
                arcname = Path(source_dir.name) / item.relative_to(source_dir)
                zf.write(item, arcname=str(arcname).replace("\\", "/"))

    def _split_for_ntfy_upload(self, archive_path: Path) -> list[Path]:
        if archive_path.stat().st_size <= self._MAX_NTFY_UPLOAD_BYTES:
            return [archive_path]
        part_paths: list[Path] = []
        with archive_path.open("rb") as src:
            part_idx = 1
            while True:
                chunk = src.read(self._UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                part_path = archive_path.parent / f"{archive_path.name}.part{part_idx:03d}"
                with part_path.open("wb") as dst:
                    dst.write(chunk)
                part_paths.append(part_path)
                part_idx += 1
        return part_paths

    def _export_db_csvs(
        self,
        *,
        export_dir: Path,
        command_norm: str,
        start: datetime,
        end: datetime,
    ) -> list[Path]:
        db_dir = export_dir / "db"
        db_dir.mkdir(parents=True, exist_ok=True)
        generated: list[Path] = []
        tables: list[tuple[str, str | None]] = [
            ("signals", "created_at"),
            ("signal_decisions", "created_at"),
            ("market_context", "fetched_at"),
            ("positions_active", None),
            ("trades_closed", "closed_at"),
            ("errors", "created_at"),
            ("heartbeats", "created_at"),
            ("runtime_state", None),
            ("ohlcv_futures", "open_time"),
        ]
        conn = sqlite3.connect(str(self.runtime.settings.storage.db_path))
        conn.row_factory = sqlite3.Row
        try:
            for table, time_col in tables:
                sql = f"SELECT * FROM {table}"
                params: tuple[str, ...] = ()
                if command_norm != "log-all" and time_col is not None:
                    sql += f" WHERE {time_col} >= ? AND {time_col} <= ? ORDER BY {time_col} ASC"
                    params = (start.isoformat(), end.isoformat())
                elif time_col is not None:
                    sql += f" ORDER BY {time_col} ASC"
                if command_norm == "log-all" and table == "ohlcv_futures":
                    generated.extend(self._export_query_csv_chunks(conn=conn, sql=sql, params=params, out_dir=db_dir, stem=table))
                    continue
                out_path = db_dir / f"{table}.csv"
                self._export_query_csv(conn=conn, sql=sql, params=params, out_path=out_path)
                generated.append(out_path)
        finally:
            conn.close()
        return generated

    @staticmethod
    def _export_query_csv(
        *,
        conn: sqlite3.Connection,
        sql: str,
        params: tuple[str, ...],
        out_path: Path,
    ) -> None:
        cur = conn.execute(sql, params)
        rows = cur.fetchall()
        fieldnames = [d[0] for d in cur.description or []]
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))

    @staticmethod
    def _export_query_csv_chunks(
        *,
        conn: sqlite3.Connection,
        sql: str,
        params: tuple[str, ...],
        out_dir: Path,
        stem: str,
        chunk_rows: int = 25000,
    ) -> list[Path]:
        cur = conn.execute(sql, params)
        fieldnames = [d[0] for d in cur.description or []]
        parts: list[Path] = []
        part_idx = 1
        while True:
            rows = cur.fetchmany(chunk_rows)
            if not rows:
                break
            out_path = out_dir / f"{stem}_part{part_idx:03d}.csv"
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(dict(row))
            parts.append(out_path)
            part_idx += 1
        if not parts:
            empty_path = out_dir / f"{stem}.csv"
            with empty_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            parts.append(empty_path)
        return parts

    @staticmethod
    def _parse_log_time(line: str) -> datetime | None:
        text = line.strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        raw = payload.get("ts")
        if not isinstance(raw, str) or not raw:
            return None
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        else:
            dt = dt.astimezone(UTC)
        return dt

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
