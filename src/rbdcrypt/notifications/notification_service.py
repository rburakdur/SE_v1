from __future__ import annotations

from datetime import datetime, timedelta
from logging import Logger
from typing import Callable

from .ntfy_client import NtfyClient


class NotificationService:
    def __init__(
        self,
        *,
        notifier: NtfyClient | None,
        logger: Logger,
        now_fn: Callable[[], datetime],
        heartbeat_interval: timedelta = timedelta(minutes=30),
        summary_interval: timedelta = timedelta(minutes=60),
    ) -> None:
        self.notifier = notifier
        self.logger = logger
        self.now_fn = now_fn
        self.heartbeat_interval = heartbeat_interval
        self.summary_interval = summary_interval
        self._last_heartbeat_at: datetime | None = None
        self._last_summary_at: datetime | None = None

    def on_engine_start(
        self,
        *,
        pnl_pct: float | None,
        active_positions: int,
        scanned_count: int,
    ) -> None:
        now = self.now_fn()
        self._maybe_send_heartbeat(
            now=now,
            symbol="ENGINE",
            pnl_pct=pnl_pct,
            active_positions=active_positions,
            scanned_count=scanned_count,
            extra_line="event: startup",
        )

    def on_cycle_completed(
        self,
        *,
        pnl_pct: float | None,
        active_positions: int,
        scanned_count: int,
        opened: int,
        closed: int,
        scan_errors: int,
    ) -> None:
        now = self.now_fn()
        self._maybe_send_heartbeat(
            now=now,
            symbol="ALL",
            pnl_pct=pnl_pct,
            active_positions=active_positions,
            scanned_count=scanned_count,
            extra_line="event: runtime loop",
        )
        if self._last_summary_at and (now - self._last_summary_at) < self.summary_interval:
            return
        self._last_summary_at = now
        message = self._format_message(
            header="📊 SUMMARY",
            symbol="ALL",
            pnl_pct=pnl_pct,
            active_positions=active_positions,
            scanned_count=scanned_count,
            extra_line=f"opened: {int(opened)} | closed: {int(closed)} | scan errors: {int(scan_errors)}",
        )
        self._send(title="RBD-CRYPT summary", message=message, priority=2, tags="information_source")

    def on_position_open(
        self,
        *,
        symbol: str,
        pnl_pct: float | None,
        active_positions: int,
        scanned_count: int,
    ) -> None:
        message = self._format_message(
            header="🟢 ENTRY",
            symbol=symbol,
            pnl_pct=pnl_pct,
            active_positions=active_positions,
            scanned_count=scanned_count,
            extra_line="event: position opened",
        )
        self._send(title=f"RBD-CRYPT entry {symbol}", message=message, priority=4, tags="chart_with_upwards_trend")

    def on_position_close(
        self,
        *,
        symbol: str,
        pnl_pct: float | None,
        active_positions: int,
        scanned_count: int,
        reason: str | None = None,
    ) -> None:
        reason_text = reason.strip() if isinstance(reason, str) and reason.strip() else "-"
        message = self._format_message(
            header="🔴 EXIT",
            symbol=symbol,
            pnl_pct=pnl_pct,
            active_positions=active_positions,
            scanned_count=scanned_count,
            extra_line=f"reason: {reason_text}",
        )
        self._send(title=f"RBD-CRYPT exit {symbol}", message=message, priority=4, tags="x")

    def on_error(
        self,
        *,
        source: str,
        error: Exception | str,
        symbol: str,
        pnl_pct: float | None,
        active_positions: int,
        scanned_count: int,
    ) -> None:
        error_type = error.__class__.__name__ if isinstance(error, Exception) else "Error"
        message = self._format_message(
            header="⚠️ ERROR",
            symbol=symbol,
            pnl_pct=pnl_pct,
            active_positions=active_positions,
            scanned_count=scanned_count,
            extra_line=f"source: {source} | {error_type}: {error}",
        )
        self._send(title="RBD-CRYPT error", message=message, priority=5, tags="rotating_light")

    def _maybe_send_heartbeat(
        self,
        *,
        now: datetime,
        symbol: str,
        pnl_pct: float | None,
        active_positions: int,
        scanned_count: int,
        extra_line: str,
    ) -> None:
        if self._last_heartbeat_at and (now - self._last_heartbeat_at) < self.heartbeat_interval:
            return
        self._last_heartbeat_at = now
        message = self._format_message(
            header="❤️ HEARTBEAT",
            symbol=symbol,
            pnl_pct=pnl_pct,
            active_positions=active_positions,
            scanned_count=scanned_count,
            extra_line=extra_line,
        )
        self._send(title="RBD-CRYPT heartbeat", message=message, priority=3, tags="heart")

    def _format_message(
        self,
        *,
        header: str,
        symbol: str,
        pnl_pct: float | None,
        active_positions: int,
        scanned_count: int,
        extra_line: str | None = None,
    ) -> str:
        clean_symbol = symbol.strip() if symbol and symbol.strip() else "-"
        lines = [
            header,
            f"symbol: {clean_symbol}",
            f"pnl%: {self._format_pct(pnl_pct)}",
            f"active positions: {max(0, int(active_positions))}",
            f"scanned count: {max(0, int(scanned_count))}",
        ]
        if extra_line:
            lines.append(extra_line)
        return "\n".join(lines)

    def _send(self, *, title: str, message: str, priority: int, tags: str | None) -> None:
        if self.notifier is None:
            return
        try:
            self.notifier.notify(title, message, priority=priority, tags=tags)
        except Exception as exc:
            self.logger.error(
                "ntfy_error",
                extra={"event": {"source": "notification_service", "title": title, "msg": str(exc)}},
            )

    @staticmethod
    def _format_pct(value: float | None) -> str:
        if value is None:
            return "-"
        return f"{value:+.2f}%"
