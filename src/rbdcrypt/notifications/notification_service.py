from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from logging import Logger
from typing import Callable, Protocol

from .ntfy_client import NtfyClient


STATE_KEY = "notifications_state"

ENTRY_EMOJI = "\U0001F7E2"
EXIT_EMOJI = "\U0001F534"
SUMMARY_EMOJI = "\U0001F4CA"
HEARTBEAT_EMOJI = "\u2764\ufe0f"
ERROR_EMOJI = "\u26A0\ufe0f"
UP_EMOJI = "\u2b06\ufe0f"
DOWN_EMOJI = "\u2b07\ufe0f"
PROFIT_EMOJI = "\u2705"
LOSS_EMOJI = "\u274c"


class NotificationStateStore(Protocol):
    def get_json(self, key: str) -> dict[str, object] | None: ...

    def set_json(self, key: str, value: dict[str, object]) -> None: ...


@dataclass(slots=True)
class PerformanceSnapshot:
    balance: float
    pnl_pct_cumulative: float
    day_pnl_quote: float
    day_pnl_pct: float
    total_trades: int
    wins: int
    losses: int
    win_rate_pct: float
    active_positions: int
    max_positions: int
    pending_signals: int = 0


@dataclass(slots=True)
class OpenPositionSnapshot:
    symbol: str
    side: str
    entry_price: float
    tp_price: float
    sl_price: float
    tp_target_pct: float
    sl_risk_pct: float
    current_pnl_pct: float
    hold_minutes: float


class NotificationService:
    def __init__(
        self,
        *,
        notifier: NtfyClient | None,
        logger: Logger,
        now_fn: Callable[[], datetime],
        state_store: NotificationStateStore | None = None,
        heartbeat_interval: timedelta = timedelta(minutes=30),
        summary_interval: timedelta = timedelta(minutes=60),
    ) -> None:
        self.notifier = notifier
        self.logger = logger
        self.now_fn = now_fn
        self.state_store = state_store
        self.heartbeat_interval = heartbeat_interval
        self.summary_interval = summary_interval
        self._last_heartbeat_at: datetime | None = None
        self._last_summary_at: datetime | None = None

    def on_engine_start(
        self,
        *,
        performance: PerformanceSnapshot | None = None,
        open_positions: list[OpenPositionSnapshot] | None = None,
        pnl_pct: float | None = None,
        active_positions: int = 0,
        scanned_count: int = 0,
    ) -> None:
        now = self.now_fn()
        perf = performance or self._legacy_perf(
            pnl_pct=pnl_pct,
            active_positions=active_positions,
            pending_signals=0,
        )
        self._maybe_send_heartbeat(now=now, perf=perf, open_positions=open_positions, event="startup")

    def on_cycle_completed(
        self,
        *,
        performance: PerformanceSnapshot | None = None,
        open_positions: list[OpenPositionSnapshot] | None = None,
        opened: int = 0,
        closed: int = 0,
        scan_errors: int = 0,
        pnl_pct: float | None = None,
        active_positions: int = 0,
        scanned_count: int = 0,
    ) -> None:
        now = self.now_fn()
        perf = performance or self._legacy_perf(
            pnl_pct=pnl_pct,
            active_positions=active_positions,
            pending_signals=0,
        )
        self._maybe_send_heartbeat(now=now, perf=perf, open_positions=open_positions, event="runtime loop")
        if not self._should_send_summary(now):
            return
        lines = [
            f"{SUMMARY_EMOJI} SUMMARY",
            *self._perf_lines(perf),
            (
                f"bu saat: acilan {int(opened)} | kapanan {int(closed)} | "
                f"hata {int(scan_errors)} | bekleyen {max(0, int(perf.pending_signals))}"
            ),
        ]
        lines.extend(self._position_lines(open_positions))
        self._send(
            title="RBD-CRYPT summary",
            message="\n".join(lines),
            priority=2,
            tags="information_source",
        )

    def on_position_open(
        self,
        *,
        symbol: str,
        side: str = "-",
        entry_price: float | None = None,
        tp_price: float | None = None,
        sl_price: float | None = None,
        tp_target_pct: float | None = None,
        sl_risk_pct: float | None = None,
        hold_minutes: float = 0.0,
        current_pnl_pct: float = 0.0,
        active_positions: int = 0,
        max_positions: int = 0,
        pending_signals: int = 0,
        pnl_pct: float | None = None,
        scanned_count: int = 0,
    ) -> None:
        direction = self._direction_text(side)
        status = self._status_text(current_pnl_pct)
        lines = [
            f"{ENTRY_EMOJI} ENTRY",
            f"sembol: {self._clean_symbol(symbol)}",
            f"yon: {direction}",
            f"durum: {status}",
            f"sure: {max(0.0, float(hold_minutes)):.1f} dk",
            (
                f"giris: {self._fmt_price(entry_price)} | "
                f"tp: {self._fmt_price(tp_price)} ({self._fmt_pct(tp_target_pct)}) | "
                f"sl: {self._fmt_price(sl_price)} ({self._fmt_pct(sl_risk_pct)})"
            ),
            (
                f"pozisyonlar: {max(0, int(active_positions))}/{max(0, int(max_positions))} | "
                f"arka plan: {max(0, int(pending_signals))}"
            ),
        ]
        self._send(
            title=f"RBD-CRYPT entry {self._clean_symbol(symbol)}",
            message="\n".join(lines),
            priority=4,
            tags="chart_with_upwards_trend",
        )

    def on_position_close(
        self,
        *,
        symbol: str,
        side: str = "-",
        entry_price: float | None = None,
        exit_price: float | None = None,
        pnl_pct: float | None = None,
        reason: str | None = None,
        hold_minutes: float = 0.0,
        active_positions: int = 0,
        max_positions: int = 0,
        pending_signals: int = 0,
        scanned_count: int = 0,
    ) -> None:
        direction = self._direction_text(side)
        pnl = float(pnl_pct or 0.0)
        status = self._status_text(pnl)
        reason_text = reason.strip() if isinstance(reason, str) and reason.strip() else "-"
        lines = [
            f"{EXIT_EMOJI} EXIT",
            f"sembol: {self._clean_symbol(symbol)}",
            f"yon: {direction}",
            f"durum: {status}",
            f"neden: {reason_text}",
            f"pnl%: {self._fmt_pct(pnl)} | sure: {max(0.0, float(hold_minutes)):.1f} dk",
            f"giris: {self._fmt_price(entry_price)} | cikis: {self._fmt_price(exit_price)}",
            (
                f"pozisyonlar: {max(0, int(active_positions))}/{max(0, int(max_positions))} | "
                f"arka plan: {max(0, int(pending_signals))}"
            ),
        ]
        self._send(
            title=f"RBD-CRYPT exit {self._clean_symbol(symbol)}",
            message="\n".join(lines),
            priority=4,
            tags="x",
        )

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
        lines = [
            f"{ERROR_EMOJI} ERROR",
            f"sembol: {self._clean_symbol(symbol)}",
            f"pnl%: {self._fmt_pct(pnl_pct)}",
            f"aktif pozisyon: {max(0, int(active_positions))}",
            f"kaynak: {source} | {error_type}: {error}",
        ]
        self._send(
            title="RBD-CRYPT error",
            message="\n".join(lines),
            priority=5,
            tags="rotating_light",
        )

    def _maybe_send_heartbeat(
        self,
        *,
        now: datetime,
        perf: PerformanceSnapshot,
        open_positions: list[OpenPositionSnapshot] | None,
        event: str,
    ) -> None:
        if not self._should_send_heartbeat(now):
            return
        lines = [
            f"{HEARTBEAT_EMOJI} HEARTBEAT",
            *self._perf_lines(perf),
            f"event: {event}",
        ]
        lines.extend(self._position_lines(open_positions))
        self._send(
            title="RBD-CRYPT heartbeat",
            message="\n".join(lines),
            priority=3,
            tags="heart",
        )

    def _perf_lines(self, perf: PerformanceSnapshot) -> list[str]:
        return [
            f"trade: toplam {perf.total_trades} | win {perf.wins} | loss {perf.losses} | winrate {perf.win_rate_pct:.1f}%",
            f"kasa: {perf.balance:.2f} | toplam pnl {self._fmt_pct(perf.pnl_pct_cumulative)}",
            f"gunluk pnl: {perf.day_pnl_quote:+.2f} ({self._fmt_pct(perf.day_pnl_pct)})",
            (
                f"pozisyonlar: {max(0, int(perf.active_positions))}/{max(0, int(perf.max_positions))} | "
                f"arka plan: {max(0, int(perf.pending_signals))}"
            ),
        ]

    def _position_lines(self, open_positions: list[OpenPositionSnapshot] | None) -> list[str]:
        if open_positions is None:
            return []
        if not open_positions:
            return ["acik pozisyon: yok"]
        lines = ["acik pozisyonlar:"]
        for idx, pos in enumerate(open_positions[:4], start=1):
            direction = self._direction_text(pos.side)
            status = self._status_text(pos.current_pnl_pct)
            lines.append(
                (
                    f"{idx}) {self._clean_symbol(pos.symbol)} {direction} {status} "
                    f"| pnl {self._fmt_pct(pos.current_pnl_pct)} | {max(0.0, pos.hold_minutes):.1f} dk "
                    f"| giris {self._fmt_price(pos.entry_price)} "
                    f"| tp {self._fmt_price(pos.tp_price)} ({self._fmt_pct(pos.tp_target_pct)}) "
                    f"| sl {self._fmt_price(pos.sl_price)} ({self._fmt_pct(pos.sl_risk_pct)})"
                )
            )
        extra = len(open_positions) - 4
        if extra > 0:
            lines.append(f"+{extra} pozisyon daha var")
        return lines

    def _should_send_heartbeat(self, now: datetime) -> bool:
        return self._should_send(name="last_heartbeat_at", now=now, interval=self.heartbeat_interval)

    def _should_send_summary(self, now: datetime) -> bool:
        return self._should_send(name="last_summary_at", now=now, interval=self.summary_interval)

    def _should_send(self, *, name: str, now: datetime, interval: timedelta) -> bool:
        cached = self._last_heartbeat_at if name == "last_heartbeat_at" else self._last_summary_at
        if cached is None:
            cached = self._load_state_time(name)
        if cached is not None and (now - cached) < interval:
            return False
        if name == "last_heartbeat_at":
            self._last_heartbeat_at = now
        else:
            self._last_summary_at = now
        self._save_state_time(name, now)
        return True

    def _load_state_time(self, name: str) -> datetime | None:
        if self.state_store is None:
            return None
        try:
            state = self.state_store.get_json(STATE_KEY) or {}
        except Exception:
            return None
        raw = state.get(name)
        if not isinstance(raw, str) or not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None

    def _save_state_time(self, name: str, value: datetime) -> None:
        if self.state_store is None:
            return
        try:
            state = self.state_store.get_json(STATE_KEY) or {}
            state[name] = value.isoformat()
            self.state_store.set_json(STATE_KEY, state)
        except Exception as exc:
            self.logger.error(
                "ntfy_state_error",
                extra={"event": {"source": "notification_service", "key": name, "msg": str(exc)}},
            )

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
    def _legacy_perf(*, pnl_pct: float | None, active_positions: int, pending_signals: int) -> PerformanceSnapshot:
        pct = float(pnl_pct or 0.0)
        return PerformanceSnapshot(
            balance=0.0,
            pnl_pct_cumulative=pct,
            day_pnl_quote=0.0,
            day_pnl_pct=0.0,
            total_trades=0,
            wins=0,
            losses=0,
            win_rate_pct=0.0,
            active_positions=max(0, int(active_positions)),
            max_positions=max(0, int(active_positions)),
            pending_signals=max(0, int(pending_signals)),
        )

    @staticmethod
    def _direction_text(side: str) -> str:
        side_norm = side.strip().lower() if side else "-"
        if side_norm in {"long", "buy"}:
            return f"{UP_EMOJI} LONG"
        if side_norm in {"short", "sell"}:
            return f"{DOWN_EMOJI} SHORT"
        return "-"

    @staticmethod
    def _status_text(pnl_pct: float | None) -> str:
        value = float(pnl_pct or 0.0)
        if value >= 0:
            return f"{PROFIT_EMOJI} artida ({value:+.2f}%)"
        return f"{LOSS_EMOJI} ekside ({value:+.2f}%)"

    @staticmethod
    def _fmt_pct(value: float | None) -> str:
        if value is None:
            return "-"
        return f"{float(value):+,.2f}%".replace(",", "")

    @staticmethod
    def _fmt_price(value: float | None) -> str:
        if value is None:
            return "-"
        return f"{float(value):.6f}"

    @staticmethod
    def _clean_symbol(symbol: str) -> str:
        clean = symbol.strip() if symbol and symbol.strip() else "-"
        return clean
