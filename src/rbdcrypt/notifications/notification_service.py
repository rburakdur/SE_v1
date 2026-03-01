from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from logging import Logger
from pathlib import Path
from typing import Callable, Protocol
from urllib.parse import quote

from .ntfy_client import NtfyClient


STATE_KEY = "notifications_state"
CMD_LAST_ID_KEY = "last_command_id"
CMD_LAST_TOPIC_KEY = "last_command_topic"

SUMMARY_EMOJI = "\U0001F4CA"
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
    realized_pnl_quote: float
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
        chart_enabled: bool = False,
    ) -> None:
        self.notifier = notifier
        self.logger = logger
        self.now_fn = now_fn
        self.state_store = state_store
        self.chart_enabled = chart_enabled
        self._last_summary_slot: str | None = None

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
        self._maybe_send_summary(
            now=now,
            perf=perf,
            open_positions=open_positions,
            opened=0,
            closed=0,
            scan_errors=0,
            source="startup",
        )

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
        self._maybe_send_summary(
            now=now,
            perf=perf,
            open_positions=open_positions,
            opened=opened,
            closed=closed,
            scan_errors=scan_errors,
            source="runtime loop",
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
        chart_points: list[float] | None = None,
        pnl_pct: float | None = None,
        scanned_count: int = 0,
    ) -> None:
        clean_symbol = self._clean_symbol(symbol)
        side_text = self._side_text(side)
        lines = [
            f"tp: {self._fmt_price(tp_price)} ({self._fmt_pct(tp_target_pct)})",
            f"sl: {self._fmt_price(sl_price)} ({self._fmt_pct(sl_risk_pct)})",
            f"giris: {self._fmt_price(entry_price)}",
            f"anlik pnl: {self._fmt_pct(current_pnl_pct)}",
            (
                f"pozisyonlar: {max(0, int(active_positions))}/{max(0, int(max_positions))} | "
                f"arka plan: {max(0, int(pending_signals))}"
            ),
        ]
        attachment = self._build_chart_attachment(
            event="entry",
            symbol=symbol,
            side=side,
            chart_points=chart_points,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
            exit_price=None,
            hold_minutes=hold_minutes,
            exit_reason=None,
        )
        self._send(
            title=f"ENTRY {clean_symbol} {side_text}",
            message="\n".join(lines),
            priority=4,
            tags="chart_with_upwards_trend",
            attach_url=attachment[0] if attachment else None,
            filename=attachment[1] if attachment else None,
        )

    def on_position_close(
        self,
        *,
        symbol: str,
        side: str = "-",
        entry_price: float | None = None,
        exit_price: float | None = None,
        tp_price: float | None = None,
        sl_price: float | None = None,
        pnl_pct: float | None = None,
        reason: str | None = None,
        hold_minutes: float = 0.0,
        active_positions: int = 0,
        max_positions: int = 0,
        pending_signals: int = 0,
        chart_points: list[float] | None = None,
        scanned_count: int = 0,
    ) -> None:
        direction = self._direction_text(side)
        pnl = float(pnl_pct or 0.0)
        clean_symbol = self._clean_symbol(symbol)
        reason_code = self._reason_code(reason)
        reason_label = self._reason_label(reason_code)
        reason_text = reason.strip().upper() if isinstance(reason, str) and reason.strip() else "-"
        lines = [
            f"islem tipi: {direction}",
            f"sure: {max(0.0, float(hold_minutes)):.1f} dk",
            f"giris: {self._fmt_price(entry_price)}",
            f"cikis: {self._fmt_price(exit_price)}",
            f"pnl: {self._fmt_pct(pnl)}",
            f"neden: {reason_text}",
            (
                f"pozisyonlar: {max(0, int(active_positions))}/{max(0, int(max_positions))} | "
                f"arka plan: {max(0, int(pending_signals))}"
            ),
        ]
        attachment = self._build_chart_attachment(
            event="exit",
            symbol=symbol,
            side=side,
            chart_points=chart_points,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
            exit_price=exit_price,
            hold_minutes=hold_minutes,
            exit_reason=reason,
        )
        self._send(
            title=f"{reason_label} {clean_symbol} {self._fmt_pct(pnl)}",
            message="\n".join(lines),
            priority=4,
            tags=self._exit_tags(reason_code=reason_code, pnl_pct=pnl),
            attach_url=attachment[0] if attachment else None,
            filename=attachment[1] if attachment else None,
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
            f"taranan: {max(0, int(scanned_count))}",
            f"kaynak: {source} | {error_type}: {error}",
        ]
        self._send(
            title="ERROR",
            message="\n".join(lines),
            priority=5,
            tags="rotating_light",
        )

    def process_ntfy_commands(self, *, export_logs_bundle: Callable[[str], Path]) -> None:
        if self.notifier is None:
            return
        cfg = self.notifier.config
        if not cfg.command_enabled:
            return
        if cfg.command_topic:
            command_topic = cfg.command_topic.strip()
        elif cfg.topic:
            # Default command topic is the same runtime topic.
            command_topic = cfg.topic
        else:
            command_topic = ""
        if not command_topic:
            return
        since = self._load_state_value(CMD_LAST_ID_KEY)
        last_topic = self._load_state_value(CMD_LAST_TOPIC_KEY)
        if last_topic and last_topic != command_topic:
            since = None
        try:
            messages = self.notifier.fetch_messages(topic=command_topic, since=since)
        except Exception as exc:
            self.logger.error(
                "ntfy_error",
                extra={"event": {"source": "notification_service", "title": "CMD_FETCH", "msg": str(exc)}},
            )
            return
        if not messages:
            return
        last_id = since
        for msg in messages:
            msg_id = str(msg.get("id") or "").strip()
            if msg_id:
                last_id = msg_id
            command = str(msg.get("message") or "").strip().lower()
            if command == "logs":
                command = "log"
            if command not in {"log", "log-all"}:
                continue
            archive_path = export_logs_bundle(command)
            try:
                self.notifier.upload_file(
                    title="LOGS ARCHIVE",
                    file_path=archive_path,
                    message=f"komut: {command} | dosya: {archive_path.name}",
                    priority=3,
                    tags="file_folder",
                )
            except Exception as exc:
                self.logger.error(
                    "ntfy_error",
                    extra={"event": {"source": "notification_service", "title": "CMD_UPLOAD", "msg": str(exc)}},
                )
        if last_id and last_id != since:
            self._save_state_value(CMD_LAST_ID_KEY, last_id)
        self._save_state_value(CMD_LAST_TOPIC_KEY, command_topic)

    def _maybe_send_summary(
        self,
        *,
        now: datetime,
        perf: PerformanceSnapshot,
        open_positions: list[OpenPositionSnapshot] | None,
        opened: int,
        closed: int,
        scan_errors: int,
        source: str,
    ) -> None:
        if not self._should_send_summary(now):
            return
        lines = [
            f"{SUMMARY_EMOJI} SUMMARY",
            *self._perf_lines_dashboard(perf),
            (
                f"donem: acilan {int(opened)} | kapanan {int(closed)} | "
                f"hata {int(scan_errors)} | kaynak {source}"
            ),
        ]
        lines.extend(self._position_lines_compact(open_positions))
        self._send(
            title="SUMMARY",
            message="\n".join(lines),
            priority=2,
            tags="bar_chart",
        )

    def _perf_lines_dashboard(self, perf: PerformanceSnapshot) -> list[str]:
        return [
            (
                f"trade: toplam {perf.total_trades} | kazanc {perf.wins} | "
                f"kayip {perf.losses} | winrate {perf.win_rate_pct:.1f}%"
            ),
            (
                f"kasa: {perf.balance:.2f} | gercek toplam pnl {perf.realized_pnl_quote:+.2f} "
                f"({self._fmt_pct(perf.pnl_pct_cumulative)})"
            ),
            f"gunluk gercek pnl: {perf.day_pnl_quote:+.2f} ({self._fmt_pct(perf.day_pnl_pct)})",
            (
                f"pozisyonlar: {max(0, int(perf.active_positions))}/{max(0, int(perf.max_positions))} | "
                f"arka plan: {max(0, int(perf.pending_signals))}"
            ),
        ]

    def _position_lines_compact(self, open_positions: list[OpenPositionSnapshot] | None) -> list[str]:
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
                    f"| {max(0.0, pos.hold_minutes):.1f} dk | pnl {self._fmt_pct(pos.current_pnl_pct)}"
                )
            )
        extra = len(open_positions) - 4
        if extra > 0:
            lines.append(f"+{extra} pozisyon daha var")
        return lines

    def _should_send_summary(self, now: datetime) -> bool:
        # Summary slots: minute 01 and 31 each hour.
        if now.minute not in {1, 31}:
            return False
        return self._should_send_slot(name="last_summary_slot", slot=now.strftime("%Y-%m-%dT%H:%M"))

    def _load_state_value(self, name: str) -> str | None:
        if self.state_store is None:
            return None
        try:
            state = self.state_store.get_json(STATE_KEY) or {}
        except Exception:
            return None
        raw = state.get(name)
        if isinstance(raw, str) and raw:
            return raw
        return None

    def _save_state_value(self, name: str, value: str) -> None:
        if self.state_store is None:
            return
        try:
            state = self.state_store.get_json(STATE_KEY) or {}
            state[name] = value
            self.state_store.set_json(STATE_KEY, state)
        except Exception as exc:
            self.logger.error(
                "ntfy_state_error",
                extra={"event": {"source": "notification_service", "key": name, "msg": str(exc)}},
            )

    def _should_send_slot(self, *, name: str, slot: str) -> bool:
        cached = self._last_summary_slot
        if cached is None:
            cached = self._load_state_value(name)
        if cached == slot:
            return False
        self._last_summary_slot = slot
        self._save_state_value(name, slot)
        return True

    def _send(
        self,
        *,
        title: str,
        message: str,
        priority: int,
        tags: str | None,
        attach_url: str | None = None,
        filename: str | None = None,
    ) -> None:
        if self.notifier is None:
            return
        try:
            self.notifier.notify(
                title,
                message,
                priority=priority,
                tags=tags,
                attach_url=attach_url,
                filename=filename,
            )
        except Exception as exc:
            self.logger.error(
                "ntfy_error",
                extra={"event": {"source": "notification_service", "title": title, "msg": str(exc)}},
            )

    def _build_chart_attachment(
        self,
        *,
        event: str,
        symbol: str,
        side: str,
        chart_points: list[float] | None,
        entry_price: float | None,
        tp_price: float | None,
        sl_price: float | None,
        exit_price: float | None,
        hold_minutes: float,
        exit_reason: str | None,
    ) -> tuple[str, str] | None:
        if not self.chart_enabled or not chart_points:
            return None
        cleaned = [float(v) for v in chart_points if float(v) > 0]
        if len(cleaned) < 8:
            return None
        series = cleaned[-25:]
        labels = list(range(1, len(series) + 1))
        side_norm = side.strip().lower() if side else "-"
        up = side_norm in {"long", "buy"}
        price_color = "#1d4ed8"
        entry_color = "#06b6d4"
        tp_color = "#15803d"
        sl_color = "#b91c1c"
        exit_color = "#0f766e"

        datasets: list[dict[str, object]] = [
            {
                "label": "price",
                "data": [round(v, 6) for v in series],
                "borderColor": price_color,
                "borderWidth": 2,
                "pointRadius": 0,
                "fill": False,
                "tension": 0.2,
            }
        ]
        entry_idx = len(series) - 1 if event == "entry" else self._estimate_entry_index(len(series), hold_minutes)
        if entry_price is not None:
            datasets.append(self._constant_line("entry", entry_price, len(series), entry_color))
            datasets.append(
                self._point_marker(
                    None,
                    series,
                    entry_price,
                    entry_color,
                    index=entry_idx,
                    point_style="circle",
                    radius=6,
                )
            )
        if tp_price is not None:
            datasets.append(self._constant_line("tp", tp_price, len(series), tp_color))
        if sl_price is not None:
            datasets.append(self._constant_line("sl", sl_price, len(series), sl_color))
        if exit_price is not None:
            datasets.append(self._constant_line("exit", exit_price, len(series), exit_color))
            exit_idx = len(series) - 1
            datasets.append(
                self._point_marker(
                    None,
                    series,
                    exit_price,
                    exit_color,
                    index=exit_idx,
                    point_style="rectRot",
                    radius=6,
                )
            )
            reason_code = self._reason_code(exit_reason)
            if reason_code in {"tp", "sl"}:
                hit_price = tp_price if reason_code == "tp" else sl_price
                if hit_price is None:
                    hit_price = exit_price
                hit_color = tp_color if reason_code == "tp" else sl_color
                datasets.append(
                    self._point_marker(
                        None,
                        series,
                        hit_price,
                        hit_color,
                        index=exit_idx,
                        point_style="crossRot",
                        radius=8,
                    )
                )

        side_text = "LONG" if up else "SHORT" if side_norm in {"short", "sell"} else "-"
        subtitle_parts = [f"YON: {side_text}"]
        if entry_price is not None:
            subtitle_parts.append(f"GIRIS {self._fmt_price(entry_price)}")
        if tp_price is not None and entry_price and entry_price > 0:
            tp_pct = abs(((tp_price - entry_price) / entry_price) * 100.0)
            subtitle_parts.append(f"TP {self._fmt_price(tp_price)} (+{tp_pct:.2f}%)")
        if sl_price is not None and entry_price and entry_price > 0:
            sl_pct = abs(((entry_price - sl_price) / entry_price) * 100.0)
            subtitle_parts.append(f"SL {self._fmt_price(sl_price)} ({sl_pct:.2f}%)")
        if exit_price is not None:
            subtitle_parts.append(f"CIKIS {self._fmt_price(exit_price)}")
        subtitle = " | ".join(subtitle_parts)
        y_min, y_max = self._y_bounds(
            series=series,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
            exit_price=exit_price,
        )

        cfg = {
            "type": "line",
            "data": {
                "labels": labels,
                "datasets": datasets,
            },
            "options": {
                "animation": False,
                "plugins": {
                    "legend": {"display": False},
                    "title": {
                        "display": True,
                        "text": [f"{self._clean_symbol(symbol)} {event.upper()} {side_text}", subtitle],
                    },
                },
                "scales": {
                    "x": {"display": False},
                    "y": {"display": True, "min": y_min, "max": y_max},
                },
            },
        }
        encoded = quote(json.dumps(cfg, separators=(",", ":")), safe="")
        url = (
            "https://quickchart.io/chart"
            f"?format=png&width=720&height=420&devicePixelRatio=2&c={encoded}"
        )
        filename = f"{self._clean_symbol(symbol).lower()}-{event}.png"
        return url, filename

    @staticmethod
    def _constant_line(label: str, value: float, size: int, color: str) -> dict[str, object]:
        return {
            "label": label,
            "data": [round(float(value), 6)] * size,
            "borderColor": color,
            "borderWidth": 1,
            "pointRadius": 0,
            "fill": False,
            "borderDash": [6, 4],
            "tension": 0,
        }

    @staticmethod
    def _point_marker(
        label: str | None,
        series: list[float],
        value: float,
        color: str,
        *,
        index: int | None = None,
        point_style: str = "circle",
        radius: int = 6,
    ) -> dict[str, object]:
        if index is not None and 0 <= int(index) < len(series):
            nearest_idx = int(index)
        else:
            # Prefer right-most candle when the distance tie happens.
            nearest_idx = max(range(len(series)), key=lambda i: (-abs(series[i] - value), i))
        points = [None] * len(series)
        points[nearest_idx] = round(float(value), 6)
        marker = {
            "data": points,
            "showLine": False,
            "pointRadius": int(radius),
            "pointHoverRadius": int(radius),
            "pointBackgroundColor": color,
            "pointBorderColor": "#111827",
            "pointBorderWidth": 1,
            "pointStyle": point_style,
        }
        if isinstance(label, str) and label.strip():
            marker["label"] = label
        return marker

    @staticmethod
    def _estimate_entry_index(series_len: int, hold_minutes: float) -> int:
        if series_len <= 1:
            return 0
        bars_back = int(round(max(0.0, float(hold_minutes)) / 5.0))
        return max(0, series_len - 1 - bars_back)

    @staticmethod
    def _y_bounds(
        *,
        series: list[float],
        entry_price: float | None,
        tp_price: float | None,
        sl_price: float | None,
        exit_price: float | None,
    ) -> tuple[float, float]:
        values = list(series)
        for level in (entry_price, tp_price, sl_price, exit_price):
            if level is None:
                continue
            values.append(float(level))
        y_min = min(values)
        y_max = max(values)
        span = max(y_max - y_min, y_max * 0.002, 1e-9)
        pad = span * 0.2
        lower = y_min - pad
        upper = y_max + pad
        if lower <= 0:
            lower = y_min * 0.98
        return round(lower, 6), round(upper, 6)

    @staticmethod
    def _legacy_perf(*, pnl_pct: float | None, active_positions: int, pending_signals: int) -> PerformanceSnapshot:
        pct = float(pnl_pct or 0.0)
        return PerformanceSnapshot(
            balance=0.0,
            realized_pnl_quote=0.0,
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
    def _side_text(side: str) -> str:
        side_norm = side.strip().lower() if side else "-"
        if side_norm in {"long", "buy"}:
            return "LONG"
        if side_norm in {"short", "sell"}:
            return "SHORT"
        return "-"

    @staticmethod
    def _reason_code(reason: str | None) -> str:
        if reason is None:
            return "exit"
        normalized = reason.strip().lower()
        if normalized == "tp":
            return "tp"
        if normalized == "sl":
            return "sl"
        return "exit"

    @staticmethod
    def _reason_label(reason_code: str) -> str:
        if reason_code == "tp":
            return "TP"
        if reason_code == "sl":
            return "SL"
        return "EXIT"

    @staticmethod
    def _exit_tags(*, reason_code: str, pnl_pct: float) -> str:
        if reason_code == "tp":
            return "white_check_mark"
        if reason_code == "sl":
            return "x"
        return "white_check_mark" if pnl_pct >= 0 else "x"

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
