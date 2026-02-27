from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from rbdcrypt.config import AppSettings
from rbdcrypt.notifications.notification_service import NotificationService, PerformanceSnapshot


class _FakeNtfyClient:
    def __init__(self, *, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.calls: list[dict[str, object]] = []

    def notify(
        self,
        title: str,
        message: str,
        *,
        priority: int | None = None,
        tags: str | None = None,
        attach_url: str | None = None,
        filename: str | None = None,
    ) -> bool:
        if self.should_fail:
            raise RuntimeError("ntfy down")
        self.calls.append(
            {
                "title": title,
                "message": message,
                "priority": priority,
                "tags": tags,
                "attach_url": attach_url,
                "filename": filename,
            }
        )
        return True


class _FakeStateStore:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, object]] = {}

    def get_json(self, key: str) -> dict[str, object] | None:
        raw = self.values.get(key)
        return dict(raw) if raw is not None else None

    def set_json(self, key: str, value: dict[str, object]) -> None:
        self.values[key] = dict(value)


def _logger() -> logging.Logger:
    logger = logging.getLogger("test_notification_service")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def _perf() -> PerformanceSnapshot:
    return PerformanceSnapshot(
        balance=1012.5,
        pnl_pct_cumulative=1.25,
        day_pnl_quote=12.5,
        day_pnl_pct=1.25,
        total_trades=10,
        wins=6,
        losses=4,
        win_rate_pct=60.0,
        active_positions=1,
        max_positions=3,
        pending_signals=0,
    )


def test_notification_service_formats_and_debounces_with_persisted_state() -> None:
    clock = {"now": datetime(2026, 2, 27, 10, 1, tzinfo=UTC)}
    notifier = _FakeNtfyClient()
    state = _FakeStateStore()
    service = NotificationService(
        notifier=notifier,
        logger=_logger(),
        now_fn=lambda: clock["now"],
        state_store=state,
    )

    service.on_engine_start(performance=_perf(), open_positions=[])
    clock["now"] += timedelta(minutes=9)
    service.on_cycle_completed(performance=_perf(), open_positions=[], opened=1, closed=0, scan_errors=0)

    # Simulate service restart: state store should preserve debounce.
    service2 = NotificationService(
        notifier=notifier,
        logger=_logger(),
        now_fn=lambda: clock["now"],
        state_store=state,
    )
    clock["now"] += timedelta(minutes=21)  # 10:31 slot => heartbeat
    service2.on_cycle_completed(performance=_perf(), open_positions=[], opened=0, closed=0, scan_errors=0)
    clock["now"] += timedelta(minutes=30)  # 11:01 slot => heartbeat + summary
    service2.on_cycle_completed(performance=_perf(), open_positions=[], opened=0, closed=0, scan_errors=1)

    headers = [str(call["message"]).splitlines()[0] for call in notifier.calls]
    assert len(headers) == 4
    assert headers[0].endswith("HEARTBEAT")
    assert headers[1].endswith("HEARTBEAT")
    assert headers[2].endswith("HEARTBEAT")
    assert headers[3].endswith("SUMMARY")

    for call in notifier.calls:
        message = str(call["message"])
        assert "trade: toplam" in message
        assert "kasa:" in message
        assert "pozisyonlar:" in message
        assert "scanned count" not in message


def test_notification_service_formats_entry_message() -> None:
    now = datetime(2026, 2, 27, 10, 0, tzinfo=UTC)
    notifier = _FakeNtfyClient()
    service = NotificationService(notifier=notifier, logger=_logger(), now_fn=lambda: now)
    service.on_position_open(
        symbol="BTCUSDT",
        side="long",
        entry_price=60000.0,
        tp_price=60600.0,
        sl_price=59700.0,
        tp_target_pct=1.0,
        sl_risk_pct=0.5,
        hold_minutes=2.0,
        current_pnl_pct=0.2,
        active_positions=2,
        max_positions=3,
        pending_signals=1,
    )
    assert len(notifier.calls) == 1
    message = str(notifier.calls[0]["message"])
    assert "ENTRY" in message
    assert "yon:" in message
    assert "durum:" not in message
    assert "sure:" not in message
    assert "giris:" in message
    assert "tp:" in message
    assert "sl:" in message
    assert "arka plan: 1" in message
    assert notifier.calls[0]["attach_url"] is None
    assert notifier.calls[0]["filename"] is None


def test_notification_service_attaches_png_chart_only_for_entry_exit() -> None:
    now = datetime(2026, 2, 27, 10, 0, tzinfo=UTC)
    notifier = _FakeNtfyClient()
    service = NotificationService(
        notifier=notifier,
        logger=_logger(),
        now_fn=lambda: now,
        chart_enabled=True,
    )
    chart_points = [60000.0 + i * 10 for i in range(64)]

    service.on_position_open(
        symbol="BTCUSDT",
        side="long",
        entry_price=60100.0,
        tp_price=60700.0,
        sl_price=59850.0,
        tp_target_pct=1.0,
        sl_risk_pct=0.4,
        active_positions=1,
        max_positions=3,
        pending_signals=0,
        chart_points=chart_points,
    )
    service.on_position_close(
        symbol="BTCUSDT",
        side="long",
        entry_price=60100.0,
        exit_price=60420.0,
        pnl_pct=0.53,
        reason="TP",
        hold_minutes=35.0,
        active_positions=0,
        max_positions=3,
        pending_signals=0,
        chart_points=chart_points,
    )

    assert len(notifier.calls) == 2
    entry_call = notifier.calls[0]
    exit_call = notifier.calls[1]
    assert entry_call["attach_url"] is not None
    assert str(entry_call["attach_url"]).startswith("https://quickchart.io/chart?format=png")
    assert entry_call["filename"] == "btcusdt-entry.png"
    assert exit_call["attach_url"] is not None
    assert str(exit_call["attach_url"]).startswith("https://quickchart.io/chart?format=png")
    assert exit_call["filename"] == "btcusdt-exit.png"


def test_point_marker_prefers_latest_when_tied() -> None:
    marker = NotificationService._point_marker("m", [1.0, 2.0, 1.0], 1.0, "#fff")
    data = marker["data"]
    assert data[0] is None
    assert data[2] == 1.0


def test_notification_service_is_fail_safe() -> None:
    clock_now = datetime(2026, 2, 27, 10, 0, tzinfo=UTC)
    service = NotificationService(
        notifier=_FakeNtfyClient(should_fail=True),
        logger=_logger(),
        now_fn=lambda: clock_now,
    )
    service.on_error(
        source="runtime.worker",
        error=RuntimeError("boom"),
        symbol="-",
        pnl_pct=None,
        active_positions=0,
        scanned_count=0,
    )


def test_ntfy_prefixed_env_keys_are_supported(monkeypatch) -> None:
    monkeypatch.delenv("NOTIFICATIONS__ENABLED", raising=False)
    monkeypatch.delenv("NOTIFICATIONS__TOPIC", raising=False)
    monkeypatch.setenv("NOTIFICATIONS__NTFY_ENABLED", "true")
    monkeypatch.setenv("NOTIFICATIONS__NTFY_TOPIC", "RBD-CRYPT")
    monkeypatch.setenv("NOTIFICATIONS__NTFY_URL", "https://ntfy.sh")

    settings = AppSettings(_env_file=None)

    assert settings.notifications.enabled is True
    assert settings.notifications.topic == "RBD-CRYPT"
    assert settings.notifications.ntfy_url == "https://ntfy.sh"
