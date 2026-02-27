from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from rbdcrypt.config import AppSettings
from rbdcrypt.notifications.notification_service import NotificationService


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
    ) -> bool:
        if self.should_fail:
            raise RuntimeError("ntfy down")
        self.calls.append(
            {
                "title": title,
                "message": message,
                "priority": priority,
                "tags": tags,
            }
        )
        return True


def _logger() -> logging.Logger:
    logger = logging.getLogger("test_notification_service")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def test_notification_service_formats_and_debounces() -> None:
    clock = {"now": datetime(2026, 2, 27, 10, 0, tzinfo=UTC)}
    notifier = _FakeNtfyClient()
    service = NotificationService(notifier=notifier, logger=_logger(), now_fn=lambda: clock["now"])

    service.on_engine_start(pnl_pct=0.0, active_positions=1, scanned_count=0)
    clock["now"] += timedelta(minutes=10)
    service.on_cycle_completed(
        pnl_pct=1.25,
        active_positions=2,
        scanned_count=40,
        opened=1,
        closed=0,
        scan_errors=0,
    )
    clock["now"] += timedelta(minutes=10)
    service.on_cycle_completed(
        pnl_pct=1.30,
        active_positions=2,
        scanned_count=41,
        opened=0,
        closed=0,
        scan_errors=0,
    )
    clock["now"] += timedelta(minutes=11)
    service.on_cycle_completed(
        pnl_pct=1.50,
        active_positions=2,
        scanned_count=39,
        opened=0,
        closed=1,
        scan_errors=0,
    )
    clock["now"] += timedelta(minutes=40)
    service.on_cycle_completed(
        pnl_pct=2.00,
        active_positions=1,
        scanned_count=38,
        opened=0,
        closed=0,
        scan_errors=1,
    )

    headers = [str(call["message"]).splitlines()[0] for call in notifier.calls]
    assert headers == [
        "❤️ HEARTBEAT",
        "📊 SUMMARY",
        "❤️ HEARTBEAT",
        "❤️ HEARTBEAT",
        "📊 SUMMARY",
    ]
    for call in notifier.calls:
        message = str(call["message"])
        assert "symbol:" in message
        assert "pnl%:" in message
        assert "active positions:" in message
        assert "scanned count:" in message


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
