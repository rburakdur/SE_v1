from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from rbdcrypt.brokers.paper_broker import PaperBroker
from rbdcrypt.config import AppSettings
from rbdcrypt.models.signal import SignalDirection, SignalEvent
from rbdcrypt.services.trade_service import TradeService


def _logger() -> logging.Logger:
    logger = logging.getLogger("test_trade_service_flapping")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def _settings(snapshot_path: Path) -> AppSettings:
    s = AppSettings(_env_file=None)
    s.strategy.profile_name = "intraday_swing_v2_default"
    s.storage.snapshot_path = snapshot_path
    s.legacy_parity.enabled = True
    s.risk.min_rr = 0.0
    s.risk.fixed_notional_per_trade = 25.0
    s.risk.max_active_positions = 5
    return s


def _candidate_signal(now: datetime, symbol: str = "ADAUSDT") -> SignalEvent:
    return SignalEvent(
        symbol=symbol,
        interval="5m",
        bar_time=now - timedelta(minutes=5),
        direction=SignalDirection.LONG,
        price=100.0,
        power_score=72.0,
        candidate_pass=True,
        auto_pass=False,
        meta={"entry_atr14": 1.0, "evaluator_outcome": "candidate"},
    )


def _candidate_signal_short_same_bar(signal: SignalEvent) -> SignalEvent:
    return SignalEvent(
        symbol=signal.symbol,
        interval=signal.interval,
        bar_time=signal.bar_time,
        direction=SignalDirection.SHORT,
        price=signal.price,
        power_score=signal.power_score,
        candidate_pass=True,
        auto_pass=False,
        meta={"entry_atr14": 1.0, "evaluator_outcome": "candidate"},
    )


def test_startup_block_marks_signal_and_next_cycle_debounces(repos, tmp_path: Path) -> None:
    now = datetime(2026, 3, 2, 12, 0, tzinfo=UTC)
    settings = _settings(tmp_path / "runtime_snapshot.json")
    service = TradeService.from_settings(
        settings=settings,
        broker=PaperBroker(),
        repos=repos,
        now_fn=lambda: now,
        logger=_logger(),
    )

    signal = _candidate_signal(now, "XRPUSDT")
    signal_id = repos.signals.insert_signal(signal)
    signal.meta["db_signal_id"] = signal_id

    first = service.handle_cycle(
        signals=[signal],
        prices_by_symbol={},
        symbol_states={},
        allow_entries=False,
        entry_block_reason="STARTUP_STABILIZATION_BLOCK",
    )
    assert first.opened == 0
    assert first.startup_blocked == 1

    second = service.handle_cycle(signals=[signal], prices_by_symbol={}, symbol_states={})
    assert second.opened == 0
    assert second.debounce_blocked == 1


def test_processed_signal_keys_restore_after_restart(repos, tmp_path: Path) -> None:
    now = datetime(2026, 3, 2, 12, 0, tzinfo=UTC)
    settings = _settings(tmp_path / "runtime_snapshot.json")
    service = TradeService.from_settings(
        settings=settings,
        broker=PaperBroker(),
        repos=repos,
        now_fn=lambda: now,
        logger=_logger(),
    )
    signal = _candidate_signal(now, "DOGEUSDT")
    signal_id = repos.signals.insert_signal(signal)
    signal.meta["db_signal_id"] = signal_id

    service.handle_cycle(
        signals=[signal],
        prices_by_symbol={},
        symbol_states={},
        allow_entries=False,
        entry_block_reason="STARTUP_STABILIZATION_BLOCK",
    )
    key = service._signal_key(signal)  # noqa: SLF001

    service_restarted = TradeService.from_settings(
        settings=settings,
        broker=PaperBroker(),
        repos=repos,
        now_fn=lambda: now + timedelta(minutes=1),
        logger=_logger(),
    )
    assert service_restarted._is_signal_key_processed(key) is True  # noqa: SLF001


def test_restart_simulation_does_not_duplicate_open_position(repos, tmp_path: Path) -> None:
    now = datetime(2026, 3, 2, 12, 0, tzinfo=UTC)
    settings = _settings(tmp_path / "runtime_snapshot.json")
    service = TradeService.from_settings(
        settings=settings,
        broker=PaperBroker(),
        repos=repos,
        now_fn=lambda: now,
        logger=_logger(),
    )
    signal = _candidate_signal(now, "SOLUSDT")
    signal_id = repos.signals.insert_signal(signal)
    signal.meta["db_signal_id"] = signal_id

    opened = service.handle_cycle(signals=[signal], prices_by_symbol={}, symbol_states={})
    assert opened.opened == 1
    assert repos.positions.count_active() == 1

    restarted = TradeService.from_settings(
        settings=settings,
        broker=PaperBroker(),
        repos=repos,
        now_fn=lambda: now + timedelta(minutes=1),
        logger=_logger(),
    )
    restarted.recover_active_positions()
    replay = restarted.handle_cycle(signals=[signal], prices_by_symbol={}, symbol_states={})

    assert replay.opened == 0
    assert repos.positions.count_active() == 1


def test_same_candle_lock_blocks_opposite_direction_retry(repos, tmp_path: Path) -> None:
    now = datetime(2026, 3, 2, 12, 0, tzinfo=UTC)
    settings = _settings(tmp_path / "runtime_snapshot.json")
    service = TradeService.from_settings(
        settings=settings,
        broker=PaperBroker(),
        repos=repos,
        now_fn=lambda: now,
        logger=_logger(),
    )
    long_signal = _candidate_signal(now, "ETHUSDT")
    long_signal_id = repos.signals.insert_signal(long_signal)
    long_signal.meta["db_signal_id"] = long_signal_id

    startup_block = service.handle_cycle(
        signals=[long_signal],
        prices_by_symbol={},
        symbol_states={},
        allow_entries=False,
        entry_block_reason="STARTUP_STABILIZATION_BLOCK",
    )
    assert startup_block.startup_blocked == 1

    short_signal = _candidate_signal_short_same_bar(long_signal)
    short_signal_id = repos.signals.insert_signal(short_signal)
    short_signal.meta["db_signal_id"] = short_signal_id
    replay = service.handle_cycle(signals=[short_signal], prices_by_symbol={}, symbol_states={})

    assert replay.opened == 0
    assert replay.debounce_blocked == 1
