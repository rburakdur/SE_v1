from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from rbdcrypt.brokers.paper_broker import PaperBroker
from rbdcrypt.config import AppSettings
from rbdcrypt.models.position import ActivePosition, PositionSide
from rbdcrypt.models.signal import SignalDirection, SignalEvent
from rbdcrypt.models.symbol_state import SymbolBarState
from rbdcrypt.services.trade_service import TradeService


def _logger():
    logger = logging.getLogger("test_trade_service")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def _base_settings() -> AppSettings:
    s = AppSettings()
    s.legacy_parity.enabled = True
    s.legacy_parity.cooldown_minutes = 20
    s.risk.max_active_positions = 3
    s.risk.min_rr = 0.0
    s.balance.starting_balance = 1000.0
    return s


def test_trade_service_blocks_signal_in_cooldown_and_records_missed(repos) -> None:
    now = datetime(2026, 2, 26, 12, 0, tzinfo=UTC)
    settings = _base_settings()
    repos.runtime_state.set_json("cooldowns", {"XRPUSDT": (now - timedelta(minutes=5)).isoformat()})

    service = TradeService.from_settings(
        settings=settings,
        broker=PaperBroker(),
        repos=repos,
        now_fn=lambda: now,
        logger=_logger(),
    )

    signal = SignalEvent(
        symbol="XRPUSDT",
        interval="5m",
        bar_time=now - timedelta(minutes=5),
        direction=SignalDirection.LONG,
        price=1.25,
        power_score=55.0,
        candidate_pass=True,
        auto_pass=True,
        meta={"entry_atr14": 0.01},
    )
    signal_id = repos.signals.insert_signal(signal)
    signal.meta["db_signal_id"] = signal_id

    result = service.handle_cycle(signals=[signal], prices_by_symbol={}, symbol_states={})
    assert result.opened == 0
    assert result.missed_signals == 1
    assert result.skipped == 1

    counters = repos.runtime_state.get_json("trade_missed_counters")
    assert counters is not None
    assert int(counters["last_cycle_missed_signals"]) == 1
    assert int(counters["hourly_missed_signals"]) >= 1

    with repos.signals.db.read_only() as conn:
        row = conn.execute(
            """
            SELECT stage, outcome, blocked_reason
            FROM signal_decisions
            WHERE signal_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (signal_id,),
        ).fetchone()
    assert row is not None
    assert row["stage"] == "execution_filter"
    assert row["outcome"] == "blocked"
    assert str(row["blocked_reason"]).startswith("COOLDOWN_")


def test_trade_service_close_persists_symbol_cooldown(repos) -> None:
    now = datetime(2026, 2, 26, 12, 0, tzinfo=UTC)
    settings = _base_settings()
    settings.risk.fee_pct_per_side = 0.0

    service = TradeService.from_settings(
        settings=settings,
        broker=PaperBroker(),
        repos=repos,
        now_fn=lambda: now,
        logger=_logger(),
    )

    pos = ActivePosition(
        position_id="p1",
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        qty=1.0,
        entry_price=100.0,
        initial_sl=98.0,
        initial_tp=104.0,
        current_sl=98.0,
        current_tp=104.0,
        opened_at=now - timedelta(minutes=30),
        recovered_at=None,
        entry_bar_time=now - timedelta(minutes=30),
        last_update_at=now - timedelta(minutes=1),
        best_pnl_pct=0.0,
        current_pnl_pct=0.0,
        leverage=1.0,
        notional=100.0,
    )
    repos.positions.upsert_active(pos)

    state = SymbolBarState(
        symbol="BTCUSDT",
        current_bar_time=now,
        current_high=104.5,
        current_low=99.5,
        current_close=103.8,
        current_trend=1,
        current_ema20=102.0,
        current_adx=25.0,
        current_rsi=60.0,
        current_vol_ratio=1.5,
        current_atr_pct=1.0,
        closed_bar_time=now - timedelta(minutes=5),
        closed_close=103.0,
        closed_atr14=1.2,
        closed_trend=1,
        closed_ema20=101.5,
        closed_rsi=58.0,
        closed_adx=24.0,
        closed_vol_ratio=1.3,
        closed_atr_pct=1.1,
        closed_macd_hist=0.2,
    )

    result = service.handle_cycle(signals=[], prices_by_symbol={}, symbol_states={"BTCUSDT": state})
    assert result.closed == 1
    cooldowns = repos.runtime_state.get_json("cooldowns")
    assert cooldowns is not None
    assert "BTCUSDT" in cooldowns
