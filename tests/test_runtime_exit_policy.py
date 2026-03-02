from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from rbdcrypt.brokers.paper_broker import PaperBroker
from rbdcrypt.config import AppSettings
from rbdcrypt.models.position import ActivePosition, PositionSide
from rbdcrypt.models.symbol_state import SymbolBarState
from rbdcrypt.services.trade_service import TradeService
from rbdcrypt.strategy.runtime_exit_policy import evaluate_runtime_exit_policy


def _logger():
    logger = logging.getLogger("test_runtime_exit_policy")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def _position(now: datetime, *, opened_at: datetime | None = None) -> ActivePosition:
    opened = opened_at or (now - timedelta(minutes=30))
    return ActivePosition(
        position_id="pos1",
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        qty=1.0,
        entry_price=100.0,
        initial_sl=98.5,
        initial_tp=102.0,
        current_sl=98.5,
        current_tp=102.0,
        opened_at=opened,
        recovered_at=None,
        entry_bar_time=opened,
        last_update_at=now,
        best_pnl_pct=0.0,
        current_pnl_pct=0.0,
        leverage=1.0,
        notional=100.0,
        strategy_tag="test",
        meta={
            "entry_layer": "candidate",
            "exit_levels": {"sl": 98.5, "tp1": 101.0, "tp2": 102.0},
            "tp1_done": False,
            "tp1_partial_fraction": 0.5,
        },
    )


def _state(now: datetime, *, high: float, low: float, close: float, trend: int = 1) -> SymbolBarState:
    return SymbolBarState(
        symbol="BTCUSDT",
        current_bar_time=now,
        current_high=high,
        current_low=low,
        current_close=close,
        current_trend=trend,
        current_ema20=99.5,
        current_adx=25.0,
        current_rsi=58.0,
        current_vol_ratio=1.4,
        current_atr_pct=1.0,
        closed_bar_time=now - timedelta(minutes=5),
        closed_close=close,
        closed_atr14=1.0,
        closed_trend=trend,
        closed_ema20=99.4,
        closed_rsi=57.0,
        closed_adx=24.0,
        closed_vol_ratio=1.3,
        closed_atr_pct=1.0,
        closed_macd_hist=0.2,
    )


def test_runtime_exit_policy_returns_partial_tp1() -> None:
    settings = AppSettings(_env_file=None)
    now = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    pos = _position(now)
    state = _state(now, high=101.2, low=100.1, close=100.9, trend=1)

    decision = evaluate_runtime_exit_policy(
        position=pos,
        state=state,
        now=now,
        strategy_profile=settings.load_strategy_profile(),
        legacy_cfg=settings.legacy_parity,
    )
    assert decision.action == "partial_tp1"
    assert decision.reason == "tp1_partial"
    assert decision.exit_price == 101.0


def test_runtime_exit_policy_returns_bias_flip() -> None:
    settings = AppSettings(_env_file=None)
    now = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    pos = _position(now)
    state = _state(now, high=100.8, low=99.7, close=100.0, trend=-1)

    decision = evaluate_runtime_exit_policy(
        position=pos,
        state=state,
        now=now,
        strategy_profile=settings.load_strategy_profile(),
        legacy_cfg=settings.legacy_parity,
    )
    assert decision.action == "close"
    assert decision.reason == "bias_flip"


def test_runtime_exit_policy_candidate_specific_hook_placeholder_flag() -> None:
    settings = AppSettings(_env_file=None)
    profile = settings.load_strategy_profile()
    profile.exit_policy.candidate_specific_exits = True
    now = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    pos = _position(now)
    state = _state(now, high=101.2, low=100.1, close=100.9, trend=1)

    decision = evaluate_runtime_exit_policy(
        position=pos,
        state=state,
        now=now,
        strategy_profile=profile,
        legacy_cfg=settings.legacy_parity,
    )
    assert decision.action == "partial_tp1"
    assert decision.payload["candidate_specific_hook_checked"] is True


def test_trade_service_applies_tp1_partial_and_keeps_position(repos) -> None:
    now = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    settings = AppSettings(_env_file=None)
    settings.legacy_parity.enabled = True
    settings.risk.min_rr = 0.0
    settings.risk.fixed_notional_per_trade = 25.0
    service = TradeService.from_settings(
        settings=settings,
        broker=PaperBroker(),
        repos=repos,
        now_fn=lambda: now,
        logger=_logger(),
    )

    pos = _position(now)
    repos.positions.upsert_active(pos)
    state = _state(now, high=101.2, low=100.1, close=100.9, trend=1)

    result = service.handle_cycle(
        signals=[],
        prices_by_symbol={},
        symbol_states={"BTCUSDT": state},
    )
    assert result.closed == 0

    active = repos.positions.list_active()
    assert len(active) == 1
    assert abs(active[0].qty - 0.5) < 1e-9
    assert abs(active[0].notional - 50.0) < 1e-9
    assert active[0].meta["tp1_done"] is True
    assert abs(active[0].current_sl - active[0].entry_price) < 1e-9
    assert abs(active[0].current_tp - 102.0) < 1e-9

    summary = repos.trades.summary()
    assert int(summary["total_trades"]) == 1


def test_trade_service_session_exit_closes_position(repos) -> None:
    now = datetime(2026, 3, 2, 0, 15, tzinfo=UTC)
    opened = datetime(2026, 3, 1, 23, 55, tzinfo=UTC)
    settings = AppSettings(_env_file=None)
    settings.legacy_parity.enabled = True
    settings.risk.min_rr = 0.0
    service = TradeService.from_settings(
        settings=settings,
        broker=PaperBroker(),
        repos=repos,
        now_fn=lambda: now,
        logger=_logger(),
    )

    pos = _position(now, opened_at=opened)
    repos.positions.upsert_active(pos)
    state = _state(now, high=100.6, low=99.8, close=100.1, trend=1)

    result = service.handle_cycle(
        signals=[],
        prices_by_symbol={},
        symbol_states={"BTCUSDT": state},
    )
    assert result.closed == 1
    assert repos.positions.count_active() == 0

    with repos.signals.db.read_only() as conn:
        row = conn.execute(
            """
            SELECT stage, outcome, blocked_reason
            FROM signal_decisions
            WHERE stage = 'exit_policy'
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    assert row is not None
    assert row["outcome"] == "close"
    assert row["blocked_reason"] == "session_exit"
