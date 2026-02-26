from __future__ import annotations

from datetime import UTC, datetime, timedelta

from rbdcrypt.config import ExitSettings
from rbdcrypt.models.position import ActivePosition, PositionSide
from rbdcrypt.strategy.exit_engine import evaluate_exit


def _pos(side: PositionSide = PositionSide.LONG) -> ActivePosition:
    now = datetime.now(tz=UTC)
    return ActivePosition(
        position_id="p1",
        symbol="X",
        side=side,
        qty=1.0,
        entry_price=100.0,
        initial_sl=98.0 if side == PositionSide.LONG else 102.0,
        initial_tp=104.0 if side == PositionSide.LONG else 96.0,
        current_sl=98.0 if side == PositionSide.LONG else 102.0,
        current_tp=104.0 if side == PositionSide.LONG else 96.0,
        opened_at=now - timedelta(minutes=10),
        recovered_at=None,
        entry_bar_time=now - timedelta(minutes=10),
        last_update_at=now,
        best_pnl_pct=0.0,
        current_pnl_pct=0.0,
        leverage=1.0,
        notional=100.0,
    )


def test_exit_tp_hit() -> None:
    p = _pos()
    decision = evaluate_exit(
        position=p,
        current_price=104.1,
        now=datetime.now(tz=UTC),
        exit_cfg=ExitSettings(),
    )
    assert decision.should_exit is True
    assert decision.reason == "tp"


def test_exit_sl_hit() -> None:
    p = _pos()
    decision = evaluate_exit(
        position=p,
        current_price=97.9,
        now=datetime.now(tz=UTC),
        exit_cfg=ExitSettings(),
    )
    assert decision.should_exit is True
    assert decision.reason == "sl"


def test_exit_moves_break_even_before_deciding() -> None:
    p = _pos()
    p.best_pnl_pct = 0.02
    cfg = ExitSettings(break_even_trigger_pct=0.005)
    decision = evaluate_exit(
        position=p,
        current_price=101.0,
        now=datetime.now(tz=UTC),
        exit_cfg=cfg,
    )
    assert decision.should_exit is False
    assert decision.break_even_moved is True
    assert p.current_sl == p.entry_price


def test_exit_on_max_hold() -> None:
    p = _pos()
    now = datetime.now(tz=UTC)
    p.opened_at = now - timedelta(minutes=999)
    decision = evaluate_exit(position=p, current_price=100.5, now=now, exit_cfg=ExitSettings(max_hold_minutes=60))
    assert decision.should_exit is True
    assert decision.reason == "max_hold"


def test_exit_on_stale() -> None:
    p = _pos()
    now = datetime.now(tz=UTC)
    p.last_update_at = now - timedelta(minutes=30)
    decision = evaluate_exit(position=p, current_price=100.5, now=now, exit_cfg=ExitSettings(stale_minutes=10))
    assert decision.should_exit is True
    assert decision.reason == "stale"


def test_exit_on_trend_flip() -> None:
    p = _pos()
    decision = evaluate_exit(
        position=p,
        current_price=100.2,
        now=datetime.now(tz=UTC),
        exit_cfg=ExitSettings(),
        trend_flip=True,
    )
    assert decision.should_exit is True
    assert decision.reason == "trend_flip"
