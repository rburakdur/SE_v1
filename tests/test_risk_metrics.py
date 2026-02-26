from __future__ import annotations

from datetime import UTC, datetime

from rbdcrypt.brokers.paper_broker import PaperBroker
from rbdcrypt.core.risk import build_risk_plan, rr_ratio
from rbdcrypt.models.position import ActivePosition, PositionSide


def test_rr_ratio_uses_initial_tp_sl() -> None:
    rr = rr_ratio(entry_price=100.0, initial_sl=98.0, initial_tp=106.0, side="long")
    assert rr == 3.0


def test_trade_rr_initial_not_affected_by_current_sl_mutation() -> None:
    broker = PaperBroker()
    now = datetime.now(tz=UTC)
    pos = ActivePosition(
        position_id="p1",
        symbol="TESTUSDT",
        side=PositionSide.LONG,
        qty=1.0,
        entry_price=100.0,
        initial_sl=98.0,
        initial_tp=106.0,
        current_sl=100.0,  # moved to break-even
        current_tp=106.0,
        opened_at=now,
        recovered_at=None,
        entry_bar_time=now,
        last_update_at=now,
        best_pnl_pct=0.02,
        current_pnl_pct=0.01,
        leverage=1.0,
        notional=100.0,
    )
    trade = broker.close_position(
        position=pos,
        exit_price=104.0,
        reason="manual",
        closed_at=now,
        fee_pct_per_side=0.0,
    )
    assert trade.rr_initial == 3.0


def test_build_risk_plan_respects_min_notional_and_returns_initial_rr() -> None:
    plan = build_risk_plan(
        balance=100.0,
        risk_per_trade_pct=0.01,
        leverage=1.0,
        entry_price=100.0,
        sl_pct=0.01,
        tp_pct=0.02,
        side="long",
        min_notional=10.0,
    )
    assert plan.notional >= 10.0
    assert round(plan.rr_initial, 5) == 2.0
