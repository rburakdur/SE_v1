from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from ..core.risk import RiskPlan, pnl_pct
from ..models.position import ActivePosition, PositionSide
from ..models.trade import ClosedTrade
from .base import BrokerInterface


@dataclass(slots=True)
class PaperBroker(BrokerInterface):
    name: str = "paper"

    def open_position(
        self,
        *,
        symbol: str,
        side: str,
        risk_plan: RiskPlan,
        opened_at: datetime,
        entry_bar_time: datetime,
        strategy_tag: str,
    ) -> ActivePosition:
        position_id = f"ppos_{uuid4().hex[:12]}"
        return ActivePosition(
            position_id=position_id,
            symbol=symbol,
            side=PositionSide(side),
            qty=risk_plan.qty,
            entry_price=risk_plan.entry_price,
            initial_sl=risk_plan.initial_sl,
            initial_tp=risk_plan.initial_tp,
            current_sl=risk_plan.initial_sl,
            current_tp=risk_plan.initial_tp,
            opened_at=opened_at,
            recovered_at=None,
            entry_bar_time=entry_bar_time,
            last_update_at=opened_at,
            best_pnl_pct=0.0,
            current_pnl_pct=0.0,
            status="active",
            leverage=1.0,
            notional=risk_plan.notional,
            strategy_tag=strategy_tag,
            meta={"broker": self.name},
        )

    def close_position(
        self,
        *,
        position: ActivePosition,
        exit_price: float,
        reason: str,
        closed_at: datetime,
        fee_pct_per_side: float,
    ) -> ClosedTrade:
        gross_pnl_pct = pnl_pct(position.entry_price, exit_price, position.side.value, position.leverage)
        gross_pnl_quote = gross_pnl_pct * position.notional
        fee_paid = (position.notional * fee_pct_per_side) * 2.0
        net_pnl_quote = gross_pnl_quote - fee_paid
        trade_id = f"ptrd_{uuid4().hex[:12]}"
        return ClosedTrade(
            trade_id=trade_id,
            position_id=position.position_id,
            symbol=position.symbol,
            side=position.side,
            qty=position.qty,
            entry_price=position.entry_price,
            exit_price=exit_price,
            initial_sl=position.initial_sl,
            initial_tp=position.initial_tp,
            current_sl=position.current_sl,
            current_tp=position.current_tp,
            opened_at=position.opened_at,
            closed_at=closed_at,
            entry_bar_time=position.entry_bar_time,
            exit_reason=reason,
            pnl_pct=gross_pnl_pct,
            pnl_quote=net_pnl_quote,
            rr_initial=(
                abs(position.initial_tp - position.entry_price)
                / max(abs(position.entry_price - position.initial_sl), 1e-12)
            ),
            fee_paid=fee_paid,
            meta={"broker": self.name},
        )
