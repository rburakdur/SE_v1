from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..core.risk import RiskPlan
from ..models.position import ActivePosition
from ..models.trade import ClosedTrade
from .base import BrokerInterface


@dataclass(slots=True)
class LiveBrokerScaffold(BrokerInterface):
    """Scaffold only: implement signed REST/WebSocket execution in a later iteration."""

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
        raise NotImplementedError("Live broker execution is not implemented in v0.1")

    def close_position(
        self,
        *,
        position: ActivePosition,
        exit_price: float,
        reason: str,
        closed_at: datetime,
        fee_pct_per_side: float,
    ) -> ClosedTrade:
        raise NotImplementedError("Live broker execution is not implemented in v0.1")
