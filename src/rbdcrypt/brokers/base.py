from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from ..core.risk import RiskPlan
from ..models.position import ActivePosition
from ..models.trade import ClosedTrade


class BrokerInterface(ABC):
    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def close_position(
        self,
        *,
        position: ActivePosition,
        exit_price: float,
        reason: str,
        closed_at: datetime,
        fee_pct_per_side: float,
    ) -> ClosedTrade:
        raise NotImplementedError
