from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from .position import PositionSide


class ClosedTrade(BaseModel):
    trade_id: str
    position_id: str
    symbol: str
    side: PositionSide
    qty: float
    entry_price: float
    exit_price: float
    initial_sl: float
    initial_tp: float
    current_sl: float
    current_tp: float
    opened_at: datetime
    closed_at: datetime
    entry_bar_time: datetime
    exit_reason: str
    pnl_pct: float
    pnl_quote: float
    rr_initial: float
    fee_paid: float = 0.0
    meta: dict[str, Any] = Field(default_factory=dict)
