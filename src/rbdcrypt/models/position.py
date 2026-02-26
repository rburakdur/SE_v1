from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class PositionSide(StrEnum):
    LONG = "long"
    SHORT = "short"


class ActivePosition(BaseModel):
    position_id: str
    symbol: str
    side: PositionSide
    qty: float
    entry_price: float
    initial_sl: float
    initial_tp: float
    current_sl: float
    current_tp: float
    opened_at: datetime
    recovered_at: datetime | None = None
    entry_bar_time: datetime
    last_update_at: datetime
    best_pnl_pct: float = 0.0
    current_pnl_pct: float = 0.0
    status: str = "active"
    leverage: float = 1.0
    notional: float = 0.0
    strategy_tag: str = "default"
    meta: dict[str, Any] = Field(default_factory=dict)


class PositionRecoveryRecord(BaseModel):
    recovered_count: int = 0
    symbols: list[str] = Field(default_factory=list)
