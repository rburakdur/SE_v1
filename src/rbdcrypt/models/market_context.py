from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class MarketContext(BaseModel):
    symbol: str = "BTCUSDT"
    interval: str = "5m"
    bar_time: datetime
    trend_direction: Literal["up", "down", "flat"] = "flat"
    trend_score: float = 0.0
    chop_state: Literal["trending", "choppy", "unknown"] = "unknown"
    metrics: dict[str, float] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
