from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class OHLCVBar(BaseModel):
    symbol: str
    interval: str = "5m"
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: datetime | None = None
    source: str = "binance_futures"
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
