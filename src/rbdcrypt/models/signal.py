from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SignalDirection(StrEnum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class SignalEvent(BaseModel):
    symbol: str
    interval: str = "5m"
    bar_time: datetime
    direction: SignalDirection
    price: float
    power_score: float
    metrics: dict[str, float] = Field(default_factory=dict)
    power_breakdown: dict[str, float] = Field(default_factory=dict)
    candidate_pass: bool = False
    auto_pass: bool = False
    blocked_reasons: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class SignalDecision(BaseModel):
    signal_id: int | None = None
    symbol: str
    bar_time: datetime
    stage: str
    outcome: str
    blocked_reason: str | None = None
    decision_payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
