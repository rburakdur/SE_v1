from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .risk import pnl_pct
from ..models.position import ActivePosition


@dataclass(slots=True)
class PositionUpdate:
    current_price: float
    now: datetime
    current_pnl_pct: float
    best_pnl_pct: float


def update_position_mark(position: ActivePosition, current_price: float, now: datetime) -> PositionUpdate:
    current = pnl_pct(position.entry_price, current_price, position.side.value, position.leverage)
    best = max(position.best_pnl_pct, current)
    position.current_pnl_pct = current
    position.best_pnl_pct = best
    position.last_update_at = now
    return PositionUpdate(
        current_price=current_price,
        now=now,
        current_pnl_pct=current,
        best_pnl_pct=best,
    )


def move_stop_to_break_even(position: ActivePosition) -> None:
    if position.side.value == "long":
        position.current_sl = max(position.current_sl, position.entry_price)
    else:
        position.current_sl = min(position.current_sl, position.entry_price)
