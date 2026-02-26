from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from ..config import ExitSettings, LegacyParitySettings
from ..core.state_machine import move_stop_to_break_even
from ..models.position import ActivePosition


@dataclass(slots=True)
class ExitDecision:
    should_exit: bool
    reason: str | None = None
    break_even_moved: bool = False
    exit_price: float | None = None


def evaluate_exit(
    *,
    position: ActivePosition,
    current_price: float,
    now: datetime,
    exit_cfg: ExitSettings,
    trend_flip: bool = False,
) -> ExitDecision:
    break_even_moved = False
    if position.best_pnl_pct >= exit_cfg.break_even_trigger_pct:
        if position.side.value == "long" and position.current_sl < position.entry_price:
            move_stop_to_break_even(position)
            break_even_moved = True
        elif position.side.value == "short" and position.current_sl > position.entry_price:
            move_stop_to_break_even(position)
            break_even_moved = True

    if position.side.value == "long":
        if current_price <= position.current_sl:
            return ExitDecision(True, "sl", break_even_moved, position.current_sl)
        if current_price >= position.current_tp:
            return ExitDecision(True, "tp", break_even_moved, position.current_tp)
    else:
        if current_price >= position.current_sl:
            return ExitDecision(True, "sl", break_even_moved, position.current_sl)
        if current_price <= position.current_tp:
            return ExitDecision(True, "tp", break_even_moved, position.current_tp)

    if now - position.opened_at >= timedelta(minutes=exit_cfg.max_hold_minutes):
        return ExitDecision(True, "max_hold", break_even_moved, current_price)
    if now - position.last_update_at >= timedelta(minutes=exit_cfg.stale_minutes):
        return ExitDecision(True, "stale", break_even_moved, current_price)
    if trend_flip:
        return ExitDecision(True, "trend_flip", break_even_moved, current_price)
    return ExitDecision(False, None, break_even_moved, None)


def evaluate_legacy_exit(
    *,
    position: ActivePosition,
    current_high: float,
    current_low: float,
    current_close: float,
    current_trend: int,
    current_ema20: float,
    now: datetime,
    legacy_cfg: LegacyParitySettings,
) -> ExitDecision:
    hold_minutes = (now - position.opened_at).total_seconds() / 60.0
    curr_pnl_live = position.current_pnl_pct * 100.0
    best_pnl = position.best_pnl_pct * 100.0
    break_even_moved = False

    if position.side.value == "long":
        if current_high >= position.current_tp:
            pnl_exit = position.current_tp
            return ExitDecision(True, "tp", break_even_moved, pnl_exit)
        if current_low <= position.current_sl:
            pnl_exit = position.current_sl
            return ExitDecision(True, "sl", break_even_moved, pnl_exit)
    else:
        if current_low <= position.current_tp:
            pnl_exit = position.current_tp
            return ExitDecision(True, "tp", break_even_moved, pnl_exit)
        if current_high >= position.current_sl:
            pnl_exit = position.current_sl
            return ExitDecision(True, "sl", break_even_moved, pnl_exit)

    if hold_minutes > legacy_cfg.max_hold_minutes:
        grace_limit = legacy_cfg.max_hold_minutes + (legacy_cfg.max_hold_st_grace_bars * 5)
        adverse_flip = (
            (position.side.value == "long" and current_trend == -1)
            or (position.side.value == "short" and current_trend == 1)
        )
        if curr_pnl_live > 0.0:
            before = position.current_sl
            move_stop_to_break_even(position)
            break_even_moved = position.current_sl != before
        if adverse_flip:
            return ExitDecision(True, "trend_flip", break_even_moved, current_close)
        if hold_minutes > grace_limit:
            ema_against = (
                (position.side.value == "long" and current_close < current_ema20)
                or (position.side.value == "short" and current_close > current_ema20)
            )
            no_progress = (
                curr_pnl_live <= float(legacy_cfg.stale_exit_min_pnl_pct)
                and best_pnl < float(legacy_cfg.stale_exit_min_best_pnl_pct)
            )
            if no_progress or ema_against:
                return ExitDecision(True, "stale", break_even_moved, current_close)
    return ExitDecision(False, None, break_even_moved, None)
