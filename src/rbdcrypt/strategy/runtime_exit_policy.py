from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from ..config import LegacyParitySettings
from ..models.position import ActivePosition
from ..models.symbol_state import SymbolBarState
from .exit_engine import evaluate_legacy_exit
from .profile_config import StrategyProfile


@dataclass(slots=True)
class RuntimeExitPolicyDecision:
    action: Literal["hold", "partial_tp1", "close"]
    reason: str | None = None
    exit_price: float | None = None
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeExitLevels:
    sl: float
    tp2: float
    tp1: float | None = None


def evaluate_runtime_exit_policy(
    *,
    position: ActivePosition,
    state: SymbolBarState,
    now: datetime,
    strategy_profile: StrategyProfile,
    legacy_cfg: LegacyParitySettings,
) -> RuntimeExitPolicyDecision:
    levels = _resolve_exit_levels(position)
    entry_layer = str(position.meta.get("entry_layer", "auto")).lower()
    candidate_hook_enabled = bool(strategy_profile.exit_policy.candidate_specific_exits)
    candidate_hook_checked = candidate_hook_enabled and entry_layer == "candidate"
    payload_base = {
        "entry_layer": entry_layer,
        "strategy_profile": strategy_profile.name,
        "candidate_specific_hook_enabled": candidate_hook_enabled,
        "candidate_specific_hook_checked": candidate_hook_checked,
    }

    if _stop_loss_hit(position, state, levels.sl):
        return RuntimeExitPolicyDecision(
            action="close",
            reason="sl",
            exit_price=levels.sl,
            payload={
                **payload_base,
                "sl_level": levels.sl,
                "tp1_level": levels.tp1,
                "tp2_level": levels.tp2,
            },
        )

    if _take_profit_hit(position, state, levels.tp2):
        return RuntimeExitPolicyDecision(
            action="close",
            reason="tp2",
            exit_price=levels.tp2,
            payload={
                **payload_base,
                "sl_level": levels.sl,
                "tp1_level": levels.tp1,
                "tp2_level": levels.tp2,
            },
        )

    tp1_done = bool(position.meta.get("tp1_done", False))
    if levels.tp1 is not None and not tp1_done and _take_profit_hit(position, state, levels.tp1):
        return RuntimeExitPolicyDecision(
            action="partial_tp1",
            reason="tp1_partial",
            exit_price=levels.tp1,
            payload={
                **payload_base,
                "tp1_done": tp1_done,
                "tp1_level": levels.tp1,
                "tp2_level": levels.tp2,
                "tp1_partial_fraction": float(position.meta.get("tp1_partial_fraction", 0.5)),
            },
        )

    if bool(strategy_profile.exit_policy.enable_session_exit) and _session_boundary_crossed(position.opened_at, now):
        return RuntimeExitPolicyDecision(
            action="close",
            reason="session_exit",
            exit_price=float(state.current_close),
            payload={
                **payload_base,
                "opened_at": position.opened_at.isoformat(),
                "now": now.isoformat(),
            },
        )

    if _bias_flip(position, state):
        return RuntimeExitPolicyDecision(
            action="close",
            reason="bias_flip",
            exit_price=float(state.current_close),
            payload={
                **payload_base,
                "current_trend": int(state.current_trend),
                "current_ema20": float(state.current_ema20),
            },
        )

    max_hold_min = max(0, int(legacy_cfg.max_hold_minutes))
    hold_min = max(0.0, (now - position.opened_at).total_seconds() / 60.0)
    if max_hold_min > 0 and hold_min >= max_hold_min:
        return RuntimeExitPolicyDecision(
            action="close",
            reason="max_hold",
            exit_price=float(state.current_close),
            payload={
                **payload_base,
                "hold_minutes": hold_min,
                "max_hold_minutes": max_hold_min,
            },
        )

    legacy = evaluate_legacy_exit(
        position=position,
        current_high=state.current_high,
        current_low=state.current_low,
        current_close=state.current_close,
        current_trend=state.current_trend,
        current_ema20=state.current_ema20,
        now=now,
        legacy_cfg=legacy_cfg,
    )
    if legacy.should_exit and legacy.reason:
        mapped_reason = _map_legacy_reason(legacy.reason)
        return RuntimeExitPolicyDecision(
            action="close",
            reason=mapped_reason,
            exit_price=legacy.exit_price,
            payload={
                **payload_base,
                "legacy_reason": legacy.reason,
                "break_even_moved": bool(legacy.break_even_moved),
            },
        )

    if legacy.break_even_moved:
        return RuntimeExitPolicyDecision(
            action="hold",
            reason=None,
            exit_price=None,
            payload={
                **payload_base,
                "break_even_moved": True,
            },
        )

    return RuntimeExitPolicyDecision(
        action="hold",
        reason=None,
        exit_price=None,
        payload=payload_base,
    )


def _resolve_exit_levels(position: ActivePosition) -> RuntimeExitLevels:
    payload = position.meta.get("exit_levels")
    if isinstance(payload, dict):
        sl = _float_or_default(payload.get("sl"), position.current_sl)
        tp2 = _float_or_default(payload.get("tp2"), position.current_tp)
        tp1_raw = payload.get("tp1")
        tp1 = float(tp1_raw) if isinstance(tp1_raw, int | float) else None
        return RuntimeExitLevels(sl=sl, tp2=tp2, tp1=tp1)
    return RuntimeExitLevels(sl=float(position.current_sl), tp2=float(position.current_tp), tp1=None)


def _float_or_default(value: object, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    return float(default)


def _stop_loss_hit(position: ActivePosition, state: SymbolBarState, level: float) -> bool:
    if position.side.value == "long":
        return float(state.current_low) <= level
    return float(state.current_high) >= level


def _take_profit_hit(position: ActivePosition, state: SymbolBarState, level: float) -> bool:
    if position.side.value == "long":
        return float(state.current_high) >= level
    return float(state.current_low) <= level


def _session_boundary_crossed(opened_at: datetime, now: datetime) -> bool:
    opened = opened_at if opened_at.tzinfo else opened_at.replace(tzinfo=UTC)
    current = now if now.tzinfo else now.replace(tzinfo=UTC)
    return opened.astimezone(UTC).date() != current.astimezone(UTC).date()


def _bias_flip(position: ActivePosition, state: SymbolBarState) -> bool:
    if position.side.value == "long":
        return int(state.current_trend) < 0
    return int(state.current_trend) > 0


def _map_legacy_reason(reason: str) -> str:
    if reason == "tp":
        return "tp2"
    if reason == "trend_flip":
        return "bias_flip"
    return reason
