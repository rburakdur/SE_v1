from __future__ import annotations

from dataclasses import dataclass


def rr_ratio(entry_price: float, initial_sl: float, initial_tp: float, side: str) -> float:
    if side == "long":
        risk = max(entry_price - initial_sl, 1e-12)
        reward = max(initial_tp - entry_price, 0.0)
    else:
        risk = max(initial_sl - entry_price, 1e-12)
        reward = max(entry_price - initial_tp, 0.0)
    return reward / risk


def pnl_pct(entry_price: float, current_price: float, side: str, leverage: float = 1.0) -> float:
    raw = (current_price - entry_price) / max(entry_price, 1e-12)
    if side == "short":
        raw *= -1.0
    return raw * leverage


@dataclass(slots=True)
class RiskPlan:
    qty: float
    notional: float
    entry_price: float
    initial_sl: float
    initial_tp: float
    rr_initial: float


def build_risk_plan(
    *,
    balance: float,
    risk_per_trade_pct: float,
    leverage: float,
    entry_price: float,
    sl_pct: float,
    tp_pct: float,
    side: str,
    min_notional: float,
) -> RiskPlan:
    risk_amount = max(balance, 0.0) * max(risk_per_trade_pct, 0.0)
    sl_move = max(entry_price * sl_pct, 1e-12)
    qty = risk_amount / sl_move if sl_move else 0.0
    notional = qty * entry_price / max(leverage, 1e-12)
    if notional < min_notional:
        qty = min_notional * max(leverage, 1e-12) / max(entry_price, 1e-12)
        notional = min_notional
    if side == "long":
        initial_sl = entry_price * (1.0 - sl_pct)
        initial_tp = entry_price * (1.0 + tp_pct)
    else:
        initial_sl = entry_price * (1.0 + sl_pct)
        initial_tp = entry_price * (1.0 - tp_pct)
    return RiskPlan(
        qty=qty,
        notional=notional,
        entry_price=entry_price,
        initial_sl=initial_sl,
        initial_tp=initial_tp,
        rr_initial=rr_ratio(entry_price, initial_sl, initial_tp, side),
    )


def build_risk_plan_from_levels(
    *,
    balance: float,
    risk_per_trade_pct: float,
    leverage: float,
    entry_price: float,
    initial_sl: float,
    initial_tp: float,
    side: str,
    min_notional: float,
) -> RiskPlan:
    risk_amount = max(balance, 0.0) * max(risk_per_trade_pct, 0.0)
    risk_per_unit = abs(entry_price - initial_sl)
    risk_per_unit = max(risk_per_unit, 1e-12)
    qty = risk_amount / risk_per_unit
    notional = qty * entry_price / max(leverage, 1e-12)
    if notional < min_notional:
        qty = min_notional * max(leverage, 1e-12) / max(entry_price, 1e-12)
        notional = min_notional
    return RiskPlan(
        qty=qty,
        notional=notional,
        entry_price=entry_price,
        initial_sl=initial_sl,
        initial_tp=initial_tp,
        rr_initial=rr_ratio(entry_price, initial_sl, initial_tp, side),
    )
