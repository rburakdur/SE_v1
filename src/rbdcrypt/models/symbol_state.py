from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class SymbolBarState:
    symbol: str
    current_bar_time: datetime
    current_high: float
    current_low: float
    current_close: float
    current_trend: int
    current_ema20: float
    current_adx: float
    current_rsi: float
    current_vol_ratio: float
    current_atr_pct: float
    closed_bar_time: datetime
    closed_close: float
    closed_atr14: float
    closed_trend: int
    closed_ema20: float
    closed_rsi: float
    closed_adx: float
    closed_vol_ratio: float
    closed_atr_pct: float
    closed_macd_hist: float
