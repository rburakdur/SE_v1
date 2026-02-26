from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..config import FilterSettings
from ..models.market_context import MarketContext


@dataclass(slots=True)
class FilterInputs:
    price: float
    ema_fast: float
    ema_slow: float
    rsi: float
    adx: float
    atr_pct: float
    volume_ratio: float


@dataclass(slots=True)
class FilterOutcome:
    direction: Literal["long", "short", "flat"]
    passed: bool
    blocked_reasons: list[str] = field(default_factory=list)
    is_choppy: bool = False
    btc_aligned: bool = True


def _infer_direction(inputs: FilterInputs, settings: FilterSettings) -> Literal["long", "short", "flat"]:
    if inputs.ema_fast > inputs.ema_slow and inputs.rsi >= settings.rsi_long_min:
        return "long"
    if inputs.ema_fast < inputs.ema_slow and inputs.rsi <= settings.rsi_short_max:
        return "short"
    return "flat"


def evaluate_filters(
    inputs: FilterInputs,
    settings: FilterSettings,
    btc_context: MarketContext | None = None,
) -> FilterOutcome:
    reasons: list[str] = []
    direction = _infer_direction(inputs, settings)
    if direction == "flat":
        reasons.append("direction_unclear")
    if inputs.adx < settings.adx_min:
        reasons.append("adx_low")
    if inputs.atr_pct < settings.atr_pct_min:
        reasons.append("atr_low")
    if inputs.volume_ratio < settings.volume_ratio_min:
        reasons.append("volume_low")

    is_choppy = inputs.atr_pct <= settings.atr_pct_max_chop and inputs.adx < (settings.adx_min + 5.0)
    if is_choppy and settings.chop_policy == "block":
        reasons.append("chop_block")

    btc_aligned = True
    if btc_context and direction in {"long", "short"}:
        if btc_context.trend_direction == "up" and direction != "long":
            btc_aligned = False
        elif btc_context.trend_direction == "down" and direction != "short":
            btc_aligned = False
        elif btc_context.trend_direction == "flat":
            btc_aligned = False
        if (not btc_aligned) and settings.btc_trend_filter_mode == "hard_block":
            reasons.append("btc_misaligned")

    return FilterOutcome(
        direction=direction,
        passed=len(reasons) == 0,
        blocked_reasons=reasons,
        is_choppy=is_choppy,
        btc_aligned=btc_aligned,
    )
