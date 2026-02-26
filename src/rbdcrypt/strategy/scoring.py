from __future__ import annotations

from dataclasses import dataclass, field
from math import tanh

from ..config import ScoreSettings
from .filters import FilterInputs, FilterOutcome


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(slots=True)
class PowerScore:
    total: float
    breakdown: dict[str, float] = field(default_factory=dict)


def compute_power_score(inputs: FilterInputs, outcome: FilterOutcome, score_cfg: ScoreSettings) -> PowerScore:
    w = score_cfg.score_weights

    trend_strength = abs(inputs.ema_fast - inputs.ema_slow) / max(inputs.price, 1e-12)
    trend_component = _clamp01(tanh(trend_strength * 200))

    if outcome.direction == "long":
        momentum_raw = (inputs.rsi - 50.0) / 25.0
    elif outcome.direction == "short":
        momentum_raw = (50.0 - inputs.rsi) / 25.0
    else:
        momentum_raw = 0.0
    momentum_component = _clamp01(momentum_raw)

    volatility_component = _clamp01(inputs.atr_pct / 0.02)
    volume_component = _clamp01((inputs.volume_ratio - 1.0) / 1.5)
    btc_component = 1.0 if outcome.btc_aligned else 0.2
    anti_chop_component = 0.2 if outcome.is_choppy else 1.0

    weighted = {
        "trend": trend_component * w.get("trend", 0.0),
        "momentum": momentum_component * w.get("momentum", 0.0),
        "volatility": volatility_component * w.get("volatility", 0.0),
        "volume": volume_component * w.get("volume", 0.0),
        "btc_alignment": btc_component * w.get("btc_alignment", 0.0),
        "anti_chop": anti_chop_component * w.get("anti_chop", 0.0),
    }

    total = sum(weighted.values())
    penalties: dict[str, float] = {}
    if not outcome.btc_aligned:
        penalties["btc_soft_penalty"] = score_cfg.soft_penalty_btc_misalignment
        total -= penalties["btc_soft_penalty"]
    if outcome.is_choppy:
        penalties["chop_penalty"] = score_cfg.chop_penalty
        total -= penalties["chop_penalty"]
    total = max(0.0, min(100.0, total))
    breakdown = {**weighted, **{f"penalty_{k}": -v for k, v in penalties.items()}}
    return PowerScore(total=round(total, 4), breakdown=breakdown)
