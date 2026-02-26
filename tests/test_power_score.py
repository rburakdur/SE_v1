from __future__ import annotations

from rbdcrypt.config import ScoreSettings
from rbdcrypt.strategy.filters import FilterInputs, FilterOutcome
from rbdcrypt.strategy.scoring import compute_power_score


def test_power_score_increases_for_stronger_setup() -> None:
    score_cfg = ScoreSettings()
    strong = compute_power_score(
        FilterInputs(
            price=100.0,
            ema_fast=102.0,
            ema_slow=98.0,
            rsi=64.0,
            adx=28.0,
            atr_pct=0.012,
            volume_ratio=2.2,
        ),
        FilterOutcome(direction="long", passed=True, blocked_reasons=[], is_choppy=False, btc_aligned=True),
        score_cfg,
    )
    weak = compute_power_score(
        FilterInputs(
            price=100.0,
            ema_fast=100.1,
            ema_slow=99.9,
            rsi=51.0,
            adx=12.0,
            atr_pct=0.0025,
            volume_ratio=0.95,
        ),
        FilterOutcome(direction="long", passed=False, blocked_reasons=["adx_low"], is_choppy=True, btc_aligned=False),
        score_cfg,
    )
    assert strong.total > weak.total
    assert "trend" in strong.breakdown
    assert any(k.startswith("penalty_") for k in weak.breakdown)
