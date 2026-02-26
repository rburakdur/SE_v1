from __future__ import annotations

from datetime import UTC, datetime

from rbdcrypt.config import FilterSettings
from rbdcrypt.models.market_context import MarketContext
from rbdcrypt.strategy.filters import FilterInputs, evaluate_filters


def test_filter_passes_for_strong_long_setup() -> None:
    outcome = evaluate_filters(
        FilterInputs(
            price=100.0,
            ema_fast=101.2,
            ema_slow=99.8,
            rsi=58.0,
            adx=26.0,
            atr_pct=0.008,
            volume_ratio=1.3,
        ),
        FilterSettings(),
        btc_context=MarketContext(
            bar_time=datetime.now(tz=UTC),
            trend_direction="up",
            chop_state="trending",
            trend_score=70.0,
        ),
    )
    assert outcome.direction == "long"
    assert outcome.passed is True
    assert outcome.blocked_reasons == []


def test_filter_blocks_on_btc_misalignment_and_weak_adx() -> None:
    settings = FilterSettings(btc_trend_filter_mode="hard_block")
    outcome = evaluate_filters(
        FilterInputs(
            price=100.0,
            ema_fast=98.5,
            ema_slow=100.5,
            rsi=44.0,
            adx=10.0,
            atr_pct=0.004,
            volume_ratio=1.2,
        ),
        settings,
        btc_context=MarketContext(
            bar_time=datetime.now(tz=UTC),
            trend_direction="up",
            chop_state="trending",
            trend_score=75.0,
        ),
    )
    assert outcome.direction == "short"
    assert outcome.passed is False
    assert "adx_low" in outcome.blocked_reasons
    assert "btc_misaligned" in outcome.blocked_reasons
