from __future__ import annotations

from datetime import UTC, datetime, timedelta

from rbdcrypt.config import AppSettings, LegacyParitySettings
from rbdcrypt.models.market_context import MarketContext
from rbdcrypt.models.position import ActivePosition, PositionSide
from rbdcrypt.models.signal import SignalDirection
from rbdcrypt.strategy.exit_engine import evaluate_legacy_exit
from rbdcrypt.strategy.legacy_parity import (
    calculate_power_score,
    evaluate_signal_filters,
    get_candidate_fail_reason,
    get_flip_candidate_signal,
    get_signal_thresholds,
)
from rbdcrypt.strategy.parity_signal_engine import ParitySignalEngine
from rbdcrypt.strategy.signal_engine import CandleSeries


def _legacy_row(**overrides):
    base = {
        "timestamp": datetime(2026, 2, 1, 12, 0, tzinfo=UTC),
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000.0,
        "RSI": 65.0,
        "ADX": 30.0,
        "EMA20": 99.5,
        "EMA50": 98.5,
        "ATR_14": 1.2,
        "ATR_PCT": 1.2,
        "BBANDS_UP": 103.0,
        "BBANDS_MID": 100.0,
        "BBANDS_LOW": 97.0,
        "MACD_HIST": 0.5,
        "VOL_RATIO": 2.0,
        "TREND": 1,
        "FLIP_LONG": True,
        "FLIP_SHORT": False,
    }
    base.update(overrides)
    return base


def test_legacy_power_score_components_and_total() -> None:
    cfg = LegacyParitySettings()
    t = get_signal_thresholds(cfg, "candidate")
    score, breakdown = calculate_power_score(_legacy_row(), t)
    assert score > 0
    assert breakdown["rsi_component"] > 0
    assert breakdown["vol_component"] > 0
    assert breakdown["adx_component"] > 0
    assert breakdown["atr_component"] > 0
    assert breakdown["macd_component"] == 10.0


def test_legacy_candidate_fail_reason_priority() -> None:
    cfg = LegacyParitySettings()
    row = _legacy_row(RSI=40.0, VOL_RATIO=0.5, ADX=5.0, ATR_PCT=0.1)
    assert get_candidate_fail_reason(row, "LONG", cfg) == "CAND_FAIL_RSI"
    row2 = _legacy_row(RSI=60.0, VOL_RATIO=0.5, ADX=5.0, ATR_PCT=0.1)
    assert get_candidate_fail_reason(row2, "LONG", cfg) == "CAND_FAIL_VOL"


def test_legacy_signal_filters_match_expected_flags() -> None:
    cfg = LegacyParitySettings()
    t = get_signal_thresholds(cfg, "auto")
    flags = evaluate_signal_filters(_legacy_row(), "LONG", t)
    assert flags["flip_ok"] is True
    assert flags["ema_ok"] is True
    assert flags["all_ok"] is True
    assert get_flip_candidate_signal(_legacy_row()) == "LONG"


def _pos(now: datetime) -> ActivePosition:
    return ActivePosition(
        position_id="p1",
        symbol="X",
        side=PositionSide.LONG,
        qty=1.0,
        entry_price=100.0,
        initial_sl=98.0,
        initial_tp=104.0,
        current_sl=98.0,
        current_tp=104.0,
        opened_at=now - timedelta(minutes=60),
        recovered_at=None,
        entry_bar_time=now - timedelta(minutes=60),
        last_update_at=now,
        best_pnl_pct=0.003,   # 0.3%
        current_pnl_pct=0.001,  # 0.1%
        leverage=1.0,
        notional=100.0,
    )


def test_legacy_exit_stale_after_grace() -> None:
    now = datetime.now(tz=UTC)
    pos = _pos(now)
    cfg = LegacyParitySettings(max_hold_minutes=45, max_hold_st_grace_bars=2, stale_exit_min_pnl_pct=0.15, stale_exit_min_best_pnl_pct=0.60)
    decision = evaluate_legacy_exit(
        position=pos,
        current_high=100.3,
        current_low=99.8,
        current_close=100.1,
        current_trend=1,
        current_ema20=100.5,  # EMA against for long
        now=now,
        legacy_cfg=cfg,
    )
    assert decision.should_exit is True
    assert decision.reason == "stale"
    assert decision.exit_price == 100.1


def test_legacy_exit_moves_breakeven_after_max_hold_profit() -> None:
    now = datetime.now(tz=UTC)
    pos = _pos(now)
    pos.current_pnl_pct = 0.004  # 0.4%
    pos.best_pnl_pct = 0.007
    cfg = LegacyParitySettings(max_hold_minutes=45)
    decision = evaluate_legacy_exit(
        position=pos,
        current_high=101.0,
        current_low=100.2,
        current_close=100.4,
        current_trend=1,
        current_ema20=99.0,
        now=now,
        legacy_cfg=cfg,
    )
    assert decision.should_exit is False
    assert pos.current_sl == pos.entry_price
    assert decision.break_even_moved is True


def test_parity_signal_engine_smoke_no_flip_returns_blocked_signal() -> None:
    settings = AppSettings()
    engine = ParitySignalEngine(settings=settings)
    base_time = datetime(2026, 2, 1, 0, 0, tzinfo=UTC)
    closes = [100 + (i * 0.05) for i in range(80)]
    candles = CandleSeries(
        open_times=[base_time + timedelta(minutes=5 * i) for i in range(80)],
        opens=[c - 0.1 for c in closes],
        highs=[c + 0.2 for c in closes],
        lows=[c - 0.2 for c in closes],
        closes=closes,
        volumes=[1000 + (i % 5) * 10 for i in range(80)],
    )
    btc_ctx = MarketContext(
        bar_time=candles.open_times[-2],
        trend_direction="up",
        trend_score=100.0,
        chop_state="trending",
        metrics={"trend_code": 1.0},
        meta={"trend_code": 1, "is_chop_market": False},
    )
    result = engine.evaluate_detailed(symbol="TESTUSDT", candles=candles, btc_context=btc_ctx)
    assert result.signal.symbol == "TESTUSDT"
    assert result.symbol_state.symbol == "TESTUSDT"
    assert result.signal.direction in {SignalDirection.FLAT, SignalDirection.LONG, SignalDirection.SHORT}
