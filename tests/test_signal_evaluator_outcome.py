from __future__ import annotations

from datetime import UTC, datetime, timedelta

import rbdcrypt.strategy.parity_signal_engine as parity_module
from rbdcrypt.config import AppSettings
from rbdcrypt.models.market_context import MarketContext
from rbdcrypt.models.symbol_state import SymbolBarState
from rbdcrypt.strategy.legacy_parity import LegacyAnalysis
from rbdcrypt.strategy.parity_signal_engine import ParitySignalEngine
from rbdcrypt.strategy.signal_engine import CandleSeries


def _series(*, descending: bool = False, bars: int = 80) -> CandleSeries:
    base = datetime(2026, 2, 1, 0, 0, tzinfo=UTC)
    if descending:
        closes = [120.0 - (i * 0.2) for i in range(bars)]
    else:
        closes = [100.0 + (i * 0.2) for i in range(bars)]
    return CandleSeries(
        open_times=[base + timedelta(hours=i) for i in range(bars)],
        opens=[c - 0.1 for c in closes],
        highs=[c + 0.2 for c in closes],
        lows=[c - 0.2 for c in closes],
        closes=closes,
        volumes=[1000.0 + i for i in range(bars)],
    )


def _analysis() -> LegacyAnalysis:
    ts = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
    row = {
        "timestamp": ts,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000.0,
        "RSI": 65.0,
        "ADX": 30.0,
        "EMA20": 99.0,
        "EMA50": 98.0,
        "ATR_14": 1.2,
        "ATR_PCT": 1.1,
        "BBANDS_UP": 103.0,
        "BBANDS_MID": 100.0,
        "BBANDS_LOW": 97.0,
        "MACD_HIST": 0.4,
        "VOL_RATIO": 2.0,
        "TREND": 1,
        "FLIP_LONG": True,
        "FLIP_SHORT": False,
    }
    state = SymbolBarState(
        symbol="",
        current_bar_time=ts + timedelta(minutes=5),
        current_high=101.2,
        current_low=99.2,
        current_close=100.6,
        current_trend=1,
        current_ema20=99.2,
        current_adx=28.0,
        current_rsi=64.0,
        current_vol_ratio=1.8,
        current_atr_pct=1.0,
        closed_bar_time=ts,
        closed_close=100.5,
        closed_atr14=1.2,
        closed_trend=1,
        closed_ema20=99.0,
        closed_rsi=65.0,
        closed_adx=30.0,
        closed_vol_ratio=2.0,
        closed_atr_pct=1.1,
        closed_macd_hist=0.4,
    )
    return LegacyAnalysis(closed_row=row, live_row=dict(row), symbol_state=state)


def _btc_ctx() -> MarketContext:
    ts = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
    return MarketContext(
        symbol="BTCUSDT",
        interval="5m",
        bar_time=ts,
        trend_direction="up",
        trend_score=100.0,
        chop_state="trending",
        metrics={},
        meta={"trend_code": 1, "is_chop_market": False},
    )


def test_signal_evaluator_outcome_candidate_when_auto_score_below_min(monkeypatch) -> None:
    settings = AppSettings(_env_file=None)
    settings.strategy.profile_name = "intraday_swing_v1_baseline"
    engine = ParitySignalEngine(settings=settings)

    monkeypatch.setattr(parity_module, "analyze_candles", lambda *_args, **_kwargs: _analysis())
    monkeypatch.setattr(parity_module, "get_flip_candidate_signal", lambda *_args, **_kwargs: "LONG")
    monkeypatch.setattr(
        parity_module,
        "evaluate_signal_filters",
        lambda *_args, **_kwargs: {
            "flip_ok": True,
            "rsi_ok": True,
            "vol_ok": True,
            "adx_ok": True,
            "atr_ok": True,
            "ema_ok": True,
            "all_ok": True,
        },
    )

    def _calc(_row, thresholds):
        if float(thresholds.min_power_score) <= 30.0:
            return 70.0, {"component": 70.0}
        return 60.0, {"component": 60.0}

    monkeypatch.setattr(parity_module, "calculate_power_score", _calc)
    monkeypatch.setattr(parity_module, "score_from_flags", lambda *_args, **_kwargs: 6)
    monkeypatch.setattr(parity_module, "btc_trend_match", lambda *_args, **_kwargs: True)

    result = engine.evaluate_detailed(
        symbol="TESTUSDT",
        candles=_series(),
        btc_context=_btc_ctx(),
        htf_candles=_series(descending=False),
    )

    assert result.signal.meta["evaluator_outcome"] == "candidate"
    assert result.signal.candidate_pass is True
    assert result.signal.auto_pass is False
    assert result.signal.blocked_reasons == []
    assert "AUTO_SCORE_BELOW_MIN" in result.signal.meta["downgrade_reason_codes"]
    assert result.signal.meta["ema_1h_bias"] == "up"


def test_signal_evaluator_outcome_blocked_on_htf_bias_mismatch(monkeypatch) -> None:
    settings = AppSettings(_env_file=None)
    settings.strategy.profile_name = "intraday_swing_v1_baseline"
    engine = ParitySignalEngine(settings=settings)

    monkeypatch.setattr(parity_module, "analyze_candles", lambda *_args, **_kwargs: _analysis())
    monkeypatch.setattr(parity_module, "get_flip_candidate_signal", lambda *_args, **_kwargs: "LONG")
    monkeypatch.setattr(
        parity_module,
        "evaluate_signal_filters",
        lambda *_args, **_kwargs: {
            "flip_ok": True,
            "rsi_ok": True,
            "vol_ok": True,
            "adx_ok": True,
            "atr_ok": True,
            "ema_ok": True,
            "all_ok": True,
        },
    )
    monkeypatch.setattr(parity_module, "calculate_power_score", lambda *_args, **_kwargs: (85.0, {"component": 85.0}))
    monkeypatch.setattr(parity_module, "score_from_flags", lambda *_args, **_kwargs: 6)
    monkeypatch.setattr(parity_module, "btc_trend_match", lambda *_args, **_kwargs: True)

    result = engine.evaluate_detailed(
        symbol="TESTUSDT",
        candles=_series(),
        btc_context=_btc_ctx(),
        htf_candles=_series(descending=True),
    )

    assert result.signal.meta["evaluator_outcome"] == "blocked"
    assert result.signal.candidate_pass is False
    assert result.signal.auto_pass is False
    assert result.signal.blocked_reasons == ["HTF_BIAS_MISMATCH"]
    assert result.signal.meta["blocked_reason_codes"] == ["HTF_BIAS_MISMATCH"]
    assert result.signal.meta["rejection_stage"] == "htf_bias_filter"
