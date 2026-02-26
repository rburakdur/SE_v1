from __future__ import annotations

from dataclasses import asdict

import pytest

from rbdcrypt.config import LegacyParitySettings
from rbdcrypt.strategy.legacy_parity import (
    calculate_power_score,
    evaluate_signal_filters,
    get_candidate_fail_reason,
    get_flip_candidate_signal,
    get_signal_thresholds,
    score_from_flags,
)


from tests import reference_legacy_engine_signals as engine_signals


def _engine_config() -> dict[str, float | int | str]:
    c = LegacyParitySettings()
    return {
        "CANDIDATE_RSI_LONG": c.candidate_rsi_long,
        "CANDIDATE_RSI_SHORT": c.candidate_rsi_short,
        "CANDIDATE_VOL_FILTER": c.candidate_vol_filter,
        "CANDIDATE_ADX_THRESHOLD": c.candidate_adx_threshold,
        "CANDIDATE_MIN_ATR_PERCENT": c.candidate_min_atr_percent,
        "CANDIDATE_MIN_POWER_SCORE": c.candidate_min_power_score,
        "AUTO_ENTRY_RSI_LONG": c.auto_entry_rsi_long,
        "AUTO_ENTRY_RSI_SHORT": c.auto_entry_rsi_short,
        "AUTO_ENTRY_VOL_FILTER": c.auto_entry_vol_filter,
        "AUTO_ENTRY_ADX_THRESHOLD": c.auto_entry_adx_threshold,
        "AUTO_ENTRY_MIN_ATR_PERCENT": c.auto_entry_min_atr_percent,
        "AUTO_ENTRY_MIN_POWER_SCORE": c.auto_entry_min_power_score,
    }


def _rows() -> list[dict[str, object]]:
    return [
        {
            "FLIP_LONG": True,
            "FLIP_SHORT": False,
            "TREND": 1,
            "RSI": 67.0,
            "VOL_RATIO": 2.1,
            "ADX": 29.0,
            "ATR_14": 1.2,
            "ATR_PCT": 1.1,
            "close": 100.0,
            "EMA20": 99.0,
            "MACD_HIST": 0.35,
            "BBANDS_UP": 104.0,
            "BBANDS_MID": 100.0,
            "BBANDS_LOW": 96.0,
        },
        {
            "FLIP_LONG": False,
            "FLIP_SHORT": True,
            "TREND": -1,
            "RSI": 33.0,
            "VOL_RATIO": 1.8,
            "ADX": 25.0,
            "ATR_14": 0.9,
            "ATR_PCT": 0.95,
            "close": 80.0,
            "EMA20": 81.5,
            "MACD_HIST": -0.28,
            "BBANDS_UP": 84.0,
            "BBANDS_MID": 80.0,
            "BBANDS_LOW": 76.0,
        },
        {
            "FLIP_LONG": False,
            "FLIP_SHORT": False,
            "TREND": 1,
            "RSI": 55.0,
            "VOL_RATIO": 1.0,
            "ADX": 14.0,
            "ATR_14": 0.3,
            "ATR_PCT": 0.25,
            "close": 50.0,
            "EMA20": 49.8,
            "MACD_HIST": 0.01,
            "BBANDS_UP": 50.8,
            "BBANDS_MID": 50.0,
            "BBANDS_LOW": 49.2,
        },
    ]


def test_thresholds_match_engine_signals() -> None:
    cfg = LegacyParitySettings()
    e_cfg = _engine_config()
    assert asdict(get_signal_thresholds(cfg, "candidate")) == engine_signals.get_signal_thresholds(e_cfg, "candidate")
    assert asdict(get_signal_thresholds(cfg, "auto")) == engine_signals.get_signal_thresholds(e_cfg, "auto")


@pytest.mark.parametrize("row", _rows())
def test_golden_parity_engine_signals_row_level(row: dict[str, object]) -> None:
    cfg = LegacyParitySettings()
    e_cfg = _engine_config()
    cand_t_engine = engine_signals.get_signal_thresholds(e_cfg, "candidate")
    auto_t_engine = engine_signals.get_signal_thresholds(e_cfg, "auto")
    cand_t = get_signal_thresholds(cfg, "candidate")
    auto_t = get_signal_thresholds(cfg, "auto")

    assert get_flip_candidate_signal(row) == engine_signals.get_flip_candidate_signal(row)

    candidate_signal = get_flip_candidate_signal(row)
    ours_cand_flags = evaluate_signal_filters(row, candidate_signal, cand_t)
    theirs_cand_flags = engine_signals.evaluate_signal_filters(row, candidate_signal, cand_t_engine)
    assert ours_cand_flags == theirs_cand_flags

    ours_auto_flags = evaluate_signal_filters(row, candidate_signal, auto_t)
    theirs_auto_flags = engine_signals.evaluate_signal_filters(row, candidate_signal, auto_t_engine)
    assert ours_auto_flags == theirs_auto_flags

    ours_candidate_score = score_from_flags(ours_cand_flags)
    theirs_candidate_score = engine_signals.hesapla_signal_score(row, config=e_cfg, signal_type=candidate_signal, thresholds=cand_t_engine)
    assert ours_candidate_score == theirs_candidate_score

    ours_candidate_power, _ = calculate_power_score(row, cand_t)
    theirs_candidate_power = engine_signals.hesapla_power_score(row, config=e_cfg, thresholds=cand_t_engine)
    assert ours_candidate_power == theirs_candidate_power

    ours_auto_power, _ = calculate_power_score(row, auto_t)
    theirs_auto_power = engine_signals.hesapla_power_score(row, config=e_cfg, thresholds=auto_t_engine)
    assert ours_auto_power == theirs_auto_power

    if candidate_signal in {"LONG", "SHORT"} and not ours_cand_flags["all_ok"]:
        assert get_candidate_fail_reason(row, candidate_signal, cfg) == engine_signals.get_candidate_fail_reason(row, candidate_signal, config=e_cfg)
