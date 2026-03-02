from __future__ import annotations

import json
from pathlib import Path

import pytest

from rbdcrypt.config import AppSettings
from rbdcrypt.strategy.profile_config import load_strategy_profile


def test_strategy_profile_baseline_loads() -> None:
    profile = load_strategy_profile(
        profile_name="intraday_swing_v1_baseline",
        profiles_dir=Path("strategies"),
    )
    assert profile.name == "intraday_swing_v1_baseline"
    assert profile.filters.session_filter is True
    assert profile.filters.candidate_min == 65.0
    assert profile.filters.auto_min == 65.0
    assert profile.filters.htf_bias.ma_type == "EMA"
    assert profile.filters.htf_bias.period == 50
    assert profile.filters.trigger_mode == "WT"
    assert profile.geometry.sl_m == 1.5
    assert profile.geometry.tp1_m == 1.0
    assert profile.geometry.tp2_m == 2.0
    assert profile.entry_policy.allow_candidate_entries is True
    assert profile.entry_policy.allow_auto_entries is True
    assert profile.entry_policy.candidate_lot_multiplier == 1.0
    assert profile.entry_policy.auto_lot_multiplier == 1.0
    assert profile.exit_policy.enable_session_exit is True


def test_strategy_profile_validation_error_is_clear(tmp_path: Path) -> None:
    broken = tmp_path / "broken_profile.json"
    broken.write_text(
        json.dumps(
            {
                "name": "broken_profile",
                "filters": {
                    "session_filter": True,
                    "candidate_min": 65.0,
                    "auto_min": 65.0,
                    "trigger_mode": "WT",
                    "ltf_trigger": "WT",
                    "htf_bias": {"timeframe": "1h", "ma_type": "EMA", "period": 50},
                },
                "entry_policy": {
                    "allow_candidate_entries": True,
                    "allow_auto_entries": True,
                    "candidate_lot_multiplier": 1.0,
                    "auto_lot_multiplier": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"Invalid strategy profile: .*broken_profile\.json"):
        load_strategy_profile(profile_name="broken_profile", profiles_dir=tmp_path)


def test_app_settings_load_strategy_profile_uses_strategy_block() -> None:
    settings = AppSettings(
        _env_file=None,
        strategy={
            "profile_name": "intraday_swing_v1_auto_only_experimental",
            "profiles_dir": "strategies",
        },
    )
    profile = settings.load_strategy_profile()
    assert profile.name == "intraday_swing_v1_auto_only_experimental"
    assert profile.entry_policy.allow_candidate_entries is False
    assert profile.entry_policy.allow_auto_entries is True


def test_strategy_profile_accepts_uppercase_keys(tmp_path: Path) -> None:
    prof = tmp_path / "uppercase_profile.json"
    prof.write_text(
        json.dumps(
            {
                "name": "uppercase_profile",
                "filters": {
                    "SESSION_FILTER": True,
                    "CANDIDATE_MIN": 65,
                    "AUTO_MIN": 65,
                    "TRIGGER_MODE": "WT",
                    "LTF_TRIGGER": "WT",
                    "htf_bias": {
                        "HTF_BIAS_MA_TYPE": "EMA",
                        "HTF_BIAS_PERIOD": 50,
                        "TIMEFRAME": "1h",
                    },
                },
                "geometry": {
                    "SL_M": 1.5,
                    "TP1_M": 1.0,
                    "TP2_M": 2.0,
                },
                "entry_policy": {
                    "ALLOW_CANDIDATE_ENTRIES": True,
                    "ALLOW_AUTO_ENTRIES": True,
                    "CANDIDATE_LOT_MULTIPLIER": 0.5,
                    "AUTO_LOT_MULTIPLIER": 1.0,
                },
                "exit_policy": {
                    "ENABLE_SESSION_EXIT": True,
                    "CANDIDATE_SPECIFIC_EXITS": False,
                },
            }
        ),
        encoding="utf-8",
    )

    parsed = load_strategy_profile(profile_name="uppercase_profile", profiles_dir=tmp_path)
    assert parsed.filters.candidate_min == 65.0
    assert parsed.geometry.tp2_m == 2.0
    assert parsed.entry_policy.candidate_lot_multiplier == 0.5


def test_strategy_profile_v2_default_loads_hma_and_session_window() -> None:
    profile = load_strategy_profile(
        profile_name="intraday_swing_v2_default",
        profiles_dir=Path("strategies"),
    )
    assert profile.filters.ltf_trigger == "WT_HMA_COMBO"
    assert profile.filters.htf_bias.ma_type == "HMA"
    assert profile.filters.session_no_entry_start_hour_utc == 1
    assert profile.filters.session_no_entry_end_hour_utc == 8
    assert profile.geometry.tp2_m == 2.3


def test_strategy_profile_rejects_partial_session_window(tmp_path: Path) -> None:
    prof = tmp_path / "bad_session_profile.json"
    prof.write_text(
        json.dumps(
            {
                "name": "bad_session_profile",
                "filters": {
                    "session_filter": True,
                    "candidate_min": 65,
                    "auto_min": 65,
                    "trigger_mode": "WT",
                    "ltf_trigger": "WT",
                    "session_no_entry_start_hour_utc": 1,
                    "htf_bias": {"timeframe": "1h", "ma_type": "EMA", "period": 50},
                },
                "geometry": {"sl_m": 1.5, "tp1_m": 1.0, "tp2_m": 2.0},
                "entry_policy": {
                    "allow_candidate_entries": True,
                    "allow_auto_entries": True,
                    "candidate_lot_multiplier": 1.0,
                    "auto_lot_multiplier": 1.0,
                },
                "exit_policy": {"enable_session_exit": True},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="session_no_entry_start_hour_utc and session_no_entry_end_hour_utc"):
        load_strategy_profile(profile_name="bad_session_profile", profiles_dir=tmp_path)
