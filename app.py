# ====================== RBD-CRYPT v87.0 Quant Research Engine ======================
# v84.1'den v87.0'a degişiklikler:
#   - Tum 'except: pass' kaldirildi, her hata error_log.csv'ye yaziliyor
#   - MIN_ATR_PERCENT filtresi gercekten uygulaniyor
#   - score / power_score hesaplamalari gercek (walk-forward'a hazir)
#   - hunter_history.csv: 20+ yeni kolon (bar detayi, market context, indicator snapshot)
#   - all_signals.csv: genişletildi, her sinyalin tam fotografi
#   - market_context.csv: her scan'in BTC/piyasa durumu
#   - error_log.csv: sessiz hatalari yakaliyor
#   - Timeout cikişinda ST flip varsa onu bekle (MAX_HOLD + 2 bar tolerans)
#   - Cooldown sonrasi missed sinyal detayi loglaniyor
# ===================================================================================

import pandas as pd
import talib
import numpy as np
import requests
import os
import time
import json
import warnings
import io
import gc
import traceback
import concurrent.futures
import threading
import signal
import atexit
import base64
import zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from engine_metrics import (
    calc_real_mdd_from_trade_log as engine_calc_real_mdd_from_trade_log,
    get_advanced_metrics as engine_get_advanced_metrics,
    get_trade_performance_snapshot as engine_get_trade_performance_snapshot,
)
from engine_messages import (
    build_status_message as engine_build_status_message,
    format_hourly_report_message as engine_format_hourly_report_message,
)
from engine_notify import (
    create_daily_backup_zip as engine_create_daily_backup_zip,
    send_ntfy_file as engine_send_ntfy_file,
    send_ntfy_notification as engine_send_ntfy_notification,
    upload_backup_to_github as engine_upload_backup_to_github,
)
from engine_indicators import (
    _talib_ready_array as engine_talib_ready_array,
    hesapla_indikatorler as engine_hesapla_indikatorler,
)
from engine_signals import (
    evaluate_signal_filters as engine_evaluate_signal_filters,
    get_candidate_fail_reason as engine_get_candidate_fail_reason,
    get_flip_candidate_signal as engine_get_flip_candidate_signal,
    get_signal_thresholds as engine_get_signal_thresholds,
    hesapla_power_score as engine_hesapla_power_score,
    hesapla_signal_score as engine_hesapla_signal_score,
    score_from_flags as engine_score_from_flags,
    sinyal_kontrol as engine_sinyal_kontrol,
)
from engine_data_runtime import (
    get_btc_context as runtime_get_btc_context,
    get_live_futures_data as runtime_get_live_futures_data,
    get_top_futures_coins as runtime_get_top_futures_coins,
    safe_api_get as runtime_safe_api_get,
)
from engine_trade_logs_runtime import (
    log_market_context as runtime_log_market_context,
    log_potential_signal as runtime_log_potential_signal,
    log_trade_to_csv as runtime_log_trade_to_csv,
)
from engine_trade_positions_runtime import (
    create_trade_chart as runtime_create_trade_chart,
    maybe_open_paper_position as runtime_maybe_open_paper_position,
    update_paper_position_for_symbol as runtime_update_paper_position_for_symbol,
)
from engine_runtime_ui import (
    draw_fund_dashboard as runtime_draw_fund_dashboard,
    flush_state_on_shutdown as runtime_flush_state_on_shutdown,
    handle_termination as runtime_handle_termination,
    log_print as runtime_log_print,
    log_storage_diagnostics as runtime_log_storage_diagnostics,
    ntfy_komut_dinle as runtime_ntfy_komut_dinle,
)
from engine_cycle_main import run_bot_cycle as runtime_run_bot_cycle
from engine_core_state_runtime import (
    HunterState,
    ascii_only,
    calc_tp_sl_metrics,
    env_bool,
    get_tr_time,
    get_trading_day_str,
    get_trading_day_time,
    json_safe,
    log_error,
    rotate_logs,
)
from engine_notifications_runtime import (
    create_daily_backup_zip,
    format_hourly_report_message,
    gunluk_dump_gonder,
    send_ntfy_file,
    send_ntfy_notification,
    upload_backup_to_github,
)
from engine_bindings_runtime import (
    calc_real_mdd_from_trade_log as binding_calc_real_mdd_from_trade_log,
    get_advanced_metrics as binding_get_advanced_metrics,
    get_candidate_fail_reason as binding_get_candidate_fail_reason,
    get_signal_thresholds as binding_get_signal_thresholds,
    get_trade_performance_snapshot as binding_get_trade_performance_snapshot,
    hesapla_indikatorler as binding_hesapla_indikatorler,
    hesapla_power_score as binding_hesapla_power_score,
    hesapla_signal_score as binding_hesapla_signal_score,
    sinyal_kontrol as binding_sinyal_kontrol,
)
from engine_bootstrap_runtime import run_main_loop as runtime_run_main_loop
warnings.filterwarnings('ignore')

# env_bool imported from engine_core_state_runtime

# ==============================================================
# 1. AYAR PANELI
# ==============================================================
CONFIG = {
    "NTFY_TOPIC": "RBD-CRYPT",
    "BASE_PATH": os.getenv("DATA_PATH", "./bot_data"),
    "MAX_POSITIONS": 3,
    "STARTING_BALANCE": 100.0,
    "FIXED_TRADE_SIZE_USD": 25.0,
    "RISK_PERCENT_PER_TRADE": 25.0,
    "MIN_TRADE_SIZE": 10.0,
    "MAX_TRADE_SIZE": 200.0,
    "TOP_COINS_LIMIT": 100,
    "CANDIDATE_TOP_COINS_LIMIT": 100,
    "ST_M": 2.8,
    "RSI_PERIOD": 9,
    # Auto-entry (seçici) eşikleri
    "AUTO_ENTRY_RSI_LONG": 62,
    "AUTO_ENTRY_RSI_SHORT": 38,
    "AUTO_ENTRY_VOL_FILTER": 1.42,
    "AUTO_ENTRY_ADX_THRESHOLD": 22,
    "AUTO_ENTRY_MIN_ATR_PERCENT": 0.85,
    "AUTO_ENTRY_MIN_POWER_SCORE": 40,
    # Candidate shortlist (gevşek) eşikleri
    "CANDIDATE_RSI_LONG": 56,
    "CANDIDATE_RSI_SHORT": 44,
    "CANDIDATE_VOL_FILTER": 1.15,
    "CANDIDATE_ADX_THRESHOLD": 17,
    "CANDIDATE_MIN_ATR_PERCENT": 0.65,
    "CANDIDATE_MIN_POWER_SCORE": 28,
    # Market policy
    "AUTO_BTC_TREND_MODE": "hard_block",   # hard_block | soft_penalty
    "AUTO_BTC_TREND_PENALTY": 12.0,
    "AUTO_CHOP_POLICY": "block",           # block | penalty | allow
    "AUTO_CHOP_PENALTY": 8.0,
    "RSI_LONG": 62,
    "RSI_SHORT": 38,
    "VOL_FILTER": 1.42,
    "ADX_THRESHOLD": 22,
    "MIN_ATR_PERCENT": 0.85,        # Artik gercekten kullaniliyor
    "SL_M": 1.65,
    "TP_M": 2.55,
    "COOLDOWN_MINUTES": 20,
    "MAX_HOLD_MINUTES": 45,
    "MAX_HOLD_ST_GRACE_BARS": 2,    # YENI: timeout dolunca ST flip icin +2 bar tolerans
    "STALE_EXIT_MIN_PNL_PCT": 0.15,     # Max hold sonrasi bu seviyenin altinda ise zayif say
    "STALE_EXIT_MIN_BEST_PNL_PCT": 0.60,# Max hold boyunca hic ivme yoksa stale exit
    "CHOP_ADX_THRESHOLD": 18,
    "BTC_VOL_THRESHOLD": 0.18,
    "MIN_POWER_SCORE": 40,
    "ENABLE_SHUTDOWN_FLUSH": True,
    "GITHUB_BACKUP_ENABLED": env_bool("GITHUB_BACKUP_ENABLED", False),
    "GITHUB_BACKUP_REPO": os.getenv("GITHUB_BACKUP_REPO", ""),          # owner/repo
    "GITHUB_BACKUP_TOKEN": os.getenv("GITHUB_BACKUP_TOKEN", ""),
    "GITHUB_BACKUP_BRANCH": os.getenv("GITHUB_BACKUP_BRANCH", "main"),
    "GITHUB_BACKUP_DIR": os.getenv("GITHUB_BACKUP_DIR", "daily-backups"),
    "TARGET_MINUTES": [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56],
    "MAX_LOG_SIZE_BYTES": 50_000_000
}

FILES = {
    "LOG":             os.path.join(CONFIG["BASE_PATH"], "hunter_history.csv"),
    "ACTIVE":          os.path.join(CONFIG["BASE_PATH"], "active_trades.json"),
    "PAPER_ACTIVE":    os.path.join(CONFIG["BASE_PATH"], "paper_active_trades.json"),
    "ALL_SIGNALS":     os.path.join(CONFIG["BASE_PATH"], "all_signals.csv"),
    "MARKET_CONTEXT":  os.path.join(CONFIG["BASE_PATH"], "market_context.csv"),   # YENI
    "ERROR_LOG":       os.path.join(CONFIG["BASE_PATH"], "error_log.csv"),         # YENI
    "STATE":           os.path.join(CONFIG["BASE_PATH"], "engine_state.json")
}

if not os.path.exists(CONFIG["BASE_PATH"]):
    os.makedirs(CONFIG["BASE_PATH"], exist_ok=True)

# ==============================================================
# 2. HATA LOGLAMA (Sessiz hatalari yakala)
# ==============================================================
# Core helpers and state are imported from engine_core_state_runtime
state = HunterState()


# ==============================================================
# 5. BILDIRIM MODULU
# ==============================================================
# Notification/backup helpers are imported from engine_notifications_runtime

# ==============================================================
# 6. VERI cEKIMI
# ==============================================================
# Compact runtime bindings
safe_api_get = runtime_safe_api_get
get_top_futures_coins = runtime_get_top_futures_coins
get_live_futures_data = runtime_get_live_futures_data
_talib_ready_array = engine_talib_ready_array
hesapla_indikatorler = binding_hesapla_indikatorler
hesapla_power_score = binding_hesapla_power_score
hesapla_signal_score = binding_hesapla_signal_score
get_flip_candidate_signal = engine_get_flip_candidate_signal
get_candidate_fail_reason = binding_get_candidate_fail_reason
get_signal_thresholds = binding_get_signal_thresholds
evaluate_signal_filters = engine_evaluate_signal_filters
score_from_flags = engine_score_from_flags
sinyal_kontrol = binding_sinyal_kontrol
get_btc_context = runtime_get_btc_context
log_trade_to_csv = runtime_log_trade_to_csv
log_potential_signal = runtime_log_potential_signal
log_market_context = runtime_log_market_context
get_trade_performance_snapshot = binding_get_trade_performance_snapshot
get_advanced_metrics = binding_get_advanced_metrics
calc_real_mdd_from_trade_log = binding_calc_real_mdd_from_trade_log
create_trade_chart = runtime_create_trade_chart
maybe_open_paper_position = runtime_maybe_open_paper_position
update_paper_position_for_symbol = runtime_update_paper_position_for_symbol
log_print = runtime_log_print
log_storage_diagnostics = runtime_log_storage_diagnostics
flush_state_on_shutdown = runtime_flush_state_on_shutdown
_handle_termination = runtime_handle_termination
draw_fund_dashboard = runtime_draw_fund_dashboard
ntfy_komut_dinle = runtime_ntfy_komut_dinle
run_bot_cycle = runtime_run_bot_cycle


if __name__ == "__main__":
    runtime_run_main_loop()
