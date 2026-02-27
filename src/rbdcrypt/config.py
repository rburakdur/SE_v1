from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrySettings(BaseModel):
    total: int = 4
    backoff_factor: float = 0.35
    jitter_max_sec: float = 0.2
    status_forcelist: tuple[int, ...] = (429, 500, 502, 503, 504)


class LoggingSettings(BaseModel):
    level: str = "INFO"
    jsonl: bool = True
    log_dir: Path = Path("bot_data/logs")
    log_file: str = "rbdcrypt.log"
    max_bytes: int = 4 * 1024 * 1024
    backup_count: int = 5

    @property
    def log_path(self) -> Path:
        return self.log_dir / self.log_file


class StorageSettings(BaseModel):
    db_path: Path = Path("bot_data/rbdcrypt.sqlite3")
    wal: bool = True
    busy_timeout_ms: int = 5000
    snapshot_enabled: bool = False
    snapshot_path: Path = Path("bot_data/snapshots/runtime_snapshot.json")


class HousekeepingSettings(BaseModel):
    signals_retention_days: int = 14
    decisions_retention_days: int = 14
    market_context_retention_days: int = 14
    errors_retention_days: int = 30
    heartbeats_retention_days: int = 7
    trades_retention_days: int = 365
    min_disk_free_mb: int = 256
    log_retention_days: int = 14


class BinanceSettings(BaseModel):
    base_url: str = "https://fapi.binance.com"
    interval: str = "5m"
    kline_limit: int = 250
    top_symbols_limit: int = 50
    quote_asset: str = "USDT"
    btc_symbol: str = "BTCUSDT"
    connect_timeout_sec: float = 5.0
    read_timeout_sec: float = 10.0
    request_min_interval_sec: float = 0.12


class RuntimeSettings(BaseModel):
    scan_interval_sec: PositiveInt = 60
    worker_count: PositiveInt = 3
    max_symbols: PositiveInt = 50
    chart_enabled: bool = False
    heavy_debug: bool = False
    enable_api_connectivity_check: bool = True
    one_shot: bool = False
    max_loop_errors_before_sleep: int = 5
    loop_error_sleep_sec: int = 30


class BalanceSettings(BaseModel):
    mode: Literal["cumulative", "daily_reset"] = "cumulative"
    starting_balance: float = 100.0
    daily_reset_enabled: bool = False

    def resolved_mode(self) -> Literal["cumulative", "daily_reset"]:
        if self.daily_reset_enabled and self.mode == "cumulative":
            return "daily_reset"
        return self.mode


class RiskSettings(BaseModel):
    risk_per_trade_pct: float = 0.01
    leverage: float = 1.0
    max_active_positions: int = 3
    fixed_notional_per_trade: float | None = 25.0
    min_rr: float = 1.2
    min_notional: float = 10.0
    fee_pct_per_side: float = 0.0004


class FilterSettings(BaseModel):
    adx_min: float = 18.0
    atr_pct_min: float = 0.002
    atr_pct_max_chop: float = 0.02
    rsi_long_min: float = 52.0
    rsi_short_max: float = 48.0
    volume_ratio_min: float = 1.05
    ema_fast_period: int = 21
    ema_slow_period: int = 55
    btc_trend_filter_mode: Literal["hard_block", "soft_penalty"] = "hard_block"
    chop_policy: Literal["block", "penalty", "allow"] = "penalty"


class ScoreSettings(BaseModel):
    candidate_score_min: float = 55.0
    auto_score_min: float = 72.0
    score_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "trend": 25.0,
            "momentum": 20.0,
            "volatility": 15.0,
            "volume": 15.0,
            "btc_alignment": 15.0,
            "anti_chop": 10.0,
        }
    )
    soft_penalty_btc_misalignment: float = 12.0
    chop_penalty: float = 10.0


class ExitSettings(BaseModel):
    tp_pct: float = 0.012
    sl_pct: float = 0.008
    max_hold_minutes: int = 180
    stale_minutes: int = 25
    break_even_trigger_pct: float = 0.006
    trend_flip_confirm_bars: int = 2


class LegacyParitySettings(BaseModel):
    enabled: bool = True
    rsi_period: int = 9
    supertrend_multiplier: float = 2.8
    supertrend_atr_period: int = 10
    allow_trend_continuation_entry: bool = False

    auto_entry_rsi_long: float = 62.0
    auto_entry_rsi_short: float = 38.0
    auto_entry_vol_filter: float = 1.42
    auto_entry_adx_threshold: float = 22.0
    auto_entry_min_atr_percent: float = 0.85
    auto_entry_min_power_score: float = 40.0

    candidate_rsi_long: float = 56.0
    candidate_rsi_short: float = 44.0
    candidate_vol_filter: float = 1.15
    candidate_adx_threshold: float = 17.0
    candidate_min_atr_percent: float = 0.65
    candidate_min_power_score: float = 28.0

    auto_btc_trend_mode: Literal["hard_block", "soft_penalty"] = "hard_block"
    auto_btc_trend_penalty: float = 12.0
    auto_chop_policy: Literal["block", "penalty", "allow"] = "block"
    auto_chop_penalty: float = 8.0
    chop_adx_threshold: float = 18.0
    btc_vol_threshold: float = 0.18  # ATR_PCT threshold (%)

    sl_atr_mult: float = 1.65
    tp_atr_mult: float = 2.55
    cooldown_minutes: int = 20
    max_hold_minutes: int = 45
    max_hold_st_grace_bars: int = 2
    stale_exit_min_pnl_pct: float = 0.15
    stale_exit_min_best_pnl_pct: float = 0.60


class NotificationSettings(BaseModel):
    ntfy_url: str | None = None
    topic: str | None = Field(default=None, validation_alias=AliasChoices("topic", "ntfy_topic"))
    enabled: bool = Field(default=False, validation_alias=AliasChoices("enabled", "ntfy_enabled"))
    timeout_sec: float = 4.0
    detail_level: Literal["compact", "detailed"] = "detailed"
    auto_signal_top_n: int = 5
    notify_on_cycle_summary: bool = False
    cycle_summary_min_interval_minutes: int = 30
    notify_on_startup: bool = True
    notify_on_recovery: bool = True
    notify_on_open: bool = True
    notify_on_close: bool = True
    notify_on_scan_degraded: bool = True
    notify_on_auto_signal_summary: bool = False
    notify_on_runtime_error: bool = True
    notify_on_missed_signal: bool = False


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
        frozen=False,
    )

    env: str = "dev"
    app_name: str = "rbdcrypt"
    timezone: str = "UTC"

    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    housekeeping: HousekeepingSettings = Field(default_factory=HousekeepingSettings)
    binance: BinanceSettings = Field(default_factory=BinanceSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    balance: BalanceSettings = Field(default_factory=BalanceSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    filters: FilterSettings = Field(default_factory=FilterSettings)
    score: ScoreSettings = Field(default_factory=ScoreSettings)
    exit: ExitSettings = Field(default_factory=ExitSettings)
    legacy_parity: LegacyParitySettings = Field(default_factory=LegacyParitySettings)
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)
    http_retry: RetrySettings = Field(default_factory=RetrySettings)

    @property
    def data_dir(self) -> Path:
        return self.storage.db_path.parent

    def ensure_runtime_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logging.log_dir.mkdir(parents=True, exist_ok=True)
        if self.storage.snapshot_enabled:
            self.storage.snapshot_path.parent.mkdir(parents=True, exist_ok=True)


def load_settings() -> AppSettings:
    env_file = resolve_env_file()
    settings = AppSettings(_env_file=env_file) if env_file else AppSettings()
    settings.ensure_runtime_dirs()
    return settings


def resolve_env_file() -> Path | None:
    """Resolve a deterministic .env file path for local and systemd runs."""
    candidates: list[Path] = []
    explicit = os.getenv("RBDCRYPT_ENV_FILE")
    if explicit:
        candidates.append(Path(explicit).expanduser())

    candidates.append(Path.cwd() / ".env")
    candidates.append(Path(__file__).resolve().parents[2] / ".env")

    seen: set[str] = set()
    for path in candidates:
        resolved = path.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.is_file():
            return resolved
    return None
