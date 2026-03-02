from __future__ import annotations

from dataclasses import dataclass
from logging import Logger

from ..brokers.paper_broker import PaperBroker
from ..config import AppSettings, load_settings
from ..core.clock import SystemClock
from ..data.binance_client import BinanceClient
from ..data.market_fetcher import MarketFetcher
from ..data.rate_limit import SimpleRateLimiter
from ..notifications.ntfy_client import NtfyClient
from ..notifications.notification_service import NotificationService
from ..services.housekeeping_service import HousekeepingService
from ..services.backfill_service import BackfillService
from ..services.metrics_service import MetricsService
from ..services.replay_service import ReplayService
from ..services.scan_service import ScanService
from ..services.trade_service import TradeService
from ..storage.db import Database
from ..storage.migrations import apply_migrations
from ..storage.repositories import Repositories, build_repositories
from ..strategy.parity_signal_engine import ParitySignalEngine
from .logging_setup import get_component_logger, setup_logging


@dataclass(slots=True)
class RuntimeContainer:
    settings: AppSettings
    db: Database
    repos: Repositories
    binance_client: BinanceClient
    fetcher: MarketFetcher
    signal_engine: ParitySignalEngine
    scan_service: ScanService
    trade_service: TradeService
    backfill_service: BackfillService
    replay_service: ReplayService
    metrics_service: MetricsService
    housekeeping_service: HousekeepingService
    notification_service: NotificationService
    notifier: NtfyClient
    logger: Logger
    signals_logger: Logger
    trades_logger: Logger
    diagnostics_logger: Logger
    clock: SystemClock

    def close(self) -> None:
        self.binance_client.close()
        self.notifier.close()


def build_runtime(settings: AppSettings | None = None) -> RuntimeContainer:
    settings = settings or load_settings()
    system_logger = setup_logging(settings.logging)
    signals_logger = get_component_logger("signals")
    trades_logger = get_component_logger("trades")
    diagnostics_logger = get_component_logger("health")
    clock = SystemClock()

    db = Database(
        path=settings.storage.db_path,
        wal=settings.storage.wal,
        busy_timeout_ms=settings.storage.busy_timeout_ms,
    )
    apply_migrations(db)
    repos = build_repositories(db)
    strategy_profile = settings.load_strategy_profile()
    system_logger.info(
        "strategy_profile_loaded",
        extra={
            "event": {
                "profile_id": strategy_profile.name,
                "trigger_mode": strategy_profile.filters.ltf_trigger,
                "bias_mode": f"{strategy_profile.filters.htf_bias.ma_type}_{strategy_profile.filters.htf_bias.timeframe}",
            }
        },
    )
    repos.system_events.insert(
        event_type="runtime_start",
        level="info",
        details={
            "profile_id": strategy_profile.name,
            "trigger_mode": strategy_profile.filters.ltf_trigger,
            "bias_mode": f"{strategy_profile.filters.htf_bias.ma_type}_{strategy_profile.filters.htf_bias.timeframe}",
        },
    )

    binance_client = BinanceClient(settings.binance, settings.http_retry)
    fetcher = MarketFetcher(
        settings=settings,
        client=binance_client,
        rate_limiter=SimpleRateLimiter(settings.binance.request_min_interval_sec),
        logger=system_logger,
    )
    signal_engine = ParitySignalEngine(settings=settings, interval=settings.binance.interval)
    notifier = NtfyClient(settings.notifications)
    notification_cfg = settings.notifications
    system_logger.info(
        "notification_config_loaded",
        extra={
            "event": {
                "enabled": bool(notification_cfg.enabled),
                "max_priority": int(notification_cfg.max_priority),
                "topic": notification_cfg.topic,
                "url": notification_cfg.ntfy_url,
                "command_enabled": bool(notification_cfg.command_enabled),
                "command_topic": notification_cfg.command_topic,
                "notify_on_startup": bool(notification_cfg.notify_on_startup),
                "notify_on_open": bool(notification_cfg.notify_on_open),
                "notify_on_close": bool(notification_cfg.notify_on_close),
                "notify_on_cycle_summary": bool(notification_cfg.notify_on_cycle_summary),
            }
        },
    )
    if notification_cfg.enabled and (not notification_cfg.ntfy_url or not notification_cfg.topic):
        system_logger.warning(
            "notification_config_incomplete",
            extra={
                "event": {
                    "enabled": bool(notification_cfg.enabled),
                    "topic_present": bool(notification_cfg.topic),
                    "url_present": bool(notification_cfg.ntfy_url),
                }
            },
        )
    notification_service = NotificationService(
        notifier=notifier,
        logger=system_logger,
        now_fn=clock.now,
        state_store=repos.runtime_state,
        chart_enabled=bool(settings.runtime.chart_enabled),
    )

    scan_service = ScanService(
        settings=settings,
        fetcher=fetcher,
        signal_engine=signal_engine,
        repos=repos,
        now_fn=clock.now,
        logger=signals_logger,
        notifier=notifier,
        notification_service=notification_service,
    )
    trade_service = TradeService.from_settings(
        settings=settings,
        broker=PaperBroker(),
        repos=repos,
        now_fn=clock.now,
        logger=trades_logger,
        notifier=notifier,
        notification_service=notification_service,
    )
    metrics_service = MetricsService(repos=repos)
    housekeeping_service = HousekeepingService(settings=settings, repos=repos)
    backfill_service = BackfillService(settings=settings, fetcher=fetcher, repos=repos, logger=system_logger)
    replay_service = ReplayService(settings=settings, repos=repos, signal_engine=signal_engine, logger=system_logger)

    return RuntimeContainer(
        settings=settings,
        db=db,
        repos=repos,
        binance_client=binance_client,
        fetcher=fetcher,
        signal_engine=signal_engine,
        scan_service=scan_service,
        trade_service=trade_service,
        backfill_service=backfill_service,
        replay_service=replay_service,
        metrics_service=metrics_service,
        housekeeping_service=housekeeping_service,
        notification_service=notification_service,
        notifier=notifier,
        logger=system_logger,
        signals_logger=signals_logger,
        trades_logger=trades_logger,
        diagnostics_logger=diagnostics_logger,
        clock=clock,
    )
