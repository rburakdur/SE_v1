from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from rbdcrypt.brokers.paper_broker import PaperBroker
from rbdcrypt.config import AppSettings
from rbdcrypt.models.signal import SignalDirection, SignalEvent
from rbdcrypt.services.trade_service import TradeService


def _logger():
    logger = logging.getLogger("test_trade_service_entry_policy")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def _base_settings() -> AppSettings:
    s = AppSettings()
    s.strategy.profile_name = "intraday_swing_v1_baseline"
    s.legacy_parity.enabled = True
    s.legacy_parity.cooldown_minutes = 20
    s.risk.max_active_positions = 3
    s.risk.min_rr = 0.0
    s.risk.fixed_notional_per_trade = 25.0
    s.risk.leverage = 1.0
    s.balance.starting_balance = 1000.0
    return s


def _candidate_signal(now: datetime, symbol: str = "ADAUSDT") -> SignalEvent:
    return SignalEvent(
        symbol=symbol,
        interval="5m",
        bar_time=now - timedelta(minutes=5),
        direction=SignalDirection.LONG,
        price=100.0,
        power_score=72.0,
        candidate_pass=True,
        auto_pass=False,
        meta={"entry_atr14": 1.0, "evaluator_outcome": "candidate"},
    )


def test_trade_service_opens_candidate_when_policy_allows(repos) -> None:
    now = datetime(2026, 2, 26, 12, 0, tzinfo=UTC)
    settings = _base_settings()

    service = TradeService.from_settings(
        settings=settings,
        broker=PaperBroker(),
        repos=repos,
        now_fn=lambda: now,
        logger=_logger(),
    )

    signal = _candidate_signal(now, "ADAUSDT")
    signal_id = repos.signals.insert_signal(signal)
    signal.meta["db_signal_id"] = signal_id

    result = service.handle_cycle(signals=[signal], prices_by_symbol={}, symbol_states={})
    assert result.opened == 1

    active = repos.positions.list_active()
    assert len(active) == 1
    assert active[0].meta["entry_layer"] == "candidate"
    assert active[0].meta["entry_policy_multiplier"] == 1.0
    assert active[0].meta["strategy_profile"] == "intraday_swing_v1_baseline"
    assert active[0].meta["execution_mode"] == "paper_dry_run"


def test_trade_service_blocks_candidate_when_auto_only_profile(repos) -> None:
    now = datetime(2026, 2, 26, 12, 0, tzinfo=UTC)
    settings = _base_settings()
    settings.strategy.profile_name = "intraday_swing_v1_auto_only_experimental"

    service = TradeService.from_settings(
        settings=settings,
        broker=PaperBroker(),
        repos=repos,
        now_fn=lambda: now,
        logger=_logger(),
    )

    signal = _candidate_signal(now, "XLMUSDT")
    signal_id = repos.signals.insert_signal(signal)
    signal.meta["db_signal_id"] = signal_id

    result = service.handle_cycle(signals=[signal], prices_by_symbol={}, symbol_states={})
    assert result.opened == 0
    assert result.skipped == 1
    assert result.missed_signals == 1

    with repos.signals.db.read_only() as conn:
        row = conn.execute(
            """
            SELECT stage, outcome, blocked_reason
            FROM signal_decisions
            WHERE signal_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (signal_id,),
        ).fetchone()
    assert row is not None
    assert row["stage"] == "entry_policy"
    assert row["outcome"] == "blocked"
    assert row["blocked_reason"] == "ENTRY_POLICY_CANDIDATE_DISABLED"


def test_trade_service_candidate_half_profile_scales_notional(repos) -> None:
    now = datetime(2026, 2, 26, 12, 0, tzinfo=UTC)
    settings = _base_settings()
    settings.strategy.profile_name = "intraday_swing_v1_candidate_half_experimental"

    service = TradeService.from_settings(
        settings=settings,
        broker=PaperBroker(),
        repos=repos,
        now_fn=lambda: now,
        logger=_logger(),
    )

    signal = _candidate_signal(now, "DOTUSDT")
    signal_id = repos.signals.insert_signal(signal)
    signal.meta["db_signal_id"] = signal_id

    result = service.handle_cycle(signals=[signal], prices_by_symbol={}, symbol_states={})
    assert result.opened == 1
    active = repos.positions.list_active()
    assert len(active) == 1
    assert abs(active[0].notional - 12.5) < 1e-9
    assert abs(active[0].meta["entry_policy_multiplier"] - 0.5) < 1e-9
