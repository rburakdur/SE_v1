from __future__ import annotations

import traceback
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Callable

from ..brokers.base import BrokerInterface
from ..config import AppSettings
from ..core.portfolio import BalanceTracker
from ..core.policies import BalanceMode
from ..core.risk import build_risk_plan, build_risk_plan_from_levels
from ..core.state_machine import update_position_mark
from ..models.error_event import ErrorEvent
from ..models.position import ActivePosition
from ..models.signal import SignalDecision, SignalEvent
from ..models.symbol_state import SymbolBarState
from ..notifications.ntfy_client import NtfyClient
from ..notifications.notification_service import NotificationService
from ..storage.repositories import Repositories
from ..strategy.exit_engine import evaluate_exit, evaluate_legacy_exit


@dataclass(slots=True)
class TradeCycleResult:
    opened: int = 0
    closed: int = 0
    skipped: int = 0
    missed_signals: int = 0
    max_pos_blocked: int = 0


class TradeService:
    def __init__(
        self,
        *,
        settings: AppSettings,
        broker: BrokerInterface,
        repos: Repositories,
        balance_tracker: BalanceTracker,
        now_fn: Callable[[], datetime],
        logger,
        notifier: NtfyClient | None = None,
        notification_service: NotificationService | None = None,
    ) -> None:
        self.settings = settings
        self.broker = broker
        self.repos = repos
        self.balance_tracker = balance_tracker
        self.now_fn = now_fn
        self.logger = logger
        self.notifier = notifier
        self.notification_service = notification_service
        self._active_cache: dict[str, ActivePosition] = {}
        self._cooldowns: dict[str, datetime] = {}
        self._hourly_missed_signals: int = 0
        self._missed_hour_anchor: str | None = None
        self._cycle_scanned_count: int = 0

    @classmethod
    def from_settings(
        cls,
        *,
        settings: AppSettings,
        broker: BrokerInterface,
        repos: Repositories,
        now_fn: Callable[[], datetime],
        logger,
        notifier: NtfyClient | None = None,
        notification_service: NotificationService | None = None,
    ) -> "TradeService":
        mode = BalanceMode(settings.balance.resolved_mode())
        tracker = BalanceTracker(
            starting_balance=settings.balance.starting_balance,
            mode=mode,
        )
        saved_balance = repos.runtime_state.get_json("portfolio")
        if saved_balance and "balance" in saved_balance:
            tracker._snapshot.balance = float(saved_balance["balance"])  # noqa: SLF001
            tracker._snapshot.realized_pnl = float(saved_balance.get("realized_pnl", 0.0))  # noqa: SLF001
            if "day_anchor" in saved_balance:
                tracker._snapshot.day_anchor = datetime.fromisoformat(saved_balance["day_anchor"])  # noqa: SLF001
        instance = cls(
            settings=settings,
            broker=broker,
            repos=repos,
            balance_tracker=tracker,
            now_fn=now_fn,
            logger=logger,
            notifier=notifier,
            notification_service=notification_service,
        )
        cooldowns_payload = repos.runtime_state.get_json("cooldowns") or {}
        cooldowns: dict[str, datetime] = {}
        for sym, ts in cooldowns_payload.items():
            if not isinstance(ts, str):
                continue
            try:
                cooldowns[sym] = datetime.fromisoformat(ts)
            except ValueError:
                continue
        instance._cooldowns = cooldowns
        missed_payload = repos.runtime_state.get_json("trade_missed_counters") or {}
        instance._hourly_missed_signals = int(missed_payload.get("hourly_missed_signals", 0) or 0)
        anchor = missed_payload.get("hour_anchor")
        instance._missed_hour_anchor = str(anchor) if anchor else None
        return instance

    def recover_active_positions(self) -> list[ActivePosition]:
        now = self.now_fn()
        recovered: list[ActivePosition] = []
        for pos in self.repos.positions.list_active():
            if pos.recovered_at is None:
                pos.recovered_at = now
                pos.last_update_at = now
                self.repos.positions.upsert_active(pos)
            self._active_cache[pos.position_id] = pos
            recovered.append(pos)
        self.repos.runtime_state.set_json(
            "recovery",
            {"recovered_count": len(recovered), "at": now.isoformat(), "symbols": [p.symbol for p in recovered]},
        )
        self.logger.info("positions_recovered", extra={"event": {"count": len(recovered)}})
        if recovered and self.settings.notifications.notify_on_recovery:
            self._notify(
                "rbdcrypt: recovered positions",
                f"count={len(recovered)} symbols={','.join(sorted(p.symbol for p in recovered))}",
                priority=4,
                tags="warning",
            )
        return recovered

    def handle_cycle(
        self,
        *,
        signals: list[SignalEvent],
        prices_by_symbol: dict[str, float],
        symbol_states: dict[str, SymbolBarState] | None = None,
        scanned_count: int | None = None,
    ) -> TradeCycleResult:
        result = TradeCycleResult()
        now = self.now_fn()
        symbol_states = symbol_states or {}
        default_scanned = len(signals)
        resolved_scanned = scanned_count if scanned_count is not None else default_scanned
        self._cycle_scanned_count = max(0, int(resolved_scanned))
        self._reset_hourly_missed_if_needed(now)
        self._refresh_cache()
        self._manage_exits(now, prices_by_symbol, symbol_states, result)
        self._open_new_positions(now, signals, result)
        self._persist_portfolio(now)
        self._persist_cooldowns(now)
        self._persist_missed_counters(
            now,
            last_cycle=result.missed_signals,
            max_pos_blocked=result.max_pos_blocked,
        )
        self.repos.heartbeats.insert(
            component="trader",
            status="ok",
            meta={
                "opened": result.opened,
                "closed": result.closed,
                "skipped": result.skipped,
                "missed_signals": result.missed_signals,
                "max_pos_blocked": result.max_pos_blocked,
                "active_positions": len(self._active_cache),
            },
        )
        return result

    def _refresh_cache(self) -> None:
        db_positions = {p.position_id: p for p in self.repos.positions.list_active()}
        self._active_cache = db_positions

    def _manage_exits(
        self,
        now: datetime,
        prices_by_symbol: dict[str, float],
        symbol_states: dict[str, SymbolBarState],
        result: TradeCycleResult,
    ) -> None:
        for position in list(self._active_cache.values()):
            state = symbol_states.get(position.symbol)
            price = state.current_close if state is not None else prices_by_symbol.get(position.symbol)
            if price is None:
                continue
            update_position_mark(position, price, now)
            if self.settings.legacy_parity.enabled and state is not None:
                decision = evaluate_legacy_exit(
                    position=position,
                    current_high=state.current_high,
                    current_low=state.current_low,
                    current_close=state.current_close,
                    current_trend=state.current_trend,
                    current_ema20=state.current_ema20,
                    now=now,
                    legacy_cfg=self.settings.legacy_parity,
                )
            else:
                trend_flip = self._trend_flip_hint(position)
                decision = evaluate_exit(
                    position=position,
                    current_price=price,
                    now=now,
                    exit_cfg=self.settings.exit,
                    trend_flip=trend_flip,
                )
            if decision.should_exit and decision.reason:
                exit_price = decision.exit_price if decision.exit_price is not None else price
                trade = self.broker.close_position(
                    position=position,
                    exit_price=exit_price,
                    reason=decision.reason,
                    closed_at=now,
                    fee_pct_per_side=self.settings.risk.fee_pct_per_side,
                )
                self.repos.trades.insert_closed_and_remove_active(trade, position.position_id)
                self.balance_tracker.apply_realized_pnl(trade.pnl_quote, now)
                self._cooldowns[trade.symbol] = now
                self._active_cache.pop(position.position_id, None)
                result.closed += 1
                self.logger.info(
                    "position_closed",
                    extra={
                        "event": {
                            "symbol": trade.symbol,
                            "reason": trade.exit_reason,
                            "pnl_quote": trade.pnl_quote,
                            "pnl_pct": trade.pnl_pct,
                            "rr_initial": trade.rr_initial,
                            "initial_sl": trade.initial_sl,
                            "initial_tp": trade.initial_tp,
                            "current_sl": trade.current_sl,
                            "current_tp": trade.current_tp,
                        }
                    },
                )
                if self.settings.notifications.notify_on_close:
                    if self.notification_service is not None:
                        hold_min = max(0.0, (trade.closed_at - trade.opened_at).total_seconds() / 60.0)
                        self.notification_service.on_position_close(
                            symbol=trade.symbol,
                            side=trade.side.value,
                            entry_price=trade.entry_price,
                            exit_price=trade.exit_price,
                            pnl_pct=trade.pnl_pct * 100.0,
                            hold_minutes=hold_min,
                            active_positions=len(self._active_cache),
                            max_positions=int(self.settings.risk.max_active_positions),
                            pending_signals=result.max_pos_blocked,
                            reason=trade.exit_reason,
                        )
                    else:
                        hold_min = max(0.0, (trade.closed_at - trade.opened_at).total_seconds() / 60.0)
                        detail = self.settings.notifications.detail_level
                        if detail == "compact":
                            message = (
                                f"{trade.side.value.upper()} reason={trade.exit_reason} "
                                f"pnl_pct={trade.pnl_pct * 100:.2f}% pnl_quote={trade.pnl_quote:.4f} "
                                f"rr={trade.rr_initial:.2f}"
                            )
                        else:
                            message = (
                                f"{trade.side.value.upper()} reason={trade.exit_reason}\n"
                                f"entry={trade.entry_price:.6f} exit={trade.exit_price:.6f} hold_min={hold_min:.1f}\n"
                                f"pnl_pct={trade.pnl_pct * 100:.2f}% pnl_quote={trade.pnl_quote:.4f} fee={trade.fee_paid:.4f}\n"
                                f"rr_initial={trade.rr_initial:.2f} sl0={trade.initial_sl:.6f} tp0={trade.initial_tp:.6f}\n"
                                f"active_positions={len(self._active_cache)} balance={self.balance_tracker.balance:.2f}"
                            )
                        self._notify(
                            f"rbdcrypt: closed {trade.symbol}",
                            message,
                            priority=4,
                            tags="moneybag" if trade.pnl_quote >= 0 else "x",
                        )
            else:
                self.repos.positions.upsert_active(position)

    def _trend_flip_hint(self, position: ActivePosition) -> bool:
        # Runtime v1 hint: use sign of current pnl as a cheap proxy until full trend-state integration.
        return position.current_pnl_pct < -0.001 and position.best_pnl_pct > 0.003

    def _open_new_positions(self, now: datetime, signals: list[SignalEvent], result: TradeCycleResult) -> None:
        existing_symbols = {p.symbol for p in self._active_cache.values()}
        candidates = [s for s in signals if s.auto_pass and s.direction.value != "flat"]
        candidates.sort(key=lambda s: s.power_score, reverse=True)
        for signal in candidates:
            try:
                if signal.symbol in existing_symbols:
                    self._block_signal(signal, now, result, "ALREADY_IN", {"stage": "execution_filter"})
                    continue
                cooldown_blocked, cooldown_left = self._cooldown_status(signal.symbol, now)
                if cooldown_blocked:
                    self._block_signal(
                        signal,
                        now,
                        result,
                        f"COOLDOWN_{round(cooldown_left, 1)}m",
                        {"stage": "execution_filter", "cooldown_minutes_left": round(cooldown_left, 2)},
                    )
                    continue
                if len(self._active_cache) >= self.settings.risk.max_active_positions:
                    self._block_signal(signal, now, result, "MAX_POS", {"stage": "execution_filter"})
                    continue
                if self.repos.positions.count_active() >= self.settings.risk.max_active_positions:
                    self._block_signal(signal, now, result, "MAX_POS", {"stage": "execution_filter", "source": "db_guard"})
                    continue
                if self.settings.legacy_parity.enabled and signal.meta.get("entry_atr14") is not None:
                    atr_val = float(signal.meta["entry_atr14"])
                    if signal.direction.value == "long":
                        initial_sl = signal.price - (self.settings.legacy_parity.sl_atr_mult * atr_val)
                        initial_tp = signal.price + (self.settings.legacy_parity.tp_atr_mult * atr_val)
                    else:
                        initial_sl = signal.price + (self.settings.legacy_parity.sl_atr_mult * atr_val)
                        initial_tp = signal.price - (self.settings.legacy_parity.tp_atr_mult * atr_val)
                    risk_plan = build_risk_plan_from_levels(
                        balance=self.balance_tracker.balance,
                        risk_per_trade_pct=self.settings.risk.risk_per_trade_pct,
                        leverage=self.settings.risk.leverage,
                        entry_price=signal.price,
                        initial_sl=initial_sl,
                        initial_tp=initial_tp,
                        side=signal.direction.value,
                        min_notional=self.settings.risk.min_notional,
                    )
                else:
                    risk_plan = build_risk_plan(
                        balance=self.balance_tracker.balance,
                        risk_per_trade_pct=self.settings.risk.risk_per_trade_pct,
                        leverage=self.settings.risk.leverage,
                        entry_price=signal.price,
                        sl_pct=self.settings.exit.sl_pct,
                        tp_pct=self.settings.exit.tp_pct,
                        side=signal.direction.value,
                        min_notional=self.settings.risk.min_notional,
                    )
                if risk_plan.rr_initial < self.settings.risk.min_rr:
                    self._block_signal(
                        signal,
                        now,
                        result,
                        f"LOW_RR_{risk_plan.rr_initial:.2f}",
                        {"stage": "risk_filter", "rr_initial": risk_plan.rr_initial, "min_rr": self.settings.risk.min_rr},
                    )
                    continue
                position = self.broker.open_position(
                    symbol=signal.symbol,
                    side=signal.direction.value,
                    risk_plan=risk_plan,
                    opened_at=now,
                    entry_bar_time=signal.bar_time,
                    strategy_tag="v1_paper",
                )
                position.leverage = self.settings.risk.leverage
                self.repos.positions.upsert_active(position)
                self._active_cache[position.position_id] = position
                existing_symbols.add(position.symbol)
                result.opened += 1
                self._record_execution_decision(
                    signal=signal,
                    now=now,
                    outcome="opened",
                    blocked_reason=None,
                    payload={
                        "rr_initial": risk_plan.rr_initial,
                        "qty": position.qty,
                        "notional": position.notional,
                        "initial_sl": position.initial_sl,
                        "initial_tp": position.initial_tp,
                    },
                )
                self.logger.info(
                    "position_opened",
                    extra={
                        "event": {
                            "symbol": position.symbol,
                            "side": position.side.value,
                            "entry": position.entry_price,
                            "rr_initial": risk_plan.rr_initial,
                            "initial_sl": position.initial_sl,
                            "initial_tp": position.initial_tp,
                            "current_sl": position.current_sl,
                            "current_tp": position.current_tp,
                            "score": signal.power_score,
                        }
                    },
                )
                if self.settings.notifications.notify_on_open:
                    if self.notification_service is not None:
                        tp_target_pct, sl_risk_pct = self._tp_sl_targets_pct(position)
                        self.notification_service.on_position_open(
                            symbol=position.symbol,
                            side=position.side.value,
                            entry_price=position.entry_price,
                            tp_price=position.current_tp,
                            sl_price=position.current_sl,
                            tp_target_pct=tp_target_pct,
                            sl_risk_pct=sl_risk_pct,
                            hold_minutes=0.0,
                            current_pnl_pct=0.0,
                            active_positions=len(self._active_cache),
                            max_positions=int(self.settings.risk.max_active_positions),
                            pending_signals=result.max_pos_blocked,
                        )
                    else:
                        detail = self.settings.notifications.detail_level
                        if detail == "compact":
                            message = (
                                f"{position.side.value.upper()} entry={position.entry_price:.6f} "
                                f"rr={risk_plan.rr_initial:.2f} score={signal.power_score:.1f}"
                            )
                        else:
                            blocked = ",".join(signal.blocked_reasons) if signal.blocked_reasons else "-"
                            message = (
                                f"{position.side.value.upper()} entry={position.entry_price:.6f} bar={signal.bar_time.isoformat()}\n"
                                f"sl0={position.initial_sl:.6f} tp0={position.initial_tp:.6f} rr_initial={risk_plan.rr_initial:.2f}\n"
                                f"qty={position.qty:.6f} notional={position.notional:.2f} lev={position.leverage:.1f}\n"
                                f"score={signal.power_score:.1f} reject_stage={signal.meta.get('rejection_stage', '-')}"
                                f" blocked={blocked}\n"
                                f"active_positions={len(self._active_cache)} balance={self.balance_tracker.balance:.2f}"
                            )
                        self._notify(
                            f"rbdcrypt: opened {position.symbol}",
                            message,
                            priority=4,
                            tags="chart_with_upwards_trend",
                        )
            except Exception as exc:
                self._block_signal(
                    signal,
                    now,
                    result,
                    "OPEN_ERROR",
                    {"stage": "execution_filter", "error_type": exc.__class__.__name__},
                )
                self._record_error("trade_service.open", exc, {"symbol": signal.symbol})

    def _persist_portfolio(self, now: datetime) -> None:
        self.balance_tracker.maybe_reset_daily(now)
        payload = self.balance_tracker.serialize()
        self.repos.runtime_state.set_json("portfolio", payload)

    def _cooldown_status(self, symbol: str, now: datetime) -> tuple[bool, float]:
        if symbol not in self._cooldowns:
            return False, 0.0
        opened = self._cooldowns[symbol]
        age_min = (now - opened).total_seconds() / 60.0
        cooldown_min = float(self.settings.legacy_parity.cooldown_minutes)
        left = cooldown_min - age_min
        if left > 0:
            return True, left
        self._cooldowns.pop(symbol, None)
        return False, 0.0

    def _persist_cooldowns(self, now: datetime) -> None:
        # prune expired cooldowns before persisting
        for sym in list(self._cooldowns.keys()):
            self._cooldown_status(sym, now)
        self.repos.runtime_state.set_json(
            "cooldowns",
            {sym: ts.isoformat() for sym, ts in self._cooldowns.items()},
        )

    def _reset_hourly_missed_if_needed(self, now: datetime) -> None:
        anchor = now.astimezone(UTC).strftime("%Y-%m-%dT%H:00:00+00:00")
        if self._missed_hour_anchor != anchor:
            self._missed_hour_anchor = anchor
            self._hourly_missed_signals = 0

    def _persist_missed_counters(self, now: datetime, *, last_cycle: int, max_pos_blocked: int) -> None:
        self.repos.runtime_state.set_json(
            "trade_missed_counters",
            {
                "hour_anchor": self._missed_hour_anchor or now.astimezone(UTC).strftime("%Y-%m-%dT%H:00:00+00:00"),
                "hourly_missed_signals": self._hourly_missed_signals,
                "last_cycle_missed_signals": int(last_cycle),
                "last_cycle_max_pos_blocked": int(max_pos_blocked),
                "updated_at": now.isoformat(),
            },
        )

    @staticmethod
    def _tp_sl_targets_pct(position: ActivePosition) -> tuple[float, float]:
        if position.entry_price <= 0:
            return 0.0, 0.0
        if position.side.value == "short":
            tp_target = ((position.entry_price - position.current_tp) / position.entry_price) * 100.0
            sl_risk = ((position.current_sl - position.entry_price) / position.entry_price) * 100.0
            return tp_target, sl_risk
        tp_target = ((position.current_tp - position.entry_price) / position.entry_price) * 100.0
        sl_risk = ((position.entry_price - position.current_sl) / position.entry_price) * 100.0
        return tp_target, sl_risk

    def _block_signal(
        self,
        signal: SignalEvent,
        now: datetime,
        result: TradeCycleResult,
        reason: str,
        payload: dict[str, object] | None = None,
    ) -> None:
        result.skipped += 1
        result.missed_signals += 1
        if reason.startswith("MAX_POS"):
            result.max_pos_blocked += 1
        self._hourly_missed_signals += 1
        self._record_execution_decision(
            signal=signal,
            now=now,
            outcome="blocked",
            blocked_reason=reason,
            payload=payload or {},
        )
        self.logger.info(
            "signal_missed",
            extra={
                "event": {
                    "symbol": signal.symbol,
                    "reason": reason,
                    "power_score": signal.power_score,
                    "hourly_missed_signals": self._hourly_missed_signals,
                }
            },
        )
        if self.settings.notifications.notify_on_missed_signal:
            detail = self.settings.notifications.detail_level
            stage = signal.meta.get("rejection_stage", "-")
            if detail == "compact":
                message = f"reason={reason} score={signal.power_score:.1f}"
            else:
                message = (
                    f"reason={reason} stage={stage}\n"
                    f"score={signal.power_score:.1f} dir={signal.direction.value.upper()} bar={signal.bar_time.isoformat()}\n"
                    f"hourly_missed={self._hourly_missed_signals}"
                )
            self._notify(
                f"rbdcrypt: missed {signal.symbol}",
                message,
                priority=2,
                tags="warning",
            )

    def _record_execution_decision(
        self,
        *,
        signal: SignalEvent,
        now: datetime,
        outcome: str,
        blocked_reason: str | None,
        payload: dict[str, object],
    ) -> None:
        signal_id_raw = signal.meta.get("db_signal_id")
        signal_id = int(signal_id_raw) if isinstance(signal_id_raw, int | float) else None
        decision = SignalDecision(
            signal_id=signal_id,
            symbol=signal.symbol,
            bar_time=signal.bar_time,
            stage="execution_filter",
            outcome=outcome,
            blocked_reason=blocked_reason,
            decision_payload={
                "recorded_at": now.isoformat(),
                **payload,
            },
        )
        self.repos.signals.insert_decisions(signal_id, [decision])

    def _record_error(self, source: str, exc: Exception, context: dict[str, object]) -> None:
        tb_single = traceback.format_exc().replace("\n", "\\n")
        self.repos.errors.insert(
            ErrorEvent(
                source=source,
                error_type=exc.__class__.__name__,
                message=str(exc),
                traceback_single_line=tb_single,
                context=context,
            )
        )
        self.logger.error(
            "trade_error",
            extra={"event": {"source": source, "type": exc.__class__.__name__, "msg": str(exc), **context}},
        )
        if self.settings.notifications.notify_on_runtime_error:
            symbol_raw = context.get("symbol")
            symbol = str(symbol_raw) if isinstance(symbol_raw, str) and symbol_raw else "-"
            if self.notification_service is not None:
                self.notification_service.on_error(
                    source=source,
                    error=exc,
                    symbol=symbol,
                    pnl_pct=None,
                    active_positions=len(self._active_cache),
                    scanned_count=self._cycle_scanned_count,
                )
            else:
                self._notify(
                    "rbdcrypt: trade error",
                    f"{source} {exc.__class__.__name__}: {exc}",
                    priority=5,
                    tags="rotating_light",
                )

    def _notify(self, title: str, message: str, *, priority: int = 3, tags: str | None = None) -> None:
        if self.notifier is None:
            return
        try:
            self.notifier.notify(title, message, priority=priority, tags=tags)
        except Exception as exc:
            self.logger.error("ntfy_error", extra={"event": {"source": "trade_service", "msg": str(exc)}})
