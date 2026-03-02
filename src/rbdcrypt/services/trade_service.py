from __future__ import annotations

import traceback
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable, Literal

from ..brokers.base import BrokerInterface
from ..config import AppSettings
from ..core.portfolio import BalanceTracker
from ..core.policies import BalanceMode
from ..core.risk import RiskPlan, build_risk_plan, build_risk_plan_from_levels
from ..core.state_machine import update_position_mark
from ..models.error_event import ErrorEvent
from ..models.position import ActivePosition
from ..models.signal import SignalDecision, SignalEvent
from ..models.symbol_state import SymbolBarState
from ..notifications.ntfy_client import NtfyClient
from ..notifications.notification_service import NotificationService
from ..storage.repositories import Repositories
from ..storage.snapshot_store import JsonSnapshotStore
from ..strategy.exit_engine import evaluate_exit
from ..strategy.profile_config import StrategyProfile
from ..strategy.runtime_exit_policy import RuntimeExitPolicyDecision, evaluate_runtime_exit_policy


@dataclass(slots=True)
class TradeCycleResult:
    opened: int = 0
    closed: int = 0
    skipped: int = 0
    missed_signals: int = 0
    max_pos_blocked: int = 0
    debounce_blocked: int = 0
    cooldown_blocked: int = 0
    startup_blocked: int = 0
    intent_lock_blocked: int = 0


@dataclass(slots=True)
class EntryPolicyResolution:
    entry_layer: Literal["candidate", "auto"] | None
    allowed: bool
    lot_multiplier: float
    blocked_reason: str | None = None


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
        broker_name = str(getattr(broker, "name", "")).lower()
        if broker_name != "paper":
            raise RuntimeError(
                "Live execution is disabled in Sprint 1. Use PaperBroker (dry-run/paper mode)."
            )
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
        self._processed_signal_keys: dict[str, datetime] = {}
        self._entry_intent_locks: set[str] = set()
        self._flap_counters: dict[str, int] = {
            "duplicate_signal_ignored": 0,
            "cooldown_blocked": 0,
            "startup_stabilization_blocked": 0,
            "intent_lock_blocked": 0,
        }
        self._snapshot_store = JsonSnapshotStore(self.settings.storage.snapshot_path)
        self.strategy_profile: StrategyProfile = self.settings.load_strategy_profile()

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
        if isinstance(saved_balance, dict):
            saved_starting_raw = saved_balance.get("starting_balance")
            try:
                saved_starting = float(saved_starting_raw) if saved_starting_raw is not None else None
            except (TypeError, ValueError):
                saved_starting = None
            if saved_starting is not None and saved_starting > 0:
                tracker.starting_balance = saved_starting

            if "balance" in saved_balance:
                tracker._snapshot.balance = float(saved_balance["balance"])  # noqa: SLF001
                tracker._snapshot.realized_pnl = float(saved_balance.get("realized_pnl", 0.0))  # noqa: SLF001
                day_anchor_raw = saved_balance.get("day_anchor")
                if isinstance(day_anchor_raw, str) and day_anchor_raw:
                    try:
                        tracker._snapshot.day_anchor = datetime.fromisoformat(day_anchor_raw)  # noqa: SLF001
                    except ValueError:
                        pass
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
        if not isinstance(saved_balance, dict):
            instance.repos.runtime_state.set_json("portfolio", instance.balance_tracker.serialize())
        snapshot_payload = instance._load_state_snapshot()
        snapshot_payload = instance._validated_snapshot_payload(snapshot_payload)

        cooldowns_payload = repos.runtime_state.get_json("cooldowns") or snapshot_payload.get("cooldowns") or {}
        instance._cooldowns = instance._parse_datetime_map(cooldowns_payload)

        processed_payload = (
            repos.runtime_state.get_json("processed_signal_keys")
            or snapshot_payload.get("processed_signal_keys")
            or {}
        )
        instance._processed_signal_keys = instance._parse_datetime_map(processed_payload)

        flap_payload = repos.runtime_state.get_json("flap_counters") or snapshot_payload.get("flap_counters") or {}
        for key in instance._flap_counters.keys():
            raw = flap_payload.get(key) if isinstance(flap_payload, dict) else None
            if isinstance(raw, int | float):
                instance._flap_counters[key] = max(0, int(raw))

        missed_payload = repos.runtime_state.get_json("trade_missed_counters") or {}
        instance._hourly_missed_signals = int(missed_payload.get("hourly_missed_signals", 0) or 0)
        anchor = missed_payload.get("hour_anchor")
        instance._missed_hour_anchor = str(anchor) if anchor else None
        return instance

    def recover_active_positions(self) -> list[ActivePosition]:
        now = self.now_fn()
        recovered: list[ActivePosition] = []
        duplicate_symbol_ids: list[str] = []
        kept_symbols: set[str] = set()
        self._active_cache = {}
        db_positions = self.repos.positions.list_active()
        for pos in db_positions:
            if pos.symbol in kept_symbols:
                duplicate_symbol_ids.append(pos.position_id)
                continue
            kept_symbols.add(pos.symbol)
            if pos.recovered_at is None:
                pos.recovered_at = now
                pos.last_update_at = now
                self.repos.positions.upsert_active(pos)
            self._active_cache[pos.position_id] = pos
            recovered.append(pos)

        for duplicate_id in duplicate_symbol_ids:
            self.repos.positions.delete_active(duplicate_id)
            self._active_cache.pop(duplicate_id, None)

        snapshot_payload = self._load_state_snapshot()
        snapshot_symbols = {
            str(item.get("symbol"))
            for item in snapshot_payload.get("active_positions", [])
            if isinstance(item, dict) and item.get("symbol")
        }
        db_symbols = {p.symbol for p in recovered}
        missing_in_db = sorted(snapshot_symbols - db_symbols)
        new_in_db = sorted(db_symbols - snapshot_symbols)
        healed = bool(duplicate_symbol_ids or missing_in_db or new_in_db)

        if duplicate_symbol_ids:
            self.logger.warning(
                "recovery_duplicate_symbol_healed",
                extra={
                    "event": {
                        "duplicate_position_ids": duplicate_symbol_ids,
                        "truth_source": "db",
                        "action": "deleted_duplicates",
                    }
                },
            )
        if missing_in_db or new_in_db:
            self.logger.warning(
                "state_recovery_mismatch_healed",
                extra={
                    "event": {
                        "missing_in_db": missing_in_db,
                        "new_in_db": new_in_db,
                        "truth_source": "db",
                    }
                },
            )

        self.repos.runtime_state.set_json(
            "recovery",
            {
                "recovered_count": len(recovered),
                "at": now.isoformat(),
                "symbols": [p.symbol for p in recovered],
                "duplicate_symbol_ids": duplicate_symbol_ids,
                "missing_in_db": missing_in_db,
                "new_in_db": new_in_db,
                "healed": healed,
                "truth_source": "db",
            },
        )
        self.repos.runtime_state.set_json(
            "active_position_index",
            {
                pos.symbol: {
                    "position_id": pos.position_id,
                    "opened_at": pos.opened_at.isoformat(),
                    "entry_layer": pos.meta.get("entry_layer"),
                }
                for pos in recovered
            },
        )
        self.logger.info("positions_recovered", extra={"event": {"count": len(recovered)}})
        self._persist_state_snapshot(now)
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
        chart_points_by_symbol: dict[str, list[float]] | None = None,
        symbol_states: dict[str, SymbolBarState] | None = None,
        scanned_count: int | None = None,
        allow_entries: bool = True,
        entry_block_reason: str | None = None,
    ) -> TradeCycleResult:
        result = TradeCycleResult()
        now = self.now_fn()
        symbol_states = symbol_states or {}
        chart_points_by_symbol = chart_points_by_symbol or {}
        default_scanned = len(signals)
        resolved_scanned = scanned_count if scanned_count is not None else default_scanned
        self._cycle_scanned_count = max(0, int(resolved_scanned))
        self._reset_hourly_missed_if_needed(now)
        self._prune_processed_signal_keys(now)
        self._refresh_cache()
        self._manage_exits(now, prices_by_symbol, chart_points_by_symbol, symbol_states, result)
        if allow_entries:
            self._open_new_positions(now, signals, chart_points_by_symbol, result)
        else:
            self._block_startup_entries(
                now=now,
                signals=signals,
                result=result,
                reason=entry_block_reason or "STARTUP_STABILIZATION_BLOCK",
            )
        self._persist_portfolio(now)
        self._persist_cooldowns(now)
        self._persist_processed_signal_keys(now)
        self._persist_missed_counters(
            now,
            last_cycle=result.missed_signals,
            max_pos_blocked=result.max_pos_blocked,
        )
        self._persist_flap_counters(now)
        self._persist_state_snapshot(now)
        self.repos.heartbeats.insert(
            component="trader",
            status="ok",
            meta={
                "opened": result.opened,
                "closed": result.closed,
                "skipped": result.skipped,
                "missed_signals": result.missed_signals,
                "max_pos_blocked": result.max_pos_blocked,
                "debounce_blocked": result.debounce_blocked,
                "cooldown_blocked": result.cooldown_blocked,
                "startup_blocked": result.startup_blocked,
                "intent_lock_blocked": result.intent_lock_blocked,
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
        chart_points_by_symbol: dict[str, list[float]],
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
                policy_decision = evaluate_runtime_exit_policy(
                    position=position,
                    state=state,
                    now=now,
                    strategy_profile=self.strategy_profile,
                    legacy_cfg=self.settings.legacy_parity,
                )
                if policy_decision.action == "partial_tp1":
                    self._record_exit_policy_decision(position=position, now=now, decision=policy_decision)
                    self._apply_tp1_partial(position=position, now=now, decision=policy_decision)
                    continue
                if policy_decision.action == "close" and policy_decision.reason:
                    self._record_exit_policy_decision(position=position, now=now, decision=policy_decision)
                    exit_price = policy_decision.exit_price if policy_decision.exit_price is not None else float(price)
                    self._close_position(
                        position=position,
                        now=now,
                        exit_price=exit_price,
                        reason=policy_decision.reason,
                        chart_points_by_symbol=chart_points_by_symbol,
                        max_pos_blocked=result.max_pos_blocked,
                    )
                    result.closed += 1
                    continue
                self.repos.positions.upsert_active(position)
                continue

            trend_flip = self._trend_flip_hint(position)
            decision = evaluate_exit(
                position=position,
                current_price=price,
                now=now,
                exit_cfg=self.settings.exit,
                trend_flip=trend_flip,
            )
            if decision.should_exit and decision.reason:
                exit_price = decision.exit_price if decision.exit_price is not None else float(price)
                self._close_position(
                    position=position,
                    now=now,
                    exit_price=exit_price,
                    reason=decision.reason,
                    chart_points_by_symbol=chart_points_by_symbol,
                    max_pos_blocked=result.max_pos_blocked,
                )
                result.closed += 1
            else:
                self.repos.positions.upsert_active(position)

    def _trend_flip_hint(self, position: ActivePosition) -> bool:
        # Runtime v1 hint: use sign of current pnl as a cheap proxy until full trend-state integration.
        return position.current_pnl_pct < -0.001 and position.best_pnl_pct > 0.003

    def _close_position(
        self,
        *,
        position: ActivePosition,
        now: datetime,
        exit_price: float,
        reason: str,
        chart_points_by_symbol: dict[str, list[float]],
        max_pos_blocked: int,
    ) -> None:
        trade = self.broker.close_position(
            position=position,
            exit_price=exit_price,
            reason=reason,
            closed_at=now,
            fee_pct_per_side=self.settings.risk.fee_pct_per_side,
        )
        trade.meta.update(
            {
                "profile_id": str(position.meta.get("strategy_profile", self.strategy_profile.name)),
                "trigger_mode": str(position.meta.get("trigger_mode", self.strategy_profile.filters.ltf_trigger)),
                "bias_mode": str(position.meta.get("bias_mode", self.strategy_profile.filters.htf_bias.ma_type)),
                "entry_layer": position.meta.get("entry_layer"),
                "execution_mode": "paper_dry_run",
            }
        )
        self.repos.trades.insert_closed_and_remove_active(trade, position.position_id)
        self.balance_tracker.apply_realized_pnl(trade.pnl_quote, now)
        self._cooldowns[trade.symbol] = now
        self._active_cache.pop(position.position_id, None)
        cooldown_min = self._symbol_cooldown_minutes()
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
                    "execution_mode": "paper_dry_run",
                    "profile_id": trade.meta.get("profile_id"),
                    "entry_layer": trade.meta.get("entry_layer"),
                    "cooldown_minutes": cooldown_min,
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
                    tp_price=trade.current_tp,
                    sl_price=trade.current_sl,
                    pnl_pct=trade.pnl_pct * 100.0,
                    hold_minutes=hold_min,
                    active_positions=len(self._active_cache),
                    max_positions=int(self.settings.risk.max_active_positions),
                    pending_signals=max_pos_blocked,
                    chart_points=chart_points_by_symbol.get(trade.symbol) if self.settings.runtime.chart_enabled else None,
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

    def _apply_tp1_partial(
        self,
        *,
        position: ActivePosition,
        now: datetime,
        decision: RuntimeExitPolicyDecision,
    ) -> None:
        tp1_price = decision.exit_price
        if tp1_price is None:
            return
        fraction = float(position.meta.get("tp1_partial_fraction", 0.5))
        fraction = min(0.99, max(0.01, fraction))
        closed_qty = position.qty * fraction
        closed_notional = position.notional * fraction
        remaining_qty = position.qty - closed_qty
        remaining_notional = position.notional - closed_notional
        if remaining_qty <= 0 or remaining_notional <= 0:
            return

        partial_position = position.model_copy(deep=True)
        partial_position.qty = closed_qty
        partial_position.notional = closed_notional
        partial_position.current_tp = float(tp1_price)
        trade = self.broker.close_position(
            position=partial_position,
            exit_price=float(tp1_price),
            reason="tp1_partial",
            closed_at=now,
            fee_pct_per_side=self.settings.risk.fee_pct_per_side,
        )
        trade.meta.update(
            {
                "profile_id": str(position.meta.get("strategy_profile", self.strategy_profile.name)),
                "trigger_mode": str(position.meta.get("trigger_mode", self.strategy_profile.filters.ltf_trigger)),
                "bias_mode": str(position.meta.get("bias_mode", self.strategy_profile.filters.htf_bias.ma_type)),
                "entry_layer": position.meta.get("entry_layer"),
                "execution_mode": "paper_dry_run",
            }
        )
        self.repos.trades.insert_closed(trade)
        self.balance_tracker.apply_realized_pnl(trade.pnl_quote, now)

        position.qty = remaining_qty
        position.notional = remaining_notional
        if position.side.value == "long" and position.current_sl < position.entry_price:
            position.current_sl = position.entry_price
        elif position.side.value == "short" and position.current_sl > position.entry_price:
            position.current_sl = position.entry_price

        exit_levels = position.meta.get("exit_levels")
        if isinstance(exit_levels, dict):
            tp2_level = exit_levels.get("tp2")
            if isinstance(tp2_level, int | float):
                position.current_tp = float(tp2_level)
        position.meta["tp1_done"] = True
        position.meta["tp1_executed_at"] = now.isoformat()
        position.meta["tp1_trade_id"] = trade.trade_id
        position.last_update_at = now
        self.repos.positions.upsert_active(position)
        self.logger.info(
            "position_tp1_partial",
            extra={
                "event": {
                    "symbol": position.symbol,
                    "position_id": position.position_id,
                    "exit_price": float(tp1_price),
                    "closed_qty": closed_qty,
                    "remaining_qty": remaining_qty,
                    "closed_notional": closed_notional,
                    "remaining_notional": remaining_notional,
                    "execution_mode": "paper_dry_run",
                }
            },
        )

    def _record_exit_policy_decision(
        self,
        *,
        position: ActivePosition,
        now: datetime,
        decision: RuntimeExitPolicyDecision,
    ) -> None:
        decision_row = SignalDecision(
            signal_id=None,
            symbol=position.symbol,
            bar_time=now,
            stage="exit_policy",
            outcome=decision.action,
            blocked_reason=decision.reason,
            decision_payload={
                "position_id": position.position_id,
                "recorded_at": now.isoformat(),
                **decision.payload,
            },
        )
        self.repos.signals.insert_decisions(None, [decision_row])

    def _open_new_positions(
        self,
        now: datetime,
        signals: list[SignalEvent],
        chart_points_by_symbol: dict[str, list[float]],
        result: TradeCycleResult,
    ) -> None:
        existing_symbols = {p.symbol for p in self._active_cache.values()}
        policy_candidates: list[tuple[SignalEvent, EntryPolicyResolution, str]] = []
        for signal in signals:
            if signal.direction.value == "flat":
                continue
            policy = self._resolve_entry_policy(signal)
            if policy.entry_layer is None:
                continue
            if not policy.allowed:
                self._block_signal(
                    signal,
                    now,
                    result,
                    policy.blocked_reason or "ENTRY_POLICY_BLOCKED",
                    {
                        "stage": "entry_policy",
                        "entry_layer": policy.entry_layer,
                        "strategy_profile": self.strategy_profile.name,
                        "lot_multiplier": policy.lot_multiplier,
                    },
                    stage="entry_policy",
                )
                continue
            signal_key = self._signal_key(signal)
            if self._is_signal_key_processed(signal_key):
                result.debounce_blocked += 1
                self._increment_flap_counter("duplicate_signal_ignored")
                self._block_signal(
                    signal,
                    now,
                    result,
                    "DUPLICATE_SIGNAL_IGNORED",
                    {
                        "stage": "debounce",
                        "signal_key": signal_key,
                        "strategy_profile": self.strategy_profile.name,
                    },
                    stage="debounce",
                    increment_missed=False,
                )
                continue
            policy_candidates.append((signal, policy, signal_key))

        policy_candidates.sort(key=lambda item: item[0].power_score, reverse=True)
        for signal, policy, signal_key in policy_candidates:
            self._mark_signal_key_processed(signal_key, now)
            try:
                self._record_execution_decision(
                    signal=signal,
                    now=now,
                    stage="entry_policy",
                    outcome="allowed",
                    blocked_reason=None,
                    payload={
                        "entry_layer": policy.entry_layer,
                        "lot_multiplier": policy.lot_multiplier,
                        "signal_key": signal_key,
                        "strategy_profile": self.strategy_profile.name,
                        "execution_mode": "paper_dry_run",
                    },
                )
                if signal.symbol in existing_symbols:
                    self._block_signal(signal, now, result, "ALREADY_IN", {"stage": "execution_filter"})
                    continue
                if signal.symbol in self._entry_intent_locks:
                    result.intent_lock_blocked += 1
                    self._increment_flap_counter("intent_lock_blocked")
                    self._block_signal(
                        signal,
                        now,
                        result,
                        "INTENT_LOCK_BLOCKED",
                        {
                            "stage": "intent_lock",
                            "symbol": signal.symbol,
                            "signal_key": signal_key,
                        },
                        stage="intent_lock",
                        increment_missed=False,
                    )
                    continue

                self._entry_intent_locks.add(signal.symbol)
                try:
                    if signal.symbol in existing_symbols:
                        self._block_signal(signal, now, result, "ALREADY_IN", {"stage": "execution_filter"})
                        continue

                    cooldown_blocked, cooldown_left = self._cooldown_status(signal.symbol, now)
                    if cooldown_blocked:
                        result.cooldown_blocked += 1
                        self._increment_flap_counter("cooldown_blocked")
                        self._block_signal(
                            signal,
                            now,
                            result,
                            f"COOLDOWN_{round(cooldown_left, 1)}m",
                            {
                                "stage": "execution_filter",
                                "cooldown_minutes_left": round(cooldown_left, 2),
                                "cooldown_minutes_config": self._symbol_cooldown_minutes(),
                            },
                        )
                        continue
                    if len(self._active_cache) >= self.settings.risk.max_active_positions:
                        self._block_signal(signal, now, result, "MAX_POS", {"stage": "execution_filter"})
                        continue
                    if self.repos.positions.count_active() >= self.settings.risk.max_active_positions:
                        self._block_signal(
                            signal,
                            now,
                            result,
                            "MAX_POS",
                            {"stage": "execution_filter", "source": "db_guard"},
                        )
                        continue

                    runtime_exit_levels: dict[str, float] | None = None
                    if self.settings.legacy_parity.enabled and signal.meta.get("entry_atr14") is not None:
                        atr_val = float(signal.meta["entry_atr14"])
                        geom = self.strategy_profile.geometry
                        if signal.direction.value == "long":
                            initial_sl = signal.price - (float(geom.sl_m) * atr_val)
                            tp1_level = signal.price + (float(geom.tp1_m) * atr_val)
                            initial_tp = signal.price + (float(geom.tp2_m) * atr_val)
                        else:
                            initial_sl = signal.price + (float(geom.sl_m) * atr_val)
                            tp1_level = signal.price - (float(geom.tp1_m) * atr_val)
                            initial_tp = signal.price - (float(geom.tp2_m) * atr_val)
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
                        runtime_exit_levels = {
                            "sl": float(initial_sl),
                            "tp1": float(tp1_level),
                            "tp2": float(initial_tp),
                        }
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
                        runtime_exit_levels = self._derive_exit_levels_from_risk_plan(
                            signal=signal,
                            risk_plan=risk_plan,
                        )
                    risk_plan = self._apply_fixed_notional(risk_plan)
                    risk_plan = self._apply_lot_multiplier(
                        risk_plan,
                        lot_multiplier=policy.lot_multiplier,
                    )
                    if risk_plan.notional < float(self.settings.risk.min_notional):
                        self._block_signal(
                            signal,
                            now,
                            result,
                            "ENTRY_POLICY_LOW_NOTIONAL",
                            {
                                "stage": "entry_policy",
                                "entry_layer": policy.entry_layer,
                                "notional_after_multiplier": risk_plan.notional,
                                "min_notional": float(self.settings.risk.min_notional),
                                "lot_multiplier": policy.lot_multiplier,
                            },
                            stage="entry_policy",
                        )
                        continue
                    if risk_plan.rr_initial < self.settings.risk.min_rr:
                        self._block_signal(
                            signal,
                            now,
                            result,
                            f"LOW_RR_{risk_plan.rr_initial:.2f}",
                            {
                                "stage": "risk_filter",
                                "rr_initial": risk_plan.rr_initial,
                                "min_rr": self.settings.risk.min_rr,
                            },
                        )
                        continue

                    position = self.broker.open_position(
                        symbol=signal.symbol,
                        side=signal.direction.value,
                        risk_plan=risk_plan,
                        opened_at=now,
                        entry_bar_time=signal.bar_time,
                        strategy_tag=f"{self.strategy_profile.name}:{policy.entry_layer}:paper",
                    )
                    position.leverage = self.settings.risk.leverage
                    if runtime_exit_levels is not None:
                        position.initial_sl = float(runtime_exit_levels["sl"])
                        position.initial_tp = float(runtime_exit_levels["tp2"])
                        position.current_sl = float(runtime_exit_levels["sl"])
                        position.current_tp = float(runtime_exit_levels["tp2"])
                    position.meta.update(
                        {
                            "entry_layer": policy.entry_layer,
                            "entry_policy_multiplier": float(policy.lot_multiplier),
                            "strategy_profile": self.strategy_profile.name,
                            "profile_id": self.strategy_profile.name,
                            "trigger_mode": self.strategy_profile.filters.ltf_trigger,
                            "bias_mode": f"{self.strategy_profile.filters.htf_bias.ma_type}_{self.strategy_profile.filters.htf_bias.timeframe}",
                            "evaluator_outcome": signal.meta.get("evaluator_outcome"),
                            "execution_mode": "paper_dry_run",
                            "exit_levels": runtime_exit_levels
                            or {
                                "sl": float(position.current_sl),
                                "tp1": None,
                                "tp2": float(position.current_tp),
                            },
                            "tp1_done": False,
                            "tp1_partial_fraction": 0.5,
                            "candidate_specific_exits_enabled": bool(self.strategy_profile.exit_policy.candidate_specific_exits),
                            "session_exit_enabled": bool(self.strategy_profile.exit_policy.enable_session_exit),
                        }
                    )
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
                            "entry_layer": policy.entry_layer,
                            "lot_multiplier": policy.lot_multiplier,
                            "strategy_profile": self.strategy_profile.name,
                            "execution_mode": "paper_dry_run",
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
                                "entry_layer": policy.entry_layer,
                                "lot_multiplier": policy.lot_multiplier,
                                "strategy_profile": self.strategy_profile.name,
                                "profile_id": self.strategy_profile.name,
                                "execution_mode": "paper_dry_run",
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
                                chart_points=chart_points_by_symbol.get(position.symbol)
                                if self.settings.runtime.chart_enabled
                                else None,
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
                finally:
                    self._entry_intent_locks.discard(signal.symbol)
            except Exception as exc:
                self._block_signal(
                    signal,
                    now,
                    result,
                    "OPEN_ERROR",
                    {"stage": "execution_filter", "error_type": exc.__class__.__name__},
                )
                self._record_error("trade_service.open", exc, {"symbol": signal.symbol})

    def _block_startup_entries(
        self,
        *,
        now: datetime,
        signals: list[SignalEvent],
        result: TradeCycleResult,
        reason: str,
    ) -> None:
        for signal in signals:
            if signal.direction.value == "flat":
                continue
            policy = self._resolve_entry_policy(signal)
            if policy.entry_layer is None or not policy.allowed:
                continue
            signal_key = self._signal_key(signal)
            self._mark_signal_key_processed(signal_key, now)
            result.startup_blocked += 1
            self._increment_flap_counter("startup_stabilization_blocked")
            self._block_signal(
                signal,
                now,
                result,
                reason,
                {
                    "stage": "startup_stabilization",
                    "entry_layer": policy.entry_layer,
                    "signal_key": signal_key,
                },
                stage="startup_stabilization",
                increment_missed=False,
            )

    def _persist_portfolio(self, now: datetime) -> None:
        self.balance_tracker.maybe_reset_daily(now)
        payload = self.balance_tracker.serialize()
        self.repos.runtime_state.set_json("portfolio", payload)

    def _cooldown_status(self, symbol: str, now: datetime) -> tuple[bool, float]:
        if symbol not in self._cooldowns:
            return False, 0.0
        opened = self._cooldowns[symbol]
        opened_ref = opened if opened.tzinfo else opened.replace(tzinfo=UTC)
        now_ref = now if now.tzinfo else now.replace(tzinfo=UTC)
        age_min = (now_ref - opened_ref).total_seconds() / 60.0
        cooldown_min = self._symbol_cooldown_minutes()
        left = cooldown_min - age_min
        if left > 0:
            return True, left
        self._cooldowns.pop(symbol, None)
        return False, 0.0

    def _symbol_cooldown_minutes(self) -> float:
        profile_cooldown = float(self.strategy_profile.entry_policy.symbol_cooldown_minutes)
        if profile_cooldown > 0:
            return profile_cooldown
        fallback = float(self.settings.legacy_parity.cooldown_minutes)
        return max(0.0, fallback)

    def _persist_cooldowns(self, now: datetime) -> None:
        # prune expired cooldowns before persisting
        for sym in list(self._cooldowns.keys()):
            self._cooldown_status(sym, now)
        self.repos.runtime_state.set_json(
            "cooldowns",
            {sym: ts.isoformat() for sym, ts in self._cooldowns.items()},
        )

    def _persist_processed_signal_keys(self, now: datetime) -> None:
        self._prune_processed_signal_keys(now)
        self.repos.runtime_state.set_json(
            "processed_signal_keys",
            {key: ts.isoformat() for key, ts in self._processed_signal_keys.items()},
        )

    def _persist_flap_counters(self, now: datetime) -> None:
        self.repos.runtime_state.set_json(
            "flap_counters",
            {
                **self._flap_counters,
                "updated_at": now.isoformat(),
            },
        )

    def _persist_state_snapshot(self, now: datetime) -> None:
        try:
            runtime_db = getattr(getattr(self.repos, "runtime_state", None), "db", None)
            runtime_db_path = getattr(runtime_db, "path", self.settings.storage.db_path)
            payload = {
                "saved_at": now.isoformat(),
                "db_path": str(Path(runtime_db_path).resolve()),
                "profile_id": self.strategy_profile.name,
                "active_positions": [
                    {
                        "position_id": pos.position_id,
                        "symbol": pos.symbol,
                        "opened_at": pos.opened_at.isoformat(),
                        "entry_layer": pos.meta.get("entry_layer"),
                    }
                    for pos in self._active_cache.values()
                ],
                "cooldowns": {sym: ts.isoformat() for sym, ts in self._cooldowns.items()},
                "processed_signal_keys": {key: ts.isoformat() for key, ts in self._processed_signal_keys.items()},
                "flap_counters": dict(self._flap_counters),
            }
            self._snapshot_store.save(payload)
        except Exception as exc:
            self.logger.error(
                "state_snapshot_save_error",
                extra={"event": {"msg": str(exc)}},
            )

    def _load_state_snapshot(self) -> dict[str, object]:
        try:
            payload = self._snapshot_store.load()
        except Exception as exc:
            self.logger.error(
                "state_snapshot_load_error",
                extra={"event": {"msg": str(exc)}},
            )
            return {}
        if isinstance(payload, dict):
            return payload
        return {}

    def _validated_snapshot_payload(self, payload: dict[str, object]) -> dict[str, object]:
        if not payload:
            return {}
        raw_db_path = payload.get("db_path")
        if not isinstance(raw_db_path, str) or not raw_db_path.strip():
            self.logger.info(
                "state_snapshot_ignored",
                extra={"event": {"reason": "missing_db_path"}},
            )
            return {}
        try:
            snapshot_db = str(Path(raw_db_path).resolve())
            runtime_db = getattr(getattr(self.repos, "runtime_state", None), "db", None)
            runtime_db_path = getattr(runtime_db, "path", self.settings.storage.db_path)
            current_db = str(Path(runtime_db_path).resolve())
        except Exception:
            self.logger.info(
                "state_snapshot_ignored",
                extra={"event": {"reason": "invalid_db_path"}},
            )
            return {}
        if snapshot_db != current_db:
            self.logger.info(
                "state_snapshot_ignored",
                extra={
                    "event": {
                        "reason": "db_path_mismatch",
                        "snapshot_db_path": snapshot_db,
                        "current_db_path": current_db,
                    }
                },
            )
            return {}
        return payload

    @staticmethod
    def _parse_datetime_map(payload: object) -> dict[str, datetime]:
        if not isinstance(payload, dict):
            return {}
        out: dict[str, datetime] = {}
        for key, raw_value in payload.items():
            if not isinstance(key, str) or not isinstance(raw_value, str):
                continue
            try:
                parsed = datetime.fromisoformat(raw_value)
                out[key] = parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
            except ValueError:
                continue
        return out

    def _prune_processed_signal_keys(self, now: datetime) -> None:
        now_ref = now if now.tzinfo else now.replace(tzinfo=UTC)
        retention_minutes = max(1, int(self.settings.runtime.processed_signal_retention_minutes))
        cutoff = now_ref - timedelta(minutes=retention_minutes)
        for key in list(self._processed_signal_keys.keys()):
            ts = self._processed_signal_keys[key]
            if ts < cutoff:
                self._processed_signal_keys.pop(key, None)

        max_keys = max(100, int(self.settings.runtime.processed_signal_max_keys))
        if len(self._processed_signal_keys) <= max_keys:
            return
        oldest = sorted(self._processed_signal_keys.items(), key=lambda item: item[1])
        overflow = len(self._processed_signal_keys) - max_keys
        for key, _ in oldest[:overflow]:
            self._processed_signal_keys.pop(key, None)

    def _is_signal_key_processed(self, signal_key: str) -> bool:
        return signal_key in self._processed_signal_keys

    def _mark_signal_key_processed(self, signal_key: str, now: datetime) -> None:
        self._processed_signal_keys[signal_key] = now if now.tzinfo else now.replace(tzinfo=UTC)

    @staticmethod
    def _signal_key(signal: SignalEvent) -> str:
        bar_time = signal.bar_time.astimezone(UTC) if signal.bar_time.tzinfo else signal.bar_time.replace(tzinfo=UTC)
        return f"{signal.symbol}|{bar_time.isoformat()}|{signal.direction.value}"

    def _increment_flap_counter(self, key: str) -> None:
        if key not in self._flap_counters:
            self._flap_counters[key] = 0
        self._flap_counters[key] += 1

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
                "debounce_blocked_total": int(self._flap_counters.get("duplicate_signal_ignored", 0)),
                "cooldown_blocked_total": int(self._flap_counters.get("cooldown_blocked", 0)),
                "startup_blocked_total": int(self._flap_counters.get("startup_stabilization_blocked", 0)),
                "intent_lock_blocked_total": int(self._flap_counters.get("intent_lock_blocked", 0)),
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

    def _apply_fixed_notional(self, plan: RiskPlan) -> RiskPlan:
        fixed = self.settings.risk.fixed_notional_per_trade
        if fixed is None:
            return plan
        fixed_value = float(fixed)
        if fixed_value <= 0:
            return plan
        leverage = max(float(self.settings.risk.leverage), 1e-12)
        qty = fixed_value * leverage / max(plan.entry_price, 1e-12)
        return RiskPlan(
            qty=qty,
            notional=fixed_value,
            entry_price=plan.entry_price,
            initial_sl=plan.initial_sl,
            initial_tp=plan.initial_tp,
            rr_initial=plan.rr_initial,
        )

    def _apply_lot_multiplier(self, plan: RiskPlan, *, lot_multiplier: float) -> RiskPlan:
        mult = float(lot_multiplier)
        if abs(mult - 1.0) <= 1e-12:
            return plan
        return RiskPlan(
            qty=plan.qty * mult,
            notional=plan.notional * mult,
            entry_price=plan.entry_price,
            initial_sl=plan.initial_sl,
            initial_tp=plan.initial_tp,
            rr_initial=plan.rr_initial,
        )

    def _resolve_entry_policy(self, signal: SignalEvent) -> EntryPolicyResolution:
        entry_layer = self._determine_entry_layer(signal)
        if entry_layer is None:
            return EntryPolicyResolution(entry_layer=None, allowed=False, lot_multiplier=1.0, blocked_reason=None)

        entry_cfg = self.strategy_profile.entry_policy
        if entry_layer == "auto":
            allowed = bool(entry_cfg.allow_auto_entries)
            lot_multiplier = float(entry_cfg.auto_lot_multiplier)
            blocked_reason = None if allowed else "ENTRY_POLICY_AUTO_DISABLED"
        else:
            allowed = bool(entry_cfg.allow_candidate_entries)
            lot_multiplier = float(entry_cfg.candidate_lot_multiplier)
            blocked_reason = None if allowed else "ENTRY_POLICY_CANDIDATE_DISABLED"

        if allowed and lot_multiplier <= 0:
            return EntryPolicyResolution(
                entry_layer=entry_layer,
                allowed=False,
                lot_multiplier=lot_multiplier,
                blocked_reason="ENTRY_POLICY_INVALID_MULTIPLIER",
            )
        return EntryPolicyResolution(
            entry_layer=entry_layer,
            allowed=allowed,
            lot_multiplier=lot_multiplier,
            blocked_reason=blocked_reason,
        )

    @staticmethod
    def _determine_entry_layer(signal: SignalEvent) -> Literal["candidate", "auto"] | None:
        outcome_raw = signal.meta.get("evaluator_outcome")
        if isinstance(outcome_raw, str):
            outcome = outcome_raw.lower()
            if outcome == "auto" and signal.auto_pass:
                return "auto"
            if outcome == "candidate" and signal.candidate_pass:
                return "candidate"
            if outcome == "blocked":
                return None
        if signal.auto_pass:
            return "auto"
        if signal.candidate_pass:
            return "candidate"
        return None

    def _derive_exit_levels_from_risk_plan(self, *, signal: SignalEvent, risk_plan: RiskPlan) -> dict[str, float]:
        sl_level = float(risk_plan.initial_sl)
        tp2_level = float(risk_plan.initial_tp)
        tp1_level = self._derive_tp1_from_sl_distance(
            signal=signal,
            sl_level=sl_level,
        )
        return {
            "sl": sl_level,
            "tp1": tp1_level,
            "tp2": tp2_level,
        }

    def _derive_tp1_from_sl_distance(self, *, signal: SignalEvent, sl_level: float) -> float:
        geom = self.strategy_profile.geometry
        entry = float(signal.price)
        sl_dist = abs(entry - float(sl_level))
        if sl_dist <= 0:
            sl_dist = max(entry * float(self.settings.exit.sl_pct), 1e-12)
        ratio = float(geom.tp1_m) / max(float(geom.sl_m), 1e-12)
        tp1_dist = sl_dist * ratio
        if signal.direction.value == "long":
            return entry + tp1_dist
        return entry - tp1_dist

    def _block_signal(
        self,
        signal: SignalEvent,
        now: datetime,
        result: TradeCycleResult,
        reason: str,
        payload: dict[str, object] | None = None,
        stage: str = "execution_filter",
        increment_missed: bool = True,
    ) -> None:
        result.skipped += 1
        if increment_missed:
            result.missed_signals += 1
            self._hourly_missed_signals += 1
        if reason.startswith("MAX_POS"):
            result.max_pos_blocked += 1
        self._record_execution_decision(
            signal=signal,
            now=now,
            stage=stage,
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
                    "stage": stage,
                    "power_score": signal.power_score,
                    "hourly_missed_signals": self._hourly_missed_signals,
                    "profile_id": self.strategy_profile.name,
                }
            },
        )
        if increment_missed and self.settings.notifications.notify_on_missed_signal:
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
        stage: str = "execution_filter",
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
            stage=stage,
            outcome=outcome,
            blocked_reason=blocked_reason,
            decision_payload={
                "recorded_at": now.isoformat(),
                "profile_id": self.strategy_profile.name,
                "trigger_mode": self.strategy_profile.filters.ltf_trigger,
                "bias_mode": f"{self.strategy_profile.filters.htf_bias.ma_type}_{self.strategy_profile.filters.htf_bias.timeframe}",
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

