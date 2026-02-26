from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from .policies import BalanceMode


@dataclass(slots=True)
class PortfolioSnapshot:
    balance: float
    day_anchor: datetime
    realized_pnl: float = 0.0


@dataclass(slots=True)
class BalanceTracker:
    starting_balance: float
    mode: BalanceMode = BalanceMode.CUMULATIVE
    _snapshot: PortfolioSnapshot = field(init=False)

    def __post_init__(self) -> None:
        now = datetime.now(tz=UTC)
        self._snapshot = PortfolioSnapshot(balance=self.starting_balance, day_anchor=now)

    @property
    def balance(self) -> float:
        return self._snapshot.balance

    def maybe_reset_daily(self, now: datetime) -> None:
        if self.mode != BalanceMode.DAILY_RESET:
            return
        if now.date() != self._snapshot.day_anchor.date():
            self._snapshot = PortfolioSnapshot(balance=self.starting_balance, day_anchor=now)

    def apply_realized_pnl(self, pnl_quote: float, now: datetime) -> None:
        self.maybe_reset_daily(now)
        self._snapshot.balance += pnl_quote
        self._snapshot.realized_pnl += pnl_quote

    def serialize(self) -> dict[str, float | str]:
        return {
            "balance": self._snapshot.balance,
            "realized_pnl": self._snapshot.realized_pnl,
            "day_anchor": self._snapshot.day_anchor.isoformat(),
        }
