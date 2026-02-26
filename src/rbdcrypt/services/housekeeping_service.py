from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ..config import AppSettings
from ..storage.repositories import Repositories


@dataclass(slots=True)
class HousekeepingService:
    settings: AppSettings
    repos: Repositories

    def prune(self) -> dict[str, int]:
        cfg = self.settings.housekeeping
        deleted = {
            "signals": self.repos.maintenance.prune(
                table="signals",
                date_column="created_at",
                retention_days=cfg.signals_retention_days,
            ),
            "signal_decisions": self.repos.maintenance.prune(
                table="signal_decisions",
                date_column="created_at",
                retention_days=cfg.decisions_retention_days,
            ),
            "market_context": self.repos.maintenance.prune(
                table="market_context",
                date_column="fetched_at",
                retention_days=cfg.market_context_retention_days,
            ),
            "errors": self.repos.maintenance.prune(
                table="errors",
                date_column="created_at",
                retention_days=cfg.errors_retention_days,
            ),
            "heartbeats": self.repos.maintenance.prune(
                table="heartbeats",
                date_column="created_at",
                retention_days=cfg.heartbeats_retention_days,
            ),
            "trades_closed": self.repos.maintenance.prune(
                table="trades_closed",
                date_column="closed_at",
                retention_days=cfg.trades_retention_days,
            ),
        }
        self.prune_logs()
        return deleted

    def prune_logs(self) -> None:
        log_dir = self.settings.logging.log_dir
        if not log_dir.exists():
            return
        cutoff = datetime.now(tz=UTC) - timedelta(days=self.settings.housekeeping.log_retention_days)
        for p in log_dir.iterdir():
            if not p.is_file():
                continue
            if p.name.startswith("."):
                continue
            mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=UTC)
            if mtime < cutoff:
                p.unlink(missing_ok=True)

    def disk_status(self, path: Path | None = None) -> dict[str, float]:
        target = path or self.settings.storage.db_path.parent
        usage = shutil.disk_usage(target)
        return {
            "total_mb": round(usage.total / (1024 * 1024), 2),
            "used_mb": round(usage.used / (1024 * 1024), 2),
            "free_mb": round(usage.free / (1024 * 1024), 2),
        }
