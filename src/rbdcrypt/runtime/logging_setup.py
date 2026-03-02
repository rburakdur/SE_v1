from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from ..config import LoggingSettings


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage().replace("\n", "\\n"),
        }
        event = getattr(record, "event", None)
        if isinstance(event, dict):
            payload.update(event)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info).replace("\n", "\\n")
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


class LoggerPrefixFilter(logging.Filter):
    def __init__(self, *prefixes: str) -> None:
        super().__init__()
        self.prefixes = tuple(prefixes)

    def filter(self, record: logging.LogRecord) -> bool:
        return any(record.name.startswith(pfx) for pfx in self.prefixes)


def _make_file_handler(path: Path, cfg: LoggingSettings, formatter: logging.Formatter) -> RotatingFileHandler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        filename=path,
        maxBytes=cfg.max_bytes,
        backupCount=cfg.backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(formatter)
    return handler


def setup_logging(cfg: LoggingSettings, logger_name: str = "rbdcrypt") -> logging.Logger:
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    namespace_logger = logging.getLogger(logger_name)
    namespace_logger.setLevel(getattr(logging, cfg.level.upper(), logging.INFO))
    namespace_logger.handlers.clear()
    namespace_logger.propagate = False

    formatter: logging.Formatter
    if cfg.jsonl:
        formatter = JsonLineFormatter()
    else:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    # Namespace-wide fallback log for backward compatibility.
    namespace_logger.addHandler(_make_file_handler(cfg.log_path, cfg, formatter))

    system_handler = _make_file_handler(cfg.log_dir / "system.log", cfg, formatter)
    system_handler.addFilter(LoggerPrefixFilter("rbdcrypt.system", "rbdcrypt.app", "rbdcrypt.runtime"))
    namespace_logger.addHandler(system_handler)

    signals_handler = _make_file_handler(cfg.log_dir / "signals.log", cfg, formatter)
    signals_handler.addFilter(LoggerPrefixFilter("rbdcrypt.signals"))
    namespace_logger.addHandler(signals_handler)

    trades_handler = _make_file_handler(cfg.log_dir / "trades.log", cfg, formatter)
    trades_handler.addFilter(LoggerPrefixFilter("rbdcrypt.trades"))
    namespace_logger.addHandler(trades_handler)

    diagnostics_handler = _make_file_handler(cfg.log_dir / "health" / "diagnostics.log", cfg, formatter)
    diagnostics_handler.addFilter(LoggerPrefixFilter("rbdcrypt.health", "rbdcrypt.diagnostics"))
    namespace_logger.addHandler(diagnostics_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    namespace_logger.addHandler(stream_handler)

    for child_name in (
        "rbdcrypt.system",
        "rbdcrypt.app",
        "rbdcrypt.runtime",
        "rbdcrypt.signals",
        "rbdcrypt.trades",
        "rbdcrypt.health",
        "rbdcrypt.diagnostics",
    ):
        child = logging.getLogger(child_name)
        child.setLevel(namespace_logger.level)
        child.propagate = True
        child.handlers.clear()

    return logging.getLogger("rbdcrypt.system")


def get_component_logger(component: str) -> logging.Logger:
    return logging.getLogger(f"rbdcrypt.{component}")
