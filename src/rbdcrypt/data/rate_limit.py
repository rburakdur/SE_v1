from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class SimpleRateLimiter:
    min_interval_sec: float
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _last_call_ts: float = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            remaining = self.min_interval_sec - (now - self._last_call_ts)
            if remaining > 0:
                time.sleep(remaining)
                now = time.monotonic()
            self._last_call_ts = now
