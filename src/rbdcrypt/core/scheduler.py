from __future__ import annotations

import random
import time
from dataclasses import dataclass
from time import monotonic


@dataclass(slots=True)
class IntervalScheduler:
    interval_sec: float
    jitter_sec: float = 0.0
    _next_ts: float | None = None

    def start(self) -> None:
        self._next_ts = monotonic()

    def sleep_until_next(self) -> None:
        if self._next_ts is None:
            self.start()
        assert self._next_ts is not None
        self._next_ts += self.interval_sec
        target = self._next_ts + (random.uniform(0, self.jitter_sec) if self.jitter_sec else 0.0)
        remaining = target - monotonic()
        if remaining > 0:
            time.sleep(remaining)
