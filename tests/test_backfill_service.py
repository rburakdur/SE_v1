from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from rbdcrypt.config import AppSettings
from rbdcrypt.services.backfill_service import BackfillService


@dataclass
class _FakeLimiter:
    calls: int = 0

    def wait(self) -> None:
        self.calls += 1


class _FakeClient:
    def __init__(self, rows: list[list[object]]):
        self._rows = rows

    def klines(
        self,
        symbol: str,
        interval: str,
        limit: int,
        *,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ) -> list[list[object]]:
        out = []
        for r in self._rows:
            ot = int(r[0])
            if start_time_ms is not None and ot < start_time_ms:
                continue
            if end_time_ms is not None and ot > end_time_ms:
                continue
            out.append(r)
        return out[:limit]


class _FakeFetcher:
    def __init__(self, rows: list[list[object]]):
        self.client = _FakeClient(rows)
        self.rate_limiter = _FakeLimiter()

    def fetch_universe_symbols(self) -> list[str]:
        return ["ETHUSDT", "SOLUSDT", "XRPUSDT"]


def _mk_rows(start: datetime, count: int) -> list[list[object]]:
    rows: list[list[object]] = []
    for i in range(count):
        open_time = start + timedelta(minutes=5 * i)
        close_time = open_time + timedelta(minutes=5) - timedelta(milliseconds=1)
        px = 100 + i
        rows.append(
            [
                int(open_time.timestamp() * 1000),
                str(px),
                str(px + 1),
                str(px - 1),
                str(px + 0.5),
                "1000",
                int(close_time.timestamp() * 1000),
            ]
        )
    return rows


def test_backfill_service_inserts_rows_into_sqlite(repos) -> None:
    settings = AppSettings()
    start = datetime(2026, 2, 1, 0, 0, tzinfo=UTC)
    rows = _mk_rows(start, 6)
    fetcher = _FakeFetcher(rows)
    svc = BackfillService(settings=settings, fetcher=fetcher, repos=repos, logger=NoneLogger())

    end_dt = start + timedelta(minutes=25)
    summary = svc.backfill(symbols=["BTCUSDT"], bars=6, days=None, chunk_limit=3, incremental=False, end_time=end_dt)

    assert summary.inserted_bars >= 6
    assert repos.candles.count(symbol="BTCUSDT", interval="5m") == 6
    assert fetcher.rate_limiter.calls >= 2


def test_backfill_service_resolve_symbols_top_with_btc(repos) -> None:
    settings = AppSettings()
    fetcher = _FakeFetcher([])
    svc = BackfillService(settings=settings, fetcher=fetcher, repos=repos, logger=NoneLogger())
    syms = svc.resolve_symbols("top:2", include_btc=True)
    assert syms[0] == settings.binance.btc_symbol
    assert syms[1:] == ["ETHUSDT", "SOLUSDT"]


class NoneLogger:
    def info(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        return None
