from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Iterable

from ..config import AppSettings
from ..data.market_fetcher import MarketFetcher
from ..models.ohlcv import OHLCVBar
from ..storage.repositories import Repositories


def interval_to_timedelta(interval: str) -> timedelta:
    unit = interval[-1].lower()
    qty = int(interval[:-1])
    if unit == "m":
        return timedelta(minutes=qty)
    if unit == "h":
        return timedelta(hours=qty)
    if unit == "d":
        return timedelta(days=qty)
    raise ValueError(f"Unsupported interval '{interval}'")


def interval_to_ms(interval: str) -> int:
    return int(interval_to_timedelta(interval).total_seconds() * 1000)


@dataclass(slots=True)
class BackfillSummary:
    symbols: list[str]
    interval: str
    inserted_bars: int
    by_symbol: dict[str, dict[str, int | str]]


class BackfillService:
    def __init__(self, *, settings: AppSettings, fetcher: MarketFetcher, repos: Repositories, logger) -> None:
        self.settings = settings
        self.fetcher = fetcher
        self.repos = repos
        self.logger = logger

    def resolve_symbols(self, spec: str, *, include_btc: bool = True) -> list[str]:
        spec = spec.strip()
        if spec.lower().startswith("top:"):
            n = int(spec.split(":", 1)[1])
            syms = self.fetcher.fetch_universe_symbols()[:n]
        elif spec.lower() == "top":
            syms = self.fetcher.fetch_universe_symbols()
        else:
            syms = [s.strip().upper() for s in spec.split(",") if s.strip()]
        if include_btc and self.settings.binance.btc_symbol not in syms:
            syms = [self.settings.binance.btc_symbol, *syms]
        # preserve order, dedupe
        seen: set[str] = set()
        out: list[str] = []
        for s in syms:
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    def backfill(
        self,
        *,
        symbols: Iterable[str],
        interval: str | None = None,
        bars: int | None = None,
        days: float | None = None,
        chunk_limit: int = 1000,
        incremental: bool = True,
        end_time: datetime | None = None,
    ) -> BackfillSummary:
        symbols = list(symbols)
        interval = interval or self.settings.binance.interval
        td = interval_to_timedelta(interval)
        end_dt = end_time or datetime.now(tz=UTC)
        total_inserted = 0
        by_symbol: dict[str, dict[str, int | str]] = {}

        for symbol in symbols:
            inserted = self._backfill_symbol(
                symbol=symbol,
                interval=interval,
                td=td,
                bars=bars,
                days=days,
                chunk_limit=chunk_limit,
                incremental=incremental,
                end_dt=end_dt,
            )
            total_inserted += inserted
            latest = self.repos.candles.latest_open_time(symbol=symbol, interval=interval)
            by_symbol[symbol] = {
                "inserted": inserted,
                "total_rows": self.repos.candles.count(symbol=symbol, interval=interval),
                "latest_open_time": latest.isoformat() if latest else "",
            }
        summary = BackfillSummary(
            symbols=symbols,
            interval=interval,
            inserted_bars=total_inserted,
            by_symbol=by_symbol,
        )
        self.logger.info(
            "backfill_completed",
            extra={"event": {"symbols": len(summary.symbols), "interval": interval, "inserted_bars": total_inserted}},
        )
        return summary

    def _backfill_symbol(
        self,
        *,
        symbol: str,
        interval: str,
        td: timedelta,
        bars: int | None,
        days: float | None,
        chunk_limit: int,
        incremental: bool,
        end_dt: datetime,
    ) -> int:
        interval_ms = int(td.total_seconds() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
        if incremental:
            latest = self.repos.candles.latest_open_time(symbol=symbol, interval=interval)
            if latest is not None:
                start_dt = latest + td
            else:
                start_dt = self._default_start(end_dt, td, bars, days)
        else:
            start_dt = self._default_start(end_dt, td, bars, days)
        start_ms = int(start_dt.timestamp() * 1000)
        if start_ms > end_ms:
            return 0

        inserted = 0
        while start_ms <= end_ms:
            # Remaining bars cap if user provided bars and non-incremental with finite window.
            req_limit = int(min(max(chunk_limit, 1), 1500))
            self.fetcher.rate_limiter.wait()
            raw = self.fetcher.client.klines(
                symbol=symbol,
                interval=interval,
                limit=req_limit,
                start_time_ms=start_ms,
                end_time_ms=end_ms,
            )
            if not raw:
                break
            bars_batch = self._raw_to_bars(raw, symbol=symbol, interval=interval)
            if not bars_batch:
                break
            inserted += self.repos.candles.upsert_many(bars_batch)
            last_open_ms = int(raw[-1][0])
            next_start = last_open_ms + interval_ms
            if next_start <= start_ms:
                break
            start_ms = next_start
            if len(raw) < req_limit:
                break
        return inserted

    @staticmethod
    def _default_start(end_dt: datetime, td: timedelta, bars: int | None, days: float | None) -> datetime:
        if bars is not None and bars > 0:
            return end_dt - (td * bars)
        if days is not None and days > 0:
            return end_dt - timedelta(days=days)
        return end_dt - timedelta(days=3)

    @staticmethod
    def _raw_to_bars(raw: list[list[object]], *, symbol: str, interval: str) -> list[OHLCVBar]:
        out: list[OHLCVBar] = []
        fetched_at = datetime.now(tz=UTC)
        for row in raw:
            open_time = datetime.fromtimestamp(int(row[0]) / 1000, tz=UTC)
            close_time = datetime.fromtimestamp(int(row[6]) / 1000, tz=UTC) if len(row) > 6 else None
            out.append(
                OHLCVBar(
                    symbol=symbol,
                    interval=interval,
                    open_time=open_time,
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    close_time=close_time,
                    source="binance_futures",
                    fetched_at=fetched_at,
                )
            )
        return out
