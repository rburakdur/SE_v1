from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from typing import Any

from ..config import AppSettings
from ..strategy.signal_engine import CandleSeries
from .binance_client import BinanceClient
from .rate_limit import SimpleRateLimiter

if TYPE_CHECKING:
    from logging import Logger


@dataclass(slots=True)
class MarketFetcher:
    settings: AppSettings
    client: BinanceClient
    rate_limiter: SimpleRateLimiter
    logger: Logger | None = None

    def fetch_universe_symbols(self) -> list[str]:
        rows = self.client.ticker_24hr()
        quote = self.settings.binance.quote_asset
        explicit_blacklist = {sym.upper() for sym in self.settings.runtime.universe_symbol_blacklist}
        filtered = [
            r
            for r in rows
            if str(r.get("symbol", "")).endswith(quote)
            and float(r.get("quoteVolume", 0.0) or 0.0) > 0
        ]
        filtered.sort(key=lambda r: float(r.get("quoteVolume", 0.0) or 0.0), reverse=True)
        symbols = [str(r["symbol"]) for r in filtered if str(r.get("symbol")) != self.settings.binance.btc_symbol]
        blocked_explicit: list[str] = []
        blocked_pattern: list[str] = []
        clean_symbols: list[str] = []
        for symbol in symbols:
            upper = symbol.upper()
            if upper in explicit_blacklist:
                blocked_explicit.append(upper)
                continue
            if self.settings.runtime.universe_stable_like_pattern_enabled and self._is_stable_like_pair(upper):
                blocked_pattern.append(upper)
                continue
            clean_symbols.append(symbol)

        if self.logger is not None and (blocked_explicit or blocked_pattern):
            self.logger.info(
                "universe_filter_applied",
                extra={
                    "event": {
                        "blocked_explicit": blocked_explicit,
                        "blocked_pattern": blocked_pattern,
                        "blocked_total": len(blocked_explicit) + len(blocked_pattern),
                    }
                },
            )
        max_symbols = min(self.settings.runtime.max_symbols, self.settings.binance.top_symbols_limit)
        return clean_symbols[:max_symbols]

    def fetch_candles(self, symbol: str) -> CandleSeries:
        return self.fetch_candles_for_interval(symbol=symbol, interval=self.settings.binance.interval)

    def fetch_candles_for_interval(self, *, symbol: str, interval: str, limit: int | None = None) -> CandleSeries:
        self.rate_limiter.wait()
        raw = self.client.klines(
            symbol=symbol,
            interval=interval,
            limit=int(limit) if limit is not None else self.settings.binance.kline_limit,
        )
        return self._parse_klines(raw)

    def fetch_many_candles(self, symbols: list[str], worker_count: int) -> tuple[dict[str, CandleSeries], dict[str, str]]:
        return self.fetch_many_candles_for_interval(
            symbols=symbols,
            interval=self.settings.binance.interval,
            worker_count=worker_count,
            limit=self.settings.binance.kline_limit,
        )

    def fetch_many_candles_for_interval(
        self,
        *,
        symbols: list[str],
        interval: str,
        worker_count: int,
        limit: int | None = None,
    ) -> tuple[dict[str, CandleSeries], dict[str, str]]:
        if worker_count <= 1:
            results: dict[str, CandleSeries] = {}
            errors: dict[str, str] = {}
            for s in symbols:
                try:
                    results[s] = self.fetch_candles_for_interval(symbol=s, interval=interval, limit=limit)
                except Exception as exc:
                    errors[s] = f"{exc.__class__.__name__}: {exc}"
            return results, errors
        results: dict[str, CandleSeries] = {}
        errors: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="scan") as pool:
            future_map = {
                pool.submit(
                    self.fetch_candles_for_interval,
                    symbol=s,
                    interval=interval,
                    limit=limit,
                ): s
                for s in symbols
            }
            for fut in as_completed(future_map):
                symbol = future_map[fut]
                try:
                    results[symbol] = fut.result()
                except Exception as exc:
                    errors[symbol] = f"{exc.__class__.__name__}: {exc}"
        return results, errors

    def fetch_btc_context_candles(self) -> CandleSeries:
        return self.fetch_candles_for_interval(
            symbol=self.settings.binance.btc_symbol,
            interval=self.settings.binance.interval,
        )

    def mark_price(self, symbol: str) -> float:
        self.rate_limiter.wait()
        return self.client.mark_price(symbol)

    @staticmethod
    def _is_stable_like_pair(symbol: str) -> bool:
        # Exclude synthetic stable-like futures pairs (e.g. USDCUSDT, FDUSDUSDT).
        stable_bases = ("USDC", "FDUSD", "USD1", "BUSD", "USDP", "TUSD", "DAI")
        if not symbol.endswith("USDT"):
            return False
        base = symbol[:-4]
        if base in stable_bases:
            return True
        if base.startswith("USD") or base.endswith("USD"):
            return True
        return False

    @staticmethod
    def _parse_klines(raw: list[list[Any]]) -> CandleSeries:
        times: list[datetime] = []
        opens: list[float] = []
        highs: list[float] = []
        lows: list[float] = []
        closes: list[float] = []
        volumes: list[float] = []
        for row in raw:
            # Binance kline: [open_time, open, high, low, close, volume, ...]
            times.append(datetime.fromtimestamp(int(row[0]) / 1000, tz=UTC))
            opens.append(float(row[1]))
            highs.append(float(row[2]))
            lows.append(float(row[3]))
            closes.append(float(row[4]))
            volumes.append(float(row[5]))
        return CandleSeries(
            open_times=times,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            volumes=volumes,
        )
