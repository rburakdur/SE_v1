from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..config import AppSettings
from ..strategy.signal_engine import CandleSeries
from .binance_client import BinanceClient
from .rate_limit import SimpleRateLimiter


@dataclass(slots=True)
class MarketFetcher:
    settings: AppSettings
    client: BinanceClient
    rate_limiter: SimpleRateLimiter

    def fetch_universe_symbols(self) -> list[str]:
        rows = self.client.ticker_24hr()
        filtered = [
            r
            for r in rows
            if str(r.get("symbol", "")).endswith(self.settings.binance.quote_asset)
            and float(r.get("quoteVolume", 0.0) or 0.0) > 0
        ]
        filtered.sort(key=lambda r: float(r.get("quoteVolume", 0.0) or 0.0), reverse=True)
        symbols = [str(r["symbol"]) for r in filtered if str(r.get("symbol")) != self.settings.binance.btc_symbol]
        max_symbols = min(self.settings.runtime.max_symbols, self.settings.binance.top_symbols_limit)
        return symbols[:max_symbols]

    def fetch_candles(self, symbol: str) -> CandleSeries:
        self.rate_limiter.wait()
        raw = self.client.klines(
            symbol=symbol,
            interval=self.settings.binance.interval,
            limit=self.settings.binance.kline_limit,
        )
        return self._parse_klines(raw)

    def fetch_many_candles(self, symbols: list[str], worker_count: int) -> tuple[dict[str, CandleSeries], dict[str, str]]:
        if worker_count <= 1:
            results: dict[str, CandleSeries] = {}
            errors: dict[str, str] = {}
            for s in symbols:
                try:
                    results[s] = self.fetch_candles(s)
                except Exception as exc:
                    errors[s] = f"{exc.__class__.__name__}: {exc}"
            return results, errors
        results: dict[str, CandleSeries] = {}
        errors: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="scan") as pool:
            future_map = {pool.submit(self.fetch_candles, s): s for s in symbols}
            for fut in as_completed(future_map):
                symbol = future_map[fut]
                try:
                    results[symbol] = fut.result()
                except Exception as exc:
                    errors[symbol] = f"{exc.__class__.__name__}: {exc}"
        return results, errors

    def fetch_btc_context_candles(self) -> CandleSeries:
        return self.fetch_candles(self.settings.binance.btc_symbol)

    def mark_price(self, symbol: str) -> float:
        self.rate_limiter.wait()
        return self.client.mark_price(symbol)

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
