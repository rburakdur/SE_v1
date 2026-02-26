from __future__ import annotations

import random
import threading
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import BinanceSettings, RetrySettings


class BinanceClient:
    def __init__(self, config: BinanceSettings, retry_cfg: RetrySettings) -> None:
        self.config = config
        self.retry_cfg = retry_cfg
        self.session = requests.Session()
        self._request_lock = threading.Lock()
        retry = Retry(
            total=retry_cfg.total,
            backoff_factor=retry_cfg.backoff_factor,
            status_forcelist=list(retry_cfg.status_forcelist),
            allowed_methods=frozenset(["GET", "POST"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def close(self) -> None:
        self.session.close()

    def _request(self, method: str, path: str, *, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.config.base_url}{path}"
        with self._request_lock:
            resp = self.session.request(
                method,
                url,
                params=params,
                timeout=(self.config.connect_timeout_sec, self.config.read_timeout_sec),
            )
        if resp.status_code == 429:
            time.sleep(random.uniform(0.05, self.retry_cfg.jitter_max_sec + 0.05))
        resp.raise_for_status()
        return resp.json()

    def ping(self) -> bool:
        try:
            self._request("GET", "/fapi/v1/ping")
            return True
        except requests.RequestException:
            return False

    def exchange_info(self) -> dict[str, Any]:
        return self._request("GET", "/fapi/v1/exchangeInfo")

    def ticker_24hr(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/fapi/v1/ticker/24hr")
        if isinstance(data, list):
            return data
        raise TypeError("Unexpected ticker_24hr response type")

    def klines(
        self,
        symbol: str,
        interval: str,
        limit: int,
        *,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ) -> list[list[Any]]:
        params: dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        if end_time_ms is not None:
            params["endTime"] = int(end_time_ms)
        data = self._request(
            "GET",
            "/fapi/v1/klines",
            params=params,
        )
        if isinstance(data, list):
            return data
        raise TypeError("Unexpected klines response type")

    def mark_price(self, symbol: str) -> float:
        data = self._request("GET", "/fapi/v1/premiumIndex", params={"symbol": symbol})
        return float(data["markPrice"])
