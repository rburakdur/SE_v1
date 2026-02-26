from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from rbdcrypt.config import AppSettings
from rbdcrypt.models.ohlcv import OHLCVBar
from rbdcrypt.services.replay_service import ReplayService
from rbdcrypt.strategy.parity_signal_engine import ParitySignalEngine


def _logger():
    l = logging.getLogger("test_replay")
    if not l.handlers:
        l.addHandler(logging.NullHandler())
    l.propagate = False
    return l


def _gen_bars(symbol: str, start: datetime, count: int, *, slope: float, swing: float) -> list[OHLCVBar]:
    out: list[OHLCVBar] = []
    for i in range(count):
        t = start + timedelta(minutes=5 * i)
        base = 100.0 + (i * slope)
        wiggle = swing if (i % 10 < 5) else -swing
        close = base + wiggle
        open_p = close - 0.1
        high = close + 0.4
        low = close - 0.4
        out.append(
            OHLCVBar(
                symbol=symbol,
                interval="5m",
                open_time=t,
                open=open_p,
                high=high,
                low=low,
                close=close,
                volume=1000 + (i % 7) * 25,
                close_time=t + timedelta(minutes=5),
            )
        )
    return out


def test_replay_service_smoke_reads_local_backfill_and_returns_report(repos) -> None:
    start = datetime(2026, 2, 1, 0, 0, tzinfo=UTC)
    btc = _gen_bars("BTCUSDT", start, 140, slope=0.03, swing=0.25)
    eth = _gen_bars("ETHUSDT", start, 140, slope=0.05, swing=0.6)
    repos.candles.upsert_many(btc + eth)

    settings = AppSettings()
    settings.risk.min_rr = 0.0
    settings.risk.fee_pct_per_side = 0.0
    engine = ParitySignalEngine(settings=settings, interval="5m")
    svc = ReplayService(settings=settings, repos=repos, signal_engine=engine, logger=_logger())

    report = svc.replay_symbol(symbol="ETHUSDT", warmup_bars=80, persist_report=True)
    assert report.symbol == "ETHUSDT"
    assert report.bars_used >= 100
    assert report.candidate_signals >= 0
    assert report.auto_signals >= 0
    stored = repos.runtime_state.get_json("last_replay:ETHUSDT:5m")
    assert stored is not None
    assert stored["symbol"] == "ETHUSDT"


def test_replay_service_raises_if_btc_backfill_missing(repos) -> None:
    start = datetime(2026, 2, 1, 0, 0, tzinfo=UTC)
    repos.candles.upsert_many(_gen_bars("ETHUSDT", start, 100, slope=0.05, swing=0.5))
    settings = AppSettings()
    engine = ParitySignalEngine(settings=settings, interval="5m")
    svc = ReplayService(settings=settings, repos=repos, signal_engine=engine, logger=_logger())
    try:
        svc.replay_symbol(symbol="ETHUSDT", warmup_bars=80)
    except ValueError as exc:
        assert "BTCUSDT" in str(exc)
    else:
        raise AssertionError("Expected ValueError when BTC backfill data is missing")
