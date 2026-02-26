from __future__ import annotations

from datetime import UTC, datetime, timedelta

from rbdcrypt.models.ohlcv import OHLCVBar


def test_candle_repository_upsert_and_query(repos) -> None:
    t0 = datetime(2026, 2, 1, 0, 0, tzinfo=UTC)
    bars = [
        OHLCVBar(
            symbol="BTCUSDT",
            interval="5m",
            open_time=t0 + timedelta(minutes=5 * i),
            open=100 + i,
            high=101 + i,
            low=99 + i,
            close=100.5 + i,
            volume=1000 + i,
            close_time=t0 + timedelta(minutes=5 * i + 5),
        )
        for i in range(3)
    ]
    inserted = repos.candles.upsert_many(bars)
    assert inserted == 3
    assert repos.candles.count(symbol="BTCUSDT", interval="5m") == 3

    listed = repos.candles.list_bars(symbol="BTCUSDT", interval="5m")
    assert [b.open_time for b in listed] == [b.open_time for b in bars]
    assert repos.candles.latest_open_time(symbol="BTCUSDT", interval="5m") == bars[-1].open_time

    # Upsert same bar with changed close value
    updated = bars[1].model_copy(update={"close": 222.0})
    repos.candles.upsert_many([updated])
    listed2 = repos.candles.list_bars(symbol="BTCUSDT", interval="5m")
    assert listed2[1].close == 222.0
