# rbdcrypt (v0.1)

Python tabanlı, modüler, düşük kaynak dostu kripto futures signal/trading engine iskeleti. Bu sürüm `paper mode` odaklıdır; live execution sadece scaffold olarak bırakılmıştır.

## Özellikler (v0.1)

- `src/` düzeni, typed code, modüler mimari (`domain/infra/runtime` ayrımı)
- `pydantic-settings` ile `.env` + environment config yönetimi
- `requests.Session` reuse + retry/backoff
- SQLite (`WAL`) persistence
- Structured logging (JSONL, rotating file)
- Paper broker runtime
- Legacy strateji parity çekirdeği (v87-style):
  - `FLIP_LONG/FLIP_SHORT` + `TREND` tabanlı 5m sinyal üretimi
  - candidate vs auto threshold ayrımı
  - legacy power score formülü (RSI/VOL/ADX/ATR/MACD/BB width)
  - BTC trend filter (`hard_block | soft_penalty`) + chop policy
  - ATR çarpanlı TP/SL (`SL_M`, `TP_M` parity)
  - timeout + ST grace + trend flip + stale exit + max-hold sonrası break-even move
  - cooldown (entry block) + missed-signal counters (SQLite runtime_state)
- CLI komutları: `run`, `doctor`, `analyze`, `rotate/prune`, `export-csv`, `backfill`, `replay`
- Testler: strategy/risk/exit/recovery/sqlite atomicity çekirdeği

## Mimari Özeti

- `models/`: domain modelleri (`Signal`, `ActivePosition`, `ClosedTrade`, `MarketContext`, `ErrorEvent`)
- `strategy/`: generic helpers + `legacy_parity.py` + `parity_signal_engine.py` + exit engine
- `data/`: Binance public futures client + market fetcher + rate limit
- `storage/`: SQLite db/migrations/repositories (+ optional JSON snapshot store)
- `services/`: scan/trade/metrics/housekeeping orchestration
- `runtime/`: dependency wiring + worker loop + logging setup
- `cli.py`: operatör komutları

## Kurulum

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
cp .env.example .env
```

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .
Copy-Item .env.example .env
```

## Çalıştırma

Tek döngü test:

```bash
rbdcrypt run --one-shot
```

Sürekli çalışma:

```bash
rbdcrypt run
```

Sağlık kontrolü:

```bash
rbdcrypt doctor
```

Özet analiz:

```bash
rbdcrypt analyze
```

Retention / prune:

```bash
rbdcrypt rotate
```

CSV export:

```bash
rbdcrypt export-csv trades_closed --out bot_data/exports
```

Historical backfill (SQLite `ohlcv_futures`):

```bash
rbdcrypt backfill --symbols top:20 --days 7
rbdcrypt backfill --symbols BTCUSDT,ETHUSDT --bars 1000 --no-incremental
```

Local replay (paper simulation, real order yok):

```bash
rbdcrypt replay ETHUSDT --days 7
rbdcrypt replay SOLUSDT --start 2026-02-01T00:00:00+00:00 --end 2026-02-10T00:00:00+00:00
```

## Config Notları

Tüm threshold ve davranışlar config üzerindedir; magic number bırakmamaya odaklanılmıştır.

- `FILTERS__BTC_TREND_FILTER_MODE`: `hard_block | soft_penalty`
- `FILTERS__CHOP_POLICY`: `block | penalty | allow`
- `SCORE__CANDIDATE_SCORE_MIN`, `SCORE__AUTO_SCORE_MIN`
- Legacy parity runtime için asıl aktif ayarlar `LEGACY_PARITY__*` altındadır (varsayılan `enabled=true`)
- `BALANCE__MODE`: `cumulative | daily_reset`
- `BALANCE__DAILY_RESET_ENABLED`: varsayılan `false`
- `RUNTIME__WORKER_COUNT`: varsayılan `3`
- `RUNTIME__MAX_SYMBOLS`: varsayılan `50`
- `RUNTIME__CHART_ENABLED`: varsayılan `false`
- `RUNTIME__HEAVY_DEBUG`: varsayılan `false`

## Ntfy Bildirimleri

`.env` içinde aşağıdakileri açın:

```env
NOTIFICATIONS__NTFY_ENABLED=true
NOTIFICATIONS__NTFY_URL=https://ntfy.sh
NOTIFICATIONS__NTFY_TOPIC=<senin_topic_adin>
NOTIFICATIONS__NOTIFY_ON_OPEN=true
NOTIFICATIONS__NOTIFY_ON_CLOSE=true
NOTIFICATIONS__NOTIFY_ON_SCAN_DEGRADED=true
NOTIFICATIONS__NOTIFY_ON_AUTO_SIGNAL_SUMMARY=true
NOTIFICATIONS__NOTIFY_ON_RUNTIME_ERROR=true
```

Aktif eventler:
- recovery bildirimi
- pozisyon açılış/kapanış
- scan degraded (hatalı cycle)
- auto signal özeti
- runtime/trade kritik hataları

## State Güvenliği / Persistence

- SQLite `WAL` aktif
- Açık pozisyonlar `positions_active`, kapanan işlemler `trades_closed`
- Sinyal ve karar aşamaları ayrı kaydedilir (`signals`, `signal_decisions`)
- `opened_at` alanı recovery/upsert sırasında korunur (overwrite edilmez)
- R:R metriği `initial_sl/initial_tp` üzerinden kaydedilir; `current_sl` mutate edilse de kayıt bozulmaz

## Gözlemlenebilirlik

- Sinyal event + power score breakdown + blocked reason saklanır
- Error kayıtları tek satır (traceback newline escape)
- `doctor` çıktısı:
  - DB integrity
  - disk free
  - last scan time
  - active positions
  - `ohlcv_rows_total`
  - cooldown symbols / count
  - trade missed counters (hourly + last cycle)
  - error count (last 1h)
  - API connectivity
  - scanner/trader heartbeat

## E2 Micro Tuning (1 GB RAM / düşük CPU)

- `RUNTIME__MAX_SYMBOLS=40` ile başlayın (50 default çoğu durumda yeterli)
- `RUNTIME__WORKER_COUNT=2-3` aralığında kalın
- `RUNTIME__HEAVY_DEBUG=false`
- chart üretimi kapalı tutun (`RUNTIME__CHART_ENABLED=false`)
- `BINANCE__KLINE_LIMIT` gereksiz yüksek olmasın (200-250 aralığı yeterli)
- düzenli `rbdcrypt rotate` veya systemd timer ile prune çalıştırın

## Paper / Live Modları

- `Paper`: aktif (default)
- `Live`: `src/rbdcrypt/brokers/live_broker.py` scaffold var, order execution henüz implement edilmedi

## Testler

```bash
pip install -e .[dev]
pytest -q
```

## Systemd (Oracle Linux / Ubuntu)

Örnek unit dosyaları `deploy/systemd/` altında:

- `rbdcrypt.service`: ana bot servisi
- `rbdcrypt-update-check.service`
- `rbdcrypt-update-check.timer`: uzaktaki GitHub branch’i periyodik kontrol edip update + restart

Kurulum örneği:

```bash
sudo cp deploy/systemd/rbdcrypt.service /etc/systemd/system/
sudo cp deploy/systemd/rbdcrypt-update-check.service /etc/systemd/system/
sudo cp deploy/systemd/rbdcrypt-update-check.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now rbdcrypt.service
sudo systemctl enable --now rbdcrypt-update-check.timer
```

### Self-update / Auto deploy notu

`scripts/check_update_and_restart.sh` şu akışı yapar:

1. `git fetch`
2. remote branch SHA değişmişse `git pull --ff-only`
3. `pip install -e .`
4. `systemctl restart rbdcrypt.service`

`rbdcrypt-update-check.service` örneği `sudo systemctl restart` kullandığı için aşağıdaki sudoers izni gerekir (örnek):

```bash
oracle ALL=(root) NOPASSWD: /bin/systemctl restart rbdcrypt.service
```

Alternatif: update-check servisini root olarak çalıştırıp repo/venv ownership modelinizi ona göre düzenleyin.

## TODO (Live Entegrasyon / Sonraki Adımlar)

- Binance signed futures order execution (auth, recvWindow, timestamp sync)
- Order state reconciliation (partial fill, cancel/replace, reduce-only)
- Exchange filters (tickSize, stepSize, minNotional) enforcement
- WebSocket market stream / user data stream
- Slippage model ve fill simulation iyileştirmesi
- Backtest/backfill pipeline + replay engine
- Prometheus endpoint veya lightweight HTTP health endpoint
- Snapshot restore path’i runtime’a opsiyonel bağlama
- Signal/scan parity için eski runtime çıktılarıyla golden dataset replay karşılaştırması (multi-bar scenario)
- Multi-symbol portfolio replay/backtest (şu an replay tek sembol + BTC benchmark odaklı)
