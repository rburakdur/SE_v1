from __future__ import annotations

from zipfile import ZipFile

from rbdcrypt.runtime.worker import RuntimeWorker


def _new_worker() -> RuntimeWorker:
    return object.__new__(RuntimeWorker)


def test_create_zip_archive_keeps_export_root_folder(tmp_path) -> None:
    export_dir = tmp_path / "ntfy_log_20260227_120000"
    logs_dir = export_dir / "logs"
    db_dir = export_dir / "db"
    logs_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "rbdcrypt.log").write_text("line1\nline2\n", encoding="utf-8")
    (db_dir / "signals.csv").write_text("id,value\n1,test\n", encoding="utf-8")
    (export_dir / "meta.json").write_text('{"ok":true}', encoding="utf-8")

    archive_path = tmp_path / "bundle.zip"
    RuntimeWorker._create_zip_archive(source_dir=export_dir, archive_path=archive_path)

    with ZipFile(archive_path, "r") as zf:
        names = set(zf.namelist())
    assert f"{export_dir.name}/meta.json" in names
    assert f"{export_dir.name}/logs/rbdcrypt.log" in names
    assert f"{export_dir.name}/db/signals.csv" in names


def test_split_for_ntfy_upload_splits_large_archive_and_preserves_bytes(tmp_path, monkeypatch) -> None:
    archive_path = tmp_path / "ntfy_log_all_20260227_120000.zip"
    payload = bytes(range(256)) * 5
    archive_path.write_bytes(payload)

    monkeypatch.setattr(RuntimeWorker, "_MAX_NTFY_UPLOAD_BYTES", 300, raising=False)
    monkeypatch.setattr(RuntimeWorker, "_UPLOAD_CHUNK_BYTES", 200, raising=False)
    worker = _new_worker()
    parts = RuntimeWorker._split_for_ntfy_upload(worker, archive_path)

    assert len(parts) == 7
    assert parts[0].name.endswith(".zip.part001")
    assert all(p.stat().st_size <= 200 for p in parts)
    rebuilt = b"".join(part.read_bytes() for part in parts)
    assert rebuilt == payload


def test_github_repo_https_builder_handles_ssh_and_https() -> None:
    ssh_url = RuntimeWorker._github_repo_https("git@github.com:rburakdur/backup.git")
    https_url = RuntimeWorker._github_repo_https("https://github.com/rburakdur/backup.git")
    assert ssh_url == "https://github.com/rburakdur/backup"
    assert https_url == "https://github.com/rburakdur/backup"


def test_db_export_specs_differs_for_log_and_log_all() -> None:
    log_specs = RuntimeWorker._db_export_specs("log")
    log_all_specs = RuntimeWorker._db_export_specs("log-all")
    log_tables = {name for name, _time_col, _limit in log_specs}
    log_all_tables = {name for name, _time_col, _limit in log_all_specs}

    assert "ohlcv_futures" not in log_tables
    assert "ohlcv_futures" in log_all_tables
    assert any(name == "signals" and limit == 2000 for name, _time_col, limit in log_specs)
    assert any(name == "signals" and limit is None for name, _time_col, limit in log_all_specs)
