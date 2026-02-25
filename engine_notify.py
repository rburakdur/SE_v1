import base64
import os
import zipfile

import requests


def _sanitize_body_text(text: str, ascii_only_fn) -> str:
    """
    Bildirimlerde mojibake/encoding bozukluğu görünmesin diye body'yi ASCII-safe yap.
    Satır kırılımlarını korur.
    """
    raw = str(text or "")
    lines = raw.splitlines() or [raw]
    cleaned = []
    for line in lines:
        cleaned_line = ascii_only_fn(line)
        cleaned.append(cleaned_line if cleaned_line is not None else "")
    body = "\n".join(cleaned).strip()
    return body or (ascii_only_fn(raw) or "")


def send_ntfy_notification(
    *,
    topic: str,
    title: str,
    message: str,
    ascii_only_fn,
    log_error_fn,
    image_buf=None,
    tags: str = "robot",
    priority: str = "3",
    print_fn=print,
):
    url = f"https://ntfy.sh/{topic}"
    safe_title = ascii_only_fn(title) or "RBD-CRYPT"
    safe_tags = ascii_only_fn(tags) or "robot"
    headers = {"Title": safe_title, "Tags": safe_tags, "Priority": str(priority)}
    try:
        if image_buf:
            safe_msg = ascii_only_fn(message.replace("\n", " | ")) or "chart"
            headers["Message"] = safe_msg
            headers["Filename"] = "chart.png"
            headers["Content-Type"] = "image/png"
            resp = requests.post(url, data=image_buf.getvalue(), headers=headers, timeout=12)
        else:
            safe_body = _sanitize_body_text(message, ascii_only_fn)
            resp = requests.post(url, data=safe_body.encode("utf-8"), headers=headers, timeout=12)
        if resp.status_code >= 400:
            raise requests.HTTPError(f"ntfy status={resp.status_code} body={resp.text[:200]}")
    except Exception as e:
        try:
            print_fn(f"!!! NTFY Hatasi: {e}")
        except Exception:
            pass
        log_error_fn("send_ntfy_notification", e, title)


def send_ntfy_file(
    *,
    topic: str,
    filepath: str,
    filename: str,
    ascii_only_fn,
    log_error_fn,
    message: str = "",
):
    url = f"https://ntfy.sh/{topic}"
    headers = {"Filename": (ascii_only_fn(filename) or "file.bin")}
    if message:
        headers["Message"] = ascii_only_fn(message.replace("\n", " | ")) or "file"
    try:
        with open(filepath, "rb") as f:
            resp = requests.put(url, data=f, headers=headers, timeout=30)
        if resp.status_code >= 400:
            raise requests.HTTPError(f"ntfy file status={resp.status_code} body={resp.text[:200]}")
    except Exception as e:
        log_error_fn("send_ntfy_file", e, filename)


def create_daily_backup_zip(*, base_path: str, filepaths: list, tarih_str: str, log_error_fn) -> str:
    zip_name = f"daily_backup_{tarih_str}.zip"
    zip_path = os.path.join(base_path, zip_name)
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in filepaths:
            if not os.path.exists(fp):
                continue
            if os.path.abspath(fp) == os.path.abspath(zip_path):
                continue
            arcname = os.path.relpath(fp, base_path).replace("\\", "/")
            try:
                zf.write(fp, arcname=arcname)
            except OSError as e:
                log_error_fn("create_daily_backup_zip_item", e, fp)
                continue
    return zip_path


def upload_backup_to_github(*, zip_path: str, tarih_str: str, config: dict, log_error_fn) -> tuple[bool, str]:
    if not config.get("GITHUB_BACKUP_ENABLED", False):
        return False, "disabled"
    repo = str(config.get("GITHUB_BACKUP_REPO", "")).strip()
    token = str(config.get("GITHUB_BACKUP_TOKEN", "")).strip()
    branch = str(config.get("GITHUB_BACKUP_BRANCH", "main")).strip() or "main"
    folder = str(config.get("GITHUB_BACKUP_DIR", "daily-backups")).strip().strip("/")
    if not repo or not token:
        return False, "missing_repo_or_token"
    try:
        with open(zip_path, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode("ascii")
        remote_name = os.path.basename(zip_path)
        remote_path = f"{folder}/{remote_name}" if folder else remote_name
        url = f"https://api.github.com/repos/{repo}/contents/{remote_path}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "RBD-CRYPT-BACKUP",
        }
        payload = {"message": f"backup: {tarih_str}", "content": content_b64, "branch": branch}
        r = requests.put(url, headers=headers, json=payload, timeout=60)
        if r.status_code in (200, 201):
            return True, f"{repo}/{remote_path}@{branch}"
        return False, f"http_{r.status_code}"
    except Exception as e:
        log_error_fn("upload_backup_to_github", e, os.path.basename(zip_path))
        return False, type(e).__name__
