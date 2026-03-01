from __future__ import annotations

import json
from pathlib import Path

import requests

from ..config import NotificationSettings


class NtfyClient:
    def __init__(self, config: NotificationSettings) -> None:
        self.config = config
        self.session = requests.Session()

    def notify(
        self,
        title: str,
        message: str,
        *,
        priority: int | None = None,
        tags: str | None = None,
        attach_url: str | None = None,
        filename: str | None = None,
    ) -> bool:
        if not self.config.enabled or not self.config.ntfy_url or not self.config.topic:
            return False
        url = self._topic_url(self.config.topic)
        headers = {"Title": title}
        if priority is not None:
            headers["Priority"] = str(int(priority))
        if tags:
            headers["Tags"] = tags
        if attach_url:
            headers["Attach"] = attach_url
            if filename:
                headers["Filename"] = filename
        resp = self.session.post(
            url,
            data=message.encode("utf-8"),
            headers=headers,
            timeout=self.config.timeout_sec,
        )
        resp.raise_for_status()
        return True

    def fetch_messages(
        self,
        *,
        topic: str,
        since: str | None = None,
    ) -> list[dict[str, object]]:
        if not self.config.enabled or not self.config.ntfy_url or not topic:
            return []
        params: dict[str, str] = {"poll": "0", "since": since or "2m"}
        resp = self.session.get(
            f"{self._topic_url(topic)}/json",
            params=params,
            timeout=self.config.timeout_sec,
        )
        resp.raise_for_status()
        items: list[dict[str, object]] = []
        for line in resp.text.splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            event = payload.get("event")
            if event is None or event == "message":
                items.append(payload)
        return items

    def upload_file(
        self,
        *,
        title: str,
        file_path: Path,
        topic: str | None = None,
        message: str | None = None,
        priority: int | None = None,
        tags: str | None = None,
        filename: str | None = None,
    ) -> bool:
        target_topic = (topic or self.config.topic or "").strip()
        if not self.config.enabled or not self.config.ntfy_url or not target_topic:
            return False
        path = Path(file_path)
        if not path.is_file():
            return False
        headers = {"Title": title, "Filename": filename or path.name}
        if message:
            headers["Message"] = message
        if priority is not None:
            headers["Priority"] = str(int(priority))
        if tags:
            headers["Tags"] = tags
        with path.open("rb") as f:
            resp = self.session.post(
                self._topic_url(target_topic),
                data=f,
                headers=headers,
                timeout=max(self.config.timeout_sec, 30.0),
            )
        resp.raise_for_status()
        return True

    def _topic_url(self, topic: str) -> str:
        return f"{self.config.ntfy_url.rstrip('/')}/{topic}"

    def close(self) -> None:
        self.session.close()
