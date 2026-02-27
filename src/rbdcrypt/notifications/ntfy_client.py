from __future__ import annotations

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
        url = f"{self.config.ntfy_url.rstrip('/')}/{self.config.topic}"
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

    def close(self) -> None:
        self.session.close()
