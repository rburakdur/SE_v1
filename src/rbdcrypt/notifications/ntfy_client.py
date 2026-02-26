from __future__ import annotations

import requests

from ..config import NotificationSettings


class NtfyClient:
    def __init__(self, config: NotificationSettings) -> None:
        self.config = config
        self.session = requests.Session()

    def notify(self, title: str, message: str) -> bool:
        if not self.config.enabled or not self.config.ntfy_url or not self.config.topic:
            return False
        url = f"{self.config.ntfy_url.rstrip('/')}/{self.config.topic}"
        resp = self.session.post(
            url,
            data=message.encode("utf-8"),
            headers={"Title": title},
            timeout=self.config.timeout_sec,
        )
        resp.raise_for_status()
        return True

    def close(self) -> None:
        self.session.close()
