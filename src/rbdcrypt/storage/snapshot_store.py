from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonSnapshotStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def save(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=True, default=str), encoding="utf-8")
        tmp.replace(self.path)

    def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        return json.loads(self.path.read_text(encoding="utf-8"))
