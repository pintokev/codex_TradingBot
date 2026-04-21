from __future__ import annotations

import json
from pathlib import Path


class RuntimeStateStore:
    def __init__(self, state_dir: str | Path | None = None) -> None:
        self.state_dir = Path(state_dir or Path.cwd() / ".runtime")
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def load(self, key: str) -> dict:
        path = self._path(key)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, key: str, payload: dict) -> Path:
        path = self._path(key)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _path(self, key: str) -> Path:
        safe_key = key.replace("/", "_").replace(":", "_")
        return self.state_dir / f"{safe_key}.json"
