"""Append-only JSONL registry for audit / workflow logging."""

import itertools
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

_registry_seq = itertools.count()


class RegistryWriter:
    """Append-only JSONL registry (thread-safe)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("timestamp", datetime.now().astimezone().isoformat())
        record.setdefault("seq", next(_registry_seq))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
