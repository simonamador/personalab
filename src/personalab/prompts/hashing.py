"""Stable hashing for prompt dicts."""

import hashlib
import json
from typing import Any


def sha256_json(data: dict[str, Any]) -> str:
    """Stable SHA256 for a dict serialized as sorted JSON."""
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
