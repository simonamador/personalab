"""Tests for RegistryWriter."""

import json
import tempfile
from pathlib import Path

from personalab import RegistryWriter


def test_registry_append_adds_timestamp():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "registry.jsonl"
        w = RegistryWriter(path=path)
        w.append({"event": "test", "id": 1})
        line = path.read_text(encoding="utf-8").strip()
        record = json.loads(line)
        assert record["event"] == "test"
        assert record["id"] == 1
        assert "timestamp" in record


def test_registry_append_multiple():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "log.jsonl"
        w = RegistryWriter(path=path)
        w.append({"n": 1})
        w.append({"n": 2})
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["n"] == 1
        assert json.loads(lines[1])["n"] == 2
