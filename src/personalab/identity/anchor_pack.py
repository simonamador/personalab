"""Anchor pack: set of reference images and manifest for a persona."""

import json
from pathlib import Path
from typing import Any

from personalab.config import load_yaml


class AnchorPack:
    """Frozen set of anchor images + cached stats."""

    __slots__ = ("persona", "root_dir", "anchor_files", "stats_path", "manifest_path")

    def __init__(
        self,
        *,
        persona: str,
        root_dir: Path,
        anchor_files: list[Path],
        stats_path: Path,
        manifest_path: Path,
    ) -> None:
        self.persona = persona
        self.root_dir = Path(root_dir)
        self.anchor_files = list(anchor_files)
        self.stats_path = Path(stats_path)
        self.manifest_path = Path(manifest_path)

    @property
    def anchors_dir(self) -> Path:
        return self.root_dir / "anchors"

    def load_manifest(self) -> dict[str, Any]:
        return load_yaml(self.manifest_path)

    def load_stats(self) -> dict[str, Any] | None:
        if not self.stats_path.exists():
            return None
        return json.loads(self.stats_path.read_text(encoding="utf-8"))

    def save_stats(self, stats: dict[str, Any]) -> None:
        self.stats_path.write_text(
            json.dumps(stats, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )


def load_anchor_pack(references_root: str | Path, persona: str) -> AnchorPack:
    """Load an AnchorPack from references/{persona}."""
    root = Path(references_root) / persona
    manifest_path = root / "anchor_manifest.yml"
    stats_path = root / "anchor_stats.json"
    anchors_dir = root / "anchors"

    manifest = load_yaml(manifest_path)
    anchors = manifest.get("anchors", [])
    files: list[Path] = []
    for i, a in enumerate(anchors):
        if not isinstance(a, dict) or "file" not in a:
            raise ValueError(f"Malformed anchor entry at index {i} in {manifest_path}: expected dict with 'file' key")
        files.append(anchors_dir / a["file"])

    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing anchor files: {missing}")

    return AnchorPack(
        persona=persona,
        root_dir=root,
        anchor_files=files,
        stats_path=stats_path,
        manifest_path=manifest_path,
    )
