"""Tests for AnchorPack loading and manifest validation."""

import json

import pytest
import yaml

from personalab.identity.anchor_pack import AnchorPack, load_anchor_pack


class TestLoadAnchorPack:
    def test_loads_valid_manifest(self, tmp_path):
        root = tmp_path / "persona_a"
        anchors = root / "anchors"
        anchors.mkdir(parents=True)

        (anchors / "frontal.png").write_bytes(b"img")
        (anchors / "side.png").write_bytes(b"img")

        manifest = {"anchors": [{"file": "frontal.png"}, {"file": "side.png"}]}
        (root / "anchor_manifest.yml").write_text(yaml.dump(manifest), encoding="utf-8")

        pack = load_anchor_pack(tmp_path, "persona_a")
        assert pack.persona == "persona_a"
        assert len(pack.anchor_files) == 2
        assert all(f.exists() for f in pack.anchor_files)

    def test_raises_on_missing_file_key(self, tmp_path):
        root = tmp_path / "persona_b"
        anchors = root / "anchors"
        anchors.mkdir(parents=True)

        manifest = {"anchors": [{"name": "frontal.png"}]}
        (root / "anchor_manifest.yml").write_text(yaml.dump(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="Malformed anchor entry"):
            load_anchor_pack(tmp_path, "persona_b")

    def test_raises_on_non_dict_entry(self, tmp_path):
        root = tmp_path / "persona_c"
        anchors = root / "anchors"
        anchors.mkdir(parents=True)

        manifest = {"anchors": ["frontal.png"]}
        (root / "anchor_manifest.yml").write_text(yaml.dump(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="Malformed anchor entry"):
            load_anchor_pack(tmp_path, "persona_c")

    def test_raises_on_missing_anchor_files(self, tmp_path):
        root = tmp_path / "persona_d"
        anchors = root / "anchors"
        anchors.mkdir(parents=True)

        manifest = {"anchors": [{"file": "does_not_exist.png"}]}
        (root / "anchor_manifest.yml").write_text(yaml.dump(manifest), encoding="utf-8")

        with pytest.raises(FileNotFoundError, match="Missing anchor files"):
            load_anchor_pack(tmp_path, "persona_d")

    def test_empty_anchors_list(self, tmp_path):
        root = tmp_path / "persona_e"
        anchors = root / "anchors"
        anchors.mkdir(parents=True)

        manifest = {"anchors": []}
        (root / "anchor_manifest.yml").write_text(yaml.dump(manifest), encoding="utf-8")

        pack = load_anchor_pack(tmp_path, "persona_e")
        assert pack.anchor_files == []


class TestAnchorPack:
    def test_load_and_save_stats(self, tmp_path):
        pack = AnchorPack(
            persona="test",
            root_dir=tmp_path,
            anchor_files=[],
            stats_path=tmp_path / "stats.json",
            manifest_path=tmp_path / "manifest.yml",
        )
        assert pack.load_stats() is None

        pack.save_stats({"score": 0.95})
        loaded = pack.load_stats()
        assert loaded["score"] == 0.95

    def test_anchors_dir_property(self, tmp_path):
        pack = AnchorPack(
            persona="test",
            root_dir=tmp_path,
            anchor_files=[],
            stats_path=tmp_path / "s.json",
            manifest_path=tmp_path / "m.yml",
        )
        assert pack.anchors_dir == tmp_path / "anchors"
