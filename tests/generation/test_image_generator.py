"""Tests for ImageGenerator with mocked LLMClient."""

import pytest
from pathlib import Path

from personalab.generation.image_generator import ImageGenerator
from personalab.llm.client import TextPart, BytesPart, LLMImageResponse
from personalab import ImagePrompt


PNG_HEADER = b"\x89PNG\r\n\x1a\nfakedata"


class TestLLMImageResponse:
    def test_extracts_first_image(self):
        resp = LLMImageResponse(images=[PNG_HEADER])
        assert resp.images[0] == PNG_HEADER

    def test_empty_images(self):
        resp = LLMImageResponse(images=[])
        assert not resp.images


class TestAnchorsToPartsNoBug:
    """Verify the base64 bug fix: _anchors_to_parts should pass raw bytes, not base64."""

    def test_raw_bytes_are_used(self, tmp_path):
        img_file = tmp_path / "anchor.png"
        img_file.write_bytes(PNG_HEADER)

        gen = ImageGenerator(
            client=None,
            model_name="test",
            aspect_ratio="4:5",
            references_root=tmp_path,
        )
        parts = gen._anchors_to_parts([img_file])
        assert len(parts) == 1
        assert isinstance(parts[0], BytesPart)
        assert parts[0].data == PNG_HEADER
        assert parts[0].mime_type == "image/png"

    def test_jpeg_mime_detection(self, tmp_path):
        img_file = tmp_path / "anchor.jpg"
        img_file.write_bytes(b"\xff\xd8\xff")

        gen = ImageGenerator(client=None, model_name="t", aspect_ratio="1:1", references_root=tmp_path)
        parts = gen._anchors_to_parts([img_file])
        assert parts[0].mime_type == "image/jpeg"

    def test_empty_paths(self):
        gen = ImageGenerator(client=None, model_name="t", aspect_ratio="1:1", references_root=".")
        assert gen._anchors_to_parts([]) == []


class TestListPersonaAnchors:
    def test_returns_empty_when_dir_missing(self, tmp_path):
        gen = ImageGenerator(client=None, model_name="t", aspect_ratio="1:1", references_root=tmp_path)
        assert gen._list_persona_anchors("nobody") == []

    def test_finds_frontal_anchor(self, tmp_path):
        anchors = tmp_path / "dani" / "anchors"
        anchors.mkdir(parents=True)
        frontal = anchors / "dani_anchor_frontal_neutral.png"
        frontal.write_bytes(PNG_HEADER)

        gen = ImageGenerator(client=None, model_name="t", aspect_ratio="1:1", references_root=tmp_path)
        found = gen._list_persona_anchors("dani")
        assert len(found) == 1
        assert found[0] == frontal

    def test_finds_frontal_plus_secondary(self, tmp_path):
        anchors = tmp_path / "dani" / "anchors"
        anchors.mkdir(parents=True)
        (anchors / "dani_anchor_frontal_neutral.png").write_bytes(PNG_HEADER)
        (anchors / "dani_sculptor_45_degree.png").write_bytes(PNG_HEADER)

        gen = ImageGenerator(client=None, model_name="t", aspect_ratio="1:1", references_root=tmp_path)
        found = gen._list_persona_anchors("dani")
        assert len(found) == 2


class TestGenerateAssetImage:
    def test_returns_bytes_and_meta(self, tmp_path, fake_client):
        anchors = tmp_path / "dani" / "anchors"
        anchors.mkdir(parents=True)
        (anchors / "dani_anchor_frontal_neutral.png").write_bytes(PNG_HEADER)

        client = fake_client(image_bytes=b"generated_img")
        gen = ImageGenerator(client=client, model_name="m", aspect_ratio="4:5", references_root=tmp_path)

        prompt = ImagePrompt(
            subject_description="desc",
            scene_description="beach",
            clothing_details="casual",
            shot_type="selfie",
        )
        template = {"dynamic_context": {"scene": "${SCENE_DESCRIPTION}"}}
        img_bytes, meta, prompt_dict = gen.generate_asset_image(
            persona_name="dani",
            scenario_template=template,
            image_prompt=prompt,
        )
        assert img_bytes == b"generated_img"
        assert meta.model == "m"
        assert meta.aspect_ratio == "4:5"
        assert meta.prompt_hash.startswith("sha256:")
        assert "dani_anchor_frontal_neutral.png" in meta.anchors_used
        assert client.calls[0]["method"] == "generate_image"

    def test_returns_none_bytes_on_failure(self, tmp_path, fake_client):
        client = fake_client(image_bytes=None)
        gen = ImageGenerator(client=client, model_name="m", aspect_ratio="4:5", references_root=tmp_path)
        template = {"scene": "${SCENE_DESCRIPTION}"}
        img_bytes, meta, _ = gen.generate_asset_image(
            persona_name="missing",
            scenario_template=template,
            variables={"SCENE_DESCRIPTION": "beach"},
        )
        assert img_bytes is None

    def test_raises_without_prompt_or_variables(self, tmp_path, fake_client):
        client = fake_client()
        gen = ImageGenerator(client=client, model_name="m", aspect_ratio="4:5", references_root=tmp_path)
        with pytest.raises(ValueError, match="Either image_prompt or variables"):
            gen.generate_asset_image(persona_name="x", scenario_template={})

    def test_parts_contain_identity_lock_and_prompt(self, tmp_path, fake_client):
        """Verify the correct structure: IDENTITY_LOCK + anchors + prompt text."""
        anchors = tmp_path / "p" / "anchors"
        anchors.mkdir(parents=True)
        (anchors / "p_anchor_frontal_neutral.png").write_bytes(PNG_HEADER)

        client = fake_client(image_bytes=b"ok")
        gen = ImageGenerator(client=client, model_name="m", aspect_ratio="4:5", references_root=tmp_path)
        gen.generate_asset_image(
            persona_name="p",
            scenario_template={"text": "go"},
            variables={"X": "Y"},
        )
        call = client.calls[0]
        # identity lock + 1 anchor BytesPart + prompt text = at least 3 parts
        assert call["num_parts"] >= 3
