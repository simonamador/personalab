"""Pytest configuration and shared fixtures."""

from typing import Any

import pytest

from personalab import Character, ImagePrompt, VideoPrompt
from personalab.config.loader import ProjectConfig
from personalab.llm.client import LLMTextResponse, LLMImageResponse, LLMVideoResponse


# ---------------------------------------------------------------------------
# Fake LLM client for generation tests (conforms to LLMClient protocol)
# ---------------------------------------------------------------------------


class FakeLLMClient:
    """In-memory LLM client that returns canned responses for unit tests."""

    def __init__(
        self,
        *,
        text_json_response: LLMTextResponse | None = None,
        image_bytes: bytes | None = None,
        video_response: LLMVideoResponse | None = None,
    ) -> None:
        self._text_json_response = text_json_response or LLMTextResponse()
        self._image_bytes = image_bytes
        self._video_response = video_response
        self.calls: list[dict[str, Any]] = []

    def generate_text_json(self, *, system_instruction, user_prompt, schema, use_search, model_name) -> LLMTextResponse:
        self.calls.append({"method": "generate_text_json", "model": model_name})
        return self._text_json_response

    def generate_image(self, *, parts, aspect_ratio, model_name) -> LLMImageResponse:
        self.calls.append({"method": "generate_image", "num_parts": len(parts), "model": model_name})
        images = [self._image_bytes] if self._image_bytes is not None else []
        return LLMImageResponse(images=images)

    def generate_video(self, *, prompt, resolution, aspect_ratio, model_name, reference_images=None) -> LLMVideoResponse:
        self.calls.append({
            "method": "generate_video",
            "model": model_name,
            "reference_images_count": len(reference_images) if reference_images else 0,
        })
        return self._video_response or LLMVideoResponse(operation_name="op-123")

    # -- async variants (mirror sync) ---------------------------------------

    async def generate_text_json_async(self, *, system_instruction, user_prompt, schema, use_search, model_name) -> LLMTextResponse:
        return self.generate_text_json(
            system_instruction=system_instruction, user_prompt=user_prompt,
            schema=schema, use_search=use_search, model_name=model_name,
        )

    async def generate_image_async(self, *, parts, aspect_ratio, model_name) -> LLMImageResponse:
        return self.generate_image(parts=parts, aspect_ratio=aspect_ratio, model_name=model_name)

    async def generate_video_async(self, *, prompt, resolution, aspect_ratio, model_name, reference_images=None) -> LLMVideoResponse:
        return self.generate_video(
            prompt=prompt, resolution=resolution, aspect_ratio=aspect_ratio,
            model_name=model_name, reference_images=reference_images,
        )


@pytest.fixture
def fake_client():
    """FakeLLMClient factory; call with keyword args to configure responses."""
    def _factory(**kwargs):
        return FakeLLMClient(**kwargs)
    return _factory


@pytest.fixture
def sample_config():
    """Minimal ProjectConfig for tests."""
    return ProjectConfig(raw={
        "paths": {"output": "./out", "references": "./refs", "prompts": "./prompts"},
        "models": {
            "text": {"provider": "gemini", "model_name": "gemini-test"},
            "image": {"provider": "gemini", "model_name": "imagen-test"},
            "video": {"provider": "gemini", "model_name": "veo-test"},
        },
        "generation": {
            "image": {"aspect_ratio": "4:5"},
            "video": {"resolution": "1080p", "aspect_ratio": "9:16"},
            "search": {"enabled": False},
        },
    })


@pytest.fixture
def sample_character():
    """Minimal Character for tests."""
    return Character(
        name="Test Persona",
        vibe="Lifestyle",
        location="Lima, Peru",
        content_pillars=["Fashion", "Food"],
        physical_description={"age_range": "25"},
    )


@pytest.fixture
def sample_image_prompt():
    """Minimal ImagePrompt for tests."""
    return ImagePrompt(
        subject_description="test",
        scene_description="cafe",
        clothing_details="casual",
        shot_type="selfie",
    )


@pytest.fixture
def sample_video_prompt():
    """Minimal VideoPrompt for tests."""
    return VideoPrompt(
        subject_description="test",
        action_details="walking",
        location_details="street",
    )
