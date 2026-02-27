"""Factory for building LLMClient instances from ProjectConfig."""

from __future__ import annotations

from typing import Any

from personalab.config.loader import ProjectConfig
from personalab.llm.client import (
    LLMClient,
    BytesPart,
    ContentPart,
    LLMTextResponse,
    LLMImageResponse,
    LLMVideoResponse,
)


class RoutingClient:
    """Composes per-modality adapters behind a single LLMClient interface.

    When every modality uses the same provider, callers can pass a single
    adapter directly.  When providers differ (e.g. OpenAI for text, Runware
    for images) this class routes each method to the correct backend.
    """

    def __init__(
        self,
        *,
        text: LLMClient,
        image: LLMClient,
        video: LLMClient,
    ) -> None:
        self._text = text
        self._image = image
        self._video = video

    def generate_text_json(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
        schema: dict[str, Any],
        use_search: bool,
        model_name: str,
    ) -> LLMTextResponse:
        return self._text.generate_text_json(
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            schema=schema,
            use_search=use_search,
            model_name=model_name,
        )

    def generate_image(
        self,
        *,
        parts: list[ContentPart],
        aspect_ratio: str,
        model_name: str,
    ) -> LLMImageResponse:
        return self._image.generate_image(
            parts=parts,
            aspect_ratio=aspect_ratio,
            model_name=model_name,
        )

    def generate_video(
        self,
        *,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        model_name: str,
        reference_images: list[ContentPart] | None = None,
    ) -> LLMVideoResponse:
        return self._video.generate_video(
            prompt=prompt,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            model_name=model_name,
            reference_images=reference_images,
        )

    # -- async variants -----------------------------------------------------

    async def generate_text_json_async(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
        schema: dict[str, Any],
        use_search: bool,
        model_name: str,
    ) -> LLMTextResponse:
        return await self._text.generate_text_json_async(
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            schema=schema,
            use_search=use_search,
            model_name=model_name,
        )

    async def generate_image_async(
        self,
        *,
        parts: list[ContentPart],
        aspect_ratio: str,
        model_name: str,
    ) -> LLMImageResponse:
        return await self._image.generate_image_async(
            parts=parts,
            aspect_ratio=aspect_ratio,
            model_name=model_name,
        )

    async def generate_video_async(
        self,
        *,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        model_name: str,
        reference_images: list[BytesPart] | None = None,
    ) -> LLMVideoResponse:
        return await self._video.generate_video_async(
            prompt=prompt,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            model_name=model_name,
            reference_images=reference_images,
        )


# ---------------------------------------------------------------------------
# Provider registry  (provider key -> callable that returns an LLMClient)
# ---------------------------------------------------------------------------

def _make_gemini(**kwargs: Any) -> LLMClient:
    from personalab.llm.gemini import GeminiAdapter
    return GeminiAdapter(**kwargs)


def _make_openai(**kwargs: Any) -> LLMClient:
    from personalab.llm.openai_adapter import OpenAIAdapter
    return OpenAIAdapter(**kwargs)


def _make_runware(**kwargs: Any) -> LLMClient:
    from personalab.llm.runware_adapter import RunwareAdapter
    return RunwareAdapter(**kwargs)


def _make_replicate(**kwargs: Any) -> LLMClient:
    from personalab.llm.replicate_adapter import ReplicateAdapter
    return ReplicateAdapter(**kwargs)


def _make_runway(**kwargs: Any) -> LLMClient:
    from personalab.llm.runway_adapter import RunwayAdapter
    return RunwayAdapter(**kwargs)


_REGISTRY: dict[str, Any] = {
    "gemini": _make_gemini,
    "openai": _make_openai,
    "runware": _make_runware,
    "replicate": _make_replicate,
    "runway": _make_runway,
}


def create_client(config: ProjectConfig) -> LLMClient:
    """Build an LLMClient from config, routing per-modality when providers differ.

    If all three modalities share the same provider key, a single adapter is
    returned.  Otherwise a :class:`RoutingClient` is constructed with one
    adapter per modality.
    """
    providers = {
        mod: config.provider(mod)
        for mod in ("text", "image", "video")
    }

    def _instantiate(provider_key: str) -> LLMClient:
        factory_fn = _REGISTRY.get(provider_key)
        if factory_fn is None:
            raise ValueError(
                f"Unknown LLM provider '{provider_key}'. "
                f"Available: {sorted(_REGISTRY)}"
            )
        return factory_fn()

    unique_providers = set(providers.values())
    if len(unique_providers) == 1:
        return _instantiate(unique_providers.pop())

    # Cache adapters so two modalities sharing a provider share an instance.
    cache: dict[str, LLMClient] = {}
    for mod in ("text", "image", "video"):
        key = providers[mod]
        if key not in cache:
            cache[key] = _instantiate(key)

    return RoutingClient(
        text=cache[providers["text"]],
        image=cache[providers["image"]],
        video=cache[providers["video"]],
    )
