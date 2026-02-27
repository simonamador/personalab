"""OpenAI adapter implementing LLMClient (text/JSON modality, sync + async)."""

from __future__ import annotations

import json
from typing import Any

from personalab.llm.client import (
    ContentPart,
    LLMTextResponse,
    LLMImageResponse,
    LLMVideoResponse,
)


class OpenAIAdapter:
    """Text generation via the OpenAI Chat Completions API with JSON mode.

    Only ``generate_text_json`` is supported; image and video methods raise
    ``NotImplementedError`` (use a dedicated adapter for those modalities via
    :class:`~personalab.llm.factory.RoutingClient`).
    """

    def __init__(self, api_key: str | None = None) -> None:
        try:
            import openai  # noqa: F811
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenAIAdapter. "
                "Install it with: pip install openai"
            ) from exc

        from dotenv import load_dotenv
        import os

        load_dotenv()
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY must be set or passed as api_key.")
        self._client = openai.OpenAI(api_key=key)

    # -- text ---------------------------------------------------------------

    def generate_text_json(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
        schema: dict[str, Any],
        use_search: bool,
        model_name: str,
    ) -> LLMTextResponse:
        """Generate a JSON response via OpenAI Chat Completions."""
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
        ]
        resp = self._client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        text = resp.choices[0].message.content or ""
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        return LLMTextResponse(parsed=parsed, text=text)

    # -- unsupported modalities ---------------------------------------------

    def generate_image(
        self,
        *,
        parts: list[ContentPart],
        aspect_ratio: str,
        model_name: str,
    ) -> LLMImageResponse:
        raise NotImplementedError("OpenAIAdapter does not support image generation. Use a dedicated image adapter.")

    def generate_video(
        self,
        *,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        model_name: str,
        reference_images: list[ContentPart] | None = None,
    ) -> LLMVideoResponse:
        raise NotImplementedError("OpenAIAdapter does not support video generation. Use a dedicated video adapter.")

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
        """Async variant using ``openai.AsyncOpenAI``."""
        import openai
        aclient = openai.AsyncOpenAI(api_key=self._client.api_key)
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
        ]
        resp = await aclient.chat.completions.create(
            model=model_name,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        text = resp.choices[0].message.content or ""
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        return LLMTextResponse(parsed=parsed, text=text)

    async def generate_image_async(
        self,
        *,
        parts: list[ContentPart],
        aspect_ratio: str,
        model_name: str,
    ) -> LLMImageResponse:
        raise NotImplementedError("OpenAIAdapter does not support image generation. Use a dedicated image adapter.")

    async def generate_video_async(
        self,
        *,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        model_name: str,
        reference_images: list[ContentPart] | None = None,
    ) -> LLMVideoResponse:
        raise NotImplementedError("OpenAIAdapter does not support video generation. Use a dedicated video adapter.")
