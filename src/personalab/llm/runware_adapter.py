"""Runware adapter implementing LLMClient (image modality)."""

from __future__ import annotations

import asyncio
from typing import Any

from personalab.llm.client import (
    ContentPart,
    TextPart,
    LLMTextResponse,
    LLMImageResponse,
    LLMVideoResponse,
)


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Return the running loop or create a new one for sync contexts."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool  # type: ignore[return-value]
    return asyncio.new_event_loop()


class RunwareAdapter:
    """Image generation via the Runware WebSocket API.

    Only ``generate_image`` is supported; text and video methods raise
    ``NotImplementedError``.

    The Runware SDK is async-first.  This adapter bridges to sync by running
    the coroutine in an event loop, making it transparent for callers.
    """

    def __init__(self, api_key: str | None = None) -> None:
        try:
            from runware import Runware  # noqa: F811
        except ImportError as exc:
            raise ImportError(
                "The 'runware' package is required for RunwareAdapter. "
                "Install it with: pip install runware"
            ) from exc

        from dotenv import load_dotenv
        import os

        load_dotenv()
        self._api_key = api_key or os.environ.get("RUNWARE_API_KEY")
        if not self._api_key:
            raise ValueError("RUNWARE_API_KEY must be set or passed as api_key.")

    async def _generate_async(
        self,
        prompt: str,
        model: str,
        width: int,
        height: int,
    ) -> list[bytes]:
        from runware import Runware, IImageInference
        import httpx

        runware = Runware(api_key=self._api_key)
        await runware.connect()
        try:
            request = IImageInference(
                positivePrompt=prompt,
                model=model,
                width=width,
                height=height,
                numberResults=1,
            )
            images = await runware.imageInference(requestImage=request)
        finally:
            await runware.disconnect()

        result: list[bytes] = []
        async with httpx.AsyncClient() as http:
            for img in images:
                url = getattr(img, "imageURL", None) or getattr(img, "image_url", "")
                if url:
                    resp = await http.get(url)
                    resp.raise_for_status()
                    result.append(resp.content)
        return result

    @staticmethod
    def _parse_aspect_ratio(aspect_ratio: str) -> tuple[int, int]:
        """Convert an aspect ratio string like ``"4:5"`` to (width, height) in pixels."""
        _MAP = {
            "1:1": (1024, 1024),
            "4:5": (832, 1024),
            "5:4": (1024, 832),
            "16:9": (1024, 576),
            "9:16": (576, 1024),
            "3:2": (1024, 680),
            "2:3": (680, 1024),
        }
        return _MAP.get(aspect_ratio, (1024, 1024))

    # -- image --------------------------------------------------------------

    def generate_image(
        self,
        *,
        parts: list[ContentPart],
        aspect_ratio: str,
        model_name: str,
    ) -> LLMImageResponse:
        """Generate an image via Runware from ContentPart inputs."""
        prompt_pieces: list[str] = []
        for p in parts:
            if isinstance(p, TextPart):
                prompt_pieces.append(p.text)
        prompt = "\n".join(prompt_pieces)

        width, height = self._parse_aspect_ratio(aspect_ratio)
        images = asyncio.run(self._generate_async(prompt, model_name, width, height))
        return LLMImageResponse(images=images)

    # -- unsupported modalities ---------------------------------------------

    def generate_text_json(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
        schema: dict[str, Any],
        use_search: bool,
        model_name: str,
    ) -> LLMTextResponse:
        raise NotImplementedError("RunwareAdapter does not support text generation. Use a dedicated text adapter.")

    def generate_video(
        self,
        *,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        model_name: str,
        reference_images: list[ContentPart] | None = None,
    ) -> LLMVideoResponse:
        raise NotImplementedError("RunwareAdapter does not support video generation. Use a dedicated video adapter.")

    # -- async variants -----------------------------------------------------

    async def generate_image_async(
        self,
        *,
        parts: list[ContentPart],
        aspect_ratio: str,
        model_name: str,
    ) -> LLMImageResponse:
        """Async image generation -- directly uses the internal coroutine."""
        prompt_pieces: list[str] = []
        for p in parts:
            if isinstance(p, TextPart):
                prompt_pieces.append(p.text)
        prompt = "\n".join(prompt_pieces)
        width, height = self._parse_aspect_ratio(aspect_ratio)
        images = await self._generate_async(prompt, model_name, width, height)
        return LLMImageResponse(images=images)

    async def generate_text_json_async(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
        schema: dict[str, Any],
        use_search: bool,
        model_name: str,
    ) -> LLMTextResponse:
        raise NotImplementedError("RunwareAdapter does not support text generation. Use a dedicated text adapter.")

    async def generate_video_async(
        self,
        *,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        model_name: str,
        reference_images: list[ContentPart] | None = None,
    ) -> LLMVideoResponse:
        raise NotImplementedError("RunwareAdapter does not support video generation. Use a dedicated video adapter.")
