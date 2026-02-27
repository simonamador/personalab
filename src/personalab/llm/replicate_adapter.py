"""Replicate adapter implementing LLMClient (image modality, sync + async)."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from personalab.llm.client import (
    ContentPart,
    TextPart,
    LLMTextResponse,
    LLMImageResponse,
    LLMVideoResponse,
)


class ReplicateAdapter:
    """Image generation via Replicate's prediction API (Stable Diffusion, Flux, etc.).

    Only ``generate_image`` is supported; text and video methods raise
    ``NotImplementedError``.
    """

    def __init__(self, api_token: str | None = None) -> None:
        try:
            import replicate as _replicate  # noqa: F811
        except ImportError as exc:
            raise ImportError(
                "The 'replicate' package is required for ReplicateAdapter. "
                "Install it with: pip install replicate"
            ) from exc

        from dotenv import load_dotenv
        import os

        load_dotenv()
        token = api_token or os.environ.get("REPLICATE_API_TOKEN")
        if not token:
            raise ValueError("REPLICATE_API_TOKEN must be set or passed as api_token.")
        os.environ["REPLICATE_API_TOKEN"] = token
        self._client = _replicate.Client(api_token=token)

    @staticmethod
    def _parse_aspect_ratio(aspect_ratio: str) -> tuple[int, int]:
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
        """Run a Replicate image model and return the generated image bytes."""
        prompt_pieces: list[str] = []
        for p in parts:
            if isinstance(p, TextPart):
                prompt_pieces.append(p.text)
        prompt = "\n".join(prompt_pieces)

        width, height = self._parse_aspect_ratio(aspect_ratio)
        output = self._client.run(
            model_name,
            input={
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_outputs": 1,
            },
        )

        images: list[bytes] = []
        urls = output if isinstance(output, list) else [output]
        with httpx.Client() as http:
            for url in urls:
                if isinstance(url, str) and url.startswith("http"):
                    resp = http.get(url)
                    resp.raise_for_status()
                    images.append(resp.content)
                elif isinstance(url, bytes):
                    images.append(url)
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
        raise NotImplementedError("ReplicateAdapter does not support text generation. Use a dedicated text adapter.")

    def generate_video(
        self,
        *,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        model_name: str,
        reference_images: list[ContentPart] | None = None,
    ) -> LLMVideoResponse:
        raise NotImplementedError("ReplicateAdapter does not support video generation. Use a dedicated video adapter.")

    # -- async variants -----------------------------------------------------

    async def generate_image_async(
        self,
        *,
        parts: list[ContentPart],
        aspect_ratio: str,
        model_name: str,
    ) -> LLMImageResponse:
        """Async variant: run model in executor, download with httpx.AsyncClient."""
        prompt_pieces: list[str] = []
        for p in parts:
            if isinstance(p, TextPart):
                prompt_pieces.append(p.text)
        prompt = "\n".join(prompt_pieces)

        width, height = self._parse_aspect_ratio(aspect_ratio)
        loop = asyncio.get_running_loop()
        output = await loop.run_in_executor(
            None,
            lambda: self._client.run(
                model_name,
                input={"prompt": prompt, "width": width, "height": height, "num_outputs": 1},
            ),
        )

        images: list[bytes] = []
        urls = output if isinstance(output, list) else [output]
        async with httpx.AsyncClient() as http:
            for url in urls:
                if isinstance(url, str) and url.startswith("http"):
                    resp = await http.get(url)
                    resp.raise_for_status()
                    images.append(resp.content)
                elif isinstance(url, bytes):
                    images.append(url)
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
        raise NotImplementedError("ReplicateAdapter does not support text generation. Use a dedicated text adapter.")

    async def generate_video_async(
        self,
        *,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        model_name: str,
        reference_images: list[ContentPart] | None = None,
    ) -> LLMVideoResponse:
        raise NotImplementedError("ReplicateAdapter does not support video generation. Use a dedicated video adapter.")
