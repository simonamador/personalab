"""Runway adapter implementing LLMClient (video modality, sync + async)."""

from __future__ import annotations

import asyncio
import io
import logging
import time
from typing import Any

import httpx

from personalab.llm.client import (
    BytesPart,
    ContentPart,
    LLMTextResponse,
    LLMImageResponse,
    LLMVideoResponse,
)

logger = logging.getLogger(__name__)


class RunwayAdapter:
    """Video generation via the RunwayML API (Gen-3 / Gen-4).

    Only ``generate_video`` is supported; text and image methods raise
    ``NotImplementedError``.
    """

    _POLL_INTERVAL = 5.0
    _MAX_POLL_SECONDS = 600.0

    def __init__(self, api_key: str | None = None) -> None:
        try:
            from runwayml import RunwayML  # noqa: F811
        except ImportError as exc:
            raise ImportError(
                "The 'runwayml' package is required for RunwayAdapter. "
                "Install it with: pip install runwayml"
            ) from exc

        from dotenv import load_dotenv
        import os

        load_dotenv()
        key = api_key or os.environ.get("RUNWAY_API_KEY") or os.environ.get("RUNWAYML_API_SECRET")
        if not key:
            raise ValueError("RUNWAY_API_KEY must be set or passed as api_key.")
        self._client = RunwayML(api_key=key)

    # -- video --------------------------------------------------------------

    def generate_video(
        self,
        *,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        model_name: str,
        reference_images: list[BytesPart] | None = None,
    ) -> LLMVideoResponse:
        """Submit a video generation task and poll until completion."""
        _RATIO_MAP = {
            "9:16": "768:1344",
            "16:9": "1344:768",
            "1:1": "1024:1024",
        }
        ratio = _RATIO_MAP.get(aspect_ratio, "1344:768")

        create_kwargs: dict[str, Any] = {
            "model": model_name,
            "prompt_text": prompt,
            "ratio": ratio,
            "duration": 5,
        }

        if reference_images:
            first = reference_images[0]
            uri = f"data:{first.mime_type};base64,"
            import base64
            uri += base64.b64encode(first.data).decode()
            create_kwargs["prompt_image"] = uri
            logger.info("RunwayAdapter: using first reference image as prompt_image")

        task = self._client.image_to_video.create(**create_kwargs)
        task_id = getattr(task, "id", "") or str(task)

        elapsed = 0.0
        video_urls: list[str] = []
        while elapsed < self._MAX_POLL_SECONDS:
            time.sleep(self._POLL_INTERVAL)
            elapsed += self._POLL_INTERVAL
            status = self._client.tasks.retrieve(id=task_id)
            state = getattr(status, "status", "PENDING")
            if state == "SUCCEEDED":
                video_urls = getattr(status, "output", []) or []
                break
            if state in ("FAILED", "CANCELLED"):
                raise RuntimeError(f"Runway task {task_id} ended with status: {state}")

        video_data: bytes | None = None
        if video_urls:
            url = video_urls[0] if isinstance(video_urls, list) else str(video_urls)
            with httpx.Client() as http:
                resp = http.get(url)
                resp.raise_for_status()
                video_data = resp.content

        return LLMVideoResponse(
            video_data=video_data,
            operation_name=task_id,
        )

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
        raise NotImplementedError("RunwayAdapter does not support text generation. Use a dedicated text adapter.")

    def generate_image(
        self,
        *,
        parts: list[ContentPart],
        aspect_ratio: str,
        model_name: str,
    ) -> LLMImageResponse:
        raise NotImplementedError("RunwayAdapter does not support image generation. Use a dedicated image adapter.")

    # -- async variants -----------------------------------------------------

    async def generate_video_async(
        self,
        *,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        model_name: str,
        reference_images: list[BytesPart] | None = None,
    ) -> LLMVideoResponse:
        """Async variant: submits via sync SDK in executor, polls with asyncio.sleep."""
        _RATIO_MAP = {"9:16": "768:1344", "16:9": "1344:768", "1:1": "1024:1024"}
        ratio = _RATIO_MAP.get(aspect_ratio, "1344:768")

        create_kwargs: dict[str, Any] = {
            "model": model_name, "prompt_text": prompt, "ratio": ratio, "duration": 5,
        }
        if reference_images:
            import base64
            first = reference_images[0]
            uri = f"data:{first.mime_type};base64," + base64.b64encode(first.data).decode()
            create_kwargs["prompt_image"] = uri

        loop = asyncio.get_running_loop()
        task = await loop.run_in_executor(
            None, lambda: self._client.image_to_video.create(**create_kwargs),
        )
        task_id = getattr(task, "id", "") or str(task)

        elapsed = 0.0
        video_urls: list[str] = []
        while elapsed < self._MAX_POLL_SECONDS:
            await asyncio.sleep(self._POLL_INTERVAL)
            elapsed += self._POLL_INTERVAL
            status = await loop.run_in_executor(
                None, lambda: self._client.tasks.retrieve(id=task_id),
            )
            state = getattr(status, "status", "PENDING")
            if state == "SUCCEEDED":
                video_urls = getattr(status, "output", []) or []
                break
            if state in ("FAILED", "CANCELLED"):
                raise RuntimeError(f"Runway task {task_id} ended with status: {state}")

        video_data: bytes | None = None
        if video_urls:
            url = video_urls[0] if isinstance(video_urls, list) else str(video_urls)
            async with httpx.AsyncClient() as http:
                resp = await http.get(url)
                resp.raise_for_status()
                video_data = resp.content

        return LLMVideoResponse(video_data=video_data, operation_name=task_id)

    async def generate_text_json_async(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
        schema: dict[str, Any],
        use_search: bool,
        model_name: str,
    ) -> LLMTextResponse:
        raise NotImplementedError("RunwayAdapter does not support text generation. Use a dedicated text adapter.")

    async def generate_image_async(
        self,
        *,
        parts: list[ContentPart],
        aspect_ratio: str,
        model_name: str,
    ) -> LLMImageResponse:
        raise NotImplementedError("RunwayAdapter does not support image generation. Use a dedicated image adapter.")
