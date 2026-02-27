"""Gemini adapter implementing LLMClient (sync + async)."""

import asyncio
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

from google.genai import types
from google import genai

from personalab.llm.client import (
    ContentPart, TextPart, BytesPart,
    LLMTextResponse, LLMImageResponse, LLMVideoResponse,
)

logger = logging.getLogger(__name__)


class GeminiAdapter:
    """Thin adapter over Google GenAI client. Implements LLMClient; config injected from outside."""

    def __init__(self, api_key: str | None = None) -> None:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY must be set or passed as api_key.")
        self._client = genai.Client(api_key=key)

    @staticmethod
    def _to_genai_parts(parts: list[ContentPart]) -> list[types.Part]:
        """Convert provider-agnostic ContentPart items to Gemini SDK Part objects."""
        out: list[types.Part] = []
        for p in parts:
            if isinstance(p, TextPart):
                out.append(types.Part.from_text(text=p.text))
            elif isinstance(p, BytesPart):
                out.append(types.Part.from_bytes(data=p.data, mime_type=p.mime_type))
            else:
                raise TypeError(f"Unknown part type: {type(p)}")
        return out

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
        """Generate a JSON response, optionally grounded with Google Search."""
        tools = [types.Tool(google_search=types.GoogleSearch())] if use_search else []
        resp = self._client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=tools,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        return LLMTextResponse(
            parsed=getattr(resp, "parsed", None) or None,
            text=getattr(resp, "text", "") or "",
        )

    # -- image --------------------------------------------------------------

    def generate_image(
        self,
        *,
        parts: list[ContentPart],
        aspect_ratio: str,
        model_name: str,
    ) -> LLMImageResponse:
        """Generate an image from multimodal ContentPart inputs."""
        genai_parts = self._to_genai_parts(parts)
        resp = self._client.models.generate_content(
            model=model_name,
            contents=genai_parts,
            config=types.GenerateContentConfig(
                image_config=types.ImageConfig(aspect_ratio=aspect_ratio)
            ),
        )
        images: list[bytes] = []
        try:
            for part in resp.candidates[0].content.parts:
                if getattr(part, "inline_data", None):
                    images.append(part.inline_data.data)
        except (IndexError, AttributeError):
            pass
        return LLMImageResponse(images=images)

    # -- video --------------------------------------------------------------

    _VIDEO_POLL_INTERVAL = 10.0
    _VIDEO_MAX_POLL_SECONDS = 600.0

    def generate_video(
        self,
        *,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        model_name: str,
        reference_images: list[BytesPart] | None = None,
    ) -> LLMVideoResponse:
        """Generate a video with optional reference images, polling until complete."""
        config_kwargs: dict[str, Any] = {
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
        }

        if reference_images:
            config_kwargs["reference_images"] = [
                types.VideoGenerationReferenceImage(
                    image=types.Image(
                        image_bytes=img.data,
                        mime_type=img.mime_type,
                    ),
                    reference_type="asset",
                )
                for img in reference_images
            ]
            config_kwargs["person_generation"] = "allow_adult"

        op = self._call_generate_videos(model_name, prompt, config_kwargs)

        op_name = getattr(op, "name", "") or ""
        logger.info("Video generation started: %s", op_name)

        elapsed = 0.0
        while not getattr(op, "done", False) and elapsed < self._VIDEO_MAX_POLL_SECONDS:
            time.sleep(self._VIDEO_POLL_INTERVAL)
            elapsed += self._VIDEO_POLL_INTERVAL
            op = self._client.operations.get(op)
            logger.debug("Polling video op %s (%.0fs elapsed)", op_name, elapsed)

        if not getattr(op, "done", False):
            logger.warning("Video generation timed out after %.0fs: %s", elapsed, op_name)
            return LLMVideoResponse(operation_name=op_name)

        video_data = self._download_video(op)
        return LLMVideoResponse(
            video_data=video_data,
            operation_name=op_name,
        )

    _RATE_LIMIT_MAX_RETRIES = 3
    _RATE_LIMIT_BASE_DELAY = 30.0

    def _call_generate_videos(
        self, model_name: str, prompt: str, config_kwargs: dict[str, Any],
    ) -> Any:
        """Call generate_videos with retry on 429 RESOURCE_EXHAUSTED."""
        from google.genai.errors import ClientError

        for attempt in range(1, self._RATE_LIMIT_MAX_RETRIES + 1):
            try:
                return self._client.models.generate_videos(
                    model=model_name,
                    prompt=prompt,
                    config=types.GenerateVideosConfig(**config_kwargs),
                )
            except ClientError as exc:
                if exc.code == 429 and attempt < self._RATE_LIMIT_MAX_RETRIES:
                    delay = self._RATE_LIMIT_BASE_DELAY * attempt
                    logger.warning(
                        "Rate-limited (429). Waiting %.0fs before retry %d/%d...",
                        delay, attempt, self._RATE_LIMIT_MAX_RETRIES,
                    )
                    time.sleep(delay)
                else:
                    raise

    def _download_video(self, operation: Any) -> bytes | None:
        """Download video bytes from a completed Veo operation.

        Tries ``operation.result`` first (standard path), then falls back to
        ``operation.response`` for older SDK versions.
        """
        try:
            container = getattr(operation, "result", None) or getattr(operation, "response", None)
            if container is None:
                logger.warning("Operation has neither .result nor .response")
                return None

            generated_videos = getattr(container, "generated_videos", None)
            if not generated_videos:
                logger.warning("No generated_videos in operation result")
                return None

            video_obj = generated_videos[0].video

            video_bytes = getattr(video_obj, "video_bytes", None)
            if video_bytes:
                return video_bytes

            if hasattr(video_obj, "uri") and video_obj.uri:
                self._client.files.download(file=video_obj)
                with tempfile.TemporaryDirectory() as tmpdir:
                    out_path = Path(tmpdir) / "video.mp4"
                    video_obj.save(str(out_path))
                    return out_path.read_bytes()

            logger.warning("Video object has no bytes or downloadable URI")
            return None
        except Exception:
            logger.exception("Failed to download video from operation")
            return None

    # ===================================================================
    # Async variants (used by AsyncRunner, non-blocking event loop)
    # ===================================================================

    async def generate_text_json_async(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
        schema: dict[str, Any],
        use_search: bool,
        model_name: str,
    ) -> LLMTextResponse:
        """Async variant using ``client.aio``."""
        tools = [types.Tool(google_search=types.GoogleSearch())] if use_search else []
        resp = await self._client.aio.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=tools,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        return LLMTextResponse(
            parsed=getattr(resp, "parsed", None) or None,
            text=getattr(resp, "text", "") or "",
        )

    async def generate_image_async(
        self,
        *,
        parts: list[ContentPart],
        aspect_ratio: str,
        model_name: str,
    ) -> LLMImageResponse:
        """Async variant using ``client.aio``."""
        genai_parts = self._to_genai_parts(parts)
        resp = await self._client.aio.models.generate_content(
            model=model_name,
            contents=genai_parts,
            config=types.GenerateContentConfig(
                image_config=types.ImageConfig(aspect_ratio=aspect_ratio)
            ),
        )
        images: list[bytes] = []
        try:
            for part in resp.candidates[0].content.parts:
                if getattr(part, "inline_data", None):
                    images.append(part.inline_data.data)
        except (IndexError, AttributeError):
            pass
        return LLMImageResponse(images=images)

    async def generate_video_async(
        self,
        *,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        model_name: str,
        reference_images: list[BytesPart] | None = None,
    ) -> LLMVideoResponse:
        """Async variant: submits via sync SDK then polls with ``asyncio.sleep``."""
        config_kwargs: dict[str, Any] = {
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
        }
        if reference_images:
            config_kwargs["reference_images"] = [
                types.VideoGenerationReferenceImage(
                    image=types.Image(
                        image_bytes=img.data,
                        mime_type=img.mime_type,
                    ),
                    reference_type="asset",
                )
                for img in reference_images
            ]
            config_kwargs["person_generation"] = "allow_adult"

        loop = asyncio.get_running_loop()
        op = await loop.run_in_executor(
            None, lambda: self._call_generate_videos(model_name, prompt, config_kwargs),
        )

        op_name = getattr(op, "name", "") or ""
        logger.info("Video generation started (async): %s", op_name)

        elapsed = 0.0
        while not getattr(op, "done", False) and elapsed < self._VIDEO_MAX_POLL_SECONDS:
            await asyncio.sleep(self._VIDEO_POLL_INTERVAL)
            elapsed += self._VIDEO_POLL_INTERVAL
            op = await loop.run_in_executor(None, lambda: self._client.operations.get(op))
            logger.debug("Polling video op %s (%.0fs elapsed)", op_name, elapsed)

        if not getattr(op, "done", False):
            logger.warning("Video generation timed out after %.0fs: %s", elapsed, op_name)
            return LLMVideoResponse(operation_name=op_name)

        video_data = self._download_video(op)
        return LLMVideoResponse(video_data=video_data, operation_name=op_name)
