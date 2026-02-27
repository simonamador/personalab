"""Protocol for LLM clients (dependency inversion)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Input parts (multimodal content sent TO the model)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TextPart:
    """Text content for multimodal generation."""
    text: str


@dataclass(frozen=True)
class BytesPart:
    """Binary content (image/audio) for multimodal generation."""
    data: bytes
    mime_type: str


ContentPart = TextPart | BytesPart


# ---------------------------------------------------------------------------
# Normalized response types (returned FROM every adapter)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMTextResponse:
    """Normalized text/JSON response. Adapters populate *parsed* when the
    provider returns structured output, otherwise fall back to *text*."""
    parsed: Any | None = None
    text: str = ""


@dataclass(frozen=True)
class LLMImageResponse:
    """Normalized image response. *images* contains raw bytes for each
    generated image (most providers return exactly one)."""
    images: list[bytes] = field(default_factory=list)


@dataclass(frozen=True)
class LLMVideoResponse:
    """Normalized video response. *video_data* is set when the provider
    returns the video synchronously; *operation_name* identifies an async job."""
    video_data: bytes | None = None
    operation_name: str = ""
    duration_s: float = 0.0


# ---------------------------------------------------------------------------
# Protocol (central contract)
# ---------------------------------------------------------------------------

class LLMClient(Protocol):
    """Interface for text, image and video generation.

    Every adapter exposes both **sync** and **async** variants.  Sync
    methods are the original API; async counterparts (``*_async``) are
    used by ``AsyncRunner`` to avoid blocking the event loop during
    I/O-bound provider calls.

    High-level code depends on this protocol, never on vendor SDKs.
    """

    # -- sync (original) ----------------------------------------------------

    def generate_text_json(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
        schema: dict[str, Any],
        use_search: bool,
        model_name: str,
    ) -> LLMTextResponse:
        """Generate a JSON response, optionally grounded with search."""
        ...

    def generate_image(
        self,
        *,
        parts: list[ContentPart],
        aspect_ratio: str,
        model_name: str,
    ) -> LLMImageResponse:
        """Generate image(s) from multimodal input."""
        ...

    def generate_video(
        self,
        *,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        model_name: str,
        reference_images: list[BytesPart] | None = None,
    ) -> LLMVideoResponse:
        """Generate video, optionally conditioned on reference images for identity."""
        ...

    # -- async --------------------------------------------------------------

    async def generate_text_json_async(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
        schema: dict[str, Any],
        use_search: bool,
        model_name: str,
    ) -> LLMTextResponse:
        """Async variant of :meth:`generate_text_json`."""
        ...

    async def generate_image_async(
        self,
        *,
        parts: list[ContentPart],
        aspect_ratio: str,
        model_name: str,
    ) -> LLMImageResponse:
        """Async variant of :meth:`generate_image`."""
        ...

    async def generate_video_async(
        self,
        *,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        model_name: str,
        reference_images: list[BytesPart] | None = None,
    ) -> LLMVideoResponse:
        """Async variant of :meth:`generate_video`."""
        ...
