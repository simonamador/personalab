"""LLM client abstraction, normalized response types, and adapters."""

from personalab.llm.client import (
    LLMClient,
    TextPart,
    BytesPart,
    ContentPart,
    LLMTextResponse,
    LLMImageResponse,
    LLMVideoResponse,
)
from personalab.llm.gemini import GeminiAdapter
from personalab.llm.factory import RoutingClient, create_client

__all__ = [
    # Protocol & input types
    "LLMClient",
    "TextPart",
    "BytesPart",
    "ContentPart",
    # Response types
    "LLMTextResponse",
    "LLMImageResponse",
    "LLMVideoResponse",
    # Built-in adapter (always available)
    "GeminiAdapter",
    # Factory & routing
    "RoutingClient",
    "create_client",
]
