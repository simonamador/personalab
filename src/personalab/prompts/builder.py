"""Contract for converting user prompts to the project's prompt format."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from personalab.config.loader import ProjectConfig
from personalab.llm.client import LLMClient
from personalab.schemas import ImagePrompt, ScenarioDetails, VideoPrompt
from personalab.schemas.llm_schemas import get_llm_schemas

logger = logging.getLogger(__name__)


class PromptBuilder(ABC):
    """Converts a user's natural-language prompt into structured ImagePrompt / VideoPrompt (and related data)."""

    @abstractmethod
    def user_to_image_prompt(self, user_prompt: str, context: dict[str, Any] | None = None) -> ImagePrompt:
        """Map user text + optional context to ImagePrompt (variables, policy_overrides)."""
        ...

    @abstractmethod
    def user_to_video_prompt(self, user_prompt: str, context: dict[str, Any] | None = None) -> VideoPrompt:
        """Map user text + optional context to VideoPrompt (variables)."""
        ...


class PassThroughPromptBuilder(PromptBuilder):
    """Maps user prompt + context to ImagePrompt without calling the LLM. Context supplies subject_description, scene_description, clothing_details, shot_type; user_prompt defaults to scene_description."""

    def user_to_image_prompt(self, user_prompt: str, context: dict[str, Any] | None = None) -> ImagePrompt:
        ctx = context or {}
        return ImagePrompt(
            subject_description=ctx.get("subject_description", ""),
            scene_description=ctx.get("scene_description", user_prompt),
            clothing_details=ctx.get("clothing_details", ""),
            shot_type=ctx.get("shot_type", ""),
            policy_overrides=ctx.get("policy_overrides"),
        )

    def user_to_video_prompt(self, user_prompt: str, context: dict[str, Any] | None = None) -> VideoPrompt:
        ctx = context or {}
        return VideoPrompt(
            subject_description=ctx.get("subject_description", ""),
            action_details=ctx.get("action_details", user_prompt),
            location_details=ctx.get("location_details", ""),
            mood_and_expression=ctx.get("mood_and_expression", ""),
            policy_overrides=ctx.get("policy_overrides"),
        )


class GeminiPromptBuilder(PromptBuilder):
    """Builds ImagePrompt by calling Gemini to generate scenario_details (scene, outfit, framing) from the user prompt. Uses Pydantic-derived schema and user_prompt_to_scenario system prompt."""

    def __init__(
        self,
        client: LLMClient,
        config: ProjectConfig,
        system_prompts: dict[str, Any],
        schemas: dict[str, Any] | None = None,
    ) -> None:
        self._client = client
        self._config = config
        self._system_prompts = system_prompts
        self._schemas = schemas if schemas is not None else get_llm_schemas()

    def user_to_image_prompt(self, user_prompt: str, context: dict[str, Any] | None = None) -> ImagePrompt:
        block = self._system_prompts.get("user_prompt_to_scenario")
        if not block:
            raise ValueError("System prompts must contain 'user_prompt_to_scenario'")
        system_instruction = block.get("system", "")
        schema = self._schemas.get("scenario_details_schema")
        if not schema:
            raise ValueError("scenario_details_schema missing; use get_llm_schemas() or pass schemas dict")
        model_name = self._config.model_name("text") or "gemini-2.0-flash"
        resp = self._client.generate_text_json(
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            schema=schema,
            use_search=False,
            model_name=model_name,
        )
        parsed = getattr(resp, "parsed", None)
        if parsed is None:
            text = getattr(resp, "text", None) or str(resp)
            try:
                parsed = json.loads(text) if isinstance(text, str) else text
            except json.JSONDecodeError as exc:
                raise ValueError(f"LLM response is not valid JSON: {text!r}") from exc
        if not isinstance(parsed, dict):
            raise TypeError(f"Expected dict from LLM, got {type(parsed).__name__}: {parsed!r}")
        scenario_details = ScenarioDetails(
            scene=parsed.get("scene", ""),
            outfit=parsed.get("outfit", ""),
            framing=parsed.get("framing", ""),
        )
        subject_description = (context or {}).get("subject_description", "")
        return ImagePrompt.from_scenario_details(scenario_details, subject_description=subject_description)

    def user_to_video_prompt(self, user_prompt: str, context: dict[str, Any] | None = None) -> VideoPrompt:
        ctx = context or {}
        return VideoPrompt(
            subject_description=ctx.get("subject_description", ""),
            action_details=ctx.get("action_details", user_prompt),
            location_details=ctx.get("location_details", ""),
            mood_and_expression=ctx.get("mood_and_expression", ""),
            policy_overrides=ctx.get("policy_overrides"),
        )
