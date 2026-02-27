"""Generates content plans using system prompts and response schema from Pydantic."""

import json
from pathlib import Path
from typing import Any

import yaml

from personalab.config.loader import ProjectConfig
from personalab.llm.client import LLMClient
from personalab.prompts.renderer import render_string
from personalab.schemas import Character
from personalab.schemas.llm_schemas import get_llm_schemas


class IdeaGenerator:
    """Generates and saves a content plan using system prompts and response schema."""

    def __init__(
        self,
        client: LLMClient,
        config: ProjectConfig,
        persona: Character | dict[str, Any],
        system_prompts: dict[str, Any],
        schemas: dict[str, Any] | None = None,
    ) -> None:
        self._client = client
        self._config = config
        self._persona = persona
        self._system_prompts = system_prompts
        self._schemas = schemas if schemas is not None else get_llm_schemas()

    def _persona_dict(self) -> dict[str, Any]:
        if isinstance(self._persona, Character):
            return self._persona.to_persona_dict()
        return self._persona

    def generate(self) -> dict[str, Any]:
        """Generate a content plan as a Python dict."""
        block = self._system_prompts.get("content_strategist")
        if not block:
            raise KeyError("System prompts must contain 'content_strategist'")
        system_tpl = block.get("system", "")
        user_prompt = block.get("user", "")
        persona = self._persona_dict()

        pillars = persona.get("content_pillars", [])
        sys_instr = render_string(system_tpl, {
            "NAME": persona.get("name", ""),
            "VIBE": persona.get("vibe", ""),
            "LOCATION": persona.get("location", ""),
            "PILLARS": ", ".join(pillars) if isinstance(pillars, list) else str(pillars),
        })

        schema = self._schemas.get("content_plan_schema")
        if not schema:
            raise ValueError("content_plan_schema missing; use get_llm_schemas() or pass schemas dict")

        use_search = bool(self._config.generation.get("search", {}).get("enabled", True))
        model_name = self._config.model_name("text")
        if not model_name:
            raise ValueError("config.models.text must be set")

        resp = self._client.generate_text_json(
            system_instruction=sys_instr,
            user_prompt=user_prompt,
            schema=schema,
            use_search=use_search,
            model_name=model_name,
        )

        if resp.parsed:
            return resp.parsed
        return json.loads(resp.text)

    def save(self, plan: dict[str, Any], path: str | Path) -> Path:
        """Save the plan to a YAML file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(plan, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return out

    def generate_and_save(self, path: str | Path) -> dict[str, Any]:
        """Generate the plan and save it to disk."""
        plan = self.generate()
        self.save(plan, path)
        return plan
