"""Build Gemini API response schemas from Pydantic models (single source of truth)."""

from typing import Any, get_origin

from pydantic import BaseModel

from personalab.schemas.character import Character
from personalab.schemas.prompts import ContentPlan, ScenarioDetails


def _pydantic_to_gemini_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model to Gemini response_schema format (OBJECT, STRING, ARRAY)."""
    schema: dict[str, Any] = {
        "type": "OBJECT",
        "properties": {},
        "required": [],
    }
    for name, field in model.model_fields.items():
        prop_schema: dict[str, Any]
        ann = field.annotation
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            prop_schema = _pydantic_to_gemini_schema(ann)
        else:
            origin = get_origin(ann)
            if origin is list:
                args = getattr(ann, "__args__", ())
                item_type = args[0] if args else None
                if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                    prop_schema = {
                        "type": "ARRAY",
                        "items": _pydantic_to_gemini_schema(item_type),
                    }
                else:
                    prop_schema = {"type": "ARRAY", "items": {"type": "STRING"}}
            elif origin is dict:
                prop_schema = {"type": "OBJECT"}
            else:
                prop_schema = {"type": "STRING"}
        if field.description:
            prop_schema["description"] = field.description
        schema["properties"][name] = prop_schema
        if field.is_required():
            schema["required"].append(name)
    if model.__doc__:
        schema["description"] = model.__doc__.split("\n")[0].strip()
    return schema


def get_scenario_details_schema() -> dict[str, Any]:
    """Gemini response_schema for a single scenario (scene, outfit, framing). Used by user_to_image_prompt."""
    schema = _pydantic_to_gemini_schema(ScenarioDetails)
    schema["required"] = ["scene", "outfit", "framing"]
    return schema


def get_content_plan_schema() -> dict[str, Any]:
    """Gemini response_schema for a content plan (planned_content list). Used by generate_ideas."""
    schema = _pydantic_to_gemini_schema(ContentPlan)
    schema["required"] = ["planned_content"]
    items = schema["properties"].get("planned_content", {}).get("items", {})
    if isinstance(items, dict) and "properties" in items:
        items["required"] = ["id", "type", "persona", "title"]
    return schema


def get_character_schema() -> dict[str, Any]:
    """Gemini response_schema for a Character. Used when LLM generates or fills character data."""
    return _pydantic_to_gemini_schema(Character)


def get_llm_schemas() -> dict[str, Any]:
    """All Gemini response schemas derived from Pydantic. Replaces response_schemas.yml."""
    return {
        "scenario_details_schema": get_scenario_details_schema(),
        "content_plan_schema": get_content_plan_schema(),
        "character_schema": get_character_schema(),
    }
