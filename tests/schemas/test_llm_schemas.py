"""Tests for Pydantic-derived Gemini LLM schemas."""

import pytest
from personalab.schemas.llm_schemas import (
    get_llm_schemas,
    get_scenario_details_schema,
    get_content_plan_schema,
)


def test_get_scenario_details_schema():
    s = get_scenario_details_schema()
    assert s["type"] == "OBJECT"
    assert "scene" in s["properties"]
    assert "outfit" in s["properties"]
    assert "framing" in s["properties"]
    assert s["required"] == ["scene", "outfit", "framing"]


def test_get_content_plan_schema():
    s = get_content_plan_schema()
    assert s["type"] == "OBJECT"
    assert "planned_content" in s["properties"]
    assert s["required"] == ["planned_content"]
    items = s["properties"]["planned_content"].get("items", {})
    assert items.get("type") == "OBJECT"
    assert "id" in items.get("properties", {})
    assert "scenario_details" in items.get("properties", {})


def test_get_llm_schemas():
    all_s = get_llm_schemas()
    assert "scenario_details_schema" in all_s
    assert "content_plan_schema" in all_s
    assert all_s["scenario_details_schema"] == get_scenario_details_schema()
    assert all_s["content_plan_schema"] == get_content_plan_schema()
