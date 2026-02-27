"""Tests for Character and persona schema."""

import pytest
from personalab import Character


def test_character_minimal():
    c = Character(name="Dani", vibe="Lifestyle", location="Lima", content_pillars=[])
    assert c.name == "Dani"
    assert c.vibe == "Lifestyle"
    assert c.location == "Lima"
    assert c.content_pillars == []
    assert c.physical_description == {}
    assert c.id is None


def test_character_to_persona_dict(sample_character):
    d = sample_character.to_persona_dict()
    assert d["name"] == sample_character.name
    assert d["vibe"] == sample_character.vibe
    assert d["location"] == sample_character.location
    assert d["content_pillars"] == sample_character.content_pillars
    assert "physical_description" in d


def test_character_with_id():
    c = Character(
        name="Dani",
        vibe="Lifestyle",
        location="Lima",
        content_pillars=[],
        id="daniperez",
    )
    assert c.id == "daniperez"
    assert c.to_persona_dict()["id"] == "daniperez"
