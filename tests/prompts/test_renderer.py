"""Tests for prompt rendering and hashing."""

from personalab.prompts import render_prompt, render_string, prompt_to_str, sha256_json


def test_render_prompt_substitutes_variables():
    template = {"scene": "${SCENE}", "nested": {"key": "${KEY}"}}
    variables = {"SCENE": "beach", "KEY": "value"}
    out = render_prompt(template, variables)
    assert out["scene"] == "beach"
    assert out["nested"]["key"] == "value"


def test_render_prompt_leaves_unknown_placeholder():
    template = {"text": "${UNKNOWN}"}
    out = render_prompt(template, {"OTHER": "x"})
    assert out["text"] == "${UNKNOWN}"


def test_render_prompt_handles_lists():
    template = {"items": ["${A}", "${B}"]}
    out = render_prompt(template, {"A": "x", "B": "y"})
    assert out["items"] == ["x", "y"]


def test_render_prompt_non_string_values_unchanged():
    template = {"count": 42, "flag": True}
    out = render_prompt(template, {"X": "Y"})
    assert out["count"] == 42
    assert out["flag"] is True


def test_render_string_basic():
    result = render_string("Hello ${NAME}, you are ${ROLE}.", {"NAME": "Dani", "ROLE": "creator"})
    assert result == "Hello Dani, you are creator."


def test_render_string_leaves_unknown():
    result = render_string("${KNOWN} and ${UNKNOWN}", {"KNOWN": "yes"})
    assert result == "yes and ${UNKNOWN}"


def test_render_string_empty_variables():
    assert render_string("no vars here", {}) == "no vars here"


def test_prompt_to_str_deterministic():
    d = {"b": 2, "a": 1}
    s = prompt_to_str(d)
    assert "a" in s and "b" in s
    assert prompt_to_str(d) == prompt_to_str({"a": 1, "b": 2})


def test_sha256_json_stable():
    d = {"b": 2, "a": 1}
    h1 = sha256_json(d)
    h2 = sha256_json({"a": 1, "b": 2})
    assert h1 == h2
    assert h1.startswith("sha256:")
