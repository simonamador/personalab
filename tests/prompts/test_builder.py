"""Tests for PromptBuilder contract."""

from personalab.prompts import PassThroughPromptBuilder


def test_passthrough_image_prompt():
    builder = PassThroughPromptBuilder()
    p = builder.user_to_image_prompt("a beach selfie")
    assert p.scene_description == "a beach selfie"
    assert p.to_variables()["SCENE_DESCRIPTION"] == "a beach selfie"


def test_passthrough_image_prompt_with_context():
    builder = PassThroughPromptBuilder()
    p = builder.user_to_image_prompt("selfie", context={"scene_description": "cafe", "shot_type": "mirror selfie"})
    assert p.scene_description == "cafe"
    assert p.shot_type == "mirror selfie"
    assert p.to_variables()["SHOT_TYPE"] == "mirror selfie"


def test_passthrough_video_prompt():
    builder = PassThroughPromptBuilder()
    p = builder.user_to_video_prompt("walking in the park")
    assert p.action_details == "walking in the park"
    assert p.to_variables()["ACTION_DETAILS"] == "walking in the park"


def test_passthrough_video_prompt_with_context():
    builder = PassThroughPromptBuilder()
    p = builder.user_to_video_prompt(
        "dancing",
        context={"subject_description": "young woman", "location_details": "beach"},
    )
    assert p.subject_description == "young woman"
    assert p.action_details == "dancing"
    assert p.location_details == "beach"
