"""Tests for ImagePrompt and VideoPrompt schemas."""

import pytest
from personalab import ImagePrompt, VideoPrompt


def test_image_prompt_defaults():
    p = ImagePrompt()
    assert p.subject_description == ""
    assert p.scene_description == ""
    assert p.policy_overrides is None


def test_image_prompt_to_variables(sample_image_prompt):
    v = sample_image_prompt.to_variables()
    assert v["SUBJECT_DESCRIPTION"] == "test"
    assert v["SCENE_DESCRIPTION"] == "cafe"
    assert v["SHOT_TYPE"] == "selfie"


def test_image_prompt_from_scenario_details():
    p = ImagePrompt.from_scenario_details(
        {"scene": "beach", "outfit": "swimwear", "framing": "selfie"},
        subject_description="person",
    )
    assert p.scene_description == "beach"
    assert p.clothing_details == "swimwear"
    assert p.shot_type == "selfie"
    assert p.subject_description == "person"


class TestVideoPrompt:
    def test_defaults(self):
        p = VideoPrompt()
        assert p.subject_description == ""
        assert p.action_details == ""
        assert p.location_details == ""
        assert p.mood_and_expression == ""
        assert p.policy_overrides is None

    def test_to_variables(self, sample_video_prompt):
        v = sample_video_prompt.to_variables()
        assert v["SUBJECT_DESCRIPTION"] == "test"
        assert v["ACTION_DETAILS"] == "walking"
        assert v["LOCATION_DETAILS"] == "street"
        assert v["MOOD_AND_FACE_EXPRESSION"] == ""

    def test_from_scenario_details_dict(self):
        scenario = {"scene": "park", "outfit": "casual", "framing": "wide"}
        vp = VideoPrompt.from_scenario_details(
            scenario,
            subject_description="young woman",
            custom_action="dancing",
        )
        assert vp.subject_description == "young woman"
        assert vp.action_details == "dancing"
        assert vp.location_details == "park"

    def test_from_scenario_details_uses_custom_action_from_dict(self):
        scenario = {"scene": "beach", "custom_action": "surfing"}
        vp = VideoPrompt.from_scenario_details(scenario)
        assert vp.action_details == "surfing"
        assert vp.location_details == "beach"

    def test_from_scenario_details_pydantic_model(self):
        from personalab.schemas.prompts import ScenarioDetails
        sd = ScenarioDetails(scene="cafe", outfit="smart casual", framing="close-up")
        vp = VideoPrompt.from_scenario_details(sd, subject_description="man")
        assert vp.subject_description == "man"
        assert vp.location_details == "cafe"

    def test_to_variables_keys(self):
        vp = VideoPrompt(
            subject_description="s",
            action_details="a",
            location_details="l",
            mood_and_expression="m",
        )
        keys = set(vp.to_variables().keys())
        assert keys == {
            "SUBJECT_DESCRIPTION",
            "ACTION_DETAILS",
            "LOCATION_DETAILS",
            "MOOD_AND_FACE_EXPRESSION",
        }
