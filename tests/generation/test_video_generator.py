"""Tests for VideoGenerator with mocked LLMClient."""

import pytest
from pathlib import Path

from personalab.generation.video_generator import VideoGenerator
from personalab.llm.client import LLMVideoResponse
from personalab.schemas import VideoPrompt


class TestGenerateScenarioVideo:
    def test_returns_response_and_meta_with_variables(self, fake_client, tmp_path):
        video_resp = LLMVideoResponse(operation_name="op-abc")
        client = fake_client(video_response=video_resp)
        vg = VideoGenerator(
            client=client,
            model_name="veo-test",
            resolution="1080p",
            aspect_ratio="9:16",
            references_root=tmp_path,
        )
        template = {
            "static_motion_constraints": {"camera_movement": "handheld"},
            "dynamic_identity": {"person_details": "${SUBJECT_DESCRIPTION}"},
            "dynamic_action": {"action_sequence": "${ACTION_DETAILS}"},
        }
        variables = {
            "SUBJECT_DESCRIPTION": "young woman",
            "ACTION_DETAILS": "walking",
        }
        resp, meta = vg.generate_scenario_video(
            persona_name="testpersona",
            video_scenario_template=template,
            variables=variables,
        )
        assert resp.operation_name == "op-abc"
        assert meta.model == "veo-test"
        assert meta.resolution == "1080p"
        assert meta.aspect_ratio == "9:16"
        assert meta.prompt_hash.startswith("sha256:")
        assert client.calls[0]["method"] == "generate_video"

    def test_returns_response_and_meta_with_video_prompt(self, fake_client, tmp_path):
        video_resp = LLMVideoResponse(operation_name="op-xyz", video_data=b"fake-video")
        client = fake_client(video_response=video_resp)
        vg = VideoGenerator(
            client=client,
            model_name="veo-test",
            resolution="1080p",
            aspect_ratio="9:16",
            references_root=tmp_path,
        )
        template = {
            "dynamic_identity": {"person_details": "${SUBJECT_DESCRIPTION}"},
            "dynamic_action": {"action_sequence": "${ACTION_DETAILS}"},
        }
        vp = VideoPrompt(
            subject_description="young woman",
            action_details="walking",
            location_details="park",
            mood_and_expression="happy",
        )
        resp, meta = vg.generate_scenario_video(
            persona_name="testpersona",
            video_scenario_template=template,
            video_prompt=vp,
        )
        assert resp.video_data == b"fake-video"
        assert meta.model == "veo-test"
        assert meta.reference_images_count == 0
        assert meta.anchors_used == []

    def test_anchor_injection_when_anchors_exist(self, fake_client, tmp_path):
        anchors_dir = tmp_path / "persona1" / "anchors"
        anchors_dir.mkdir(parents=True)
        (anchors_dir / "persona1_anchor_frontal_neutral.png").write_bytes(b"fake-anchor")

        client = fake_client()
        vg = VideoGenerator(
            client=client,
            model_name="veo-test",
            resolution="1080p",
            aspect_ratio="9:16",
            references_root=tmp_path,
        )
        template = {"text": "${SUBJECT_DESCRIPTION}"}
        vp = VideoPrompt(subject_description="test person")

        resp, meta = vg.generate_scenario_video(
            persona_name="persona1",
            video_scenario_template=template,
            video_prompt=vp,
        )
        assert meta.anchors_used == ["persona1_anchor_frontal_neutral.png"]
        assert meta.reference_images_count == 1

    def test_no_anchors_still_works(self, fake_client, tmp_path):
        client = fake_client()
        vg = VideoGenerator(
            client=client,
            model_name="m",
            resolution="720p",
            aspect_ratio="9:16",
            references_root=tmp_path,
        )
        template = {"text": "${SUBJECT_DESCRIPTION}"}
        vp = VideoPrompt(subject_description="test")
        resp, meta = vg.generate_scenario_video(
            persona_name="nonexistent",
            video_scenario_template=template,
            video_prompt=vp,
        )
        assert meta.anchors_used == []
        assert meta.reference_images_count == 0

    def test_meta_hash_changes_with_variables(self, fake_client, tmp_path):
        client = fake_client()
        vg = VideoGenerator(
            client=client, model_name="m", resolution="720p", aspect_ratio="9:16",
            references_root=tmp_path,
        )
        template = {"text": "${SUBJECT_DESCRIPTION}"}

        vp1 = VideoPrompt(subject_description="a")
        vp2 = VideoPrompt(subject_description="b")

        _, meta1 = vg.generate_scenario_video(
            persona_name="p", video_scenario_template=template, video_prompt=vp1,
        )
        _, meta2 = vg.generate_scenario_video(
            persona_name="p", video_scenario_template=template, video_prompt=vp2,
        )
        assert meta1.prompt_hash != meta2.prompt_hash

    def test_requires_video_prompt_or_variables(self, fake_client, tmp_path):
        client = fake_client()
        vg = VideoGenerator(
            client=client, model_name="m", resolution="720p", aspect_ratio="9:16",
            references_root=tmp_path,
        )
        with pytest.raises(ValueError, match="Either video_prompt or variables"):
            vg.generate_scenario_video(
                persona_name="p",
                video_scenario_template={"t": "v"},
            )
