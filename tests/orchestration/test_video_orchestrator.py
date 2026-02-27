"""Tests for VideoOrchestrator with fakes."""

import pytest
from pathlib import Path

from personalab.generation.video_generator import VideoGenerator
from personalab.identity.identity_policy import IdentityPolicy, Decision
from personalab.identity.video_evaluator import StubVideoEvaluator
from personalab.identity.schemas import VideoEvaluationResult
from personalab.llm.client import LLMVideoResponse
from personalab.orchestration.video_orchestrator import (
    VideoOrchestrator,
    VideoOrchestrationResult,
    VideoGenerationAttempt,
)
from personalab.schemas import VideoPrompt


class TestVideoOrchestrator:
    @pytest.fixture
    def setup(self, fake_client, tmp_path):
        """Build a VideoOrchestrator with fakes, returning a dict of components."""
        video_resp = LLMVideoResponse(video_data=b"fake-mp4-data", operation_name="op-1")
        client = fake_client(video_response=video_resp)

        vid_gen = VideoGenerator(
            client=client,
            model_name="veo-test",
            resolution="1080p",
            aspect_ratio="9:16",
            references_root=tmp_path,
        )

        video_evaluator = StubVideoEvaluator()
        policy = IdentityPolicy(
            accept_threshold=0.55,
            retry_threshold=0.35,
            max_attempts=3,
            policy_mode="composite_only",
        )

        orchestrator = VideoOrchestrator(
            video_generator=vid_gen,
            video_evaluator=video_evaluator,
            policy=policy,
            output_dir=tmp_path / "output",
            references_root=tmp_path,
        )

        return {
            "orchestrator": orchestrator,
            "client": client,
            "tmp_path": tmp_path,
        }

    def test_accepts_on_first_attempt_with_stub(self, setup):
        orch = setup["orchestrator"]
        vp = VideoPrompt(subject_description="test", action_details="walking")

        result = orch.run(
            persona="testpersona",
            scenario_template={"text": "${SUBJECT_DESCRIPTION}"},
            video_prompt=vp,
        )

        assert isinstance(result, VideoOrchestrationResult)
        assert result.accepted is True
        assert result.final_attempt == 1
        assert result.final_score == pytest.approx(1.0)
        assert result.final_video_path is not None
        assert Path(result.final_video_path).exists()

    def test_saves_video_to_disk(self, setup):
        orch = setup["orchestrator"]
        tmp_path = setup["tmp_path"]
        vp = VideoPrompt(subject_description="test")

        result = orch.run(
            persona="testpersona",
            scenario_template={"text": "${SUBJECT_DESCRIPTION}"},
            video_prompt=vp,
        )

        video_path = Path(result.final_video_path)
        assert video_path.exists()
        assert video_path.read_bytes() == b"fake-mp4-data"

    def test_attempts_recorded(self, setup):
        orch = setup["orchestrator"]
        vp = VideoPrompt(subject_description="test")

        result = orch.run(
            persona="testpersona",
            scenario_template={"text": "${SUBJECT_DESCRIPTION}"},
            video_prompt=vp,
        )

        assert len(result.attempts) == 1
        attempt = result.attempts[0]
        assert isinstance(attempt, VideoGenerationAttempt)
        assert attempt.attempt == 1
        assert attempt.video_bytes_size == len(b"fake-mp4-data")
        assert attempt.generation_ms > 0

    def test_no_video_bytes_continues_loop(self, fake_client, tmp_path):
        empty_resp = LLMVideoResponse(video_data=None, operation_name="op-empty")
        client = fake_client(video_response=empty_resp)

        vid_gen = VideoGenerator(
            client=client,
            model_name="veo-test",
            resolution="1080p",
            aspect_ratio="9:16",
            references_root=tmp_path,
        )
        policy = IdentityPolicy(
            accept_threshold=0.55,
            retry_threshold=0.35,
            max_attempts=2,
            policy_mode="composite_only",
        )
        orch = VideoOrchestrator(
            video_generator=vid_gen,
            video_evaluator=StubVideoEvaluator(),
            policy=policy,
            output_dir=tmp_path / "output",
            references_root=tmp_path,
        )

        vp = VideoPrompt(subject_description="test")
        result = orch.run(
            persona="testpersona",
            scenario_template={"text": "${SUBJECT_DESCRIPTION}"},
            video_prompt=vp,
        )
        assert result.accepted is False
        assert len(result.attempts) == 2

    def test_generation_id_propagated(self, setup):
        orch = setup["orchestrator"]
        vp = VideoPrompt(subject_description="test")

        result = orch.run(
            persona="testpersona",
            scenario_template={"text": "${SUBJECT_DESCRIPTION}"},
            video_prompt=vp,
            generation_id="custom-id-123",
        )
        assert result.generation_id == "custom-id-123"
