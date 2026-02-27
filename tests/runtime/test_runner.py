"""Tests for runtime.runner: SyncRunner and AsyncRunner."""

import tempfile
from pathlib import Path

import pytest

from personalab.generation.image_generator import ImageGenerator
from personalab.identity.evaluator import StubEvaluator
from personalab.identity.identity_policy import IdentityPolicy
from personalab.runtime.job import JobSpec
from personalab.runtime.runner import SyncRunner, AsyncRunner
from personalab.runtime.retry_policy import RetryPolicy
from personalab.schemas.prompts import ImagePrompt


@pytest.fixture
def workspace(tmp_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    return tmp_path, refs, out


@pytest.fixture
def image_job():
    return JobSpec(
        job_id="test-job-01",
        seed=42,
        config_hash="abc123",
        persona="testpersona",
        scenario_template={"prompt_template": "A photo of ${SUBJECT_DESCRIPTION}"},
        prompt=ImagePrompt(
            subject_description="a person",
            scene_description="park",
            clothing_details="casual",
            shot_type="selfie",
        ),
        asset_type="image",
        output_prefix="test",
    )


class TestSyncRunner:
    def test_run_single_image_job_accepted_no_anchor(self, fake_client, workspace, image_job):
        tmp_path, refs, out = workspace
        client = fake_client(image_bytes=b"\x89PNG\r\n" + b"\x00" * 100)
        img_gen = ImageGenerator(
            client=client,
            model_name="test-model",
            aspect_ratio="4:5",
            references_root=str(refs),
        )
        policy = IdentityPolicy(max_attempts=3, accept_threshold=0.55)
        runner = SyncRunner(
            image_generator=img_gen,
            evaluator=StubEvaluator(),
            policy=policy,
            output_dir=str(out),
            references_root=str(refs),
        )

        results = runner.run([image_job])
        assert len(results) == 1
        r = results[0]
        assert r.job_id == "test-job-01"
        assert r.seed == 42
        assert r.config_hash == "abc123"
        assert r.accepted is True
        assert r.asset_type == "image"

    def test_run_empty_jobs(self, fake_client, workspace):
        tmp_path, refs, out = workspace
        client = fake_client()
        img_gen = ImageGenerator(
            client=client, model_name="m", aspect_ratio="1:1",
            references_root=str(refs),
        )
        runner = SyncRunner(
            image_generator=img_gen,
            evaluator=StubEvaluator(),
            policy=IdentityPolicy(),
            output_dir=str(out),
            references_root=str(refs),
        )
        results = runner.run([])
        assert results == []

    def test_job_failure_captured(self, fake_client, workspace, image_job):
        tmp_path, refs, out = workspace
        client = fake_client(image_bytes=None)
        img_gen = ImageGenerator(
            client=client, model_name="m", aspect_ratio="1:1",
            references_root=str(refs),
        )
        runner = SyncRunner(
            image_generator=img_gen,
            evaluator=StubEvaluator(),
            policy=IdentityPolicy(max_attempts=1),
            output_dir=str(out),
            references_root=str(refs),
        )
        results = runner.run([image_job])
        assert len(results) == 1
        assert results[0].accepted is False


class TestAsyncRunner:
    def test_run_single_image_job(self, fake_client, workspace, image_job):
        tmp_path, refs, out = workspace
        client = fake_client(image_bytes=b"\x89PNG\r\n" + b"\x00" * 100)
        img_gen = ImageGenerator(
            client=client, model_name="test-model", aspect_ratio="4:5",
            references_root=str(refs),
        )
        policy = IdentityPolicy(max_attempts=2, accept_threshold=0.55)
        runner = AsyncRunner(
            client=client,
            image_generator=img_gen,
            policy=policy,
            output_dir=str(out),
            references_root=str(refs),
        )
        results = runner.run([image_job])
        assert len(results) == 1
        r = results[0]
        assert r.job_id == "test-job-01"
        assert r.accepted is True

    def test_run_multiple_jobs_parallel(self, fake_client, workspace):
        tmp_path, refs, out = workspace
        client = fake_client(image_bytes=b"\x89PNG\r\n" + b"\x00" * 50)
        img_gen = ImageGenerator(
            client=client, model_name="m", aspect_ratio="1:1",
            references_root=str(refs),
        )
        jobs = [
            JobSpec(
                job_id=f"par-{i}",
                persona="p",
                scenario_template={"prompt_template": "test ${SUBJECT_DESCRIPTION}"},
                prompt=ImagePrompt(subject_description=f"subject-{i}"),
                asset_type="image",
            )
            for i in range(3)
        ]
        runner = AsyncRunner(
            client=client, image_generator=img_gen,
            policy=IdentityPolicy(max_attempts=1),
            output_dir=str(out), references_root=str(refs),
        )
        results = runner.run(jobs)
        assert len(results) == 3
        job_ids = {r.job_id for r in results}
        assert job_ids == {"par-0", "par-1", "par-2"}
