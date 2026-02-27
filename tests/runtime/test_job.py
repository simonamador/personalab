"""Tests for runtime.job: JobSpec, JobResult, config hashing."""

from personalab.runtime.job import JobSpec, JobResult, compute_config_hash
from personalab.schemas.prompts import ImagePrompt, VideoPrompt


class TestComputeConfigHash:
    def test_deterministic(self):
        cfg = {"a": 1, "b": [2, 3]}
        assert compute_config_hash(cfg) == compute_config_hash(cfg)

    def test_key_order_invariant(self):
        h1 = compute_config_hash({"a": 1, "b": 2})
        h2 = compute_config_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_different_configs_differ(self):
        h1 = compute_config_hash({"a": 1})
        h2 = compute_config_hash({"a": 2})
        assert h1 != h2


class TestJobSpec:
    def test_auto_job_id(self):
        j = JobSpec(
            persona="test",
            scenario_template={},
            prompt=ImagePrompt(subject_description="x"),
            asset_type="image",
        )
        assert len(j.job_id) == 12

    def test_explicit_fields(self):
        j = JobSpec(
            job_id="custom123",
            seed=42,
            config_hash="abc",
            persona="alice",
            scenario_template={"k": "v"},
            prompt=VideoPrompt(subject_description="y"),
            asset_type="video",
        )
        assert j.job_id == "custom123"
        assert j.seed == 42
        assert j.config_hash == "abc"
        assert j.asset_type == "video"

    def test_frozen(self):
        j = JobSpec(
            persona="test",
            scenario_template={},
            prompt=ImagePrompt(),
            asset_type="image",
        )
        try:
            j.seed = 99  # type: ignore[misc]
            assert False, "Should be frozen"
        except Exception:
            pass


class TestJobResult:
    def test_monotonic_sequence(self):
        r1 = JobResult(job_id="a")
        r2 = JobResult(job_id="b")
        assert r2.sequence_number > r1.sequence_number

    def test_defaults(self):
        r = JobResult(job_id="x")
        assert r.accepted is False
        assert r.error is None
        assert r.final_attempt == 0
