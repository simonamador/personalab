"""Tests for runtime.retry_policy: RetryPolicy."""

import asyncio

import pytest

from personalab.identity.identity_policy import Decision
from personalab.runtime.retry_policy import RetryPolicy


class TestRetryPolicy:
    def test_should_retry_on_retry_action(self):
        rp = RetryPolicy(max_retries=3)
        d = Decision(action="RETRY", score=0.4)
        assert rp.should_retry(1, d) is True
        assert rp.should_retry(2, d) is True
        assert rp.should_retry(3, d) is False  # at max

    def test_should_not_retry_on_accept(self):
        rp = RetryPolicy(max_retries=3)
        d = Decision(action="ACCEPT", score=0.8)
        assert rp.should_retry(1, d) is False

    def test_should_not_retry_on_reject(self):
        rp = RetryPolicy(max_retries=3)
        d = Decision(action="REJECT_FINAL", score=0.1)
        assert rp.should_retry(1, d) is False

    def test_backoff_exponential(self):
        rp = RetryPolicy(backoff_base=1.0, backoff_factor=2.0, backoff_max=10.0)
        assert rp.backoff_seconds(1) == 1.0
        assert rp.backoff_seconds(2) == 2.0
        assert rp.backoff_seconds(3) == 4.0
        assert rp.backoff_seconds(4) == 8.0
        assert rp.backoff_seconds(5) == 10.0  # clamped

    def test_log_attempt_returns_record(self):
        d = Decision(action="RETRY", score=0.45, notes="test")
        rec = RetryPolicy.log_attempt(
            job_id="j1", seed=42, attempt=1,
            prompt_hash="abc", prompt_applied={"k": "v"},
            decision=d,
        )
        assert rec["job_id"] == "j1"
        assert rec["seed"] == 42
        assert rec["attempt"] == 1
        assert rec["prompt_hash"] == "abc"
        assert rec["decision_action"] == "RETRY"
        assert rec["decision_score"] == 0.45


class TestRetryPolicyAsync:
    @pytest.mark.asyncio
    async def test_wait_async_returns(self):
        rp = RetryPolicy(backoff_base=0.001, backoff_max=0.01)
        await rp.wait_async(1)
