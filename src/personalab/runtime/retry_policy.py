"""Explicit retry governance for the runtime layer.

``RetryPolicy`` wraps the existing ``IdentityPolicy.decide()`` logic with:
* mandatory logging of the *final prompt* applied on every attempt,
* configurable backoff,
* a hard cap on total retries independent of the identity policy.

The identity policy still owns accept/retry/reject *decisions*; this module
controls the *execution* of those decisions and ensures full auditability.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from personalab.identity.identity_policy import Decision

logger = logging.getLogger(__name__)


class RetryPolicy:
    """Governs retry execution and logs every attempt with its full prompt."""

    def __init__(
        self,
        *,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        backoff_max: float = 30.0,
        backoff_factor: float = 2.0,
    ) -> None:
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.backoff_factor = backoff_factor

    def should_retry(self, attempt: int, decision: Decision) -> bool:
        """Return True when the decision is RETRY and attempts remain."""
        if decision.action != "RETRY":
            return False
        return attempt < self.max_retries

    def backoff_seconds(self, attempt: int) -> float:
        """Exponential backoff clamped to *backoff_max*."""
        delay = self.backoff_base * (self.backoff_factor ** (attempt - 1))
        return min(delay, self.backoff_max)

    def wait_sync(self, attempt: int) -> None:
        """Blocking sleep for sync runners."""
        delay = self.backoff_seconds(attempt)
        if delay > 0:
            logger.debug("RetryPolicy: sync backoff %.2fs (attempt %d)", delay, attempt)
            time.sleep(delay)

    async def wait_async(self, attempt: int) -> None:
        """Non-blocking sleep for async runners."""
        delay = self.backoff_seconds(attempt)
        if delay > 0:
            logger.debug("RetryPolicy: async backoff %.2fs (attempt %d)", delay, attempt)
            await asyncio.sleep(delay)

    @staticmethod
    def log_attempt(
        *,
        job_id: str,
        seed: int,
        attempt: int,
        prompt_hash: str,
        prompt_applied: dict[str, Any],
        decision: Decision,
    ) -> dict[str, Any]:
        """Log the full prompt and decision for audit.  Returns the record."""
        record = {
            "job_id": job_id,
            "seed": seed,
            "attempt": attempt,
            "prompt_hash": prompt_hash,
            "prompt_applied": prompt_applied,
            "decision_action": decision.action,
            "decision_score": decision.score,
            "decision_notes": decision.notes,
        }
        logger.info(
            "RetryPolicy attempt: job=%s attempt=%d action=%s score=%.4f hash=%s",
            job_id, attempt, decision.action, decision.score, prompt_hash,
        )
        return record
