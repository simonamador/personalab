"""Provider-level concurrency controls for async generation.

* ``ProviderSemaphore`` -- limits the number of simultaneous in-flight
  requests to each LLM provider (prevents connection-pool exhaustion and
  server-side throttling).
* ``RateLimiter`` -- token-bucket algorithm per provider that smooths
  request bursts and avoids 429 responses.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any


_DEFAULT_CONCURRENCY: dict[str, int] = {
    "gemini": 3,
    "openai": 5,
    "runware": 5,
    "replicate": 3,
    "runway": 2,
}

_DEFAULT_RATES: dict[str, float] = {
    "gemini": 5.0,
    "openai": 10.0,
    "runware": 10.0,
    "replicate": 5.0,
    "runway": 2.0,
}


class ProviderSemaphore:
    """One ``asyncio.Semaphore`` per provider key, lazily created.

    Usage inside an async task::

        async with provider_sem["gemini"]:
            resp = await client.generate_image_async(...)
    """

    def __init__(self, limits: dict[str, int] | None = None) -> None:
        merged = dict(_DEFAULT_CONCURRENCY)
        if limits:
            merged.update(limits)
        self._limits = merged
        self._sems: dict[str, asyncio.Semaphore] = {}

    def __getitem__(self, provider: str) -> asyncio.Semaphore:
        if provider not in self._sems:
            limit = self._limits.get(provider, 3)
            self._sems[provider] = asyncio.Semaphore(limit)
        return self._sems[provider]


class RateLimiter:
    """Async token-bucket rate limiter, one bucket per provider.

    ``acquire`` sleeps until a token is available, guaranteeing the
    average request rate stays at or below *rate* req/s.
    """

    def __init__(self, rates: dict[str, float] | None = None) -> None:
        merged = dict(_DEFAULT_RATES)
        if rates:
            merged.update(rates)
        self._rates = merged
        self._buckets: dict[str, _Bucket] = {}

    def _get_bucket(self, provider: str) -> "_Bucket":
        if provider not in self._buckets:
            rate = self._rates.get(provider, 5.0)
            self._buckets[provider] = _Bucket(rate)
        return self._buckets[provider]

    async def acquire(self, provider: str) -> None:
        """Wait until a token is available for *provider*."""
        bucket = self._get_bucket(provider)
        await bucket.acquire()


class _Bucket:
    """Single token-bucket with *rate* tokens/second and a burst of 1."""

    def __init__(self, rate: float) -> None:
        self._interval = 1.0 / max(rate, 0.01)
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait = self._last + self._interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = asyncio.get_event_loop().time()
