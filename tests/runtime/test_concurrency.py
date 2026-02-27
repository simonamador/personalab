"""Tests for runtime.concurrency: ProviderSemaphore and RateLimiter."""

import asyncio
import time

import pytest

from personalab.runtime.concurrency import ProviderSemaphore, RateLimiter


class TestProviderSemaphore:
    def test_default_limits(self):
        sem = ProviderSemaphore()
        s = sem["gemini"]
        assert isinstance(s, asyncio.Semaphore)

    def test_custom_limits(self):
        sem = ProviderSemaphore(limits={"custom_provider": 1})
        s = sem["custom_provider"]
        assert isinstance(s, asyncio.Semaphore)

    def test_unknown_provider_gets_default(self):
        sem = ProviderSemaphore(limits={})
        s = sem["unknown"]
        assert isinstance(s, asyncio.Semaphore)

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        sem = ProviderSemaphore(limits={"test": 2})
        running = 0
        max_running = 0

        async def task():
            nonlocal running, max_running
            async with sem["test"]:
                running += 1
                max_running = max(max_running, running)
                await asyncio.sleep(0.01)
                running -= 1

        await asyncio.gather(*[task() for _ in range(6)])
        assert max_running <= 2


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_acquire_does_not_error(self):
        rl = RateLimiter(rates={"test": 100.0})
        await rl.acquire("test")

    @pytest.mark.asyncio
    async def test_rate_limiting_enforced(self):
        rl = RateLimiter(rates={"slow": 10.0})
        t0 = time.monotonic()
        for _ in range(3):
            await rl.acquire("slow")
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.15
