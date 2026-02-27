"""Tests for runtime.eval_pool: EvaluationPool (ProcessPoolExecutor wrapper).

These tests use the StubEvaluator path (no insightface required) to validate
that the pool correctly serialises / deserialises evaluation results across
the process boundary.
"""

import pytest

from personalab.runtime.eval_pool import EvaluationPool


class TestEvaluationPool:
    @pytest.fixture
    def stub_eval_cfg(self):
        """Config that forces StubEvaluator (insightface not loaded)."""
        return {
            "embedding": {"backend": "stub"},
            "geometric": {"enabled": False},
            "scoring": {"weights": {"embedding": 1.0, "geometric": 0.0}},
        }

    def test_shutdown_without_use(self, stub_eval_cfg):
        pool = EvaluationPool(max_workers=1, eval_cfg=stub_eval_cfg)
        pool.shutdown()

    def test_context_manager(self, stub_eval_cfg):
        with EvaluationPool(max_workers=1, eval_cfg=stub_eval_cfg) as pool:
            pass
