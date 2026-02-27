"""CPU/GPU-bound evaluation pool backed by ``ProcessPoolExecutor``.

Identity evaluation (ArcFace embeddings, geometric landmarks, OpenCV frame
extraction) is compute-heavy and blocks the event loop.  This module wraps
a process pool so the ``AsyncRunner`` can ``await`` evaluation without
stalling I/O-bound generation tasks.

Each worker process initialises its own ``FaceRuntime`` (ONNX sessions are
not pickleable, so they cannot be sent across the process boundary).
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from personalab.identity.schemas import EvaluationResult

logger = logging.getLogger(__name__)

# ---- worker-local global (one per process) --------------------------------

_worker_evaluator: Any = None


def _init_worker(eval_cfg: dict[str, Any]) -> None:
    """Called once per worker process to build a local evaluator."""
    global _worker_evaluator
    from personalab.identity.factory import build_evaluator
    _worker_evaluator = build_evaluator(eval_cfg)


def _evaluate_in_worker(anchor_path: str, candidate_paths: list[str]) -> dict[str, Any]:
    """Top-level function executed inside the worker (must be pickleable)."""
    if _worker_evaluator is None:
        raise RuntimeError("Worker evaluator not initialised")
    result: EvaluationResult = _worker_evaluator.evaluate(anchor_path, candidate_paths)
    return result.model_dump()


# ---- public API -----------------------------------------------------------


class EvaluationPool:
    """Async-friendly wrapper around a ``ProcessPoolExecutor``.

    Usage::

        pool = EvaluationPool(max_workers=2, eval_cfg=config.evaluation)
        result = await pool.evaluate(anchor, [candidate])
        pool.shutdown()
    """

    def __init__(
        self,
        *,
        max_workers: int = 2,
        eval_cfg: dict[str, Any],
    ) -> None:
        self._eval_cfg = eval_cfg
        self._pool = ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(eval_cfg,),
        )

    async def evaluate(
        self,
        anchor_path: str,
        candidate_paths: list[str],
    ) -> EvaluationResult:
        """Submit evaluation to the process pool and ``await`` the result."""
        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(
            self._pool,
            _evaluate_in_worker,
            anchor_path,
            candidate_paths,
        )
        return EvaluationResult.model_validate(raw)

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)

    def __enter__(self) -> "EvaluationPool":
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()
