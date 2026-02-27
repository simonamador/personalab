"""Automated anchor selection from frontal candidates.

Strategies:
  - ``central_similarity``: pick the candidate with highest average cosine
    similarity to all other candidates (most "central" face).
  - ``first``: pick the first candidate (index 0).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)

SelectionStrategy = Literal["central_similarity", "first"]


def select_anchor(
    persona: str,
    references_root: str | Path,
    *,
    strategy: SelectionStrategy = "central_similarity",
    runtime: Any | None = None,
) -> Path | None:
    """Promote the best frontal candidate to the official anchor.

    Returns the promoted anchor path, or None if no candidates exist
    or promotion fails.
    """
    refs = Path(references_root)
    candidates_dir = refs / persona / "anchors" / "candidates"
    anchors_dir = refs / persona / "anchors"
    target = anchors_dir / f"{persona}_anchor_frontal_neutral.png"

    if target.exists():
        logger.info("Anchor already promoted for %s: %s", persona, target)
        return target

    candidates = sorted(candidates_dir.glob(f"{persona}_anchor_frontal_neutral_candidate_*"))
    if not candidates:
        logger.warning("No frontal candidates found for persona '%s' in %s", persona, candidates_dir)
        return None

    if len(candidates) == 1 or strategy == "first":
        chosen = candidates[0]
    elif strategy == "central_similarity":
        chosen = _pick_central(candidates, runtime)
    else:
        chosen = candidates[0]

    shutil.copy2(chosen, target)
    logger.info("Promoted %s -> %s (strategy=%s)", chosen.name, target.name, strategy)
    return target


def _pick_central(candidates: list[Path], runtime: Any | None) -> Path:
    """Choose the candidate whose embedding is most similar to all others on average."""
    if runtime is None:
        try:
            from personalab.identity.face_runtime import FaceRuntime
            runtime = FaceRuntime()
        except ImportError:
            logger.warning("FaceRuntime unavailable; falling back to first candidate")
            return candidates[0]

    embeddings: list[tuple[Path, np.ndarray]] = []
    for c in candidates:
        emb = runtime.extract_embedding(str(c))
        if emb is not None:
            embeddings.append((c, emb))

    if len(embeddings) < 2:
        return candidates[0]

    best_path = candidates[0]
    best_avg = -1.0
    for i, (path_i, emb_i) in enumerate(embeddings):
        sims = []
        for j, (_, emb_j) in enumerate(embeddings):
            if i == j:
                continue
            dot = float(np.dot(emb_i, emb_j))
            norm = float(np.linalg.norm(emb_i) * np.linalg.norm(emb_j))
            sims.append(dot / norm if norm > 0 else 0.0)
        avg = sum(sims) / len(sims) if sims else 0.0
        if avg > best_avg:
            best_avg = avg
            best_path = path_i

    logger.info("Central similarity selection: best avg=%.4f from %d candidates", best_avg, len(embeddings))
    return best_path
