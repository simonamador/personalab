"""Post-processing gate: decides whether to recommend visual post-processing.

Fundamental rule: post-processing is ONLY recommended when identity is
verified stable AND the degradation is purely textural.  Identity
thresholds are never adjusted based on visual quality.
"""

from __future__ import annotations

from personalab.identity.schemas import VideoEvaluationResult
from personalab.quality.schemas import PostProcessingAdvice, VideoQualityResult


class PostProcessingGate:
    """Combine identity and quality verdicts into post-processing advice.

    Parameters
    ----------
    identity_accept_threshold:
        Minimum mean composite score to consider identity stable.
    max_embedding_std:
        Maximum allowed std of cosine similarities for identity stability.
    max_geometric_variance:
        Maximum allowed mean geometric ratio variance for stability.
    """

    def __init__(
        self,
        *,
        identity_accept_threshold: float = 0.55,
        max_embedding_std: float = 0.06,
        max_geometric_variance: float = 0.002,
    ) -> None:
        self.identity_accept_threshold = identity_accept_threshold
        self.max_embedding_std = max_embedding_std
        self.max_geometric_variance = max_geometric_variance

    def evaluate(
        self,
        identity_result: VideoEvaluationResult,
        quality_result: VideoQualityResult,
    ) -> PostProcessingAdvice:
        identity_stable = self._is_identity_stable(identity_result)

        textural_degraded = not quality_result.sharpness_ok
        illumination_degraded = not quality_result.illumination_ok

        if textural_degraded and illumination_degraded:
            deg_type = "both"
        elif textural_degraded:
            deg_type = "textural"
        elif illumination_degraded:
            deg_type = "illumination"
        else:
            deg_type = "none"

        quality_degraded = textural_degraded or illumination_degraded

        recommend_sharpening = False
        recommend_super_res = False
        notes_parts: list[str] = []

        if not identity_stable:
            notes_parts.append("identity_unstable:no_postprocessing")
        elif quality_degraded and deg_type == "textural":
            recommend_sharpening = True
            recommend_super_res = True
            notes_parts.append("identity_ok+textural_loss:recommend_sharpen+super_res")
        elif quality_degraded and deg_type == "both":
            notes_parts.append("identity_ok+mixed_degradation:review_manually")
        elif quality_degraded and deg_type == "illumination":
            notes_parts.append("identity_ok+illumination_only:no_auto_fix")
        else:
            notes_parts.append("all_ok")

        return PostProcessingAdvice(
            identity_stable=identity_stable,
            quality_degraded=quality_degraded,
            degradation_type=deg_type,
            recommend_sharpening=recommend_sharpening,
            recommend_super_resolution=recommend_super_res,
            notes=";".join(notes_parts),
        )

    def _is_identity_stable(self, result: VideoEvaluationResult) -> bool:
        if result.frames_evaluated == 0:
            return False
        if result.mean_composite_score < self.identity_accept_threshold:
            return False

        emb = result.embedding_stats
        if emb is not None and emb.std_cosine > self.max_embedding_std:
            return False

        geo = result.geometric_stats
        if geo is not None and geo.mean_ratio_variance > self.max_geometric_variance:
            return False

        return True
