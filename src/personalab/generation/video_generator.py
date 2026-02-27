"""Video generation with identity-conditioned reference images."""

import logging
from pathlib import Path
from typing import Any

from personalab.llm.client import LLMClient, LLMVideoResponse, BytesPart, ContentPart
from personalab.prompts import render_prompt, render_string, prompt_to_str, sha256_json
from personalab.schemas import VideoGenMeta, VideoPrompt

logger = logging.getLogger(__name__)


class VideoGenerator:
    """Builds prompts from scenario templates, loads anchor references,
    and calls LLM client for identity-conditioned video generation."""

    def __init__(
        self,
        client: LLMClient,
        model_name: str,
        resolution: str,
        aspect_ratio: str,
        references_root: str | Path = "./references",
    ) -> None:
        self._client = client
        self.model_name = model_name
        self.resolution = resolution
        self.aspect_ratio = aspect_ratio
        self.references_root = Path(references_root)

    # -- public API ---------------------------------------------------------

    def generate_scenario_video(
        self,
        persona_name: str,
        video_scenario_template: dict[str, Any],
        video_prompt: VideoPrompt | None = None,
        *,
        variables: dict[str, str] | None = None,
    ) -> tuple[LLMVideoResponse, VideoGenMeta]:
        """Generate a video conditioned on persona anchors.

        Pass *video_prompt* (preferred) or raw *variables* dict.
        Returns ``(response, metadata)``.
        """
        if video_prompt is not None:
            variables = video_prompt.to_variables()
        elif variables is None:
            raise ValueError("Either video_prompt or variables must be provided")

        prompt_str = self._build_prompt_text(video_scenario_template, variables)
        prompt_hash = sha256_json({"prompt": prompt_str})

        anchor_paths = self._list_persona_anchors(persona_name)
        ref_parts = self._anchors_to_parts(anchor_paths)

        resp = self._client.generate_video(
            prompt=prompt_str,
            resolution=self.resolution,
            aspect_ratio=self.aspect_ratio,
            model_name=self.model_name,
            reference_images=ref_parts or None,
        )

        meta = VideoGenMeta(
            model=self.model_name,
            resolution=self.resolution,
            aspect_ratio=self.aspect_ratio,
            prompt_hash=prompt_hash,
            anchors_used=[p.name for p in anchor_paths],
            reference_images_count=len(ref_parts),
        )
        return resp, meta

    # -- prompt helpers -----------------------------------------------------

    @staticmethod
    def _build_prompt_text(
        template: dict[str, Any], variables: dict[str, str],
    ) -> str:
        """Build a clean natural-language prompt from the scenario template.

        If the template contains a ``prompt_template`` key (a flat string
        template), substitute variables directly into it.  Otherwise fall back
        to rendering the full dict and serialising to JSON (legacy path).
        """
        raw_template = template.get("prompt_template")
        if isinstance(raw_template, str):
            return render_string(raw_template, variables).strip()
        rendered = render_prompt(template, variables)
        return prompt_to_str(rendered)

    # -- anchor helpers (mirrors ImageGenerator) ----------------------------

    def _list_persona_anchors(self, persona_name: str) -> list[Path]:
        """Collect frontal + three-quarter anchor paths for a persona."""
        anchors_dir = self.references_root / persona_name / "anchors"
        if not anchors_dir.exists():
            logger.warning(
                "No anchors directory for '%s'. Video will be generated WITHOUT identity reference.",
                persona_name,
            )
            return []

        frontal = sorted(anchors_dir.glob(f"{persona_name}_anchor_frontal_neutral.*"))
        if not frontal:
            logger.warning(
                "No frontal anchor found for '%s'. Video will lack identity reference.",
                persona_name,
            )
            return []

        chosen: list[Path] = frontal[:1]
        threeq = sorted(anchors_dir.glob(f"{persona_name}_sculptor_45_degree.*"))
        if threeq:
            chosen.append(threeq[0])
        return chosen

    @staticmethod
    def _anchors_to_parts(paths: list[Path]) -> list[BytesPart]:
        """Read anchor image files and wrap them as BytesPart for reference_images."""
        out: list[BytesPart] = []
        for p in paths:
            mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
            out.append(BytesPart(data=p.read_bytes(), mime_type=mime))
        return out
