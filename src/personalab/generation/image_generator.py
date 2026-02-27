"""Image generation: anchors and scenario-based assets."""

import json
import logging
from pathlib import Path
from typing import Any

from personalab.llm.client import LLMClient, LLMImageResponse, TextPart, BytesPart, ContentPart
from personalab.prompts import render_prompt, prompt_to_str, sha256_json
from personalab.schemas import ImageGenMeta, ImagePrompt

logger = logging.getLogger(__name__)


def _stable_json_str(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


IDENTITY_LOCK_TEXT = (
    "IDENTITY LOCK (non-negotiable): Depict the exact same person as the FIRST reference image. "
    "Preserve facial geometry (jawline, nose width/bridge/tip, eyebrow shape/thickness, eye spacing, lips, hairline). "
    "Do not recast. Do not change age/ethnicity/facial proportions. No beautification or 'generic influencer face'."
)


class ImageGenerator:
    """Builds prompts, loads anchors, calls LLM client for image generation."""

    def __init__(
        self,
        client: LLMClient,
        model_name: str,
        aspect_ratio: str,
        references_root: str | Path,
    ) -> None:
        self._client = client
        self.model_name = model_name
        self.aspect_ratio = aspect_ratio
        self.references_root = Path(references_root)

    def _find_frontal_anchor(self, persona_name: str) -> Path | None:
        """Return the frontal neutral anchor path for a persona, or None if missing."""
        anchors_dir = self.references_root / persona_name / "anchors"
        if not anchors_dir.exists():
            return None
        matches = sorted(anchors_dir.glob(f"{persona_name}_anchor_frontal_neutral.*"))
        return matches[0] if matches else None

    def _generate_frontal_candidates(
        self,
        persona_name: str,
        frontal_template: dict[str, Any],
        variables: dict[str, str],
        output_dir: Path,
        k: int,
    ) -> list[Path]:
        """Generate *k* frontal neutral candidates and save them to disk."""
        cand_dir = output_dir / "candidates"
        cand_dir.mkdir(parents=True, exist_ok=True)
        prompt_dict = render_prompt(frontal_template, variables)
        prompt_str = prompt_to_str(prompt_dict)
        saved: list[Path] = []
        for i in range(k):
            resp = self._client.generate_image(
                parts=[TextPart(text=prompt_str)],
                aspect_ratio=self.aspect_ratio,
                model_name=self.model_name,
            )
            img_bytes = resp.images[0] if resp.images else None
            if not img_bytes:
                continue
            p = cand_dir / f"{persona_name}_anchor_frontal_neutral_candidate_{i:02d}.png"
            p.write_bytes(img_bytes)
            saved.append(p)
        return saved

    def _generate_secondary_anchor(
        self,
        persona_name: str,
        template: dict[str, Any],
        variables: dict[str, str],
        frontal_path: Path,
        out_path: Path,
    ) -> Path | None:
        """Generate a secondary anchor conditioned on the frontal reference."""
        prompt_dict = render_prompt(template, variables)
        prompt_str = prompt_to_str(prompt_dict)
        mime = "image/png" if frontal_path.suffix.lower() == ".png" else "image/jpeg"
        parts: list[ContentPart] = [
            TextPart(text=IDENTITY_LOCK_TEXT),
            BytesPart(data=frontal_path.read_bytes(), mime_type=mime),
            TextPart(text=prompt_str),
        ]
        resp = self._client.generate_image(
            parts=parts,
            aspect_ratio=self.aspect_ratio,
            model_name=self.model_name,
        )
        img_bytes = resp.images[0] if resp.images else None
        if not img_bytes:
            return None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(img_bytes)
        return out_path

    def generate_reference_anchors(
        self,
        persona_name: str,
        character_prompt: dict[str, Any],
        physical_description: dict[str, Any],
        out_dir: str | Path | None = None,
        frontal_candidates_k: int = 16,
    ) -> list[Path]:
        """Generate anchors with identity stability: frontal first, then secondaries conditioned on it."""
        output_dir = Path(out_dir) if out_dir else (self.references_root / persona_name / "anchors")
        output_dir.mkdir(parents=True, exist_ok=True)
        variables = {"PHYSICAL_DESCRIPTION": _stable_json_str(physical_description)}
        frontal_key = "anchor_frontal_neutral"
        if frontal_key not in character_prompt:
            raise KeyError("Anchor templates must include 'anchor_frontal_neutral'.")
        frontal_path = self._find_frontal_anchor(persona_name)
        if frontal_path is None:
            return self._generate_frontal_candidates(
                persona_name=persona_name,
                frontal_template=character_prompt[frontal_key],
                variables=variables,
                output_dir=output_dir,
                k=frontal_candidates_k,
            )
        saved: list[Path] = [frontal_path]
        for key, ref in character_prompt.items():
            if key == frontal_key:
                continue
            out_path = output_dir / f"{persona_name}_{key}.png"
            p = self._generate_secondary_anchor(
                persona_name=persona_name,
                template=ref,
                variables=variables,
                frontal_path=frontal_path,
                out_path=out_path,
            )
            if p:
                saved.append(p)
        return saved

    def generate_asset_image(
        self,
        persona_name: str,
        scenario_template: dict[str, Any],
        image_prompt: ImagePrompt | None = None,
        *,
        variables: dict[str, str] | None = None,
        policy_overrides: dict[str, Any] | None = None,
    ) -> tuple[bytes | None, ImageGenMeta, dict[str, Any]]:
        """Generate an image using persona anchors. Pass image_prompt (scenario as ImagePrompt) or variables+policy_overrides."""
        if image_prompt is not None:
            variables = image_prompt.to_variables()
            policy_overrides = image_prompt.policy_overrides
        elif variables is None:
            raise ValueError("Either image_prompt or variables must be provided")
        prompt_dict = render_prompt(scenario_template, variables)
        if policy_overrides:
            prompt_dict["policy_overrides"] = policy_overrides
        prompt_hash = sha256_json(prompt_dict)
        anchor_paths = self._list_persona_anchors(persona_name)
        anchor_parts = self._anchors_to_parts(anchor_paths)
        prompt_str = prompt_to_str(prompt_dict)
        parts: list[ContentPart] = (
            [TextPart(text=IDENTITY_LOCK_TEXT)]
            + anchor_parts
            + [TextPart(text=prompt_str)]
        )
        resp = self._client.generate_image(
            parts=parts,
            aspect_ratio=self.aspect_ratio,
            model_name=self.model_name,
        )
        img_bytes = resp.images[0] if resp.images else None
        meta = ImageGenMeta(
            model=self.model_name,
            aspect_ratio=self.aspect_ratio,
            prompt_hash=prompt_hash,
            anchors_used=[p.name for p in anchor_paths],
        )
        return img_bytes, meta, prompt_dict

    def _list_persona_anchors(self, persona_name: str) -> list[Path]:
        """Collect frontal + three-quarter anchor paths for a persona."""
        anchors_dir = self.references_root / persona_name / "anchors"
        if not anchors_dir.exists():
            logger.warning(
                "No anchors directory for '%s'. Image will be generated WITHOUT identity reference. "
                "Run anchor generation first, then call select_anchor() or manually promote a candidate.",
                persona_name,
            )
            return []
        frontal = sorted(anchors_dir.glob(f"{persona_name}_anchor_frontal_neutral.*"))
        if not frontal:
            candidates_dir = anchors_dir / "candidates"
            has_candidates = candidates_dir.exists() and any(candidates_dir.iterdir())
            if has_candidates:
                logger.warning(
                    "Frontal anchor not promoted for '%s'. Candidates exist in %s. "
                    "Call select_anchor('%s', references_root) to auto-select, "
                    "or manually copy a candidate to %s/%s_anchor_frontal_neutral.png",
                    persona_name, candidates_dir, persona_name,
                    anchors_dir, persona_name,
                )
            else:
                logger.warning(
                    "No frontal anchor found for '%s'. Generate anchors first.",
                    persona_name,
                )
            return []
        chosen: list[Path] = frontal[:1]
        threeq = sorted(anchors_dir.glob(f"{persona_name}_sculptor_45_degree.*"))
        if threeq:
            chosen.append(threeq[0])
        return chosen

    @staticmethod
    def _anchors_to_parts(paths: list[Path]) -> list[ContentPart]:
        """Read anchor image files and wrap them as BytesPart for the LLM client."""
        out: list[ContentPart] = []
        for p in paths:
            mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
            out.append(BytesPart(data=p.read_bytes(), mime_type=mime))
        return out
