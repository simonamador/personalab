"""Generate identity-conditioned video for a persona with evaluation and retry.

Usage:
    python scripts/gen_videos.py daniperez
    python scripts/gen_videos.py daniperez --index 2
    python scripts/gen_videos.py daniperez --no-eval      # skip identity evaluation
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from personalab import (
    load_config,
    create_client,
    VideoGenerator,
    VideoPrompt,
    IdentityPolicy,
    DriftTracker,
    GenerationMetricsLogger,
    build_evaluator,
    FrameExtractor,
    VideoIdentityEvaluator,
    StubVideoEvaluator,
    VideoOrchestrator,
)
from personalab.config import load_yaml
from personalab.prompts import load_personas, load_asset_scenarios

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gen_videos")


def _summarize_physical(phys: dict[str, Any]) -> str:
    """Turn a physical_description dict into a concise natural-language string."""
    if not phys:
        return ""
    parts = [
        f"{phys.get('age_range', '')}-year-old",
        f"{phys.get('ancestry_expression', {}).get('region', '')} person",
        f"skin: {phys.get('skin', {}).get('tone', '')}",
        f"hair: {phys.get('hair', {}).get('base_color', '')} {phys.get('hair', {}).get('wave_pattern', '')}",
        f"eyes: {phys.get('eyes', {}).get('color', '')} {phys.get('eyes', {}).get('shape', '')}",
        f"face: {phys.get('facial_structure', {}).get('face_shape', '')}",
        f"build: {phys.get('body', {}).get('build', '')}",
    ]
    return ", ".join(p for p in parts if not p.endswith(": "))


def main(persona: str, content_index: int = 0, *, skip_eval: bool = False) -> None:
    cfg = load_config()
    client = create_client(cfg)

    vg = VideoGenerator(
        client=client,
        model_name=cfg.model_name("video"),
        resolution=cfg.generation["video"]["resolution"],
        aspect_ratio=cfg.generation["video"]["aspect_ratio"],
        references_root=cfg.paths["references"],
    )

    plan_path = Path(cfg.paths["output"]) / persona / "content_plan.yml"
    if not plan_path.exists():
        log.error("Content plan not found: %s  -- run gen_ideas.py first", plan_path)
        sys.exit(1)

    content_plan = load_yaml(plan_path)
    persona_details = load_personas(cfg)
    asset_scenarios = load_asset_scenarios(cfg)

    planned = content_plan.get("planned_content", [])
    if content_index >= len(planned):
        log.error(
            "content_index=%d out of range (planned_content has %d items)",
            content_index, len(planned),
        )
        sys.exit(1)

    persona_info = persona_details.get(persona)
    if persona_info is None:
        log.error("Persona '%s' not found in persona definitions", persona)
        sys.exit(1)

    item = planned[content_index]
    scenario = item.get("scenario_details", {})

    video_prompt = VideoPrompt(
        subject_description=_summarize_physical(persona_info.get("physical_description", {})),
        action_details=scenario.get("action", item.get("custom_action", "")),
        location_details=scenario.get("scene", ""),
        mood_and_expression=scenario.get("expression", ""),
    )

    log.info("Persona: %s", persona)
    log.info("Content item [%d]: %s", content_index, item.get("title", "(untitled)"))
    log.info("Action: %s", video_prompt.action_details or "(none)")
    log.info("Location: %s", video_prompt.location_details or "(none)")

    eval_cfg = cfg.evaluation
    video_eval_cfg = eval_cfg.get("video", {})

    if skip_eval:
        log.info("Identity evaluation DISABLED (--no-eval)")
        video_evaluator = StubVideoEvaluator()
    else:
        evaluator = build_evaluator(eval_cfg)
        frame_extractor = FrameExtractor()
        video_evaluator = VideoIdentityEvaluator(
            evaluator=evaluator,
            frame_extractor=frame_extractor,
        )

    policy = IdentityPolicy.from_config(eval_cfg)

    orchestrator = VideoOrchestrator(
        video_generator=vg,
        video_evaluator=video_evaluator,
        policy=policy,
        drift_tracker=DriftTracker(log_dir=eval_cfg["drift"]["log_dir"]),
        metrics_logger=GenerationMetricsLogger(output_dir=cfg.paths["output"]),
        output_dir=cfg.paths["output"],
        references_root=cfg.paths["references"],
        max_frames=video_eval_cfg.get("max_frames", 8),
        frame_strategy=video_eval_cfg.get("frame_strategy", "uniform"),
    )

    result = orchestrator.run(
        persona=persona,
        scenario_template=asset_scenarios["video_scenarios"],
        video_prompt=video_prompt,
    )

    print()
    print("=" * 50)
    print(f"  Accepted:  {result.accepted}")
    print(f"  Score:     {result.final_score:.4f}")
    print(f"  Attempts:  {result.final_attempt}")
    if result.final_video_path:
        print(f"  Video:     {result.final_video_path}")
    print("=" * 50)

    for att in result.attempts:
        tag = att.decision.action if att.decision else "N/A"
        score = att.decision.score if att.decision else 0.0
        print(f"  attempt {att.attempt}: {tag} (score={score:.4f}, gen={att.generation_ms:.0f}ms, eval={att.evaluation_ms:.0f}ms)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate identity-conditioned video for a persona")
    parser.add_argument("persona", help="Persona key (e.g. daniperez)")
    parser.add_argument("--index", type=int, default=0, help="Content plan index (default: 0)")
    parser.add_argument("--no-eval", action="store_true", help="Skip identity evaluation (faster)")
    args = parser.parse_args()
    main(args.persona, args.index, skip_eval=args.no_eval)
