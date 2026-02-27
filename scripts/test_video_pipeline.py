"""Smoke-test: generate test videos for Dani Perez to validate the full video pipeline.

Self-contained -- does NOT require a content plan. Uses hardcoded IG-style
scenarios that exercise different video generation patterns.

Usage:
    python scripts/test_video_pipeline.py                  # all scenarios
    python scripts/test_video_pipeline.py --pick 0         # single scenario by index
    python scripts/test_video_pipeline.py --no-eval        # skip identity evaluation
    python scripts/test_video_pipeline.py --pick 0 --no-eval

Output goes to  ./generated_content/daniperez/video_tests/
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

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
from personalab.prompts import load_asset_scenarios, load_personas

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_video_pipeline")

# ── Hardcoded IG Reels scenarios for Dani Perez ──────────────────────────────

TEST_SCENARIOS = [
    {
        "title": "GRWM Morning Routine",
        "action": (
            "Dani picks up her phone from the nightstand, walks to the bathroom mirror, "
            "ties her hair in a messy bun, splashes water on her face, "
            "and smiles at the camera with a 'buenos días' expression"
        ),
        "location": (
            "Small bright Lima apartment bathroom, morning golden light through frosted window, "
            "white tiles, skincare bottles on shelf, slightly steamy mirror"
        ),
        "expression": "Sleepy transitioning to cheerful, natural yawn, then warm smile",
    },
    {
        "title": "Street Food Taste Test",
        "action": (
            "Dani walks through a busy street market, stops at a food cart, "
            "receives an anticucho skewer, takes a bite, closes eyes in delight, "
            "then looks at the camera and nods approvingly"
        ),
        "location": (
            "Bustling Lima street market at golden hour, colorful food carts, "
            "warm tungsten lights mixed with sunset, people walking in background, "
            "smoke rising from grills"
        ),
        "expression": "Curious and excited, then genuine pleasure while tasting food",
    },
    {
        "title": "Outfit Check in Mirror",
        "action": (
            "Dani steps back from a full-length mirror, does a slow turn showing her outfit, "
            "adjusts her jacket, fixes a strand of hair, "
            "then poses with a confident hand-on-hip stance facing the camera"
        ),
        "location": (
            "Cozy bedroom with warm afternoon light, full-length mirror leaning against wall, "
            "minimalist decor, a few plants, wooden floor, "
            "clothes draped casually on a chair in the background"
        ),
        "expression": "Self-assured, slight head tilt, subtle satisfied smile, confident eye contact",
    },
    {
        "title": "Walking Through Barranco",
        "action": (
            "Dani walks along a colorful street in Barranco, trailing her hand along a painted wall, "
            "turns to look at the camera over her shoulder with a playful grin, "
            "then continues walking while the camera follows from behind"
        ),
        "location": (
            "Barranco district Lima, vibrant street murals on old colonial walls, "
            "cobblestone sidewalk, bougainvillea hanging over walls, "
            "soft overcast afternoon light, a few people in the distance"
        ),
        "expression": "Playful, carefree, genuine laugh, eyes lit up",
    },
]


def _get_dani_description(cfg) -> str:
    """Build a concise natural-language appearance summary for Dani."""
    personas = load_personas(cfg)
    info = personas.get("daniperez", {})
    phys = info.get("physical_description", {})
    if not phys:
        return (
            "23-year-old Peruvian woman, light olive skin, dark blonde hair "
            "with loose natural waves, brown almond eyes, soft oval face, "
            "athletic natural build"
        )
    parts = [
        f"{phys.get('age_range', '23')}-year-old",
        f"{phys.get('ancestry_expression', {}).get('region', 'Peruvian')} woman",
        f"skin tone: {phys.get('skin', {}).get('tone', 'light olive')}",
        f"hair: {phys.get('hair', {}).get('base_color', 'dark blonde')} "
        f"{phys.get('hair', {}).get('wave_pattern', 'loose waves')}",
        f"eyes: {phys.get('eyes', {}).get('color', 'brown')} {phys.get('eyes', {}).get('shape', 'almond')}",
        f"face: {phys.get('facial_structure', {}).get('face_shape', 'oval')}",
        f"build: {phys.get('body', {}).get('build', 'athletic')}",
    ]
    return ", ".join(parts)


def run_scenario(
    *,
    index: int,
    scenario: dict,
    orchestrator: VideoOrchestrator,
    asset_template: dict,
    subject_desc: str,
) -> None:
    title = scenario["title"]
    log.info("-" * 60)
    log.info("Scenario %d: %s", index, title)
    log.info("-" * 60)

    vp = VideoPrompt(
        subject_description=subject_desc,
        action_details=scenario["action"],
        location_details=scenario["location"],
        mood_and_expression=scenario["expression"],
    )

    log.info("  Action:     %s", scenario["action"][:80] + "...")
    log.info("  Location:   %s", scenario["location"][:80] + "...")
    log.info("  Expression: %s", scenario["expression"][:60])

    t0 = time.time()
    result = orchestrator.run(
        persona="daniperez",
        scenario_template=asset_template,
        video_prompt=vp,
        output_prefix=f"test_{index:02d}",
    )
    elapsed = time.time() - t0

    print()
    status = "ACCEPTED" if result.accepted else "REJECTED"
    print(f"  [{status}]  score={result.final_score:.4f}  attempts={result.final_attempt}  time={elapsed:.1f}s")
    if result.final_video_path:
        size_mb = Path(result.final_video_path).stat().st_size / (1024 * 1024)
        print(f"  Video: {result.final_video_path}  ({size_mb:.1f} MB)")
    else:
        print("  No video produced.")
    print()


def main(*, pick: int | None = None, skip_eval: bool = False) -> None:
    cfg = load_config()
    client = create_client(cfg)

    output_dir = Path(cfg.paths["output"]) / "daniperez" / "video_tests"
    output_dir.mkdir(parents=True, exist_ok=True)

    vg = VideoGenerator(
        client=client,
        model_name=cfg.model_name("video"),
        resolution=cfg.generation["video"]["resolution"],
        aspect_ratio=cfg.generation["video"]["aspect_ratio"],
        references_root=cfg.paths["references"],
    )

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
    asset_scenarios = load_asset_scenarios(cfg)
    subject_desc = _get_dani_description(cfg)

    orchestrator = VideoOrchestrator(
        video_generator=vg,
        video_evaluator=video_evaluator,
        policy=policy,
        drift_tracker=DriftTracker(log_dir=eval_cfg["drift"]["log_dir"]),
        metrics_logger=GenerationMetricsLogger(output_dir=str(output_dir)),
        output_dir=str(output_dir),
        references_root=cfg.paths["references"],
        max_frames=video_eval_cfg.get("max_frames", 8),
        frame_strategy=video_eval_cfg.get("frame_strategy", "uniform"),
    )

    scenarios = TEST_SCENARIOS
    if pick is not None:
        if pick < 0 or pick >= len(scenarios):
            log.error("--pick %d out of range (0-%d)", pick, len(scenarios) - 1)
            sys.exit(1)
        scenarios = [scenarios[pick]]
        start_idx = pick
    else:
        start_idx = 0

    print()
    print("=" * 60)
    print("  Dani Perez -- Video Pipeline Smoke Test")
    print(f"  Model:     {cfg.model_name('video')}")
    print(f"  Scenarios: {len(scenarios)}")
    print(f"  Eval:      {'OFF (stub)' if skip_eval else 'ON (ArcFace + Geometric)'}")
    print(f"  Output:    {output_dir}")
    print("=" * 60)
    print()

    total_t0 = time.time()
    for i, scenario in enumerate(scenarios):
        if i > 0:
            delay = 15
            log.info("Pausing %ds between scenarios (rate limit)...", delay)
            time.sleep(delay)
        run_scenario(
            index=start_idx + i,
            scenario=scenario,
            orchestrator=orchestrator,
            asset_template=asset_scenarios["video_scenarios"],
            subject_desc=subject_desc,
        )

    total_elapsed = time.time() - total_t0
    print("-" * 60)
    print(f"Done. {len(scenarios)} scenario(s) in {total_elapsed:.1f}s")
    print(f"Videos saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smoke-test: generate IG Reels test videos for Dani Perez",
    )
    parser.add_argument(
        "--pick", type=int, default=None,
        help="Run only scenario N (0-3). Default: all.",
    )
    parser.add_argument(
        "--no-eval", action="store_true",
        help="Skip identity evaluation (faster, useful for first test run)",
    )
    args = parser.parse_args()
    main(pick=args.pick, skip_eval=args.no_eval)
