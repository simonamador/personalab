"""
Quickstart: full workflow from character creation to identity-evaluated content.

Run from project root:
  python examples/quickstart.py

Requires:
  - GOOGLE_API_KEY in .env or environment
  - pip install -e ".[eval]"  (for identity evaluation; works without it using StubEvaluator)
"""

from pathlib import Path

from personalab import (
    Character,
    load_config,
    GeminiAdapter,
    IdeaGenerator,
    ImageGenerator,
    ImagePrompt,
    VideoGenerator,
    VideoPrompt,
    PassThroughPromptBuilder,
    select_anchor,
    build_evaluator,
    IdentityPolicy,
    GenerationOrchestrator,
    VideoOrchestrator,
    FrameExtractor,
    VideoIdentityEvaluator,
    DriftTracker,
    GenerationMetricsLogger,
)
from personalab.prompts import load_system_prompts, load_asset_scenarios, load_anchor_templates


def main() -> None:
    config = load_config()

    character = Character(
        name="Daniela 'Dani' Perez",
        vibe="Lifestyle & Urban Chic",
        location="Lima, Peru",
        content_pillars=[
            "Sustainable Andean Fashion",
            "Peruvian Gastronomy",
            "Hidden Urban Gems in Lima",
            "Travel",
        ],
        physical_description={"age_range": "23"},
        id="daniperez",
    )

    try:
        client = GeminiAdapter()
    except ValueError as e:
        print("LLM client:", e)
        return

    # -- Step 1: Generate anchor references --
    img_gen = ImageGenerator(
        client=client,
        model_name=config.model_name("image"),
        aspect_ratio=config.generation["image"]["aspect_ratio"],
        references_root=config.paths["references"],
    )
    anchor_templates = load_anchor_templates(config)
    anchors = img_gen.generate_reference_anchors(
        persona_name="daniperez",
        character_prompt=anchor_templates,
        physical_description=character.physical_description,
        frontal_candidates_k=4,
    )
    print(f"Generated {len(anchors)} anchor files")

    # -- Step 2: Auto-select the best anchor --
    anchor_path = select_anchor("daniperez", config.paths["references"])
    print(f"Selected anchor: {anchor_path}")

    # -- Step 3: Generate a content plan --
    system_prompts = load_system_prompts(config)
    idea_gen = IdeaGenerator(
        client=client,
        config=config,
        persona=character,
        system_prompts=system_prompts,
    )
    plan_path = Path(config.paths["output"]) / "daniperez" / "content_plan.yml"
    plan = idea_gen.generate_and_save(str(plan_path))
    print(f"Content plan: {len(plan.get('planned_content', []))} items -> {plan_path}")

    # -- Step 4: Generate with identity-aware retry --
    eval_cfg = config.evaluation
    evaluator = build_evaluator(eval_cfg)
    policy = IdentityPolicy.from_config(eval_cfg)

    orchestrator = GenerationOrchestrator(
        image_generator=img_gen,
        evaluator=evaluator,
        policy=policy,
        drift_tracker=DriftTracker(log_dir=eval_cfg["drift"]["log_dir"]),
        metrics_logger=GenerationMetricsLogger(output_dir=config.paths["output"]),
        output_dir=config.paths["output"],
        references_root=config.paths["references"],
    )

    planned = plan.get("planned_content", [])
    if planned:
        scenario = planned[0]["scenario_details"]
        image_prompt = ImagePrompt.from_scenario_details(
            scenario, subject_description="young Peruvian woman, 23, dark blonde hair",
        )
        result = orchestrator.run(
            persona="daniperez",
            scenario_template=load_asset_scenarios(config)["image_scenarios"],
            image_prompt=image_prompt,
        )
        print(f"Accepted: {result.accepted}, Score: {result.final_score:.4f}, "
              f"Attempts: {result.final_attempt}")

    # -- Step 5: Video generation with identity evaluation --
    asset_scenarios = load_asset_scenarios(config)
    vid_gen = VideoGenerator(
        client=client,
        model_name=config.model_name("video"),
        resolution=config.generation["video"]["resolution"],
        aspect_ratio=config.generation["video"]["aspect_ratio"],
        references_root=config.paths["references"],
    )

    frame_extractor = FrameExtractor()
    video_evaluator = VideoIdentityEvaluator(
        evaluator=evaluator,
        frame_extractor=frame_extractor,
    )

    video_orchestrator = VideoOrchestrator(
        video_generator=vid_gen,
        video_evaluator=video_evaluator,
        policy=policy,
        drift_tracker=DriftTracker(log_dir=eval_cfg["drift"]["log_dir"]),
        metrics_logger=GenerationMetricsLogger(output_dir=config.paths["output"]),
        output_dir=config.paths["output"],
        references_root=config.paths["references"],
    )

    if planned:
        scenario = planned[0]["scenario_details"]
        video_prompt = VideoPrompt.from_scenario_details(
            scenario,
            subject_description="young Peruvian woman, 23, dark blonde hair",
            custom_action=planned[0].get("custom_action", "walking through the market"),
        )
        vid_result = video_orchestrator.run(
            persona="daniperez",
            scenario_template=asset_scenarios["video_scenarios"],
            video_prompt=video_prompt,
        )
        print(f"Video accepted: {vid_result.accepted}, Score: {vid_result.final_score:.4f}, "
              f"Attempts: {vid_result.final_attempt}")

    # -- Step 6: Prompt builder (no LLM call) --
    builder = PassThroughPromptBuilder()
    prompt = builder.user_to_image_prompt(
        "beach selfie at sunset",
        context={"subject_description": "young woman, dark hair", "shot_type": "iPhone selfie"},
    )
    print(f"Prompt builder -> {prompt.to_variables().keys()}")


if __name__ == "__main__":
    main()
