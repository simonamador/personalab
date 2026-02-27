# scripts/auto_generate_image.py
"""CLI script that generates an asset image with identity-aware retry.

Now delegates to GenerationOrchestrator for the evaluate-retry loop.
"""
import argparse
import json
from pathlib import Path

from personalab import (
    load_config,
    GeminiAdapter,
    ImageGenerator,
    ImagePrompt,
    IdentityPolicy,
    DriftTracker,
    GenerationMetricsLogger,
    GenerationOrchestrator,
    build_evaluator,
    select_anchor,
)
from personalab.config import load_yaml
from personalab.prompts import load_personas, load_asset_scenarios


def main(persona: str, content_index: int = 0) -> None:
    cfg = load_config()
    eval_cfg = cfg.evaluation
    client = GeminiAdapter()

    img_gen = ImageGenerator(
        client=client,
        model_name=cfg.model_name("image"),
        aspect_ratio=cfg.generation["image"]["aspect_ratio"],
        references_root=cfg.paths["references"],
    )

    evaluator = build_evaluator(eval_cfg)
    policy = IdentityPolicy.from_config(eval_cfg)
    drift = DriftTracker(log_dir=eval_cfg["drift"]["log_dir"])
    metrics = GenerationMetricsLogger(output_dir=cfg.paths["output"])

    # Auto-promote anchor if candidates exist but no anchor promoted yet
    select_anchor(persona, cfg.paths["references"])

    content_plan = load_yaml(
        Path(cfg.paths["output"]) / persona / "content_plan.yml"
    )
    persona_details = load_personas(cfg)
    asset_scenarios = load_asset_scenarios(cfg)

    planned = content_plan.get("planned_content", [])
    if content_index >= len(planned):
        raise IndexError(
            f"content_index={content_index} out of range (planned_content has {len(planned)} items)"
        )

    persona_info = persona_details.get(persona)
    if persona_info is None:
        raise KeyError(f"Persona '{persona}' not found in persona definitions")

    scenario = planned[content_index]["scenario_details"]
    subject_desc = json.dumps(
        persona_info.get("physical_description", {}),
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    image_prompt = ImagePrompt.from_scenario_details(scenario, subject_description=subject_desc)

    orchestrator = GenerationOrchestrator(
        image_generator=img_gen,
        evaluator=evaluator,
        policy=policy,
        drift_tracker=drift,
        metrics_logger=metrics,
        output_dir=cfg.paths["output"],
        references_root=cfg.paths["references"],
    )

    result = orchestrator.run(
        persona=persona,
        scenario_template=asset_scenarios["image_scenarios"],
        image_prompt=image_prompt,
        output_prefix="auto_asset",
    )

    if result.accepted:
        print(f"Accepted at attempt {result.final_attempt} (score={result.final_score:.4f})")
        print(f"  -> {result.final_image_path}")
    else:
        print(f"Not accepted after {result.final_attempt} attempts (score={result.final_score:.4f})")

    for att in result.attempts:
        cand = att.eval_result.first() if att.eval_result else None
        emb_str = f"emb={cand.embedding.cosine_similarity:.4f}" if cand and cand.embedding else "emb=N/A"
        geo_str = f"geo_err={cand.geometric.normalized_error:.4f}" if cand and cand.geometric else "geo=N/A"
        score_str = f"composite={cand.composite_score:.4f}" if cand else "score=N/A"
        action = att.decision.action if att.decision else "N/A"
        print(f"  Attempt {att.attempt}: {score_str}  {emb_str}  {geo_str}  -> {action}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("persona")
    parser.add_argument("--index", type=int, default=0, help="Index into planned_content")
    args = parser.parse_args()
    main(args.persona, content_index=args.index)
