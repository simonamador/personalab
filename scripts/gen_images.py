import argparse
import json
from pathlib import Path

from personalab import (
    load_config,
    GeminiAdapter,
    ImageGenerator,
    ImagePrompt,
    apply_instagram_realism,
    load_realism_config,
)
from personalab.config import load_yaml
from personalab.prompts import load_personas, load_asset_scenarios
from personalab.tools.post.io import encode_jpeg


def main(persona: str, content_index: int = 0) -> None:
    cfg = load_config()
    client = GeminiAdapter()
    img_gen = ImageGenerator(
        client=client,
        model_name=cfg.model_name("image"),
        aspect_ratio=cfg.generation["image"]["aspect_ratio"],
        references_root=cfg.paths["references"],
    )

    content_plan = load_yaml(
        Path(cfg.paths["output"]) / persona / "content_plan.yml"
    )
    persona_details = load_personas(cfg)

    planned = content_plan.get("planned_content", [])
    if content_index >= len(planned):
        raise IndexError(
            f"content_index={content_index} out of range (planned_content has {len(planned)} items)"
        )

    persona_info = persona_details.get(persona)
    if persona_info is None:
        raise KeyError(f"Persona '{persona}' not found in persona definitions")

    scenario_details = planned[content_index]["scenario_details"]
    subject_desc = json.dumps(
        persona_info.get("physical_description", {}),
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    image_prompt = ImagePrompt.from_scenario_details(scenario_details, subject_description=subject_desc)

    asset_scenarios = load_asset_scenarios(cfg)
    print("Generating image for scenario", content_index, "...")
    img_bytes, meta, _ = img_gen.generate_asset_image(
        persona_name=persona,
        scenario_template=asset_scenarios["image_scenarios"],
        image_prompt=image_prompt,
    )
    if not img_bytes:
        raise RuntimeError("Image generation returned no bytes")

    realism_cfg = load_realism_config(cfg.raw)
    bgr = apply_instagram_realism(img_bytes, realism_cfg)
    final_bytes = encode_jpeg(bgr, realism_cfg.jpeg_quality)

    out_path = Path(cfg.paths["output"]) / persona / "generated_image.png"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    out_path.write_bytes(final_bytes)
    print("Saved image:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("persona")
    parser.add_argument("--index", type=int, default=0, help="Index into planned_content")
    args = parser.parse_args()
    main(args.persona, content_index=args.index)
