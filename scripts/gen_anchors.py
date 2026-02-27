import argparse
from pathlib import Path

from personalab import load_config, GeminiAdapter, ImageGenerator
from personalab.prompts import load_personas, load_anchor_templates


def main(persona: str = "daniperez") -> None:
    cfg = load_config()
    client = GeminiAdapter()

    anchor_templates = load_anchor_templates(cfg)
    personas = load_personas(cfg)
    physical = personas[persona]["physical_description"]

    img_gen = ImageGenerator(
        client=client,
        model_name=cfg.model_name("image"),
        aspect_ratio=cfg.generation["image"]["aspect_ratio"],
        references_root=cfg.paths["references"],
    )

    saved = img_gen.generate_reference_anchors(
        persona_name=persona,
        character_prompt=anchor_templates,
        physical_description=physical,
        frontal_candidates_k=8,
    )

    anchors_dir = Path(cfg.paths["references"]) / persona / "anchors"
    lock_path = anchors_dir / f"{persona}_anchor_frontal_neutral.png"

    if lock_path.exists():
        print(f"✅ Anchor lock exists: {lock_path}")
        print("Saved anchors:\n" + "\n".join(str(p) for p in saved))
    else:
        print("⚠️ No frontal lock found. Generated candidates only.")
        print(f"Pick ONE candidate and rename/copy it to:\n  {lock_path}\n")
        print("Candidates:\n" + "\n".join(str(p) for p in saved))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("persona", help="Persona key from personas.yml")
    args = parser.parse_args()
    main(args.persona)
