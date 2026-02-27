import argparse
from pathlib import Path

from personalab import load_config, GeminiAdapter, IdeaGenerator
from personalab.prompts import load_personas, load_system_prompts


def main(persona: str, output: str) -> None:
    cfg = load_config()
    personas = load_personas(cfg)
    system_prompts = load_system_prompts(cfg)

    client = GeminiAdapter()

    ig = IdeaGenerator(
        client=client,
        config=cfg,
        persona=personas[persona],
        system_prompts=system_prompts,
    )

    plan = ig.generate_and_save(output)
    print(f"Saved content plan to {output}")
    print(plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("persona")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    if args.output is None:
        cfg = load_config()
        args.output = str(Path(cfg.paths["output"]) / args.persona / "content_plan.yml")

    main(args.persona, args.output)