import argparse
from pathlib import Path

from personalab import load_config, build_evaluator


def main(persona: str, asset_file: str) -> None:
    cfg = load_config()
    eval_cfg = cfg.evaluation
    evaluator = build_evaluator(eval_cfg)

    anchors_dir = Path(cfg.paths["references"]) / persona / "anchors"
    anchors = sorted(anchors_dir.glob("*.*"))

    if not anchors:
        raise FileNotFoundError("No anchors found")

    result = evaluator.evaluate(str(anchors[0]), [asset_file])
    cand = result.first()

    print(f"Composite score: {cand.composite_score:.4f}")
    print(f"OK: {cand.ok}")

    if cand.embedding:
        print(f"Embedding similarity: {cand.embedding.cosine_similarity:.4f} (threshold: {cand.embedding.threshold})")

    if cand.geometric:
        print(f"Geometric error: {cand.geometric.normalized_error:.4f} (max: {cand.geometric.max_allowed})")
        if cand.geometric.landmark_deltas:
            for name, delta in cand.geometric.landmark_deltas.items():
                print(f"  {name}: {delta:.4f}")

    if cand.failure_reasons:
        print(f"Failures: {cand.failure_reasons}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("persona")
    parser.add_argument("asset_file")
    args = parser.parse_args()
    main(args.persona, args.asset_file)
